#include <neural-graphics-primitives/nerf_device.cuh>

#include <synerfgine/nerf_world.h>

#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/reduce_sum.h>

namespace sng {

using namespace tcnn;

bool NerfWorld::handle(CudaDevice& device, const ivec2& resolution) {
    return true;
}

void NerfWorld::load_snapshot(const nlohmann::json& config, const fs::path& data_path = {}) {
	using namespace ngp;
	if (!data_path.empty() && data_path.exists()) {
		m_data_path = data_path;
	}

	const auto& snapshot = config["snapshot"];
	if (snapshot.value("version", 0) < SNAPSHOT_FORMAT_VERSION) {
		throw std::runtime_error{"Snapshot uses an old format and can not be loaded."};
	}

	if (!snapshot.contains("nerf") && snapshot.contains("mode") && mode_from_string(snapshot["mode"]) != ETestbedMode::Nerf) {
		throw std::runtime_error{"Unsupported (non-NeRF) snapshot mode. Snapshot must be regenerated with a new version of instant-ngp."};
	}

	m_aabb = snapshot.value("aabb", m_aabb);
	m_bounding_radius = snapshot.value("bounding_radius", m_bounding_radius);

	// if (m_testbed_mode == ETestbedMode::Nerf) {
	if (snapshot["density_grid_size"] != NERF_GRIDSIZE()) {
		throw std::runtime_error{"Incompatible grid size."};
	}

	m_nerf.training.counters_rgb.rays_per_batch = snapshot["nerf"]["rgb"]["rays_per_batch"];
	m_nerf.training.counters_rgb.measured_batch_size = snapshot["nerf"]["rgb"]["measured_batch_size"];
	m_nerf.training.counters_rgb.measured_batch_size_before_compaction = snapshot["nerf"]["rgb"]["measured_batch_size_before_compaction"];

	// If we haven't got a nerf dataset loaded, load dataset metadata from the snapshot
	// and render using just that.
	if (m_data_path.empty() && snapshot["nerf"].contains("dataset")) {
		m_nerf.training.dataset = snapshot["nerf"]["dataset"];
		// TODO: FIX
		// ngp::load_nerf(m_data_path);
	} else {
		if (snapshot["nerf"].contains("aabb_scale")) {
			m_nerf.training.dataset.aabb_scale = snapshot["nerf"]["aabb_scale"];
		}

		if (snapshot["nerf"].contains("dataset")) {
			m_nerf.training.dataset.n_extra_learnable_dims = snapshot["nerf"]["dataset"].value("n_extra_learnable_dims", m_nerf.training.dataset.n_extra_learnable_dims);
		}
	}

	m_nerf.rgb_activation = m_nerf.training.dataset.is_hdr ? ENerfActivation::Exponential : ENerfActivation::Logistic;

	m_nerf.training.n_images_for_training = (int)m_nerf.training.dataset.n_images;

	m_nerf.training.dataset.update_metadata();

	m_nerf.training.cam_pos_gradient.resize(m_nerf.training.dataset.n_images, vec3(0.0f));
	m_nerf.training.cam_pos_gradient_gpu.resize_and_copy_from_host(m_nerf.training.cam_pos_gradient);

	m_nerf.training.cam_exposure.resize(m_nerf.training.dataset.n_images, AdamOptimizer<vec3>(1e-3f));
	m_nerf.training.cam_pos_offset.resize(m_nerf.training.dataset.n_images, AdamOptimizer<vec3>(1e-4f));
	m_nerf.training.cam_rot_offset.resize(m_nerf.training.dataset.n_images, RotationAdamOptimizer(1e-4f));
	m_nerf.training.cam_focal_length_offset = AdamOptimizer<vec2>(1e-5f);

	m_nerf.training.cam_rot_gradient.resize(m_nerf.training.dataset.n_images, vec3(0.0f));
	m_nerf.training.cam_rot_gradient_gpu.resize_and_copy_from_host(m_nerf.training.cam_rot_gradient);

	m_nerf.training.cam_exposure_gradient.resize(m_nerf.training.dataset.n_images, vec3(0.0f));
	m_nerf.training.cam_exposure_gpu.resize_and_copy_from_host(m_nerf.training.cam_exposure_gradient);
	m_nerf.training.cam_exposure_gradient_gpu.resize_and_copy_from_host(m_nerf.training.cam_exposure_gradient);

	m_nerf.training.cam_focal_length_gradient = vec2(0.0f);
	m_nerf.training.cam_focal_length_gradient_gpu.resize_and_copy_from_host(&m_nerf.training.cam_focal_length_gradient, 1);

	m_nerf.reset_extra_dims(m_rng);
	m_nerf.training.optimize_extra_dims = m_nerf.training.dataset.n_extra_learnable_dims > 0;

	if (m_nerf.training.dataset.has_rays) {
		m_nerf.training.near_distance = 0.0f;
	}

	m_nerf.training.update_transforms();

	if (!m_nerf.training.dataset.metadata.empty()) {
		m_nerf.render_lens = m_nerf.training.dataset.metadata[0].lens;
		m_screen_center = vec2(1.f) - m_nerf.training.dataset.metadata[0].principal_point;
	}

	if (!is_pot(m_nerf.training.dataset.aabb_scale)) {
		throw std::runtime_error{fmt::format("NeRF dataset's `aabb_scale` must be a power of two, but is {}.", m_nerf.training.dataset.aabb_scale)};
	}

	int max_aabb_scale = 1 << (NERF_CASCADES()-1);
	if (m_nerf.training.dataset.aabb_scale > max_aabb_scale) {
		throw std::runtime_error{fmt::format(
			"NeRF dataset must have `aabb_scale <= {}`, but is {}. "
			"You can increase this limit by factors of 2 by incrementing `NERF_CASCADES()` and re-compiling.",
			max_aabb_scale, m_nerf.training.dataset.aabb_scale
		)};
	}

	m_aabb = BoundingBox{vec3(0.5f), vec3(0.5f)};
	m_aabb.inflate(0.5f * std::min(1 << (NERF_CASCADES()-1), m_nerf.training.dataset.aabb_scale));
	m_raw_aabb = m_aabb;
	m_render_aabb = m_aabb;
	m_render_aabb_to_local = m_nerf.training.dataset.render_aabb_to_local;
	if (!m_nerf.training.dataset.render_aabb.is_empty()) {
		m_render_aabb = m_nerf.training.dataset.render_aabb.intersection(m_aabb);
	}

	m_nerf.max_cascade = 0;
	while ((1 << m_nerf.max_cascade) < m_nerf.training.dataset.aabb_scale) {
		++m_nerf.max_cascade;
	}

	// Perform fixed-size stepping in unit-cube scenes (like original NeRF) and exponential
	// stepping in larger scenes.
	m_nerf.cone_angle_constant = m_nerf.training.dataset.aabb_scale <= 1 ? 0.0f : (1.0f / 256.0f);

	m_up_dir = m_nerf.training.dataset.up;

	GPUMemory<__half> density_grid_fp16 = snapshot["density_grid_binary"];
	m_nerf.density_grid.resize(density_grid_fp16.size());

	parallel_for_gpu(density_grid_fp16.size(), [density_grid=m_nerf.density_grid.data(), density_grid_fp16=density_grid_fp16.data()] __device__ (size_t i) {
		density_grid[i] = (float)density_grid_fp16[i];
	});

	if (m_nerf.density_grid.size() == NERF_GRID_N_CELLS() * (m_nerf.max_cascade + 1)) {
		// TODO: Port over.
		// update_density_grid_mean_and_bitfield(nullptr, m_nerf);
		// set devices dirty
	} else if (m_nerf.density_grid.size() != 0) {
		// A size of 0 indicates that the density grid was never populated, which is a valid state of a (yet) untrained model.
		throw std::runtime_error{"Incompatible number of grid cascades."};
	}

	// }

}

void sng::update_density_grid_mean_and_bitfield(cudaStream_t stream, Testbed::Nerf& m_nerf) {
	const uint32_t n_elements = NERF_GRID_N_CELLS();

	size_t size_including_mips = grid_mip_offset(NERF_CASCADES())/8;
	m_nerf.density_grid_bitfield.enlarge(size_including_mips);
	m_nerf.density_grid_mean.enlarge(reduce_sum_workspace_size(n_elements));

	CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.density_grid_mean.data(), 0, sizeof(float), stream));
	// reduce_sum(m_nerf.density_grid.data(), [n_elements] __device__ (float val) { return fmaxf(val, 0.f) / (n_elements); }, m_nerf.density_grid_mean.data(), n_elements, stream);

	// linear_kernel(grid_to_bitfield, 0, stream, n_elements/8 * NERF_CASCADES(), n_elements/8 * (m_nerf.max_cascade + 1), m_nerf.density_grid.data(), m_nerf.density_grid_bitfield.data(), m_nerf.density_grid_mean.data());

	// for (uint32_t level = 1; level < NERF_CASCADES(); ++level) {
	// 	linear_kernel(bitfield_max_pool, 0, stream, n_elements/64, m_nerf.get_density_grid_bitfield_mip(level-1), m_nerf.get_density_grid_bitfield_mip(level));
	// }
}

}