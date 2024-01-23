#include <synerfgine/nerf_world.h>

namespace sng {

// Returns network from file
json reload_network_from_file(const fs::path& path, bool& is_snapshot);

bool NerfWorld::handle(CudaDevice& device, const ivec2& resolution) {
    return true;
}

void NerfWorld::load_network(const fs::path& path) {
	if (!path.exists()) {
		tlog::error() << "File '" << path.str() << "' does not exist.";
		return;
	}

    bool is_snapshot;
    m_network_info = reload_network_from_file(path, is_snapshot);

	if (equals_case_insensitive(path.extension(), "ingp") || equals_case_insensitive(path.extension(), "msgpack")) {
		load_snapshot(m_network_info);
		return;
	}

	// If we get a json file, we need to parse it to determine its purpose.
	if (equals_case_insensitive(path.extension(), "json")) {
		json file;
		{
			std::ifstream f{native_string(path)};
			file = json::parse(f, nullptr, true, true);
		}

		// Snapshot in json format... inefficient, but technically supported.
		if (file.contains("snapshot")) {
            load_snapshot(m_network_info);
			return;
		}

		// Regular network config
		if (file.contains("parent") || file.contains("network") || file.contains("encoding") || file.contains("loss") || file.contains("optimizer")) {
            load_snapshot(m_network_info);
			return;
		}

		// Camera path
		if (file.contains("path")) {
            tlog::error() << "Camera path at '" << path.str() << "' is not supported.";
			return;
		}
	}

	// If the dragged file isn't any of the above, assume that it's training data
	try {
		bool was_training_data_available = m_training_data_available;
		load_training_data(path);

		if (!was_training_data_available) {
			// If we previously didn't have any training data and only now dragged
			// some into the window, it is very unlikely that the user doesn't
			// want to immediately start training on that data. So: go for it.
			m_train = true;
		}
	} catch (const std::runtime_error& e) {
		tlog::error() << "Failed to load training data: " << e.what();
	}
}

json reload_network_from_file(const fs::path& path, bool& is_snapshot) {
	fs::path full_network_config_path = find_network_config(path);
	is_snapshot = equals_case_insensitive(full_network_config_path.extension(), "msgpack");

	if (!full_network_config_path.exists()) {
        std::string msg = "Network path does not exist: " + full_network_config_path.string();
		throw std::runtime_error(msg);
	}

	// Reset training if we haven't loaded a snapshot of an already trained model, in which case, presumably the network
	// configuration changed and the user is interested in seeing how it trains from scratch.
    return load_network_config(full_network_config_path);
}

void NerfWorld::load_snapshot(nlohmann::json config) {
	const auto& snapshot = config["snapshot"];
	if (snapshot.value("version", 0) < SNAPSHOT_FORMAT_VERSION) {
		throw std::runtime_error{"Snapshot uses an old format and can not be loaded."};
	}

    bool is_nerf = mode_from_string(snapshot["mode"]) == ETestbedMode::Nerf || snapshot.contains("nerf");
	if (!is_nerf) {
		throw std::runtime_error{"Unsupported (or non-NeRF) snapshot. Snapshot must be regenerated with a new version of instant-ngp."};
	}

	m_aabb = snapshot.value("aabb", m_aabb);
	m_bounding_radius = snapshot.value("bounding_radius", m_bounding_radius);

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
        load_nerf(m_data_path);
    } else {
        if (snapshot["nerf"].contains("aabb_scale")) {
            m_nerf.training.dataset.aabb_scale = snapshot["nerf"]["aabb_scale"];
        }

        if (snapshot["nerf"].contains("dataset")) {
            m_nerf.training.dataset.n_extra_learnable_dims = snapshot["nerf"]["dataset"].value("n_extra_learnable_dims", m_nerf.training.dataset.n_extra_learnable_dims);
        }
    }

    load_nerf_post();

    GPUMemory<__half> density_grid_fp16 = snapshot["density_grid_binary"];
    m_nerf.density_grid.resize(density_grid_fp16.size());

    parallel_for_gpu(density_grid_fp16.size(), [density_grid=m_nerf.density_grid.data(), density_grid_fp16=density_grid_fp16.data()] __device__ (size_t i) {
        density_grid[i] = (float)density_grid_fp16[i];
    });

    if (m_nerf.density_grid.size() == NERF_GRID_N_CELLS() * (m_nerf.max_cascade + 1)) {
        update_density_grid_mean_and_bitfield(nullptr);
    } else if (m_nerf.density_grid.size() != 0) {
        // A size of 0 indicates that the density grid was never populated, which is a valid state of a (yet) untrained model.
        throw std::runtime_error{"Incompatible number of grid cascades."};
    }

	// Needs to happen after `load_nerf_post()`
	m_sun_dir = snapshot.value("sun_dir", m_sun_dir);
	m_exposure = snapshot.value("exposure", m_exposure);

	m_background_color = snapshot.value("background_color", m_background_color);

	if (snapshot.contains("camera")) {
		m_camera = snapshot["camera"].value("matrix", m_camera);
		m_fov_axis = snapshot["camera"].value("fov_axis", m_fov_axis);
		if (snapshot["camera"].contains("relative_focal_length")) from_json(snapshot["camera"]["relative_focal_length"], m_relative_focal_length);
		if (snapshot["camera"].contains("screen_center")) from_json(snapshot["camera"]["screen_center"], m_screen_center);
		m_zoom = snapshot["camera"].value("zoom", m_zoom);
		m_scale = snapshot["camera"].value("scale", m_scale);

		m_aperture_size = snapshot["camera"].value("aperture_size", m_aperture_size);
		if (m_aperture_size != 0) {
			m_dlss = false;
		}

		m_autofocus = snapshot["camera"].value("autofocus", m_autofocus);
		if (snapshot["camera"].contains("autofocus_target")) from_json(snapshot["camera"]["autofocus_target"], m_autofocus_target);
		m_slice_plane_z = snapshot["camera"].value("autofocus_depth", m_slice_plane_z);
	}

	if (snapshot.contains("render_aabb_to_local")) from_json(snapshot.at("render_aabb_to_local"), m_render_aabb_to_local);
	m_render_aabb = snapshot.value("render_aabb", m_render_aabb);
	if (snapshot.contains("up_dir")) from_json(snapshot.at("up_dir"), m_up_dir);

	m_network_config = std::move(config);

	reset_network(false);

	m_training_step = m_network_config["snapshot"]["training_step"];
	m_loss_scalar.set(m_network_config["snapshot"]["loss"]);

	m_trainer->deserialize(m_network_config["snapshot"]);

	if (m_testbed_mode == ETestbedMode::Nerf) {
		// If the snapshot appears to come from the same dataset as was already present
		// (or none was previously present, in which case it came from the snapshot
		// in the first place), load dataset-specific optimized quantities, such as
		// extrinsics, exposure, latents.
		if (snapshot["nerf"].contains("dataset") && m_nerf.training.dataset.is_same(snapshot["nerf"]["dataset"])) {
			if (snapshot["nerf"].contains("cam_pos_offset")) m_nerf.training.cam_pos_offset = snapshot["nerf"].at("cam_pos_offset").get<std::vector<AdamOptimizer<vec3>>>();
			if (snapshot["nerf"].contains("cam_rot_offset")) m_nerf.training.cam_rot_offset = snapshot["nerf"].at("cam_rot_offset").get<std::vector<RotationAdamOptimizer>>();
			if (snapshot["nerf"].contains("extra_dims_opt")) m_nerf.training.extra_dims_opt = snapshot["nerf"].at("extra_dims_opt").get<std::vector<VarAdamOptimizer>>();
			m_nerf.training.update_transforms();
			m_nerf.training.update_extra_dims();
		}
	}

	set_all_devices_dirty();
}

}