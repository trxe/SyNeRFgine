#include <synerfgine/virtual_object.cuh>

namespace sng {

static float g_vo_pos_bound = 2.0f;

VirtualObject::VirtualObject(uint32_t id, const nlohmann::json& config) 
    : id(id), pos(0.0), rot(mat3::identity()), scale(1.0) {
	std::string warn;
	std::string err;
    file_path = fs::path(config["file"].get<std::string>());
	scale = config.contains("scale") ? config["scale"].get<float>() : 1.0;
	g_vo_pos_bound = 2.0f * scale;
    auto prims_per_leaf = config.contains("primitives-per-leaf") ? config["primitives-per-leaf"].get<int>() : 4;
	if (config.contains("pos")) {
		auto a = config["pos"];
		pos = { a[0].get<float>(), a[1].get<float>(), a[2].get<float>() };
	}
	if (config.contains("rot")) {
		auto a = config["rot"];
		rot = { 
			a[0].get<float>(), a[1].get<float>(), a[2].get<float>(),
			a[3].get<float>(), a[4].get<float>(), a[5].get<float>(),
			a[6].get<float>(), a[7].get<float>(), a[8].get<float>(),
		};
	}
	if (config.contains("anim")) {
		auto a = config["anim"]["rot_center"];
		anim_rot_centre = { a[0].get<float>(), a[1].get<float>(), a[2].get<float>() };
		a = config["anim"]["rot_axis"];
		vec3 ax = { a[0].get<float>(), a[1].get<float>(), a[2].get<float>() };
		float angle = config["anim"]["rot_angle"];
		float cost = std::cos(angle);
		float sint = std::sin(angle);
		anim_next_rot = mat3{
			cost + ax.x * ax.x * (1.0f - cost),        ax.x * ax.y * (1.0f - cost) - ax.z * sint, ax.x * ax.z * (1.0f - cost) + ax.y * sint, 
			ax.x * ax.y * (1.0f - cost) + ax.z * sint, cost + ax.y * ax.y * (1.0f - cost),        ax.y * ax.z * (1.0f - cost) - ax.x * sint, 
			ax.z * ax.y * (1.0f - cost) - ax.y * sint, ax.z * ax.y * (1.0f - cost) + ax.x * sint, cost + ax.z * ax.z * (1.0f - cost) 
		};
	}

    name = fs::path(file_path).basename();
    material_id = config["material"].get<uint32_t>();

    if (!file_path.exists()) {
        throw std::runtime_error(fmt::format("Error loading file: {}", file_path.str()));
    }

	std::ifstream f{file_path.str(), std::ios::in | std::ios::binary};

	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, &f);

	if (!warn.empty()) {
		tlog::warning() << warn << " while loading '" << file_path.str() << "'";
	}

	if (!err.empty()) {
        throw std::runtime_error(fmt::format("Error loading file: {}", file_path.str()));
	}

	std::vector<vec3> result;

	tlog::success() << "Loaded mesh \"" << file_path.str() << "\" file with " << shapes.size() << " shapes.";

	vec3 center{0.0};
	uint32_t tri_count = 0;
    for (auto& shape : shapes) {
		auto& idxs = shape.mesh.indices;
		auto& verts = attrib.vertices;
		auto get_vec = [verts=verts, idxs=idxs](size_t i) {
			return vec3(
				verts[idxs[i].vertex_index * 3], 
				verts[idxs[i].vertex_index * 3 + 1], 
				verts[idxs[i].vertex_index * 3 + 2]
			);
		};
        for (size_t i = 0; i < shape.mesh.indices.size(); i += 3) {
            Triangle triangle {
                get_vec(i), get_vec(i+1), get_vec(i+2)
            };
			center += triangle.a + triangle.b + triangle.c;
			tri_count += 3;
			// std::cout << triangle << std::endl
            triangles_cpu.push_back(triangle);
        }
    }

	if (tri_count) center = center / (float)tri_count;
	triangles_bvh = TriangleBvh::make();
	// orig_triangles_gpu.resize_and_copy_from_host(triangles_cpu);
	// cam_triangles_gpu.resize_and_copy_from_host(triangles_cpu);
	triangles_bvh->build(triangles_cpu, prims_per_leaf);
	triangles_gpu.resize_and_copy_from_host(triangles_cpu);
	// TODO: build a bvh implementation that can be updated
}

void VirtualObject::imgui() {
	// std::string unique_pos = fmt::format("[{}] pos", id);
	std::string unique_scale = fmt::format("[{}] scale", id);
	std::string title = fmt::format("Object [{}]", id);
	std::string unique_pos = fmt::format("Vals [{}]", id);
	std::string unique_mat_id = fmt::format("[{}] mat id", id);
	std::string info = fmt::format(
		"\"pos\" : [ {:.3f}, {:.3f}, {:.3f} ],\n"
		"\"rot\" : [ {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f} ],\n"
		"\"scale\" : {:.3f},", 
		pos.x, pos.y, pos.z, 
		rot[0][0], rot[0][1], rot[0][2],
		rot[1][0], rot[1][1], rot[1][2],
		rot[2][0], rot[2][1], rot[2][2],
		scale);
	if (ImGui::TreeNode(title.c_str())) {
		ImGui::InputTextMultiline(unique_pos.c_str(), info.data(), info.size() + 1, {300, 50});
		if (ImGui::SliderFloat(unique_scale.c_str(), &this->scale, 0.0, g_vo_pos_bound)) { 
			is_dirty = true;
		}
		if (ImGui::InputInt(unique_mat_id.c_str(), (int*)&this->material_id)) {
			is_dirty = false;
		}
		ImGui::TreePop();
	}
	ImGui::Separator();
}

}