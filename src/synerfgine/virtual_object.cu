#include <synerfgine/virtual_object.cuh>

namespace sng {

static float g_vo_pos_bound = 1.0f;

VirtualObject::VirtualObject(uint32_t id, const nlohmann::json& config) 
    : id(id), pos(0.0), rot(mat3::identity()), scale(1.0) {
	std::string warn;
	std::string err;
    file_path = fs::path(config["file"].get<std::string>());
    auto a = config["pos"];
    auto prims_per_leaf = config["primitives-per-leaf"].get<int>();
    pos = { a[0].get<float>(), a[1].get<float>(), a[2].get<float>() };
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
    for(auto& shape : shapes) {
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
	std::string unique_pos = fmt::format("[{}] pos", id);
	std::string unique_scale = fmt::format("[{}] scale", id);
	std::string title = fmt::format("Object [{}]", id);
	if (ImGui::TreeNode(title.c_str())) {
		ImGui::InputFloat("Draggable bounds", &g_vo_pos_bound);
		if (ImGui::SliderFloat3(unique_pos.c_str(), pos.data(), -g_vo_pos_bound, g_vo_pos_bound)) { }
		if (ImGui::SliderFloat(unique_scale.c_str(), &this->scale, 0.0, g_vo_pos_bound)) { }
		ImGui::TreePop();
	}
	ImGui::Separator();
}

}