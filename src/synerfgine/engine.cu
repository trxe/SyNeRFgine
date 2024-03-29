#include <chrono>
#include <synerfgine/engine.cuh>
#include <synerfgine/common.cuh>
#include <filesystem/path.h>
#include <iostream>
#include <type_traits>
#include <imguizmo/ImGuizmo.h>

namespace sng {

void Engine::set_virtual_world(const std::string& config_fp) {
    nlohmann::json config = File::read_json(config_fp);
    if (config.count("camera")) {
        nlohmann::json& cam_conf = config["camera"];
        if (cam_conf.count("view")) {
            m_default_view_dir = {cam_conf["view"][0], cam_conf["view"][1], cam_conf["view"][2]};
        }
        if (cam_conf.count("at")) {
            m_default_at_pos = {cam_conf["at"][0], cam_conf["at"][1], cam_conf["at"][2]};
        }
        if (cam_conf.count("zoom")) {
            m_default_zoom = cam_conf["zoom"];
        }
        if (cam_conf.count("clear_color")) {
            m_default_clear_color = {cam_conf["clear_color"][0], cam_conf["clear_color"][1], cam_conf["clear_color"][2]};
        }
        if (cam_conf.count("show_ui_start")) {
            m_show_ui = cam_conf["show_ui_start"];
        }
        if (cam_conf.count("vo_scale")) {
            m_relative_vo_scale = cam_conf["vo_scale"];
        }
        if (cam_conf.count("animation_speed")) {
            m_anim_speed = cam_conf["animation_speed"];
            m_enable_animations = true;
        }
        if (cam_conf.count("probe_resolution")) {
            m_probe_resolution.y = m_probe_resolution.x = cam_conf["probe_resolution"];
        }
        if (cam_conf.count("path_trace_depth")) {
            m_raytracer.m_ray_iters = cam_conf["path_trace_depth"];
        }
        if (cam_conf.count("shadow_rays")) {
            m_raytracer.m_shadow_iters = cam_conf["shadow_rays"];
        }
        if (cam_conf.count("attenuation")) {
            m_raytracer.m_attenuation_coeff = cam_conf["attenuation"];
        }
    }
    if (config.count("output")) {
        nlohmann::json& output_conf = config["output"];
        m_output_dest = output_conf["folder"];
        if (output_conf.count("img_count")) {
            m_display.m_img_count_max = output_conf["img_count"];
        }
        if (output_conf.count("res_factor")) {
            m_factor_constant = output_conf["res_factor"];
        }
    }
    if (config.count("shader")) {
        nlohmann::json& shader_conf = config["shader"];
        if (shader_conf.count("nerf_blur_kernel_size")) {
            m_display.m_nerf_blur_kernel_size = shader_conf["nerf_blur_kernel_size"];
        }
        if (shader_conf.count("syn_blur_kernel_size")) {
            m_display.m_syn_blur_kernel_size =  shader_conf["syn_blur_kernel_size"];
        }
        if (shader_conf.count("syn_bsigma")) {
            m_display.m_syn_bsigma = shader_conf["syn_bsigma"];
        }
        if (shader_conf.count("syn_sigma")) {
            m_display.m_syn_sigma = shader_conf["syn_sigma"];
        }
        if (shader_conf.count("nerf_expand_mult")) {
            m_display.m_nerf_expand_mult = shader_conf["nerf_expand_mult"];
        }
        if (shader_conf.count("nerf_shadow_blur_threshold")) {
            m_display.m_nerf_shadow_blur_threshold = shader_conf["nerf_shadow_blur_threshold"];
        }
    }
    nlohmann::json& mat_conf = config["materials"];
    for (uint32_t i = 0; i < mat_conf.size(); ++i) {
        m_materials.emplace_back(i, mat_conf[i]);
    }
    nlohmann::json& obj_conf = config["objfile"];
    for (uint32_t i = 0; i < obj_conf.size(); ++i) {
        m_objects.emplace_back(i, obj_conf[i]);
        m_probes.emplace_back();
    }
    nlohmann::json& light_conf = config["lights"];
    for (uint32_t i = 0; i < light_conf.size(); ++i) {
        m_lights.emplace_back(i, light_conf[i]);
    }
}

void Engine::update_world_objects() {
    bool needs_reset = false;
    bool is_any_obj_dirty = false;
    for (auto& m : m_objects) {
        is_any_obj_dirty = is_any_obj_dirty || m.is_dirty;
        m.is_dirty = false;
    }
    if (is_any_obj_dirty || m_enable_animations) {
        needs_reset = true;
        std::vector<ObjectTransform> h_world;
        for (auto& obj : m_objects) {
            if (m_enable_animations) obj.next_frame(m_anim_speed);
            if (m_enable_reflections) {
                uint32_t obj_id = obj.get_id();
                auto& probe = m_probes[obj_id];
                const uint32_t padded_output_width = m_testbed->m_network->padded_output_width();
                const uint32_t n_extra_dimensions = m_testbed->m_nerf.training.dataset.n_extra_dims();
                const float depth_scale = 1.0f / m_testbed->m_nerf.training.dataset.scale;
                constexpr uint32_t target_n_queries = 2 * 1024 * 1024;
                uint32_t n_steps_between_compaction = clamp(target_n_queries / product(m_probe_resolution), (uint32_t)1, (uint32_t)8);
                probe.init_rays_in_sphere(
                    m_probe_resolution, 
                    obj.get_translate(), 
                    0, 
                    padded_output_width, n_extra_dimensions,
                    m_testbed->m_render_aabb,
                    m_testbed->m_render_aabb_to_local, 
                    m_testbed->m_nerf.density_grid_bitfield.data(),
                    m_testbed->m_nerf.max_cascade,
                    m_testbed->m_nerf.cone_angle_constant,
                    n_steps_between_compaction,
                    m_stream_id
                );
                vec2 focal_length = {}; // dummy
                auto n_hit = probe.trace_alt(
                    m_testbed->m_nerf_network,
                    m_testbed->m_render_aabb,
                    m_testbed->m_render_aabb_to_local,
                    m_testbed->m_aabb,
                    focal_length,
                    m_testbed->m_nerf.cone_angle_constant,
                    m_testbed->m_nerf.density_grid_bitfield.data(),
                    m_testbed->m_render_mode,
                    m_testbed->m_camera,
                    depth_scale,
                    m_testbed->m_visualized_layer,
                    m_testbed->m_visualized_dimension,
                    m_testbed->m_nerf.rgb_activation,
                    m_testbed->m_nerf.density_activation,
                    m_testbed->m_nerf.show_accel,
                    m_testbed->m_nerf.max_cascade,
                    m_testbed->m_nerf.render_min_transmittance,
                    m_testbed->m_nerf.glow_y_cutoff,
                    m_testbed->m_nerf.glow_mode,
                    m_testbed->m_nerf.get_rendering_extra_dims(m_stream_id),
                    m_stream_id
                );
                CudaRenderBufferView view = probe.m_render_buffer.view();
                probe.shade(
                    n_hit,
                    depth_scale,
                    view,
                    m_stream_id
                );
            }
            h_world.emplace_back(obj.gpu_node(), obj.gpu_triangles(), obj.get_rotate(), 
                obj.get_translate(), obj.get_scale(), obj.get_mat_idx());
        }
        d_world.check_guards();
        d_world.resize_and_copy_from_host(h_world);
    }
    is_any_obj_dirty = false;
    for (auto& m : m_materials) {
        is_any_obj_dirty = is_any_obj_dirty || m.is_dirty;
        m.is_dirty = false;
    }
    if (is_any_obj_dirty) {
        needs_reset = true;
        d_materials.check_guards();
        d_materials.resize_and_copy_from_host(m_materials);
    }
    is_any_obj_dirty = false;
    for (auto& l : m_lights) {
        is_any_obj_dirty = is_any_obj_dirty || l.is_dirty;
        l.is_dirty = false;
    }
    if (is_any_obj_dirty) {
        needs_reset = true;
        d_lights.check_guards();
        d_lights.resize_and_copy_from_host(m_lights);
    }
    if (needs_reset && m_testbed) {
        m_testbed->reset_accumulation();
    }
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
}

void Engine::init(int res_width, int res_height, const std::string& frag_fp, Testbed* nerf) {
	m_testbed = nerf;
    m_testbed->m_train = false;
    m_testbed->set_n_views(1);
    m_next_frame_resolution = {res_width, res_height};
    GLFWwindow* glfw_window = m_display.init_window(res_width, res_height, frag_fp);
    glfwSetWindowUserPointer(glfw_window, this);
    glfwSetWindowSizeCallback(glfw_window, [](GLFWwindow* window, int width, int height) {
        Engine* engine = (Engine*)glfwGetWindowUserPointer(window);
        if (engine) {
            engine->m_next_frame_resolution = {width, height};
            engine->redraw_next_frame();
        }
    });
    glfwSetWindowCloseCallback(glfw_window, [](GLFWwindow* window) {
        Engine* engine = (Engine*)glfwGetWindowUserPointer(window);
        if (engine) { engine->set_dead(); }
    });
    Testbed::CudaDevice& device = m_testbed->primary_device();
    if (length2(m_default_view_dir) != 0.0f) {
        m_testbed->set_view_dir(m_default_view_dir);
        m_testbed->set_look_at(m_default_at_pos);
        m_testbed->set_scale(m_default_zoom);
    }
    m_stream_id = device.stream();
    m_testbed->m_imgui.enabled = m_show_ui;
}

void Engine::resize() {
    m_display.set_window_res(m_next_frame_resolution);
    m_testbed->m_window_res = m_next_frame_resolution;
    auto& view = nerf_render_buffer_view();
    m_last_target_fps = m_testbed->m_dynamic_res_target_fps;
    float factor = min (1.0f, m_factor_constant / m_testbed->m_dynamic_res_target_fps);
    // tlog::success() << "Scaling full resolution by " << factor;
    auto new_res = downscale_resolution(m_next_frame_resolution, factor);
    view.resize(new_res);
    d_rand_state.resize(product(new_res));
    linear_kernel(init_rand_state, 0, m_stream_id, d_rand_state.size(), d_rand_state.data());
    sync(m_stream_id);

    m_raytracer.enlarge(min(scale_resolution(new_res, m_relative_vo_scale), m_next_frame_resolution));
}

void Engine::imgui() {
    auto& io = ImGui::GetIO();
    if (ImGui::IsKeyPressed(ImGuiKey_Tab)) {
        m_show_ui = !m_show_ui;
    }
    if (m_show_ui) {
        if (ImGui::Begin("Synthetic World")) {
            if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (ImGui::Button("Reset Camera") && length2(m_default_view_dir) != 0.0f) {
                    m_testbed->set_view_dir(m_default_view_dir);
                    m_testbed->set_look_at(m_default_at_pos);
                    m_testbed->set_scale(m_default_zoom);
                }
            }
            if (ImGui::CollapsingHeader("Materials", ImGuiTreeNodeFlags_DefaultOpen)) {
                for (auto& m : m_materials) { m.imgui(); }
            }
            if (ImGui::CollapsingHeader("Objects", ImGuiTreeNodeFlags_DefaultOpen)) {
                for (auto& m : m_objects) { m.imgui(); }
            }
            if (ImGui::CollapsingHeader("Lights", ImGuiTreeNodeFlags_DefaultOpen)) {
                for (auto& m : m_lights) { m.imgui(); }
            }
            m_raytracer.imgui();
            if (ImGui::CollapsingHeader("NeRF", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Checkbox("View Virtual Object shadows on NeRF", &m_view_syn_shadow);
                if (ImGui::SliderFloat("NeRF Epsilon offset", &m_depth_epsilon_shadow, 0.0, 0.1)) {
                    m_is_dirty = true;
                }
                auto& view = m_testbed->m_views.front();
                int max_scale = m_display.get_window_res().x / max(1, view.render_buffer->out_resolution().x);
                if (ImGui::SliderFloat("Relative scale of Virtual Scene", &m_relative_vo_scale, 0.5, max_scale)) {
                    resize();
                }
            }
            if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
                if (ImGui::Combo("Mode", (int*)&m_transform_type, world_object_names, sizeof(world_object_names) / sizeof(const char*))) {
                    m_transform_idx = 0;
                }
                int max_count = 0;
                switch (m_transform_type) {
                case WorldObjectType::LightObj:
                    max_count = m_lights.size() - 1;
                    break;
                case WorldObjectType::VirtualObjectObj:
                    max_count = m_objects.size() - 1;
                    break;
                default:
                    break;
                }
                ImGui::SliderInt("Obj idx", (int*)&m_transform_idx, 0, max_count);
            }
        }
        if (ImGui::CollapsingHeader("Shader", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderInt("nerf_expand_mult", &m_display.m_nerf_expand_mult, 0, 20);
            ImGui::SliderInt("nerf_blur_kernel_size", &m_display.m_nerf_blur_kernel_size, 0, 20);
            ImGui::SliderInt("syn_blur_kernel_size", &m_display.m_syn_blur_kernel_size, 0, 20);
            ImGui::SliderFloat("nerf_shadow_blur_threshold", &m_display.m_nerf_shadow_blur_threshold, 0.0, 1.0);
            ImGui::SliderFloat("syn_sigma", &m_display.m_syn_sigma, 1.0, 32.0);
            ImGui::SliderFloat("syn_bsigma", &m_display.m_syn_bsigma, 0.0, 4.0);
        }
        if (ImGui::CollapsingHeader("Animation", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat("speed", &m_anim_speed, 0.0, 4.0);
        }
        ImGui::End();
    }

    if (m_transform_type == WorldObjectType::LightObj && !m_lights.empty()) { 
        m_pos_to_translate = &(m_lights[m_transform_idx].pos);
        m_rot_to_rotate = nullptr;
        m_scale_to_scale = nullptr;
        m_obj_dirty_marker = &(m_lights[m_transform_idx].is_dirty);
    } else if (m_transform_type == WorldObjectType::VirtualObjectObj && !m_objects.empty()) {
        m_pos_to_translate = &(m_objects[m_transform_idx].get_translate_mut());
        m_rot_to_rotate = &(m_objects[m_transform_idx].get_rotate_mut());
        m_scale_to_scale = &(m_objects[m_transform_idx].get_scale_mut());
        m_obj_dirty_marker = &(m_objects[m_transform_idx].is_dirty);
    } else {
        m_pos_to_translate = nullptr;
        m_rot_to_rotate = nullptr;
        m_scale_to_scale = nullptr;
        m_obj_dirty_marker = nullptr;
    }
}

bool Engine::frame() {
    if (!m_display.is_alive()) return false;
    Testbed::CudaDevice& device = m_testbed->primary_device();
    device.device_guard();
    m_display.begin_frame();
    imgui();
    ivec2 curr_window_res = m_display.get_window_res();
    if (curr_window_res != m_next_frame_resolution || !m_testbed->m_render_skip_due_to_lack_of_camera_movement_counter || 
            m_last_target_fps != m_testbed->m_dynamic_res_target_fps) {
        resize();
    }
    sync(m_stream_id);
    m_testbed->handle_user_input();
    ImDrawList* list = ImGui::GetBackgroundDrawList();
    m_testbed->draw_visualizations(list, m_testbed->m_smoothed_camera, m_pos_to_translate, m_rot_to_rotate, m_scale_to_scale, m_obj_dirty_marker);
    update_world_objects();
    m_testbed->apply_camera_smoothing(__timer.get_ave_time("nerf"));

    auto& view = nerf_render_buffer_view();

    auto nerf_view = view.render_buffer->view();
    __timer.reset();

    m_testbed->render( m_stream_id, view, d_world, d_lights, d_rand_state, m_view_syn_shadow, m_depth_epsilon_shadow);

    vec2 focal_length = m_testbed->calc_focal_length(
        m_raytracer.resolution(),
        m_testbed->m_relative_focal_length, 
        m_testbed->m_fov_axis, 
        m_testbed->m_zoom);
    vec2 screen_center = m_testbed->render_screen_center(view.screen_center);
    m_raytracer.render(
        m_objects,
        m_probes,
        d_materials,
        d_lights,
        view, 
        screen_center,
        nerf_view.spp,
        focal_length,
        m_testbed->m_snap_to_pixel_centers,
        m_testbed->m_nerf.density_grid_bitfield.data(),
        d_world,
        m_enable_reflections
    );

    sync(m_stream_id);
    view.prev_camera = view.camera0;
    view.prev_foveation = view.foveation;

    ivec2 nerf_res = nerf_view.resolution;
    auto n_elements = product(nerf_res);
    m_render_ms = (float)__timer.log_time("nerf");
    m_testbed->m_frame_ms.set(m_render_ms);
    m_testbed->m_rgba_render_textures.front()->load_gpu(nerf_view.frame_buffer, nerf_view.resolution, m_nerf_rgba_cpu);
    m_testbed->m_depth_render_textures.front()->load_gpu(nerf_view.depth_buffer, nerf_view.resolution, 1, m_nerf_depth_cpu);

    m_raytracer.load(m_syn_rgba_cpu, m_syn_depth_cpu);
    GLuint nerf_rgba_texid = m_testbed->m_rgba_render_textures.front()->texture();
    GLuint nerf_depth_texid = m_testbed->m_depth_render_textures.front()->texture();
    GLuint syn_rgba_texid = m_raytracer.m_rgba_texture->texture();
    GLuint syn_depth_texid = m_raytracer.m_depth_texture->texture();
    auto rt_res = m_raytracer.resolution();
    m_display.present(m_default_clear_color, nerf_rgba_texid, nerf_depth_texid, syn_rgba_texid, syn_depth_texid, view.render_buffer->out_resolution(), rt_res, view.foveation, m_raytracer.filter_type());
    if (has_output()) {
        auto fp = m_output_dest.str();
        m_display.save_image(fp.c_str());
    }
    return m_display.is_alive();
}

}