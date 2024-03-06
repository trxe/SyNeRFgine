#include <synerfgine/camera.cuh>
#include <synerfgine/input.h>

#include <tiny-cuda-nn/common.h>

namespace sng {

bool Camera::handle_mouse_wheel() {
	bool is_moved = false;
	if (ImGui::IsMouseDragging(0) && ImGui::IsKeyDown(ImGuiKey_LeftShift)) {
		float delta = ImGui::GetMouseDragDelta(0).y / (float)m_resolution[m_fov_axis];
		// float scale_factor = pow(2.5f, -delta * 10.0f);
		set_scale(m_scale - delta);
		is_moved = true;
	}
	return is_moved;
}

bool Camera::handle_mouse_drag() {
	vec2 rel = vec2{ImGui::GetIO().MouseDelta.x, ImGui::GetIO().MouseDelta.y} / (float)m_resolution[m_fov_axis];
	vec2 mouse = {ImGui::GetMousePos().x, ImGui::GetMousePos().y};

	vec3 side = m_camera[0];
	bool is_moved = false;

	// Left held
	if (ImGui::GetIO().MouseDown[0] && !ImGui::IsKeyDown(ImGuiKey_LeftShift)) {
		float rot_sensitivity = m_fps_camera ? 0.35f : 1.0f;
		mat3 rot = rotation_from_angles(-rel * 2.0f * PI() * rot_sensitivity);

		rot *= mat3(m_camera);
		m_camera = mat4x3(rot[0], rot[1], rot[2], m_camera[3]);
		is_moved = true;
	}

	// Middle held
	if (ImGui::GetIO().MouseDown[2]) {
		vec3 translation = vec3{-rel.x, -rel.y, 0.0f} / m_zoom;

		// If we have a valid depth value, scale the scene translation by it such that the
		// hovered point in 3D space stays under the cursor.
		if (m_drag_depth < 256.0f) {
			translation *= m_drag_depth / m_relative_focal_length[m_fov_axis];
		}

		translate_camera(translation, mat3(m_camera));
		is_moved = true;
	}
	return is_moved;
}

bool Camera::handle_user_input() {
	if (ImGui::IsAnyItemActive() || ImGui::GetIO().WantCaptureMouse) {
		return false;
	}
	bool is_wheeled = handle_mouse_wheel();
	bool is_dragged = handle_mouse_drag();
	is_buffer_outdated = true;
	return is_dragged || is_wheeled;
	// return is_dragged;
}

void Camera::translate_camera(const vec3& rel, const mat3& rot, bool allow_up_down) {
	vec3 movement = rot * rel;
	if (!allow_up_down) {
		movement -= dot(movement, m_up_dir) * m_up_dir;
	}

	m_camera[3] += movement;
}

mat3 Camera::rotation_from_angles(const vec2& angles) const {
	vec3 up = m_up_dir;
	vec3 side = m_camera[0];
	return rotmat(angles.x, up) * rotmat(angles.y, side);
}

vec3 Camera::look_at() const {
	return view_pos() + view_dir() * m_scale;
}

void Camera::set_look_at(const vec3& pos) {
	m_camera[3] += pos - look_at();
}

void Camera::set_scale(float scale) {
	auto prev_look_at = look_at();
	m_camera[3] = m_default_camera[3] + scale * view_dir();
	m_scale = scale;
}

void Camera::set_default_matrix(const mat4x3& matrix) {
	m_default_camera = m_camera = matrix;
}

void Camera::set_view_dir(const vec3& dir) {
	auto old_look_at = look_at();
	m_camera[0] = normalize(cross(dir, m_up_dir));
	m_camera[1] = normalize(cross(dir, m_camera[0]));
	m_camera[2] = normalize(dir);
	set_look_at(old_look_at);
}

void Camera::reset_camera() {
	m_fov_axis = camera_default::fov_axis;
	m_zoom = camera_default::zoom;
	m_screen_center = camera_default::screen_center;

    set_fov(50.625f);
    m_scale = camera_default::scale;

	m_camera = m_default_camera;

	m_camera[3] -= m_scale * view_dir();
}

float Camera::fov() const {
	return focal_length_to_fov(1.0f, m_relative_focal_length[m_fov_axis]);
}

void Camera::set_fov(float val) {
	m_relative_focal_length = vec2(fov_to_focal_length(1, val));
}

vec2 Camera::fov_xy() const {
	return focal_length_to_fov(ivec2(1), m_relative_focal_length);
}

void Camera::set_fov_xy(const vec2& val) {
	m_relative_focal_length = fov_to_focal_length(ivec2(1), val);
}

vec2 Camera::calc_focal_length(const ivec2& resolution, const vec2& relative_focal_length, int fov_axis, float zoom) const {
	return relative_focal_length * (float)resolution[fov_axis] * zoom;
}

vec2 Camera::render_screen_center(const vec2& screen_center) const {
	// see pixel_to_ray for how screen center is used; 0.5, 0.5 is 'normal'. we flip so that it becomes the point in the original image we want to center on.
	return (0.5f - screen_center) * m_zoom + 0.5f;
}

void Camera::imgui() {
	if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
		if (ImGui::Button("Reset Camera")) {
			reset_camera();
		}
		if (ImGui::SliderFloat("Cam Scale", &m_scale, -20.0, 20.0)) {
			set_scale(m_scale);
		}
		if (ImGui::TreeNode("Cam Info")) {
			auto rd = view_pos();
			ImGui::Text("View Pos: %f, %f, %f", rd.r, rd.g, rd.b);
			rd = view_dir();
			ImGui::Text("View Dir: %f, %f, %f", rd.r, rd.g, rd.b);
			rd = look_at();
			ImGui::Text("Look At: %f, %f, %f", rd.r, rd.g, rd.b);
            ImGui::TreePop();
		}
	}
}


}