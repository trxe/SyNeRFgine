#pragma once

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <synerfgine/common.cuh>
#include <fmt/format.h>
#include <neural-graphics-primitives/testbed.h>
#include <vector>
#include <sstream>

namespace sng {

struct CamKeyframe {
    uint32_t id;
    vec3 view;
    vec3 at;
    float zoom;

    CamKeyframe(uint32_t id, vec3 view, vec3 at, float zoom) : id(id), view(view), at(at), zoom(zoom) {}

    CamKeyframe(uint32_t id, const nlohmann::json& config) : id(id) {
        auto a = config["view"];
        view = { a[0].get<float>(), a[1].get<float>(), a[2].get<float>() };
        a = config["at"];
        at = { a[0].get<float>(), a[1].get<float>(), a[2].get<float>() };
        zoom = config["zoom"].get<float>();
    }

    void interpolate(const CamKeyframe& next, uint32_t fid, uint32_t frames_per_interval, Testbed& testbed) const {
        float k = (float) (fid % frames_per_interval) / (float) frames_per_interval;
        float invk = 1.0 - k;
        auto _view = invk * view + k * next.view;
        testbed.set_view_dir(_view);
        auto _at = invk * at + k * next.at;
        testbed.set_look_at(_at);
        auto _scale = invk * zoom + k * next.zoom;
        testbed.set_scale(_scale);
    }

    std::string get_info() const {
        std::string fmtted_str = fmt::format("\"view\" : [ {}, {}, {} ], \"at\" : [ {}, {}, {} ], \"zoom\" : {}",
            view.r, view.g, view.b,
            at.r, at.g, at.b,
            zoom
        );
        return fmtted_str;
    }

    void imgui() {
        std::string label = fmt::format("Keyframe [{}]", id);
        std::string label_info = fmt::format("Info [{}]", id);
        std::string fmtted_str = get_info();
        if (ImGui::TreeNode(label.c_str())) {
            ImGui::InputTextMultiline(label_info.c_str(), fmtted_str.data(), fmtted_str.size() + 1, {50, 350});
            ImGui::TreePop();
        }
    }
};

class CamPath {
public:
    CamPath() {}

    void imgui(Testbed& testbed) {
        if (ImGui::Button("Add Keyframe")) {
            keyframes.emplace_back(
                static_cast<uint32_t>(keyframes.size()),
                testbed.view_dir(),
                testbed.look_at(),
                testbed.scale()
            );
            update_cam_path();
        }
        if (!is_playing && ImGui::Button("Play")) {
            is_playing = true;
        } else if (is_playing && ImGui::Button("Pause")) {
            is_playing = false;
        }

        // HAPPENS EVERY FRAME!!
        if (is_playing) advance_frame(testbed);

        if (ImGui::InputInt("Duration", &total_time_ms)) {
            update_cam_path();
        }
        if (ImGui::SliderInt("Current Frame", &current_frame, 0, total_frames)) {
            set_to_frame(testbed);
        }
        std::ostringstream keyframes_str;
        for (const auto& kf : keyframes) {
            keyframes_str << "{" << kf.get_info() << "},\n";
        }
        std::string str = keyframes_str.str();
        ImGui::InputTextMultiline("Keyframe Info", str.data(), str.size() + 1, {150, 100});
    }

    CamPath(const nlohmann::json& config) {
        uint32_t tid = 0;
        auto& frames_conf = config["frames"];
        for (auto& frame_conf : frames_conf) {
            keyframes.emplace_back(tid++, frame_conf);
        }
        total_time_ms = config["total_time_ms"].get<int>();
        if (config.count("fps")) {
            fps = config["fps"];
        }
        total_frames = total_time_ms / fps;
        frames_between_keyframes = total_frames / max((int)keyframes.size(), 1);
        if (config.count("move_on_start")) is_playing = config["move_on_start"].get<bool>();
        if (config.count("path")) {
            auto& path_conf = config["path"];
            for (const auto& p : path_conf) {
                keyframes.emplace_back(static_cast<uint32_t>(keyframes.size()), p);
            }
        }
    }

    void update(Testbed& testbed) {
        if (is_playing) advance_frame(testbed);
    }

private:
    void update_cam_path() {
        total_frames = total_time_ms / fps;
        frames_between_keyframes = total_frames / (uint32_t)max(keyframes.size(), (size_t)1);
    }

    void set_to_frame(Testbed& testbed) {
        current_keyframe = current_frame / frames_between_keyframes;
        uint32_t next_keyframe = current_keyframe + 1;
        if (next_keyframe >= keyframes.size()) {
            current_frame = 0;
            current_keyframe = 0;
            next_keyframe = 1;
        }
        keyframes[current_keyframe].interpolate(keyframes[next_keyframe], current_frame, frames_between_keyframes, testbed);
    }

    void advance_frame(Testbed& testbed) {
        current_frame = current_frame + 1;
        set_to_frame(testbed);
    }

    std::vector<CamKeyframe> keyframes;
    int total_time_ms{10000};
    int fps{24};
    int total_frames{0};
    int frames_between_keyframes{1};
    int current_frame{0};
    int current_keyframe{0};
    bool is_playing{false};
};

}