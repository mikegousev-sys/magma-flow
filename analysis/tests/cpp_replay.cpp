// Воспроизведение записи через НЕИЗМЕНЁННЫЕ алгоритмы C++ magma_vision (code/*.hpp).
//
// Повторяет RunBurst() из magma_vision.cpp без камеры: на каждую серию новый FlowEstimator,
// профиль вдоль потока с каждого кадра, кромки уровня с каждого level_every-го кадра,
// CombineLevelObservations (медиана), отчёт FormatVisionReport. Настройки читаются тем же
// ConfigFile и теми же ключами/умолчаниями, что в ApplySchedule/ApplyGeometryAndThresholds/
// ApplyStrips/ApplyCalibration (ручная калибровка). Кадры приходят со stdin от cpp_feed.py:
//   'B' uint32 n   - начало серии из n кадров
//   uint32 w, uint32 h, double t, w*h байт  - кадр (серый) и его время, с
//
//   g++ -std=c++17 -O2 -I../../code cpp_replay.cpp -o cpp_replay
//   python3 cpp_feed.py video.mkv --burst 120 | ./cpp_replay magma_vision.conf

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "magma_config.hpp"
#include "magma_edge.hpp"
#include "magma_flow.hpp"
#include "magma_level.hpp"
#include "magma_vision_protocol.hpp"

using namespace magma;

static bool ReadExact(void* p, std::size_t n) { return std::fread(p, 1, n, stdin) == n; }

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "cpp_replay <magma_vision.conf>\n");
        return 2;
    }
    ConfigFile config;
    if (!config.Load(argv[1])) {
        std::fprintf(stderr, "не прочитан %s\n", argv[1]);
        return 2;
    }
    // --- как ApplySchedule / ApplyGeometryAndThresholds / ApplyStrips / ApplyCalibration ---
    const std::size_t burst_frames = config.GetSize("schedule.burst_frames", 120);
    const std::size_t level_frames = config.GetSize("schedule.level_frames", 9);
    const std::size_t warmup_frames = config.GetSize("schedule.warmup_frames", 3);

    LauncherGeometry geometry;
    geometry.channel_radius_mm = config.GetDouble("geometry.channel_radius_mm", 150.0);
    geometry.max_level_mm = config.GetDouble("geometry.max_level_mm", 105.0);
    LevelConfig level_config;
    level_config.min_sharpness = config.GetDouble("level.min_sharpness", 5.0);
    level_config.melt_present_brightness = config.GetDouble("level.melt_present_brightness", 40.0);
    level_config.outlier_sigmas = config.GetDouble("level.outlier_sigmas", 3.0);
    level_config.edge_sigma_px = config.GetDouble("level.edge_sigma_px", 0.2);
    level_config.systematic_sigma_mm = config.GetDouble("level.systematic_sigma_mm", 0.5);
    const double center_reference_px = config.GetDouble("level.center_reference_px", 0.0);
    EdgeConfig edge_config;
    edge_config.smooth_radius = config.GetSize("edge.smooth_radius", 3);
    edge_config.min_gradient = config.GetDouble("edge.min_gradient", 3.0);
    edge_config.margin_fraction = config.GetDouble("edge.margin_fraction", 0.05);
    edge_config.min_separation = config.GetSize("edge.min_separation", 20);
    FlowConfig flow_config;   // mm_per_px в magma_vision.cpp не задаётся - остаётся 0.5333
    flow_config.min_speed_ms = config.GetDouble("flow.min_speed_ms", 0.4);
    flow_config.max_speed_ms = config.GetDouble("flow.max_speed_ms", 5.0);
    flow_config.velocity_steps = config.GetSize("flow.velocity_steps", 400);
    flow_config.max_interval_frames = config.GetSize("flow.max_interval_frames", 10);
    flow_config.background_tau_s = config.GetDouble("flow.background_tau_s", 3.0);
    flow_config.min_snr = config.GetDouble("flow.min_snr", 4.0);
    flow_config.snr_exclusion = config.GetInt("flow.snr_exclusion", 8);
    flow_config.min_profiles = config.GetSize("flow.min_profiles", 20);

    StripDefinition level_strip;
    level_strip.center_x_px = config.GetDouble("level.strip.center_x_px", 400.0);
    level_strip.center_y_px = config.GetDouble("level.strip.center_y_px", 512.0);
    level_strip.across_angle_deg = config.GetDouble("level.strip.angle_deg", 90.0);
    level_strip.length_px = config.GetDouble("level.strip.length_px", 500.0);
    level_strip.average_px = config.GetDouble("level.strip.average_px", 40.0);
    level_strip.samples = config.GetSize("level.strip.samples", 500);
    level_strip.average_lines = config.GetSize("level.strip.average_lines", 9);
    FlowStrip flow_strip;
    flow_strip.center_x_px = config.GetDouble("flow.strip.center_x_px", 640.0);
    flow_strip.center_y_px = config.GetDouble("flow.strip.center_y_px", 512.0);
    flow_strip.along_angle_deg = config.GetDouble("flow.strip.angle_deg", 0.0);
    flow_strip.length_px = config.GetDouble("flow.strip.length_px", 900.0);
    flow_strip.average_px = config.GetDouble("flow.strip.average_px", 60.0);
    flow_strip.samples = config.GetSize("flow.strip.samples", 900);
    flow_strip.average_lines = config.GetSize("flow.strip.average_lines", 15);

    CameraCalibration calibration;   // ручной путь
    calibration.elevation_deg = config.GetDouble("calibration.beta_deg", 31.0);
    calibration.scale_mm_per_px = config.GetDouble("calibration.scale_mm_per_px", 0.4135);
    calibration.flow_angle_deg = config.GetDouble("calibration.flow_angle_deg", 0.0);
    calibration.valid = true;

    const std::size_t level_every =
        std::max<std::size_t>(1, burst_frames / std::max<std::size_t>(1, level_frames));
    std::fprintf(stderr, "серия %zu кадров, прогрев %zu, уровень каждые %zu; полоса скорости "
                 "(%.0f,%.0f) угол %.1f длина %.0f ширина %.0f; калибровка β=%.1f° %.4f мм/px, "
                 "FlowConfig.mm_per_px=%.4f\n",
                 burst_frames, warmup_frames, level_every, flow_strip.center_x_px, flow_strip.center_y_px,
                 flow_strip.along_angle_deg, flow_strip.length_px, flow_strip.average_px,
                 calibration.elevation_deg, calibration.scale_mm_per_px, flow_config.mm_per_px);

    std::vector<uint8_t> buffer;
    char tag;
    while (ReadExact(&tag, 1) && tag == 'B') {
        uint32_t n = 0;
        if (!ReadExact(&n, 4)) break;
        FlowEstimator flow(flow_config);
        EdgeFinder edges(edge_config);
        std::vector<EdgeObservation> observations;
        double t_first = -1.0, t_last = 0.0;
        std::size_t used = 0;
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t w = 0, h = 0;
            double t = 0.0;
            if (!ReadExact(&w, 4) || !ReadExact(&h, 4) || !ReadExact(&t, 8)) return 1;
            buffer.resize(static_cast<std::size_t>(w) * h);
            if (!ReadExact(buffer.data(), buffer.size())) return 1;
            if (i < warmup_frames) continue;                       // как CaptureBurst(..., warmup)
            GrayImage image{buffer.data(), w, h, 0};
            if (t_first < 0) t_first = t;
            t_last = t;
            flow.AddProfile(ProfileExtractor::Extract(image, flow_strip), t - t_first);
            if (used % level_every == 0) observations.push_back(edges.Find(image, level_strip));
            ++used;
        }
        const FlowResult fr = flow.Estimate();

        // как CombineLevelObservations: медиана уровней по кадрам серии
        const LevelEstimator level(geometry, level_config);
        std::vector<LevelResult> per_frame;
        for (const auto& o : observations) per_frame.push_back(level.Estimate(o, calibration, center_reference_px));
        std::vector<std::pair<double, std::size_t>> ranked;
        for (std::size_t i = 0; i < per_frame.size(); ++i)
            if (per_frame[i].level_mm) ranked.push_back({*per_frame[i].level_mm, i});
        LevelResult lr = per_frame.empty() ? LevelResult{} : per_frame.front();
        if (!ranked.empty()) {
            std::sort(ranked.begin(), ranked.end());
            const double median = ranked[ranked.size() / 2].first;
            std::size_t closest = ranked.front().second;
            double gap = 1e300;
            for (const auto& [v, idx] : ranked)
                if (std::fabs(v - median) < gap) { gap = std::fabs(v - median); closest = idx; }
            lr = per_frame[closest];
            lr.level_mm = median;
        }

        VisionReport r;
        r.speed_valid = static_cast<bool>(fr.speed_ms);
        r.speed_ms = fr.speed_ms.value_or(0.0);
        r.speed_snr = fr.snr;
        r.speed_profiles = fr.profiles_used;
        r.speed_intervals = fr.intervals_used;
        r.speed_shift_px_per_frame = fr.shift_px_per_frame;
        r.level_valid = static_cast<bool>(lr.level_mm);
        r.level_mm = lr.level_mm.value_or(0.0);
        r.level_sigma_mm = lr.sigma_mm;
        const char* names[] = {"measured", "empty", "below_visible", "single_edge", "unreliable"};
        r.level_state = names[static_cast<int>(lr.state)];
        r.level_disagreement_mm = lr.disagreement_mm;
        for (const auto& p : lr.participants)
            if (p.used) r.level_sources.push_back(p.name);
        r.level_frames = observations.size();
        r.camera_frames = used;
        r.camera_fps = t_last > t_first ? (used - 1) / (t_last - t_first) : 0.0;
        r.calibration_valid = true;
        r.calibration_beta_deg = calibration.elevation_deg;
        r.calibration_scale_mm_per_px = calibration.scale_mm_per_px;
        r.calibration_manual = true;
        std::printf("%s\n", FormatVisionReport(r).c_str());

        // диагностика кромок: медианы по кадрам серии
        std::vector<double> nears, fars, sn, sf;
        std::size_t both = 0;
        for (const auto& o : observations) {
            if (o.near_found) { nears.push_back(o.near_px); sn.push_back(o.near_sharpness); }
            if (o.far_found) { fars.push_back(o.far_px); sf.push_back(o.far_sharpness); }
            both += o.near_found && o.far_found;
        }
        auto med = [](std::vector<double> v) {
            if (v.empty()) return -1.0;
            std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
            return v[v.size() / 2];
        };
        std::fprintf(stderr, "  t=%.2f с: скорость %s (snr %.1f, сдвиг %.2f px/кадр, %zu профилей, %zu интервалов); "
                     "кромки near %.0f (резкость %.1f) far %.0f (%.1f), обе в %zu/%zu кадрах; участники:",
                     t_first, fr.speed_ms ? std::to_string(*fr.speed_ms).c_str() : "нет", fr.snr,
                     fr.shift_px_per_frame, fr.profiles_used, fr.intervals_used, med(nears), med(sn), med(fars),
                     med(sf), both, observations.size());
        for (const auto& p : lr.participants)
            std::fprintf(stderr, " %s=%.1f мм%s", p.name.c_str(), p.level_mm, p.used ? "" : "(отброшен)");
        std::fprintf(stderr, "\n");
    }
    return 0;
}
