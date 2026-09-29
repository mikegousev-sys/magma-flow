// magma_axis_level.hpp — уровень по оси потока и краю корки («ось + край»).
//
// Для установки, где камера видит только ПОЛОВИНУ жёлоба (вторую закрывает
// стенка): двух кромок зеркала в кадре нет, и участники «ширина»/«центр»
// из magma_level.hpp неприменимы. Здесь уровень строится по полуширине:
//
//   ОСЬ   — полоса наибольшей скорости. Поток разбивается поперёк на
//           несколько параллельных полос, скорость каждой считается тем же
//           FlowEstimator; максимум (с уточнением параболой) — ось.
//   КРАЙ  — граница «поток — корка». Профили поперёк потока, усреднённые
//           по времени (медиана по кадрам серии: текстура уходит, корка
//           остаётся), в нескольких местах вдоль потока. В каждом идём от оси
//           к краю и берём самую дальнюю точку ярче середины между яркостью
//           потока и фона за краем — тёмные пятна текстуры внутри потока не
//           мешают. Край рваный, поэтому по местам берётся медиана, а разброс
//           P10..P90 идёт в погрешность.
//   УРОВЕНЬ — полуширина b = |ось − край|·масштаб_поперёк + корка (на сколько
//           корка заходит на зеркало) → уровень по профилю жёлоба.
//
// Отсчёты за кадром в расчёт не идут. Тот же алгоритм — в прототипе
// analysis/vision_series.py (update_level) и analysis/flow_rate.py.

#ifndef MAGMA_AXIS_LEVEL_HPP
#define MAGMA_AXIS_LEVEL_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

#include "magma_edge.hpp"
#include "magma_flow.hpp"
#include "magma_level.hpp"

namespace magma {

struct AxisEdgeConfig {
    std::size_t bands = 8;              ///< полос скорости поперёк потока
    double width_px = 0.0;              ///< суммарная ширина полос; 0 — flow.strip.average_px
    std::size_t band_lines = 5;         ///< линий усреднения в каждой полосе
    double edge_length_px = 800.0;      ///< длина профиля края поперёк потока (по центру полосы)
    double edge_step_px = 2.0;          ///< шаг профиля края
    std::size_t edge_columns = 64;      ///< мест вдоль потока, где ищется край
    int edge_side = 0;                  ///< +1/−1 — в какую сторону от оси край; 0 — к медленным полосам
    double min_contrast = 5.0;          ///< поток ярче фона за краем хотя бы на столько
    std::size_t min_columns = 10;       ///< мест с найденным краем для результата
    double scale_across_mm_per_px = 0.638;  ///< масштаб поперёк потока на поверхности
    double crust_mm = 25.0;             ///< на сколько корка заходит на зеркало
};

/// Диагностика расчёта (для журнала и cpp_replay).
struct AxisEdgeDetail {
    std::vector<std::optional<double>> band_speed_ms;
    std::vector<double> band_offset_px;
    std::optional<double> axis_offset_px;
    std::optional<double> axis_speed_ms;
    bool axis_on_edge = false;
    std::optional<double> edge_offset_px;
    double edge_ragged_px = 0.0;
    std::size_t edge_columns_found = 0;
    std::size_t edge_frames = 0;
    std::optional<double> half_width_mm;
};

/// Накопитель одной серии: кадры подаются по мере захвата, сами не хранятся.
class AxisEdgeLevel {
public:
    AxisEdgeLevel(const FlowStrip& strip, const FlowConfig& flow, const AxisEdgeConfig& config)
        : strip_(strip), config_(config) {
        const std::size_t n = std::max<std::size_t>(3, config_.bands);
        const double width = config_.width_px > 0.0 ? config_.width_px : strip_.average_px;
        const double angle = strip_.along_angle_deg * M_PI / 180.0;
        nx_ = -std::sin(angle);
        ny_ = std::cos(angle);
        for (std::size_t k = 0; k < n; ++k) {
            const double offset =
                -width / 2.0 + width * (static_cast<double>(k) + 0.5) / static_cast<double>(n);
            FlowStrip band = strip_;
            band.center_x_px += nx_ * offset;
            band.center_y_px += ny_ * offset;
            band.average_px = width / static_cast<double>(n);
            band.average_lines = std::max<std::size_t>(1, config_.band_lines);
            bands_.push_back(band);
            offsets_.push_back(offset);
            flows_.emplace_back(flow);
        }
    }

    /// Кадр серии. take_edge — снять и профили края (на части кадров).
    void AddFrame(const GrayImage& image, double timestamp_s, bool take_edge) {
        for (std::size_t k = 0; k < bands_.size(); ++k) {
            flows_[k].AddProfile(ProfileExtractor::Extract(image, bands_[k]), timestamp_s);
        }
        if (take_edge) AddEdgeFrame(image);
    }

    LevelResult Estimate(const LauncherGeometry& geometry, const LevelConfig& level_config,
                         AxisEdgeDetail* detail = nullptr) const {
        AxisEdgeDetail d;
        LevelResult result;   // по умолчанию Empty
        d.band_offset_px = offsets_;
        d.edge_frames = edge_frames_.size();

        // --- ось: максимум скорости по полосам, вершина параболы по трём точкам ---
        std::vector<double> xs, vs;
        for (std::size_t k = 0; k < flows_.size(); ++k) {
            const FlowResult r = flows_[k].Estimate();
            d.band_speed_ms.push_back(r.speed_ms);
            if (r.speed_ms) { xs.push_back(offsets_[k]); vs.push_back(*r.speed_ms); }
        }
        if (xs.size() < 3) {
            // Поток не виден ни в одной или почти ни в одной полосе.
            result.state = xs.empty() ? LevelState::Empty : LevelState::Unreliable;
            if (detail) *detail = d;
            return result;
        }
        // Парабола через максимум и соседей (у края охвата — через три крайние
        // точки). Ось найдена, если вершина лежит внутри охвата полос: при
        // плоской вершине максимум часто попадает на крайнюю полосу, хотя сама
        // ось видна. Узлы могут идти неравномерно — полосы без скорости пропущены.
        const std::size_t i = static_cast<std::size_t>(
            std::max_element(vs.begin(), vs.end()) - vs.begin());
        const std::size_t m = std::clamp<std::size_t>(i, 1, xs.size() - 2);
        const double x0 = xs[m - 1], x1 = xs[m], x2 = xs[m + 1];
        const double y0 = vs[m - 1], y1 = vs[m], y2 = vs[m + 1];
        const double denom = (x0 - x1) * (x0 - x2) * (x1 - x2);
        const double a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom;
        const double b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom;
        double axis = xs[i], axis_speed = vs[i];
        d.axis_on_edge = true;
        if (a < 0.0) {
            const double vertex = -b / (2.0 * a);
            if (vertex >= xs.front() && vertex <= xs.back()) {
                axis = std::clamp(vertex, x0, x2);
                const double c = y0 - a * x0 * x0 - b * x0;
                axis_speed = std::min(a * axis * axis + b * axis + c, 1.05 * vs[i]);
                d.axis_on_edge = false;
            }
        }
        d.axis_offset_px = axis;
        d.axis_speed_ms = axis_speed;

        // --- край: сторона — к медленным полосам, если не задана ---
        int side = config_.edge_side;
        if (side == 0) {
            const std::size_t slow = static_cast<std::size_t>(
                std::min_element(vs.begin(), vs.end()) - vs.begin());
            side = xs[slow] < axis ? -1 : 1;
        }
        std::vector<double> edges = FindCrustEdges(axis, side);
        d.edge_columns_found = edges.size();
        if (edges.size() < std::max<std::size_t>(1, config_.min_columns)) {
            result.state = LevelState::SingleEdge;   // ось есть, края нет
            if (detail) *detail = d;
            return result;
        }
        std::sort(edges.begin(), edges.end());
        const double edge = Quantile(edges, 0.5);
        d.edge_offset_px = edge;
        d.edge_ragged_px = Quantile(edges, 0.9) - Quantile(edges, 0.1);

        const double half = std::fabs(edge - axis) * config_.scale_across_mm_per_px + config_.crust_mm;
        d.half_width_mm = half;
        const auto level = geometry.LevelFromHalfWidth(half);
        if (!level || *level > geometry.max_level_mm || d.axis_on_edge) {
            // За пределами профиля или ось упёрлась в крайнюю полосу — не доверяем.
            result.state = LevelState::Unreliable;
            if (detail) *detail = d;
            return result;
        }

        // Погрешность: рваный край, P10..P90 ≈ ±1.28σ.
        const double db = d.edge_ragged_px / 2.56 * config_.scale_across_mm_per_px;
        const auto lo = geometry.LevelFromHalfWidth(std::max(0.0, half - db));
        const auto hi = geometry.LevelFromHalfWidth(half + db);
        const double ragged_sigma = (lo && hi) ? std::fabs(*hi - *lo) / 2.0 : 0.0;

        Participant participant;
        participant.name = "axis_edge";
        participant.level_mm = *level;
        participant.sigma_mm = std::hypot(ragged_sigma, level_config.systematic_sigma_mm);
        participant.weight = 1.0 / (participant.sigma_mm * participant.sigma_mm);
        participant.used = true;
        result.participants.push_back(participant);
        result.level_mm = *level;
        result.sigma_mm = participant.sigma_mm;
        result.state = LevelState::Measured;
        if (detail) *detail = d;
        return result;
    }

private:
    std::size_t EdgeSamples() const {
        return static_cast<std::size_t>(config_.edge_length_px / std::max(0.5, config_.edge_step_px)) + 1;
    }
    double EdgeOffset(std::size_t i) const {
        return -config_.edge_length_px / 2.0 + config_.edge_step_px * static_cast<double>(i);
    }

    /// Профили поперёк потока в edge_columns местах вдоль полосы; за кадром — NaN.
    void AddEdgeFrame(const GrayImage& image) {
        if (!image.Valid()) return;
        const std::size_t columns = std::max<std::size_t>(1, config_.edge_columns);
        const std::size_t samples = EdgeSamples();
        const double angle = strip_.along_angle_deg * M_PI / 180.0;
        const double ax = std::cos(angle), ay = std::sin(angle);
        std::vector<float> frame(columns * samples);
        for (std::size_t c = 0; c < columns; ++c) {
            const double u = -strip_.length_px / 2.0 +
                strip_.length_px * (static_cast<double>(c) + 0.5) / static_cast<double>(columns);
            for (std::size_t i = 0; i < samples; ++i) {
                const double v = EdgeOffset(i);
                const double x = strip_.center_x_px + ax * u + nx_ * v;
                const double y = strip_.center_y_px + ay * u + ny_ * v;
                frame[c * samples + i] = image.Contains(x, y)
                    ? static_cast<float>(image.Sample(x, y))
                    : std::numeric_limits<float>::quiet_NaN();
            }
        }
        edge_frames_.push_back(std::move(frame));
    }

    /// Край корки в каждом месте вдоль потока (как crust_edges в flow_rate.py).
    std::vector<double> FindCrustEdges(double axis_px, int side) const {
        std::vector<double> edges;
        if (edge_frames_.empty()) return edges;
        const std::size_t columns = std::max<std::size_t>(1, config_.edge_columns);
        const std::size_t samples = EdgeSamples();
        const double step = std::max(0.5, config_.edge_step_px);
        const long axis_index = std::lround((axis_px + config_.edge_length_px / 2.0) / step);
        if (axis_index < 0 || axis_index >= static_cast<long>(samples)) return edges;

        // Сколько отсчётов профиля нужно: как в прототипе при шаге 2 px — 40 от оси,
        // 15 — фон за краем.
        constexpr std::size_t kMinSamples = 40, kBackground = 15;
        std::vector<float> values;
        for (std::size_t c = 0; c < columns; ++c) {
            // Медиана по кадрам: движущаяся текстура уходит, корка и фон остаются.
            std::vector<double> profile;
            std::vector<double> position;
            for (long i = axis_index; i >= 0 && i < static_cast<long>(samples); i += side) {
                values.clear();
                for (const auto& frame : edge_frames_) {
                    const float value = frame[c * samples + static_cast<std::size_t>(i)];
                    if (std::isfinite(value)) values.push_back(value);
                }
                if (values.size() * 2 <= edge_frames_.size()) {
                    if (i == axis_index) break;   // ось за кадром
                    continue;
                }
                std::nth_element(values.begin(),
                                 values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2),
                                 values.end());
                profile.push_back(values[values.size() / 2]);
                position.push_back(EdgeOffset(static_cast<std::size_t>(i)));
            }
            if (profile.size() < kMinSamples) continue;
            profile = Smooth(profile, 4);

            std::vector<double> tail(profile.end() - kBackground, profile.end());
            std::sort(tail.begin(), tail.end());
            const double background = tail[tail.size() / 2];
            std::vector<double> head(profile.begin(),
                profile.begin() + static_cast<std::ptrdiff_t>(std::max<std::size_t>(5, profile.size() / 3)));
            std::sort(head.begin(), head.end());
            const double top = Quantile(head, 0.9);
            if (top - background < config_.min_contrast) continue;

            const double mid = (top + background) / 2.0;
            std::size_t last = profile.size();
            for (std::size_t i = 0; i < profile.size(); ++i) {
                if (profile[i] >= mid) last = i;
            }
            if (last == profile.size() || profile.size() - last < kBackground) continue;  // фон не виден
            edges.push_back(position[last]);
        }
        return edges;
    }

    static std::vector<double> Smooth(const std::vector<double>& input, std::size_t r) {
        std::vector<double> output(input.size());
        for (std::size_t i = 0; i < input.size(); ++i) {
            const std::size_t lo = i > r ? i - r : 0;
            const std::size_t hi = std::min(input.size() - 1, i + r);
            double sum = 0.0;
            for (std::size_t k = lo; k <= hi; ++k) sum += input[k];
            output[i] = sum / static_cast<double>(hi - lo + 1);
        }
        return output;
    }

    /// Квантиль отсортированного массива, линейная интерполяция (как numpy).
    static double Quantile(const std::vector<double>& sorted, double q) {
        if (sorted.empty()) return 0.0;
        const double position = q * static_cast<double>(sorted.size() - 1);
        const std::size_t i = static_cast<std::size_t>(position);
        if (i + 1 >= sorted.size()) return sorted.back();
        const double f = position - static_cast<double>(i);
        return sorted[i] * (1.0 - f) + sorted[i + 1] * f;
    }

    FlowStrip strip_;
    AxisEdgeConfig config_;
    double nx_ = 0.0, ny_ = 1.0;
    std::vector<FlowStrip> bands_;
    std::vector<double> offsets_;
    std::vector<FlowEstimator> flows_;
    std::vector<std::vector<float>> edge_frames_;
};

}  // namespace magma

#endif  // MAGMA_AXIS_LEVEL_HPP
