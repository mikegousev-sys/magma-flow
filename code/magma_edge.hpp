// magma_edge.hpp — поиск кромок зеркала расплава в кадре.
//
// Задача: в полосе, проведённой поперёк жёлоба, найти две границы «стенка —
// расплав» и определить их положение точнее одного пикселя.
//
// Профиль яркости поперёк жёлоба выглядит так:
//
//     тёмная стенка |‾‾‾‾‾ яркий расплав ‾‾‾‾‾| тёмная стенка
//                   ^                         ^
//              ближняя кромка            дальняя кромка
//              (подъём яркости)          (спад яркости)
//
// Кромки ищутся по экстремумам производной яркости: ближняя даёт наибольший
// положительный градиент, дальняя — наибольший отрицательный. Такой критерий
// не зависит от абсолютной яркости расплава, которая меняется с температурой,
// и от общего уровня засветки.
//
// Величина градиента в найденной точке возвращается как мера резкости. Она
// служит признаком доверия: заросшая шлаком или закрытая загрязнённым стеклом
// граница РАЗМЫВАЕТСЯ раньше, чем сдвигается, поэтому падение резкости
// предупреждает об ухудшении заранее, до появления ошибки в уровне.
//
// Зависимостей нет: на вход подаётся готовый кадр в оттенках серого.

#ifndef MAGMA_EDGE_HPP
#define MAGMA_EDGE_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <vector>

#include "magma_level.hpp"

namespace magma {

// =============================================================================
// Кадр
// =============================================================================

/// Кадр в оттенках серого, без владения памятью.
struct GrayImage {
    const uint8_t* data = nullptr;
    std::size_t width = 0;
    std::size_t height = 0;
    std::size_t stride = 0;   ///< байт на строку; 0 означает width

    bool Valid() const { return data != nullptr && width > 1 && height > 1; }

    std::size_t Stride() const { return stride > 0 ? stride : width; }

    /// Значение в дробных координатах (билинейная интерполяция).
    /// Нужна потому, что полоса проводится под произвольным углом и почти
    /// никогда не попадает в центры пикселей.
    double Sample(double x, double y) const {
        if (x < 0.0 || y < 0.0) return 0.0;
        if (x >= static_cast<double>(width - 1)) return 0.0;
        if (y >= static_cast<double>(height - 1)) return 0.0;

        const std::size_t x0 = static_cast<std::size_t>(x);
        const std::size_t y0 = static_cast<std::size_t>(y);
        const double fx = x - static_cast<double>(x0);
        const double fy = y - static_cast<double>(y0);
        const std::size_t s = Stride();

        const double v00 = data[y0 * s + x0];
        const double v10 = data[y0 * s + x0 + 1];
        const double v01 = data[(y0 + 1) * s + x0];
        const double v11 = data[(y0 + 1) * s + x0 + 1];

        return v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) +
               v01 * (1 - fx) * fy + v11 * fx * fy;
    }
};

// =============================================================================
// Полоса выборки
// =============================================================================

/// Где и как брать профиль яркости.
///
/// Полоса проводится поперёк потока и усредняется вдоль него: продольное
/// усреднение подавляет шум и мелкую рябь, не размывая при этом кромку,
/// поскольку кромка вдоль потока идёт почти прямо.
struct StripDefinition {
    double center_x_px = 0.0;      ///< середина полосы в кадре
    double center_y_px = 0.0;
    double across_angle_deg = 0.0; ///< направление поперёк потока
    double length_px = 400.0;      ///< длина выборки поперёк жёлоба
    double average_px = 40.0;      ///< ширина усреднения вдоль потока
    std::size_t samples = 400;     ///< точек в профиле
    std::size_t average_lines = 9; ///< сколько параллельных линий усредняем
};

// =============================================================================
// Поиск кромок
// =============================================================================

struct EdgeConfig {
    /// Полуширина сглаживания профиля перед дифференцированием, отсчётов.
    /// Слишком малое значение оставляет шум и даёт ложные экстремумы,
    /// слишком большое размывает саму кромку и снижает точность.
    std::size_t smooth_radius = 3;
    /// Минимальный градиент, при котором кромка считается найденной.
    double min_gradient = 3.0;
    /// Доля профиля с каждого края, где кромки не ищутся: там полоса может
    /// выходить за пределы жёлоба или кадра.
    double margin_fraction = 0.05;
    /// Минимальное расстояние между кромками, отсчётов. Защищает от того,
    /// чтобы обе кромки нашлись на одном и том же перепаде.
    std::size_t min_separation = 20;
};

/// Извлекает профиль яркости и находит по нему кромки.
class EdgeFinder {
public:
    explicit EdgeFinder(EdgeConfig config = {}) : config_(config) {}

    /// Профиль яркости поперёк жёлоба, усреднённый вдоль потока.
    std::vector<double> ExtractProfile(const GrayImage& image,
                                       const StripDefinition& strip) const {
        std::vector<double> profile(strip.samples, 0.0);
        if (!image.Valid() || strip.samples < 8) return profile;

        const double angle = strip.across_angle_deg * M_PI / 180.0;
        const double ax = std::cos(angle), ay = std::sin(angle);   // поперёк
        const double lx = -ay, ly = ax;                            // вдоль потока

        const std::size_t lines = std::max<std::size_t>(1, strip.average_lines);
        const double step = strip.length_px / static_cast<double>(strip.samples - 1);

        for (std::size_t i = 0; i < strip.samples; ++i) {
            const double offset =
                -strip.length_px / 2.0 + step * static_cast<double>(i);
            double sum = 0.0;
            std::size_t count = 0;
            for (std::size_t k = 0; k < lines; ++k) {
                const double t = lines > 1
                    ? -strip.average_px / 2.0 +
                      strip.average_px * static_cast<double>(k) /
                      static_cast<double>(lines - 1)
                    : 0.0;
                const double x = strip.center_x_px + ax * offset + lx * t;
                const double y = strip.center_y_px + ay * offset + ly * t;
                sum += image.Sample(x, y);
                ++count;
            }
            profile[i] = count > 0 ? sum / static_cast<double>(count) : 0.0;
        }
        return profile;
    }

    /// Найти кромки по готовому профилю.
    ///
    /// Координаты возвращаются в отсчётах профиля. Перевод в миллиметры
    /// выполняется снаружи через калибровку, поскольку здесь неизвестен
    /// шаг полосы в мировых единицах.
    EdgeObservation FindEdges(const std::vector<double>& profile) const {
        EdgeObservation observation;
        if (profile.size() < 16) return observation;

        observation.brightness =
            std::accumulate(profile.begin(), profile.end(), 0.0) /
            static_cast<double>(profile.size());

        const std::vector<double> smooth = Smooth(profile);
        const std::vector<double> gradient = Gradient(smooth);

        const std::size_t margin = std::max<std::size_t>(
            2, static_cast<std::size_t>(profile.size() * config_.margin_fraction));
        if (margin * 2 + config_.min_separation >= gradient.size()) return observation;

        // Ближняя кромка — наибольший подъём яркости, дальняя — наибольший спад.
        std::size_t rise = margin, fall = margin;
        double best_rise = 0.0, best_fall = 0.0;
        for (std::size_t i = margin; i + margin < gradient.size(); ++i) {
            if (gradient[i] > best_rise) { best_rise = gradient[i]; rise = i; }
            if (-gradient[i] > best_fall) { best_fall = -gradient[i]; fall = i; }
        }

        // Спад обязан идти ПОСЛЕ подъёма: иначе это не полоса расплава,
        // а посторонний перепад (блик, край кадра, тень).
        const bool ordered = fall > rise + config_.min_separation;

        if (best_rise >= config_.min_gradient && ordered) {
            observation.near_px =
                static_cast<double>(rise) + Parabolic(gradient, rise);
            observation.near_sharpness = best_rise;
            observation.near_found = true;
        }
        if (best_fall >= config_.min_gradient && ordered) {
            observation.far_px =
                static_cast<double>(fall) + Parabolic(gradient, fall, true);
            observation.far_sharpness = best_fall;
            observation.far_found = true;
        }
        return observation;
    }

    /// Полный проход: кадр -> профиль -> кромки.
    EdgeObservation Find(const GrayImage& image, const StripDefinition& strip) const {
        return FindEdges(ExtractProfile(image, strip));
    }

private:
    /// Скользящее среднее. Простое усреднение предпочтено гауссову: при
    /// небольшом радиусе разница в результате незначительна, а стоимость
    /// заметно ниже — профиль обрабатывается для каждого кадра серии.
    std::vector<double> Smooth(const std::vector<double>& input) const {
        const std::size_t r = config_.smooth_radius;
        if (r == 0) return input;
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

    /// Центральная разность.
    static std::vector<double> Gradient(const std::vector<double>& input) {
        std::vector<double> output(input.size(), 0.0);
        for (std::size_t i = 1; i + 1 < input.size(); ++i) {
            output[i] = (input[i + 1] - input[i - 1]) / 2.0;
        }
        return output;
    }

    /// Субпиксельное уточнение экстремума по трём соседним точкам.
    /// Без него положение кромки квантуется целым отсчётом, а именно доли
    /// отсчёта и дают требуемую точность по уровню.
    static double Parabolic(const std::vector<double>& values, std::size_t peak,
                            bool negative = false) {
        if (peak == 0 || peak + 1 >= values.size()) return 0.0;
        const double sign = negative ? -1.0 : 1.0;
        const double a = sign * values[peak - 1];
        const double b = sign * values[peak];
        const double c = sign * values[peak + 1];
        const double denominator = a - 2.0 * b + c;
        if (std::fabs(denominator) < 1e-12) return 0.0;
        return std::clamp(0.5 * (a - c) / denominator, -0.5, 0.5);
    }

    EdgeConfig config_;
};

}  // namespace magma

#endif  // MAGMA_EDGE_HPP
