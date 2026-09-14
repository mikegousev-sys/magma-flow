// magma_signal.hpp — обработка сигналов скоростемера расплава.
//
// Оценка скорости ведётся по ВСЕМ шести базам между четырьмя фотодиодами
// одновременно. Прежний подход искал пик корреляции в каждой паре отдельно
// и на практике давал противоречивые результаты: базы по 125 мм показывали
// задержку впятеро меньше, чем базы по 75 мм, чего физически быть не может.
// Причины — периодичность сигнала (пик мог «проскочить» на целый период) и
// ложный максимум около нулевого лага от синхронной наводки.
//
// Здесь ищется одна скорость, объясняющая задержки на всех базах сразу:
// задержка обязана расти пропорционально расстоянию, и оценка, нарушающая
// это, отбрасывается. Ложный пик у нуля не может согласоваться сразу с
// шестью базами, проскок цикла — тоже.
//
// Зависимостей нет намеренно: БПФ реализовано здесь (radix-2), поэтому
// сборка на Raspberry Pi не требует ни FFTW, ни пакетного менеджера.

#ifndef MAGMA_SIGNAL_HPP
#define MAGMA_SIGNAL_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <numeric>
#include <optional>
#include <vector>

namespace magma {

// =============================================================================
// Геометрия датчика
// =============================================================================

/// Положение фотодиодов вдоль потока и все базы между ними.
///
/// Входы Arduino подключены с чередованием: A2 и A3 стоят на позициях 1 и 3,
/// A4 и A5 — на позициях 2 и 4. Поэтому пары (A2,A3) и (A4,A5), которые
/// использовал прежний код, разнесены на 125 мм, а соседние диоды — на 75 и
/// 50 мм. Порядок индексов здесь — порядок каналов в кадре, а не вдоль потока.
struct SensorGeometry {
    /// Позиция канала вдоль потока, м. Индекс = номер канала в кадре.
    /// A2 = 0 мм, A3 = 125 мм, A4 = 75 мм, A5 = 200 мм.
    std::array<double, 4> position_m{0.000, 0.125, 0.075, 0.200};

    struct Baseline {
        std::size_t upstream;    // канал выше по потоку
        std::size_t downstream;  // канал ниже по потоку
        double distance_m;
    };

    /// Все базы, упорядоченные так, что upstream всегда выше по потоку.
    std::vector<Baseline> Baselines() const {
        std::vector<Baseline> result;
        for (std::size_t i = 0; i < position_m.size(); ++i) {
            for (std::size_t j = i + 1; j < position_m.size(); ++j) {
                const double delta = position_m[j] - position_m[i];
                if (delta > 0.0) result.push_back({i, j, delta});
                else result.push_back({j, i, -delta});
            }
        }
        return result;
    }
};

// =============================================================================
// Конфигурация обработки
// =============================================================================

struct DetectionConfig {
    // Окно корреляции, отсчётов. При частоте оцифровки 1000 Гц это 1.02 с.
    //
    // Прежнее значение 500 досталось от первой версии и было мало. При лаге L
    // в окне N перекрываются лишь (N - L) отсчётов, поэтому надёжно измеряется
    // задержка примерно до N/3. Для базы 200 мм окно 500 ограничивало метод
    // скоростью 1.2 м/с снизу, хотя в настройках стояло 0.3 м/с — нижняя часть
    // заявленного диапазона была недостижима.
    //
    // Окно 1024 поднимает предел до 0.59 м/с по самой длинной базе и заметно
    // улучшает устойчивость к шуму: при сильном шуме доля принятых оценок
    // выросла с 17 из 30 до 27 из 30, а ошибка упала вдвое. Стоимость —
    // около 4% ядра Raspberry Pi при пересчёте 20 раз в секунду.
    //
    // Ограничения по времени нет: наружу сводка уходит раз в секунду, так что
    // окно длиной около секунды не ухудшает временное разрешение выдачи.
    std::size_t window_size = 1024;
    // Ниже 0.6 м/с самая длинная база (200 мм) выходит за предел окна;
    // короткие базы ещё работают, но запас достоверности падает.
    double min_speed_ms = 0.6;
    double max_speed_ms = 6.0;
    std::size_t velocity_steps = 400; // узлов перебора скорости
    double min_snr = 4.0;             // высота пика над шумом, в сигмах
    int snr_peak_exclusion = 8;       // узлов у пика, скрытых от оценки шума
    std::size_t min_baselines = 4;    // сколько баз обязано дать вклад
    double compute_interval_s = 0.05; // минимальный интервал между пересчётами
};

struct FilterConfig {
    double kalman_process_noise_per_sec = 20.0;  // изменчивость за секунду
    double kalman_base_measurement_noise = 1.0;
    std::size_t median_buffer_size = 5;
};

// =============================================================================
// Быстрое преобразование Фурье
// =============================================================================

inline std::size_t NextPowerOfTwo(std::size_t value) {
    std::size_t result = 1;
    while (result < value) result <<= 1;
    return result;
}

/// Комплексное БПФ на месте (radix-2, итеративное, с бит-реверсом).
/// inverse делит на n, что соответствует соглашению numpy.irfft.
inline void Fft(std::vector<std::complex<double>>& data, bool inverse) {
    const std::size_t n = data.size();
    if (n <= 1) return;

    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(data[i], data[j]);
    }

    for (std::size_t len = 2; len <= n; len <<= 1) {
        const double angle = 2.0 * M_PI / static_cast<double>(len) * (inverse ? 1.0 : -1.0);
        const std::complex<double> step(std::cos(angle), std::sin(angle));
        for (std::size_t i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (std::size_t k = 0; k < len / 2; ++k) {
                const std::complex<double> u = data[i + k];
                const std::complex<double> v = data[i + k + len / 2] * w;
                data[i + k] = u + v;
                data[i + k + len / 2] = u - v;
                w *= step;
            }
        }
    }
    if (inverse) for (auto& value : data) value /= static_cast<double>(n);
}

// =============================================================================
// Взаимная корреляция
// =============================================================================

/// Уточнить положение пика между узлами по трём соседним точкам.
/// Возвращает поправку к индексу в долях шага, в пределах [-0.5, 0.5].
inline double ParabolicInterpolation(const std::vector<double>& values,
                                     std::size_t peak_index) {
    if (peak_index == 0 || peak_index + 1 >= values.size()) return 0.0;
    const double a = values[peak_index - 1];
    const double b = values[peak_index];
    const double c = values[peak_index + 1];
    const double denominator = a - 2.0 * b + c;
    if (std::fabs(denominator) < 1e-12) return 0.0;
    return std::clamp(0.5 * (a - c) / denominator, -0.5, 0.5);
}

/// Вычесть среднее; возвращает СКО центрированного сигнала.
inline double Center(std::vector<double>& values) {
    if (values.empty()) return 0.0;
    const double mean =
        std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
    double sum_squares = 0.0;
    for (double& value : values) {
        value -= mean;
        sum_squares += value * value;
    }
    return std::sqrt(sum_squares / static_cast<double>(values.size()));
}

/// Взаимная корреляция с фазовым преобразованием (GCC-PHAT).
/// Устойчивее обычной корреляции к амплитудному шуму: нормирует спектр по
/// модулю, оставляя только временную информацию. Ненормированная корреляция
/// вдобавок смещала пик к малым лагам, поскольку при лаге L перекрываются
/// лишь (N - L) отсчётов, — здесь этого смещения нет.
/// Индекс size - 1 в результате соответствует нулевой задержке.
inline std::vector<double> GccPhat(const std::vector<double>& signal_a,
                                   const std::vector<double>& signal_b) {
    const std::size_t size = signal_a.size();
    const std::size_t n_fft = NextPowerOfTwo(2 * size - 1);

    std::vector<std::complex<double>> spectrum_a(n_fft, {0.0, 0.0});
    std::vector<std::complex<double>> spectrum_b(n_fft, {0.0, 0.0});
    for (std::size_t i = 0; i < size; ++i) {
        spectrum_a[i] = {signal_a[i], 0.0};
        spectrum_b[i] = {signal_b[i], 0.0};
    }
    Fft(spectrum_a, false);
    Fft(spectrum_b, false);

    for (std::size_t i = 0; i < n_fft; ++i) {
        const std::complex<double> cross = spectrum_b[i] * std::conj(spectrum_a[i]);
        const double magnitude = std::abs(cross);
        spectrum_a[i] = magnitude > 1e-12 ? cross / magnitude
                                          : std::complex<double>(0.0, 0.0);
    }
    Fft(spectrum_a, true);

    std::vector<double> correlation(2 * size - 1);
    for (std::size_t i = 0; i < size - 1; ++i) {
        correlation[i] = spectrum_a[n_fft - (size - 1) + i].real();
    }
    for (std::size_t i = 0; i < size; ++i) {
        correlation[size - 1 + i] = spectrum_a[i].real();
    }
    return correlation;
}

// =============================================================================
// Оценка скорости по нескольким базам
// =============================================================================

/// Результат оценки скорости.
struct SpeedResult {
    std::optional<double> speed_ms;  ///< пусто, если уверенной оценки нет
    double snr = 0.0;                ///< высота пика в шкале скоростей, в сигмах
    std::size_t baselines_used = 0;  ///< сколько баз дало вклад
};

/// Оценивает скорость потока согласованно по всем базам сразу.
///
/// Для каждой базы строится корреляционная функция, затем перебираются
/// скорости: каждой скорости соответствует своя ожидаемая задержка на каждой
/// базе, и вклады складываются. Верная скорость набирает вклад со всех баз
/// одновременно, ложный пик — лишь с одной. Перебор ведётся равномерно по
/// величине, обратной скорости, поскольку задержка пропорциональна именно ей.
class SpeedEstimator {
public:
    explicit SpeedEstimator(DetectionConfig config = {}, SensorGeometry geometry = {})
        : config_(config), baselines_(geometry.Baselines()) {}

    /// Оценить скорость по снимку всех каналов.
    ///
    /// Спектр каждого канала вычисляется ОДИН раз и переиспользуется всеми
    /// базами, в которые этот канал входит. Прямолинейная реализация считала
    /// бы для каждой из шести баз по два прямых преобразования, то есть 18
    /// БПФ вместо 10, — при четырёх каналах это лишняя работа почти вдвое.
    SpeedResult Estimate(const std::vector<std::vector<double>>& channels,
                         double sample_rate) const {
        const std::size_t size = channels.empty() ? 0 : channels[0].size();
        if (size < 10) return {};
        for (const auto& channel : channels) {
            if (channel.size() != size) return {};
        }

        std::vector<std::vector<double>> centered = channels;
        std::vector<double> deviation(centered.size());
        for (std::size_t i = 0; i < centered.size(); ++i) {
            deviation[i] = Center(centered[i]);
        }

        // Прямые преобразования всех каналов — по одному на канал.
        const std::size_t n_fft = NextPowerOfTwo(2 * size - 1);
        std::vector<std::vector<std::complex<double>>> spectra(centered.size());
        for (std::size_t i = 0; i < centered.size(); ++i) {
            if (deviation[i] < 1e-6) continue;
            spectra[i].assign(n_fft, {0.0, 0.0});
            for (std::size_t k = 0; k < size; ++k) spectra[i][k] = {centered[i][k], 0.0};
            Fft(spectra[i], false);
        }

        // Перебор равномерен по 1/v: задержка = расстояние * (1/v).
        const double slowness_min = 1.0 / config_.max_speed_ms;
        const double slowness_max = 1.0 / config_.min_speed_ms;
        const std::size_t steps = config_.velocity_steps;
        std::vector<double> score(steps, 0.0);
        std::size_t used = 0;

        for (const auto& baseline : baselines_) {
            if (deviation[baseline.upstream] < 1e-6) continue;
            if (deviation[baseline.downstream] < 1e-6) continue;

            const std::vector<double> correlation = CrossCorrelate(
                spectra[baseline.upstream], spectra[baseline.downstream], size, n_fft);
            const double zero_index = static_cast<double>(size) - 1.0;
            ++used;

            for (std::size_t k = 0; k < steps; ++k) {
                const double slowness = slowness_min +
                    (slowness_max - slowness_min) * static_cast<double>(k) /
                    static_cast<double>(steps - 1);
                const double lag = baseline.distance_m * slowness * sample_rate;
                score[k] += SampleAt(correlation, zero_index + lag);
            }
        }

        if (used < config_.min_baselines) return {};
        return Peak(score, slowness_min, slowness_max, used);
    }

private:
    /// Взаимная корреляция по готовым спектрам: PHAT-нормировка и обратное
    /// преобразование. Прямые БПФ уже посчитаны вызывающей стороной.
    static std::vector<double> CrossCorrelate(
        const std::vector<std::complex<double>>& spectrum_a,
        const std::vector<std::complex<double>>& spectrum_b,
        std::size_t size, std::size_t n_fft) {
        std::vector<std::complex<double>> cross(n_fft);
        for (std::size_t i = 0; i < n_fft; ++i) {
            const std::complex<double> value = spectrum_b[i] * std::conj(spectrum_a[i]);
            const double magnitude = std::abs(value);
            cross[i] = magnitude > 1e-12 ? value / magnitude
                                         : std::complex<double>(0.0, 0.0);
        }
        Fft(cross, true);

        std::vector<double> correlation(2 * size - 1);
        for (std::size_t i = 0; i < size - 1; ++i) {
            correlation[i] = cross[n_fft - (size - 1) + i].real();
        }
        for (std::size_t i = 0; i < size; ++i) {
            correlation[size - 1 + i] = cross[i].real();
        }
        return correlation;
    }

    /// Значение корреляции в дробной позиции (линейная интерполяция).
    static double SampleAt(const std::vector<double>& values, double position) {
        if (position < 0.0 || position + 1.0 >= static_cast<double>(values.size())) {
            return 0.0;
        }
        const std::size_t index = static_cast<std::size_t>(position);
        const double fraction = position - static_cast<double>(index);
        return values[index] * (1.0 - fraction) + values[index + 1] * fraction;
    }

    /// Найти максимум накопленной оценки и перевести его в скорость.
    SpeedResult Peak(const std::vector<double>& score, double slowness_min,
                     double slowness_max, std::size_t used) const {
        const std::size_t peak =
            static_cast<std::size_t>(std::max_element(score.begin(), score.end()) -
                                     score.begin());

        // Шум оценивается вне окрестности пика: иначе пик завышает собственный
        // фон и занижает отношение сигнал/шум тем сильнее, чем он острее.
        std::vector<double> noise;
        noise.reserve(score.size());
        for (std::size_t i = 0; i < score.size(); ++i) {
            const long distance = std::labs(static_cast<long>(i) - static_cast<long>(peak));
            if (distance > config_.snr_peak_exclusion) noise.push_back(score[i]);
        }
        if (noise.size() < 10) return {};

        const double mean =
            std::accumulate(noise.begin(), noise.end(), 0.0) / static_cast<double>(noise.size());
        double sum_squares = 0.0;
        for (double value : noise) sum_squares += (value - mean) * (value - mean);
        const double deviation = std::sqrt(sum_squares / static_cast<double>(noise.size()));
        if (deviation < 1e-12) return {};

        const double snr = (score[peak] - mean) / deviation;
        if (snr < config_.min_snr) return {std::nullopt, snr, used};

        const double step = (slowness_max - slowness_min) /
                            static_cast<double>(score.size() - 1);
        const double slowness = slowness_min +
            step * (static_cast<double>(peak) + ParabolicInterpolation(score, peak));
        if (slowness <= 0.0) return {std::nullopt, snr, used};
        return {1.0 / slowness, snr, used};
    }

    DetectionConfig config_;
    std::vector<SensorGeometry::Baseline> baselines_;
};

// =============================================================================
// Сглаживание
// =============================================================================

/// Скалярный фильтр Калмана для сглаживания скорости во времени.
///
/// Шум измерения обратно пропорционален качеству замера, поэтому шумные
/// оценки влияют слабее без бинарного отбрасывания. Шум процесса задаётся на
/// секунду и умножается на реальный dt, что делает настройку независимой от
/// частоты пересчёта.
class SpeedKalman {
public:
    SpeedKalman(double process_noise_per_sec, double base_measurement_noise)
        : process_noise_per_sec_(process_noise_per_sec),
          base_measurement_noise_(base_measurement_noise) {}

    void Reset() { state_.reset(); variance_ = 1.0; }

    std::optional<double> Update(std::optional<double> measurement, double quality,
                                 double dt) {
        const double process_noise = process_noise_per_sec_ * std::max(dt, 1e-6);
        if (!measurement) {
            variance_ += process_noise;
            return state_;
        }
        const double measurement_noise = base_measurement_noise_ / std::max(quality, 1e-3);
        if (!state_) {
            state_ = *measurement;
            variance_ = measurement_noise;
            return state_;
        }
        const double predicted = variance_ + process_noise;
        const double gain = predicted / (predicted + measurement_noise);
        state_ = *state_ + gain * (*measurement - *state_);
        variance_ = (1.0 - gain) * predicted;
        return state_;
    }

private:
    double process_noise_per_sec_;
    double base_measurement_noise_;
    std::optional<double> state_;
    double variance_ = 1.0;
};

/// Скользящая медиана достоверных значений — защита от единичных выбросов.
class MedianBuffer {
public:
    explicit MedianBuffer(std::size_t size) : size_(size) {}

    double Push(std::optional<double> value) {
        if (value && *value > 0.0) {
            values_.push_back(*value);
            if (values_.size() > size_) values_.erase(values_.begin());
        }
        if (values_.empty()) return 0.0;
        std::vector<double> sorted = values_;
        std::sort(sorted.begin(), sorted.end());
        const std::size_t middle = sorted.size() / 2;
        return sorted.size() % 2 == 1 ? sorted[middle]
                                      : (sorted[middle - 1] + sorted[middle]) / 2.0;
    }

private:
    std::size_t size_;
    std::vector<double> values_;
};

}  // namespace magma

#endif  // MAGMA_SIGNAL_HPP
