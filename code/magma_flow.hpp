// magma_flow.hpp — измерение скорости потока по последовательности кадров.
//
// МЕТОД. Из каждого кадра берётся одномерный профиль яркости ВДОЛЬ потока.
// Профили, поставленные в ряд по времени, образуют картину «положение —
// время», на которой движущаяся неоднородность оставляет наклонную полосу;
// наклон полосы и есть скорость. Наклон измеряется взаимной корреляцией
// профилей, снятых с разным интервалом.
//
// Это тот же алгоритм, что работает по фотодиодам, с одной подстановкой:
// там вклады складывались по шести БАЗАМ между датчиками, здесь — по
// нескольким ИНТЕРВАЛАМ между кадрами. Верная скорость набирает вклад со
// всех интервалов сразу, ложный пик — только с одного.
//
// ВЫЧИТАНИЕ НЕПОДВИЖНОГО ФОНА ОБЯЗАТЕЛЬНО.
// Это не тонкая настройка, а условие работоспособности. На архивных записях
// расплава прямая корреляция кадров устойчиво показывала НУЛЕВОЕ смещение:
// неподвижная корка на стенках жёлоба даёт контраст много сильнее, чем сама
// плывущая текстура, и корреляция цеплялась за неё. После вычитания
// накопленного фона тот же расчёт дал 51 пиксель за кадр с разбросом в один
// пиксель. Разница между «не работает вовсе» и «работает точно» — целиком
// в этом шаге.
//
// ПОЧЕМУ НЕ ОТСЛЕЖИВАНИЕ ОТДЕЛЬНЫХ ПЯТЕН.
// Такая попытка была и провалилась: поверхность расплава не содержит
// различимых объектов, а представляет собой сплошную текстуру. При двух сотнях
// «пятен» в кадре число ложных пар растёт как квадрат, и пик истинного
// смещения возвышался над фоном случайных совпадений всего в 1.2 раза при
// необходимых трёх. Корреляция целых профилей не требует ничего выделять и
// работает со всей текстурой сразу.

#ifndef MAGMA_FLOW_HPP
#define MAGMA_FLOW_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <deque>
#include <optional>
#include <vector>

#include "magma_edge.hpp"
#include "magma_signal.hpp"

namespace magma {

// =============================================================================
// Полоса выборки вдоль потока
// =============================================================================

/// Откуда брать профиль яркости.
///
/// Полоса вытянута ВДОЛЬ потока и усредняется ПОПЕРЁК — противоположно полосе
/// для измерения уровня. Поперечное усреднение подавляет шум, не размывая
/// продольную структуру, которая и несёт информацию о переносе.
struct FlowStrip {
    double center_x_px = 0.0;
    double center_y_px = 0.0;
    double along_angle_deg = 0.0;  ///< направление потока в кадре
    double length_px = 900.0;      ///< длина выборки вдоль потока
    double average_px = 60.0;      ///< ширина усреднения поперёк
    std::size_t samples = 900;
    std::size_t average_lines = 9;
};

// =============================================================================
// Конфигурация
// =============================================================================

struct FlowConfig {
    double mm_per_px = 0.5333;
    double min_speed_ms = 0.4;
    double max_speed_ms = 5.0;
    std::size_t velocity_steps = 400;

    /// Наибольший интервал между кадрами, участвующий в расчёте.
    ///
    /// Ограничен двумя явлениями. Текстура постепенно меняется сама по себе —
    /// по архивным записям постоянная этого затухания около 657 мс. И
    /// неоднородность уходит за край кадра: при поле зрения 683 мм и скорости
    /// 1.5 м/с на это уходит 0.46 с. Практический предел — примерно треть
    /// времени пролёта, дальше перекрытие профилей становится слишком малым.
    std::size_t max_interval_frames = 10;

    /// Постоянная времени накопления фона, секунд.
    ///
    /// Слишком малая съест и саму движущуюся текстуру, слишком большая оставит
    /// в остатке медленный дрейф освещённости. Несколько секунд — разумный
    /// компромисс: неподвижная корка держится минутами, текстура живёт доли
    /// секунды.
    double background_tau_s = 3.0;

    /// Порог превышения пика над фоном, в сигмах.
    double min_snr = 4.0;
    /// Узлов около пика, исключаемых из оценки фона.
    int snr_exclusion = 8;
    /// Сколько профилей нужно накопить, чтобы считать.
    std::size_t min_profiles = 20;
};

// =============================================================================
// Извлечение профиля
// =============================================================================

/// Берёт из кадра профиль яркости вдоль потока.
class ProfileExtractor {
public:
    // Класс не хранит состояния, поэтому метод сделан статическим: вызов
    // без создания временного объекта на каждый кадр серии (их около 120)
    // яснее сигнализирует об отсутствии состояния и не полагается на то,
    // что компилятор сам уберёт конструирование пустого объекта.
    static std::vector<double> Extract(const GrayImage& image, const FlowStrip& strip) {
        std::vector<double> profile(strip.samples, 0.0);
        if (!image.Valid() || strip.samples < 16) return profile;

        const double angle = strip.along_angle_deg * M_PI / 180.0;
        const double ax = std::cos(angle), ay = std::sin(angle);   // вдоль потока
        const double nx = -ay, ny = ax;                            // поперёк

        const std::size_t lines = std::max<std::size_t>(1, strip.average_lines);
        const double step = strip.length_px / static_cast<double>(strip.samples - 1);

        for (std::size_t i = 0; i < strip.samples; ++i) {
            const double offset =
                -strip.length_px / 2.0 + step * static_cast<double>(i);
            double sum = 0.0;
            for (std::size_t k = 0; k < lines; ++k) {
                const double t = lines > 1
                    ? -strip.average_px / 2.0 +
                      strip.average_px * static_cast<double>(k) /
                      static_cast<double>(lines - 1)
                    : 0.0;
                sum += image.Sample(strip.center_x_px + ax * offset + nx * t,
                                    strip.center_y_px + ay * offset + ny * t);
            }
            profile[i] = sum / static_cast<double>(lines);
        }
        return profile;
    }
};

// =============================================================================
// Результат
// =============================================================================

struct FlowResult {
    std::optional<double> speed_ms;
    double snr = 0.0;
    std::size_t profiles_used = 0;
    std::size_t intervals_used = 0;
    double frame_rate = 0.0;       ///< фактическая, по меткам времени
    double shift_px_per_frame = 0.0;
};

// =============================================================================
// Оценка скорости
// =============================================================================

/// Накапливает профили серии и вычисляет по ним скорость.
///
/// Профили добавляются по мере поступления кадров, сами кадры при этом не
/// хранятся: серия из ста кадров разрешением 1280x1024 заняла бы 130 мегабайт,
/// тогда как сто профилей укладываются в один.
class FlowEstimator {
public:
    explicit FlowEstimator(FlowConfig config = {}) : config_(config) {}

    /// Добавить профиль очередного кадра с меткой времени в секундах.
    void AddProfile(const std::vector<double>& profile, double timestamp_s) {
        if (profile.empty()) return;
        if (!background_.empty() && background_.size() != profile.size()) Reset();

        if (background_.empty()) {
            background_ = profile;
            length_ = profile.size();
        } else {
            // Экспоненциальное накопление фона по РЕАЛЬНОМУ интервалу времени,
            // а не по числу кадров: при пропуске кадров постоянная времени
            // иначе поплыла бы.
            const double dt = timestamps_.empty()
                ? 1.0 / 60.0 : timestamp_s - timestamps_.back();
            const double alpha =
                1.0 - std::exp(-std::max(1e-6, dt) / config_.background_tau_s);
            for (std::size_t i = 0; i < background_.size(); ++i) {
                background_[i] += alpha * (profile[i] - background_[i]);
            }
        }

        // В работу идёт ОСТАТОК после вычитания фона: только то, что движется.
        std::vector<double> residual(profile.size());
        for (std::size_t i = 0; i < profile.size(); ++i) {
            residual[i] = profile[i] - background_[i];
        }
        residuals_.push_back(std::move(residual));
        timestamps_.push_back(timestamp_s);
    }

    void Reset() {
        residuals_.clear();
        timestamps_.clear();
        background_.clear();
        length_ = 0;
    }

    std::size_t ProfileCount() const { return residuals_.size(); }

    /// Вычислить скорость по накопленной серии.
    FlowResult Estimate() const {
        FlowResult result;
        const std::size_t count = residuals_.size();
        result.profiles_used = count;
        if (count < config_.min_profiles || length_ < 32) return result;

        result.frame_rate = MeasuredFrameRate();
        if (result.frame_rate <= 0.0) return result;

        // Спектр каждого профиля считается ОДИН раз и переиспользуется во всех
        // интервалах, куда этот профиль входит. Прямолинейная реализация
        // выполняла бы прямое преобразование заново для каждой пары.
        const std::size_t n_fft = NextPowerOfTwo(2 * length_ - 1);
        std::vector<std::vector<std::complex<double>>> spectra;
        std::vector<bool> usable;
        spectra.reserve(count);
        usable.reserve(count);
        for (const std::vector<double>& residual : residuals_) {
            std::vector<double> centered = residual;
            const double deviation = Center(centered);
            usable.push_back(deviation >= 1e-9);
            std::vector<std::complex<double>> spectrum(n_fft, {0.0, 0.0});
            if (deviation >= 1e-9) {
                for (std::size_t i = 0; i < length_; ++i) spectrum[i] = {centered[i], 0.0};
                Fft(spectrum, false);
            }
            spectra.push_back(std::move(spectrum));
        }

        const double slowness_min = 1.0 / config_.max_speed_ms;
        const double slowness_max = 1.0 / config_.min_speed_ms;
        const std::size_t steps = config_.velocity_steps;
        std::vector<double> score(steps, 0.0);
        std::size_t intervals = 0;

        const std::size_t max_gap =
            std::min(config_.max_interval_frames, count - 1);
        for (std::size_t gap = 1; gap <= max_gap; ++gap) {
            // Взаимные спектры всех пар с данным интервалом СУММИРУЮТСЯ до
            // обратного преобразования: сумма корреляций равна обратному
            // преобразованию суммы спектров. Это заменяет сотни обратных
            // преобразований одним на каждый интервал.
            std::vector<std::complex<double>> accumulated(n_fft, {0.0, 0.0});
            double dt_sum = 0.0;
            std::size_t pairs = 0;

            for (std::size_t i = 0; i + gap < count; ++i) {
                if (!usable[i] || !usable[i + gap]) continue;
                for (std::size_t f = 0; f < n_fft; ++f) {
                    const std::complex<double> cross =
                        spectra[i + gap][f] * std::conj(spectra[i][f]);
                    const double magnitude = std::abs(cross);
                    // Фазовое взвешивание: оставляем только временную
                    // информацию, подавляя влияние отдельных ярких всплесков.
                    if (magnitude > 1e-12) accumulated[f] += cross / magnitude;
                }
                dt_sum += timestamps_[i + gap] - timestamps_[i];
                ++pairs;
            }
            if (pairs == 0) continue;

            Fft(accumulated, true);
            const double dt = dt_sum / static_cast<double>(pairs);
            if (dt <= 0.0) continue;

            std::vector<double> correlation(2 * length_ - 1);
            for (std::size_t i = 0; i < length_ - 1; ++i) {
                correlation[i] = accumulated[n_fft - (length_ - 1) + i].real();
            }
            for (std::size_t i = 0; i < length_; ++i) {
                correlation[length_ - 1 + i] = accumulated[i].real();
            }

            const double zero_index = static_cast<double>(length_) - 1.0;
            const double norm = 1.0 / static_cast<double>(pairs);
            for (std::size_t k = 0; k < steps; ++k) {
                const double slowness = slowness_min +
                    (slowness_max - slowness_min) * static_cast<double>(k) /
                    static_cast<double>(steps - 1);
                // Ожидаемый сдвиг за этот интервал: скорость на время, в пикселях.
                const double shift = dt * 1000.0 / (slowness * config_.mm_per_px);
                score[k] += SampleAt(correlation, zero_index + shift) * norm;
            }
            ++intervals;
        }

        result.intervals_used = intervals;
        if (intervals == 0) return result;

        return Peak(score, slowness_min, slowness_max, result);
    }

private:
    double MeasuredFrameRate() const {
        if (timestamps_.size() < 2) return 0.0;
        const double span = timestamps_.back() - timestamps_.front();
        if (span <= 0.0) return 0.0;
        return static_cast<double>(timestamps_.size() - 1) / span;
    }

    static double SampleAt(const std::vector<double>& values, double position) {
        if (position < 0.0 || position + 1.0 >= static_cast<double>(values.size())) {
            return 0.0;
        }
        const std::size_t index = static_cast<std::size_t>(position);
        const double fraction = position - static_cast<double>(index);
        return values[index] * (1.0 - fraction) + values[index + 1] * fraction;
    }

    FlowResult Peak(const std::vector<double>& score, double slowness_min,
                    double slowness_max, FlowResult result) const {
        const std::size_t peak = static_cast<std::size_t>(
            std::max_element(score.begin(), score.end()) - score.begin());

        // Окрестность пика исключается из оценки фона: иначе пик завышает
        // собственный фон и занижает отношение сигнал/шум тем сильнее, чем
        // он острее.
        std::vector<double> noise;
        noise.reserve(score.size());
        for (std::size_t i = 0; i < score.size(); ++i) {
            if (std::labs(static_cast<long>(i) - static_cast<long>(peak)) >
                config_.snr_exclusion) {
                noise.push_back(score[i]);
            }
        }
        if (noise.size() < 10) return result;

        double mean = 0.0;
        for (double value : noise) mean += value;
        mean /= static_cast<double>(noise.size());
        double variance = 0.0;
        for (double value : noise) variance += (value - mean) * (value - mean);
        const double deviation =
            std::sqrt(variance / static_cast<double>(noise.size()));
        if (deviation < 1e-12) return result;

        result.snr = (score[peak] - mean) / deviation;
        if (result.snr < config_.min_snr) return result;

        const double step =
            (slowness_max - slowness_min) / static_cast<double>(score.size() - 1);
        const double refined = static_cast<double>(peak) +
                               ParabolicInterpolation(score, peak);
        const double slowness = slowness_min + step * refined;
        if (slowness <= 0.0) return result;

        result.speed_ms = 1.0 / slowness;
        result.shift_px_per_frame =
            *result.speed_ms * 1000.0 / config_.mm_per_px / result.frame_rate;
        return result;
    }

    FlowConfig config_;
    std::vector<std::vector<double>> residuals_;
    std::vector<double> timestamps_;
    std::vector<double> background_;
    std::size_t length_ = 0;
};

}  // namespace magma

#endif  // MAGMA_FLOW_HPP
