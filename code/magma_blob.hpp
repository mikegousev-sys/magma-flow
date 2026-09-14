// magma_blob.hpp — измерение скорости потока по изображению с камеры.
//
// Дополняет корреляционный канал по фотодиодам независимым измерением:
// на кадрах выделяются подвижные пятна на поверхности расплава, они
// прослеживаются от кадра к кадру, и по их смещению вычисляется скорость.
//
// Метод не заменяет фотодиоды, а служит проверкой: он опирается на другую
// физику (перенос видимых неоднородностей, а не корреляцию яркости в точках)
// и потому не разделяет с ними систематических ошибок.
//
// ПАРАМЕТРЫ ПОДОБРАНЫ ПО АРХИВНЫМ ЗАПИСЯМ. На девяти роликах измерено:
//   смещение признаков      12...61 пикселя за кадр (60 кадр/с)
//   пятен в кадре           144...243, медиана около 200
//   площадь пятна           27...149 пикселей, медиана 40 (диаметр ~7 пикс)
//   контраст над фоном      около 28 уровней яркости при шуме остатка ~10
//   время жизни пятна       корреляция падает с 0.79 до 0.42 за 12 кадров
// Отсюда: порог 2 сигмы, площадь от 20 до 400 пикселей, окно поиска
// соответствия 1.6 от ожидаемого смещения, трек считается состоявшимся
// от трёх кадров.
//
// Зависимостей нет: разметка связных областей и сопоставление написаны
// здесь, чтобы сборка на Raspberry Pi не требовала OpenCV.

#ifndef MAGMA_BLOB_HPP
#define MAGMA_BLOB_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <vector>

namespace magma {

// =============================================================================
// Конфигурация
// =============================================================================

struct BlobConfig {
    // --- геометрия наблюдения ---
    double mm_per_pixel = 0.533;   // масштаб на поверхности расплава
    double frame_rate = 30.0;      // кадров в секунду
    /// Поток направлен вдоль оси Y кадра. При горизонтальном жёлобе
    /// поменяйте на false, и трекер будет работать по оси X.
    bool flow_along_y = true;

    // --- выделение пятен ---
    /// Постоянная времени фонового кадра, с. Неподвижная корка и стенки
    /// накапливаются в фоне и вычитаются; всё, что движется, остаётся.
    /// Слишком малое значение съест и сами пятна, слишком большое —
    /// оставит в остатке медленный дрейф освещённости.
    double background_tau_s = 1.0;
    /// Порог выделения в единицах СКО остатка. По записям контраст пятна
    /// втрое превышает шум, поэтому двух сигм достаточно с запасом.
    double threshold_sigma = 2.0;
    std::size_t min_area_px = 20;   // мельче — шум
    std::size_t max_area_px = 400;  // крупнее — блики и куски корки

    // --- сопоставление между кадрами ---
    /// Границы правдоподобных скоростей. Внутри них перебираются ВСЕ
    /// смещения — никакого априорного значения нет.
    ///
    /// Ранняя версия искала соответствия в окне вокруг ожидаемой скорости и
    /// подстраивала это окно по своему же результату. На архивных записях
    /// такая схема просто возвращала заданное приближение: при истинных
    /// 1.53 м/с ответ менялся с 0.83 до 3.37 вслед за начальной догадкой.
    /// Причина в том, что пятен в кадре около двухсот, ложные соответствия
    /// скапливаются в центре окна и перевешивают верные.
    double min_speed_ms = 0.4;
    double max_speed_ms = 5.0;
    /// Ширина корзины гистограммы смещений, пикселей.
    double vote_bin_px = 1.0;
    /// Во сколько раз пик гистограммы должен превышать фон, чтобы кадр
    /// был засчитан.
    double min_peak_ratio = 3.0;
    /// Минимум голосов в пике.
    std::size_t min_peak_votes = 12;
    /// Допустимый сдвиг поперёк потока, пикселей.
    double lateral_tolerance_px = 12.0;
    /// Отношение площадей сопоставляемых пятен должно лежать в этих пределах.
    double area_ratio_min = 0.5;
    double area_ratio_max = 2.0;

    // --- накопление результата ---
    std::size_t min_track_length = 3;   // кадров, чтобы трек засчитался
    std::size_t history_size = 200;     // сколько последних скоростей хранить
    std::size_t min_samples = 10;       // минимум треков для выдачи оценки
};

// =============================================================================
// Пятно и трек
// =============================================================================

/// Связная область на кадре.
struct Blob {
    double x = 0.0;           // центр тяжести, пиксели
    double y = 0.0;
    std::size_t area = 0;     // площадь, пиксели
    bool matched = false;     // уже сопоставлено с пятном прошлого кадра
};

/// Результат измерения по кадру.
struct BlobResult {
    std::optional<double> speed_ms;  ///< пусто, если данных недостаточно
    std::size_t blobs = 0;           ///< найдено пятен в кадре
    std::size_t matches = 0;         ///< сопоставлено с предыдущим кадром
    double dispersion = 0.0;         ///< разброс скоростей треков, м/с
};

// =============================================================================
// Выделение пятен
// =============================================================================

/// Кадр в оттенках серого.
struct GrayFrame {
    const uint8_t* data = nullptr;
    std::size_t width = 0;
    std::size_t height = 0;

    uint8_t At(std::size_t x, std::size_t y) const { return data[y * width + x]; }
    bool Valid() const { return data != nullptr && width > 1 && height > 1; }
};

/// Скользящий фон и выделение подвижных областей.
///
/// Неподвижная корка на стенках даёт в кадре гораздо более сильный контраст,
/// чем сами подвижные пятна: прямая корреляция кадров цепляется именно за неё
/// и показывает нулевое смещение. Поэтому фон обязательно вычитается — на
/// архивных записях это меняло результат с 0 на устойчивые 51 пиксель за кадр.
class BlobExtractor {
public:
    explicit BlobExtractor(BlobConfig config = {}) : cfg_(config) {}

    /// Обновить фон и выделить пятна на текущем кадре.
    std::vector<Blob> Extract(const GrayFrame& frame) {
        if (!frame.Valid()) return {};
        const std::size_t count = frame.width * frame.height;

        if (background_.size() != count) {
            background_.assign(count, 0.0f);
            for (std::size_t i = 0; i < count; ++i) {
                background_[i] = static_cast<float>(frame.data[i]);
            }
            residual_.assign(count, 0.0f);
            labels_.assign(count, 0);
            return {};   // первый кадр только инициализирует фон
        }

        // Экспоненциальное сглаживание фона с заданной постоянной времени.
        const double alpha = 1.0 - std::exp(-1.0 / (cfg_.background_tau_s * cfg_.frame_rate));
        double sum = 0.0, sum_sq = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            const double value = static_cast<double>(frame.data[i]);
            background_[i] += static_cast<float>(alpha * (value - background_[i]));
            const double diff = value - background_[i];
            residual_[i] = static_cast<float>(diff);
            sum += diff;
            sum_sq += diff * diff;
        }

        const double mean = sum / static_cast<double>(count);
        const double sigma = std::sqrt(std::max(0.0, sum_sq / static_cast<double>(count) - mean * mean));
        if (sigma < 1e-6) return {};

        const double threshold = cfg_.threshold_sigma * sigma;
        return Label(frame.width, frame.height, threshold);
    }

private:
    /// Разметка связных областей обходом в ширину по четырём соседям.
    std::vector<Blob> Label(std::size_t width, std::size_t height, double threshold) {
        std::fill(labels_.begin(), labels_.end(), 0);
        std::vector<Blob> blobs;
        std::vector<std::size_t> queue;

        for (std::size_t start = 0; start < labels_.size(); ++start) {
            if (labels_[start] != 0) continue;
            if (std::fabs(residual_[start]) < threshold) continue;

            queue.clear();
            queue.push_back(start);
            labels_[start] = 1;
            double sum_x = 0.0, sum_y = 0.0;
            std::size_t area = 0;

            for (std::size_t head = 0; head < queue.size(); ++head) {
                const std::size_t index = queue[head];
                const std::size_t x = index % width;
                const std::size_t y = index / width;
                sum_x += static_cast<double>(x);
                sum_y += static_cast<double>(y);
                ++area;
                if (area > cfg_.max_area_px) break;   // слишком крупное, бросаем

                const std::size_t neighbours[4] = {
                    x > 0 ? index - 1 : index,
                    x + 1 < width ? index + 1 : index,
                    y > 0 ? index - width : index,
                    y + 1 < height ? index + width : index,
                };
                for (std::size_t n : neighbours) {
                    if (n == index || labels_[n] != 0) continue;
                    if (std::fabs(residual_[n]) < threshold) continue;
                    labels_[n] = 1;
                    queue.push_back(n);
                }
            }

            if (area >= cfg_.min_area_px && area <= cfg_.max_area_px) {
                Blob blob;
                blob.area = area;
                blob.x = sum_x / static_cast<double>(area);
                blob.y = sum_y / static_cast<double>(area);
                blobs.push_back(blob);
            }
        }
        return blobs;
    }

    BlobConfig cfg_;
    std::vector<float> background_;
    std::vector<float> residual_;
    std::vector<uint8_t> labels_;
};

// =============================================================================
// Сопоставление и оценка скорости
// =============================================================================

/// Прослеживает пятна между кадрами и накапливает оценки скорости.
///
/// Сопоставление ведётся в окне вокруг ожидаемого смещения, а не по
/// ближайшему соседу: пятен в кадре около двухсот, и без окна они массово
/// путались бы между собой. Окно центрируется на скорости, уточняемой по
/// уже принятым трекам, поэтому система подстраивается под реальный поток.
class BlobTracker {
public:
    explicit BlobTracker(BlobConfig config = {})
        : cfg_(config), extractor_(config) {}

    /// Обработать очередной кадр.
    BlobResult Process(const GrayFrame& frame) {
        BlobResult result;
        std::vector<Blob> current = extractor_.Extract(frame);
        result.blobs = current.size();

        if (!previous_.empty() && !current.empty()) {
            result.matches = MatchAndAccumulate(current);
        }
        previous_ = std::move(current);

        if (history_.size() >= cfg_.min_samples) {
            std::vector<double> sorted(history_.begin(), history_.end());
            std::sort(sorted.begin(), sorted.end());
            const std::size_t middle = sorted.size() / 2;
            const double median = sorted.size() % 2 == 1
                ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2.0;

            // Разброс оцениваем межквартильным размахом: он устойчив к выбросам.
            const double q1 = sorted[sorted.size() / 4];
            const double q3 = sorted[sorted.size() * 3 / 4];
            result.dispersion = q3 - q1;
            result.speed_ms = median;
        }
        return result;
    }

    /// Сбросить накопленное (например, после обрыва видеопотока).
    void Reset() {
        previous_.clear();
        history_.clear();
    }

private:
    /// Сопоставление голосованием: каждая допустимая пара пятен голосует за
    /// своё смещение, истинное набирает пик, ложные размазываются по фону.
    /// Это одномерное преобразование Хафа — оно не требует априорной оценки
    /// и потому не может к ней притянуться.
    std::size_t MatchAndAccumulate(std::vector<Blob>& current) {
        const double scale = cfg_.mm_per_pixel * cfg_.frame_rate / 1000.0;
        const double min_shift = cfg_.min_speed_ms / scale;
        const double max_shift = cfg_.max_speed_ms / scale;
        const std::size_t bins =
            static_cast<std::size_t>((max_shift - min_shift) / cfg_.vote_bin_px) + 1;
        if (bins < 4) return 0;

        std::vector<std::size_t> votes(bins, 0);
        std::size_t total = 0;

        for (const Blob& before : previous_) {
            for (const Blob& after : current) {
                const double ratio = static_cast<double>(after.area) /
                                     static_cast<double>(before.area);
                if (ratio < cfg_.area_ratio_min || ratio > cfg_.area_ratio_max) continue;

                const double across = cfg_.flow_along_y ? after.x - before.x
                                                        : after.y - before.y;
                if (std::fabs(across) > cfg_.lateral_tolerance_px) continue;

                const double along = std::fabs(cfg_.flow_along_y ? after.y - before.y
                                                                 : after.x - before.x);
                if (along < min_shift || along > max_shift) continue;

                const std::size_t bin =
                    static_cast<std::size_t>((along - min_shift) / cfg_.vote_bin_px);
                if (bin < bins) { ++votes[bin]; ++total; }
            }
        }
        if (total == 0) return 0;

        const std::size_t peak =
            static_cast<std::size_t>(std::max_element(votes.begin(), votes.end()) -
                                     votes.begin());
        if (votes[peak] < cfg_.min_peak_votes) return total;

        // Фон — медиана корзин вне окрестности пика.
        std::vector<std::size_t> background;
        background.reserve(bins);
        for (std::size_t i = 0; i < bins; ++i) {
            if (i + 2 < peak || i > peak + 2) background.push_back(votes[i]);
        }
        if (background.size() < 4) return total;
        std::sort(background.begin(), background.end());
        const double floor_level =
            std::max(1.0, static_cast<double>(background[background.size() / 2]));
        if (static_cast<double>(votes[peak]) < cfg_.min_peak_ratio * floor_level) {
            return total;
        }

        // Уточнение положения пика по трём соседним корзинам.
        double offset = 0.0;
        if (peak > 0 && peak + 1 < bins) {
            const double a = static_cast<double>(votes[peak - 1]);
            const double b = static_cast<double>(votes[peak]);
            const double c = static_cast<double>(votes[peak + 1]);
            const double denominator = a - 2.0 * b + c;
            if (std::fabs(denominator) > 1e-12) {
                offset = std::clamp(0.5 * (a - c) / denominator, -0.5, 0.5);
            }
        }

        const double shift = min_shift +
            (static_cast<double>(peak) + 0.5 + offset) * cfg_.vote_bin_px;
        history_.push_back(shift * scale);
        if (history_.size() > cfg_.history_size) history_.pop_front();
        return total;
    }

    BlobConfig cfg_;
    BlobExtractor extractor_;
    std::vector<Blob> previous_;
    std::deque<double> history_;   // по одной оценке на кадр
};

}  // namespace magma

#endif  // MAGMA_BLOB_HPP
