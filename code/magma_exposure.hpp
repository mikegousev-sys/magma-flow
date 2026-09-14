// magma_exposure.hpp — адаптивное управление выдержкой и усилением.
//
// ЗАЧЕМ СВОЙ РЕГУЛЯТОР, А НЕ АВТОМАТИКА КАМЕРЫ.
// Штатная автоматика стремится к «правильной» средней яркости и ради этого
// свободно удлиняет выдержку. Для нас это разрушительно: при выдержке в одну
// тридцатую секунды расплав, идущий со скоростью полтора метра в секунду,
// смазывается на девяносто пикселей, и текстура, по которой измеряется
// скорость, исчезает полностью. Выдержкой нужно управлять по собственному
// критерию, а автоматику камеры отключать.
//
// ДВА ОГРАНИЧЕНИЯ, МЕЖДУ КОТОРЫМИ ИЩЕТСЯ РЕШЕНИЕ.
// Сверху выдержку ограничивает смаз: он равен скорости, умноженной на
// выдержку и делённой на масштаб. Снизу — шум: слишком короткая выдержка
// оставляет мало света, и текстура тонет в шуме сенсора.
//
// Потолок по смазу не задаётся константой, а ВЫЧИСЛЯЕТСЯ по измеренной
// скорости. Это возможно потому, что скорость измеряется тем же прибором:
// при медленном потоке допустима более длинная выдержка и больше света, при
// быстром регулятор сам её укорачивает.
//
// ПОЧЕМУ УСИЛЕНИЕ ПРЕДПОЧТИТЕЛЬНЕЕ ДЛИННОЙ ВЫДЕРЖКИ.
// Когда света не хватает, есть два пути: удлинить выдержку или поднять
// усиление. Выбрано усиление. Смаз УНИЧТОЖАЕТ текстуру необратимо и вносит
// систематическую ошибку, тогда как шум усиления лишь добавляет случайный
// разброс, который гасится усреднением по многим кадрам и интервалам.
// Поэтому выдержка наращивается только до потолка по смазу, а дальше в дело
// идёт усиление.

#ifndef MAGMA_EXPOSURE_HPP
#define MAGMA_EXPOSURE_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace magma {

// =============================================================================
// Статистика кадра
// =============================================================================

/// То, по чему принимается решение о выдержке.
///
/// Управление ведётся не по средней яркости, а по ВЫСОКОМУ ПРОЦЕНТИЛЮ.
/// Средняя яркость зависит от того, какую долю кадра занимает расплав, и при
/// изменении уровня менялась бы сама по себе, заставляя регулятор гоняться за
/// геометрией вместо освещённости. Высокий процентиль отражает яркость самого
/// расплава и от его площади в кадре почти не зависит.
struct FrameStats {
    double high_percentile = 0.0;  ///< яркость, ниже которой лежит заданная доля
    double clipped_fraction = 0.0; ///< доля пикселей в насыщении
    double mean = 0.0;
    double texture_sigma = 0.0;    ///< разброс яркости — мера контраста текстуры
    bool melt_present = false;     ///< есть ли расплав в кадре

    /// Собрать статистику по области кадра.
    static FrameStats Measure(const uint8_t* data, std::size_t count,
                              double percentile = 0.995,
                              double melt_threshold = 40.0) {
        FrameStats stats;
        if (data == nullptr || count == 0) return stats;

        std::size_t histogram[256] = {0};
        double sum = 0.0, sum_squares = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            const uint8_t value = data[i];
            ++histogram[value];
            sum += value;
            sum_squares += static_cast<double>(value) * value;
        }

        const double n = static_cast<double>(count);
        stats.mean = sum / n;
        stats.texture_sigma =
            std::sqrt(std::max(0.0, sum_squares / n - stats.mean * stats.mean));
        stats.clipped_fraction = static_cast<double>(histogram[255]) / n;

        const std::size_t target =
            static_cast<std::size_t>(percentile * n);
        std::size_t accumulated = 0;
        for (int value = 0; value < 256; ++value) {
            accumulated += histogram[value];
            if (accumulated >= target) { stats.high_percentile = value; break; }
        }

        stats.melt_present = stats.high_percentile > melt_threshold;
        return stats;
    }
};

// =============================================================================
// Настройки регулятора
// =============================================================================

struct ExposureConfig {
    // --- пределы, продиктованные паспортом MV-MIPI-SC130M ---

    /// Выдержка квантуется временем одной строки: точность до микросекунды
    /// сенсором не поддерживается. При полном кадре 1280x1024 и предельных
    /// 214 к/с время строки составляет около 4.6 мкс, поэтому заданное
    /// значение округляется до кратного этой величине.
    double line_time_us = 4.6;
    /// Минимум — одна строка. Ставить меньше бессмысленно.
    int min_exposure_us = 5;

    /// Верхний предел выдержки задан паспортом как 1000000/fps: выдержка не
    /// может превышать период кадра, поскольку у этого сенсора экспозиция и
    /// считывание идут ПОСЛЕДОВАТЕЛЬНО, а не одновременно, как у сенсоров Sony.
    /// Значение вычисляется из частоты кадров, а не задаётся константой.
    double frame_rate = 60.0;

    /// Усиление задаётся в децибелах: диапазон 0-40 дБ с шагом 0.1 дБ.
    /// Сорок децибел соответствуют усилению в сто раз.
    double min_gain_db = 0.0;
    double max_gain_db = 40.0;
    double gain_step_db = 0.1;
    /// Шаг подстройки усиления за одну серию.
    double gain_increment_db = 2.0;

    /// Допустимый смаз в пикселях. Текстура имеет размер порядка семи
    /// пикселей, поэтому смаз до трёх ещё оставляет её различимой.
    ///
    /// На практике этот предел почти никогда не связывает: расплав настолько
    /// ярок, что насыщение наступает много раньше. При диафрагме F/6 и
    /// температуре 1150 градусов матрица насыщается за 114 мкс, а смаз при
    /// такой выдержке составляет треть пикселя. Предел оставлен страховкой
    /// на случай холодного расплава или разогнавшегося потока.
    double max_blur_px = 3.0;
    double mm_per_px = 0.5333;

    /// Целевое значение высокого процентиля. Ниже насыщения оставлен запас:
    /// пиксель, ушедший в насыщение, теряет текстуру безвозвратно, тогда как
    /// небольшая недодержка лишь немного увеличивает шум.
    double target_level = 210.0;
    /// Собственная автоматика камеры (режимы AE и AG) намеренно НЕ
    /// используется: она сводит к цели СРЕДНЮЮ яркость в заданной области,
    /// а средняя зависит от того, какую долю кадра занимает расплав, и
    /// менялась бы при изменении уровня сама по себе. Камеру следует
    /// перевести в ручной режим командами expmode и gainmode.
    double tolerance = 15.0;       ///< зона нечувствительности

    /// Доля насыщенных пикселей, выше которой выдержка снижается принудительно,
    /// независимо от процентиля.
    double max_clipped_fraction = 0.002;

    /// Демпфирование: доля от вычисленной поправки, применяемая за один шаг.
    /// Полная поправка приводит к раскачке, поскольку отклик сенсора нелинеен
    /// у краёв диапазона.
    double damping = 0.5;

    /// Скорость по умолчанию, пока измерение недоступно.
    double fallback_speed_ms = 1.5;
};

/// Решение регулятора.
struct ExposureSetting {
    int exposure_us = 200;
    double gain_db = 0.0;
    double blur_px = 0.0;          ///< ожидаемый смаз при этой выдержке
    bool at_blur_limit = false;    ///< выдержка упёрлась в предел по смазу
    bool changed = false;
};

// =============================================================================
// Регулятор
// =============================================================================

/// Подбирает выдержку и усиление между сериями кадров.
///
/// ВНУТРИ СЕРИИ ПАРАМЕТРЫ НЕ МЕНЯЮТСЯ. Изменение выдержки посреди серии дало бы
/// ступеньку яркости между кадрами, а корреляционный расчёт скорости сравнивает
/// кадры между собой — ступенька внесла бы ложный сигнал. Между сериями проходят
/// секунды, условия за это время меняются мало, поэтому подстройки раз в серию
/// достаточно.
class ExposureController {
public:
    explicit ExposureController(ExposureConfig config = {})
        : config_(config) {
        current_.exposure_us = Quantize(200);
        current_.gain_db = config.min_gain_db;
    }

    /// Верхний предел выдержки по паспорту: период кадра.
    int FrameLimitUs() const {
        return static_cast<int>(1e6 / std::max(1.0, config_.frame_rate));
    }

    /// Предел выдержки по смазу при заданной скорости.
    int BlurLimitUs(double speed_ms) const {
        const double speed = std::max(0.1, speed_ms);
        const double limit_s = config_.max_blur_px * config_.mm_per_px /
                               (speed * 1000.0);
        return static_cast<int>(limit_s * 1e6);
    }

    /// Пересчитать параметры по итогам прошедшей серии.
    ///
    /// speed_ms — последняя измеренная скорость; если она недоступна,
    /// используется значение по умолчанию из настроек.
    ExposureSetting Update(const FrameStats& stats, double speed_ms = -1.0) {
        const double speed = speed_ms > 0.0 ? speed_ms : config_.fallback_speed_ms;
        // Действующий потолок — меньшее из двух: предел по смазу и
        // паспортный предел по периоду кадра.
        const int blur_limit =
            std::clamp(std::min(BlurLimitUs(speed), FrameLimitUs()),
                       config_.min_exposure_us, FrameLimitUs());

        ExposureSetting next = current_;
        next.changed = false;

        // Пустой жёлоб: настройки удерживаются, а не подстраиваются.
        //
        // Это важнее, чем кажется. Тёмный кадр заставил бы регулятор поднять
        // выдержку и усиление до предела, и в момент прихода расплава первая
        // серия ушла бы в глубокое насыщение — как раз тогда, когда измерение
        // особенно нужно.
        if (!stats.melt_present) {
            next.blur_px = BlurAt(next.exposure_us, speed);
            next.at_blur_limit = next.exposure_us >= blur_limit;
            current_ = next;
            return next;
        }

        // Насыщение отрабатывается вне общей логики: оно уничтожает текстуру,
        // и реагировать на него нужно решительнее, чем на отклонение уровня.
        if (stats.clipped_fraction > config_.max_clipped_fraction) {
            if (next.gain_db > config_.min_gain_db) {
                next.gain_db = std::max(config_.min_gain_db,
                                        next.gain_db - config_.gain_increment_db);
            } else {
                next.exposure_us = std::max(config_.min_exposure_us,
                                            static_cast<int>(next.exposure_us * 0.7));
            }
            next.changed = true;
            Finish(next, speed, blur_limit);
            return next;
        }

        const double error = config_.target_level - stats.high_percentile;
        if (std::fabs(error) <= config_.tolerance) {
            Finish(next, speed, blur_limit);
            return next;   // в зоне нечувствительности не трогаем
        }

        // Отклик яркости на выдержку линеен, поэтому поправка мультипликативна.
        const double ratio = stats.high_percentile > 1.0
            ? config_.target_level / stats.high_percentile
            : 2.0;
        const double damped = std::pow(ratio, config_.damping);

        if (error > 0.0) {
            // Темно. Сначала выдержка — до предела по смазу, затем усиление.
            const int wanted = static_cast<int>(next.exposure_us * damped);
            if (next.exposure_us < blur_limit) {
                next.exposure_us = std::min(wanted, blur_limit);
                next.changed = true;
            } else if (next.gain_db < config_.max_gain_db) {
                next.gain_db = std::min(config_.max_gain_db,
                                        next.gain_db + config_.gain_increment_db);
                next.changed = true;
            }
        } else {
            // Светло. Сначала снимаем усиление, потом укорачиваем выдержку:
            // усиление даёт шум и убирать его выгоднее в первую очередь.
            if (next.gain_db > config_.min_gain_db) {
                next.gain_db = std::max(config_.min_gain_db,
                                        next.gain_db - config_.gain_increment_db);
                next.changed = true;
            } else {
                const int wanted = static_cast<int>(next.exposure_us * damped);
                next.exposure_us = std::max(config_.min_exposure_us, wanted);
                next.changed = true;
            }
        }

        Finish(next, speed, blur_limit);
        return next;
    }

    const ExposureSetting& Current() const { return current_; }

    /// Подставить новые настройки, СОХРАНИВ текущее состояние регулятора.
    ///
    /// Используется при перечитывании файла настроек на ходу. Простое
    /// пересоздание объекта с новым конфигом сбросило бы уже подобранную
    /// регулятором выдержку к значению по умолчанию, и после каждой правки
    /// файла настроек первая серия снова снималась бы «на глазок», пока
    /// регулятор заново не сойдётся, — при том что менялся, возможно, всего
    /// один порог. Текущие выдержка и усиление лишь подрезаются под новые
    /// границы, если они сузились.
    void UpdateConfig(const ExposureConfig& config) {
        config_ = config;
        current_.exposure_us =
            std::clamp(current_.exposure_us, config_.min_exposure_us, FrameLimitUs());
        current_.gain_db =
            std::clamp(current_.gain_db, config_.min_gain_db, config_.max_gain_db);
    }

    void Reset(int exposure_us, double gain_db) {
        current_.exposure_us =
            Quantize(std::clamp(exposure_us, config_.min_exposure_us, FrameLimitUs()));
        current_.gain_db =
            std::clamp(gain_db, config_.min_gain_db, config_.max_gain_db);
    }

private:
    static double BlurAtStatic(int exposure_us, double speed_ms, double mm_per_px) {
        return speed_ms * 1000.0 * (exposure_us / 1e6) / mm_per_px;
    }

    double BlurAt(int exposure_us, double speed_ms) const {
        return speed_ms * 1000.0 * (exposure_us / 1e6) / config_.mm_per_px;
    }

    /// Округление к целому числу строк: сенсор всё равно так поступит,
    /// и лучше знать заданное значение точно, чем гадать о фактическом.
    int Quantize(int exposure_us) const {
        if (config_.line_time_us <= 0.0) return exposure_us;
        const int lines = std::max(1, static_cast<int>(
            std::lround(exposure_us / config_.line_time_us)));
        return static_cast<int>(std::lround(lines * config_.line_time_us));
    }

    void Finish(ExposureSetting& setting, double speed, int blur_limit) {
        // Предел применяется в самом конце и безусловно.
        setting.exposure_us = std::clamp(setting.exposure_us,
                                         config_.min_exposure_us, blur_limit);
        setting.exposure_us = Quantize(setting.exposure_us);
        setting.gain_db = std::clamp(
            std::round(setting.gain_db / config_.gain_step_db) * config_.gain_step_db,
            config_.min_gain_db, config_.max_gain_db);
        setting.blur_px = BlurAt(setting.exposure_us, speed);
        setting.at_blur_limit = setting.exposure_us >= blur_limit;
        current_ = setting;
    }

    ExposureConfig config_;
    ExposureSetting current_;
};

}  // namespace magma

#endif  // MAGMA_EXPOSURE_HPP
