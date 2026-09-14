// magma_level.hpp — измерение уровня расплава в жёлобе по изображению.
//
// Камера видит зеркало расплава как светлую полосу между тёмными стенками.
// Положения двух кромок дают две независимые величины, и обе несут информацию
// об уровне:
//
//   ШИРИНА полосы  — канал круглый, поэтому чем выше уровень, тем шире зеркало.
//   ЦЕНТР  полосы  — кромки не только расходятся, но и ПОДНИМАЮТСЯ вместе с
//                    уровнем, а подъём проецируется в кадр.
//
// Их чувствительности ведут себя противоположно: у поверхности стенка почти
// вертикальна и ширина перестаёт реагировать, тогда как центр даёт постоянные
// cos(β)/s пикселей на миллиметр по всему диапазону. Внизу диапазона точнее
// ширина, вверху — центр, и вместе они покрывают весь рабочий ход.
//
// Третий участник — МЕТКИ на неподвижных частях жёлоба. Уровень они не мерят,
// но служат системой отсчёта: только по ним можно отличить реальное изменение
// уровня от смещения камеры, поскольку сдвиг камеры двигает центр полосы точно
// так же, как это сделал бы поднявшийся расплав.
//
// Зависимостей нет: только вычисления, без захвата кадров и без внешних
// библиотек. Это позволяет проверять логику на синтетике и на записанных
// кадрах, не подключая камеру.

#ifndef MAGMA_LEVEL_HPP
#define MAGMA_LEVEL_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace magma {

// =============================================================================
// 1. ГЕОМЕТРИЯ ЖЁЛОБА
// =============================================================================

/// Сечение канала — дуга окружности. Уровень отсчитывается от нижней точки.
struct LauncherGeometry {
    double channel_radius_mm = 150.0;  ///< R150 по чертежу
    double max_level_mm = 105.0;       ///< рабочий максимум

    /// Половина ширины зеркала на уровне h: sqrt(2Rh - h^2).
    double HalfWidth(double level_mm) const {
        const double h = std::clamp(level_mm, 0.0, 2.0 * channel_radius_mm);
        const double value = 2.0 * channel_radius_mm * h - h * h;
        return value > 0.0 ? std::sqrt(value) : 0.0;
    }

    /// Производная полуширины по уровню — чувствительность ширинного участника.
    /// У поверхности стремится к нулю: стенка там почти вертикальна.
    double HalfWidthSlope(double level_mm) const {
        const double w = HalfWidth(level_mm);
        if (w < 1e-6) return 0.0;
        return (channel_radius_mm - level_mm) / w;
    }

    /// Уровень по известной ширине зеркала (обратная задача).
    std::optional<double> LevelFromWidth(double width_mm) const {
        const double half = width_mm / 2.0;
        if (half < 0.0 || half > channel_radius_mm) return std::nullopt;
        return channel_radius_mm -
               std::sqrt(channel_radius_mm * channel_radius_mm - half * half);
    }
};

// =============================================================================
// 2. КАЛИБРОВКА ПО МЕТКАМ
// =============================================================================

/// Метка — неподвижная деталь жёлоба с известным положением.
///
/// Мировые координаты задаются в плоскости жёлоба: u вдоль потока,
/// v поперёк. Метки должны лежать на одной высоте (например, на кромке
/// борта), иначе разная высота внесёт в подгонку систематику.
struct CalibrationMark {
    std::string name;
    double image_x_px = 0.0;   ///< где найдена в кадре
    double image_y_px = 0.0;
    double world_u_mm = 0.0;   ///< вдоль потока
    double world_v_mm = 0.0;   ///< поперёк потока
};

/// Результат калибровки: связь кадра с плоскостью жёлоба.
///
/// Модель — слабая перспектива. При поле зрения около 10 градусов она даёт
/// погрешность заметно ниже прочих источников, а параметров содержит вчетверо
/// меньше полной проективной, что делает подгонку устойчивой при трёх метках.
struct CameraCalibration {
    double scale_mm_per_px = 0.0;   ///< масштаб вдоль потока (не сжат перспективой)
    double elevation_deg = 0.0;     ///< β — угол над плоскостью расплава
    double flow_angle_deg = 0.0;    ///< наклон оси потока в кадре
    double residual_px = 0.0;       ///< невязка подгонки, мера доверия
    bool valid = false;

    double SinBeta() const { return std::sin(elevation_deg * M_PI / 180.0); }
    double CosBeta() const { return std::cos(elevation_deg * M_PI / 180.0); }

    /// Поперечная координата точки в кадре: p = (v·sin β + z·cos β) / s.
    ///
    /// Это ключевое соотношение всего метода. Поперечное смещение сжимается
    /// перспективой как sin β, вертикальное — как cos β, и обе величины
    /// попадают в одну и ту же координату кадра.
    double ProjectAcross(double world_v_mm, double world_z_mm) const {
        return (world_v_mm * SinBeta() + world_z_mm * CosBeta()) / scale_mm_per_px;
    }
};

/// Вычисляет калибровку по набору меток.
///
/// Подгоняется аффинное преобразование из мировых координат в кадр, после чего
/// из него извлекаются масштаб и угол: длина образа единичного вектора вдоль
/// потока даёт масштаб, отношение длин образов поперечного и продольного
/// векторов даёт sin β.
///
/// Именно так угол и был определён по опорному снимку: две линии известной
/// длины — одна вдоль потока, другая поперёк — дали отношение масштабов, а
/// из него угол. Здесь то же самое, но по произвольному числу меток и с
/// оценкой невязки.
class CalibrationSolver {
public:
    void AddMark(const CalibrationMark& mark) { marks_.push_back(mark); }
    void Clear() { marks_.clear(); }
    std::size_t MarkCount() const { return marks_.size(); }

    /// Минимум три метки: шесть уравнений на шесть параметров аффинности.
    /// Метки не должны лежать на одной прямой, иначе задача вырождена.
    CameraCalibration Solve() const {
        CameraCalibration result;
        if (marks_.size() < 3) return result;

        // Аффинная модель: x = a·u + b·v + x0,  y = c·u + d·v + y0.
        // Решаем две независимые задачи наименьших квадратов (для x и для y)
        // с общей матрицей плана.
        double su = 0, sv = 0, suu = 0, svv = 0, suv = 0;
        double sx = 0, sy = 0, sux = 0, svx = 0, suy = 0, svy = 0;
        const double n = static_cast<double>(marks_.size());
        for (const CalibrationMark& m : marks_) {
            su += m.world_u_mm;  sv += m.world_v_mm;
            suu += m.world_u_mm * m.world_u_mm;
            svv += m.world_v_mm * m.world_v_mm;
            suv += m.world_u_mm * m.world_v_mm;
            sx += m.image_x_px;  sy += m.image_y_px;
            sux += m.world_u_mm * m.image_x_px;
            svx += m.world_v_mm * m.image_x_px;
            suy += m.world_u_mm * m.image_y_px;
            svy += m.world_v_mm * m.image_y_px;
        }

        // Центрируем, чтобы исключить смещение и свести задачу к 2x2.
        const double mu = su / n, mv = sv / n, mx = sx / n, my = sy / n;
        const double cuu = suu - n * mu * mu;
        const double cvv = svv - n * mv * mv;
        const double cuv = suv - n * mu * mv;
        const double cux = sux - n * mu * mx;
        const double cvx = svx - n * mv * mx;
        const double cuy = suy - n * mu * my;
        const double cvy = svy - n * mv * my;

        const double det = cuu * cvv - cuv * cuv;
        if (std::fabs(det) < 1e-9) return result;   // метки на одной прямой

        const double a = (cvv * cux - cuv * cvx) / det;
        const double b = (cuu * cvx - cuv * cux) / det;
        const double c = (cvv * cuy - cuv * cvy) / det;
        const double d = (cuu * cvy - cuv * cuy) / det;

        // Столбцы матрицы — образы единичных мировых векторов в кадре.
        const double len_u = std::hypot(a, c);   // вдоль потока: не сжат
        const double len_v = std::hypot(b, d);   // поперёк: сжат как sin β
        if (len_u < 1e-9) return result;

        const double sin_beta = std::clamp(len_v / len_u, 0.0, 1.0);
        result.scale_mm_per_px = 1.0 / len_u;
        result.elevation_deg = std::asin(sin_beta) * 180.0 / M_PI;
        result.flow_angle_deg = std::atan2(c, a) * 180.0 / M_PI;
        result.residual_px = Residual(a, b, c, d, mu, mv, mx, my);
        result.valid = true;
        return result;
    }

private:
    /// Среднеквадратичное расхождение меток с подогнанной моделью.
    /// Растущая невязка означает, что модель перестала описывать сцену:
    /// камера сдвинулась, метку перекрыло или её координаты заданы неверно.
    double Residual(double a, double b, double c, double d,
                    double mu, double mv, double mx, double my) const {
        double sum = 0.0;
        for (const CalibrationMark& m : marks_) {
            const double du = m.world_u_mm - mu;
            const double dv = m.world_v_mm - mv;
            const double ex = (mx + a * du + b * dv) - m.image_x_px;
            const double ey = (my + c * du + d * dv) - m.image_y_px;
            sum += ex * ex + ey * ey;
        }
        return std::sqrt(sum / static_cast<double>(marks_.size()));
    }

    std::vector<CalibrationMark> marks_;
};

/// Слежение за метками между кадрами — независимый детектор сдвига камеры.
///
/// Смещение камеры двигает центр полосы расплава ровно так же, как поднявшийся
/// уровень, и по одному изображению расплава эти причины неразличимы. Метки
/// разрывают эту неоднозначность: они неподвижны, поэтому их смещение
/// однозначно указывает на камеру.
class MarkTracker {
public:
    void SetReference(const CameraCalibration& calibration) {
        reference_ = calibration;
        has_reference_ = true;
    }

    struct Drift {
        double angle_deg = 0.0;      ///< уход угла с момента опорной калибровки
        double scale_percent = 0.0;  ///< уход масштаба, %
        bool shifted = false;        ///< превышен порог
    };

    /// Порог по углу подобран из чувствительности: 1 градус даёт около 3%
    /// систематической ошибки в ширинном участнике.
    Drift Compare(const CameraCalibration& current,
                  double angle_threshold_deg = 0.5,
                  double scale_threshold_percent = 1.0) const {
        Drift drift;
        if (!has_reference_ || !current.valid) return drift;
        drift.angle_deg = current.elevation_deg - reference_.elevation_deg;
        drift.scale_percent =
            (current.scale_mm_per_px / reference_.scale_mm_per_px - 1.0) * 100.0;
        drift.shifted = std::fabs(drift.angle_deg) > angle_threshold_deg ||
                        std::fabs(drift.scale_percent) > scale_threshold_percent;
        return drift;
    }

private:
    CameraCalibration reference_;
    bool has_reference_ = false;
};

// =============================================================================
// 3. НАБЛЮДЕНИЕ КРОМОК
// =============================================================================

/// Положения кромок зеркала в поперечной координате кадра.
///
/// Резкость перепада яркости служит мерой доверия: заросшая шлаком или
/// закрытая загрязнённым окном граница размывается раньше, чем сдвигается,
/// поэтому падение резкости предупреждает об ухудшении заранее.
struct EdgeObservation {
    double near_px = 0.0;        ///< ближняя к камере кромка
    double far_px = 0.0;         ///< дальняя
    double near_sharpness = 0.0; ///< градиент яркости на границе
    double far_sharpness = 0.0;
    bool near_found = false;
    bool far_found = false;
    double brightness = 0.0;     ///< средняя яркость в области жёлоба

    double WidthPx() const { return far_px - near_px; }
    double CenterPx() const { return (far_px + near_px) / 2.0; }
};

/// Состояние измерения — различается по кромкам и яркости.
///
/// Различение однозначно, поскольку задымления в жёлобе практически нет:
/// либо расплава мало и он скрыт бортом, либо жёлоб пуст.
enum class LevelState {
    Measured,        ///< обе кромки найдены, уровень вычислен
    Empty,           ///< кромок нет, область тёмная — жёлоб пуст
    BelowVisible,    ///< кромок нет, область яркая — уровень ниже видимого
    SingleEdge,      ///< найдена одна кромка, точность понижена
    Unreliable,      ///< резкость упала, вероятно загрязнено окно
};

// =============================================================================
// 4. ОЦЕНКА УРОВНЯ
// =============================================================================

struct LevelConfig {
    /// Порог резкости, ниже которого кромке не доверяем.
    double min_sharpness = 5.0;
    /// Яркость, выше которой считаем, что расплав в жёлобе есть.
    double melt_present_brightness = 40.0;
    /// Расхождение участников, при котором один отбрасывается, в сигмах.
    /// Отбраковка применяется только при ТРЁХ и более участниках: из двух
    /// значений нельзя определить, какое ошибочно, и медиана вырождается в
    /// большее из них, отбрасывая меньшее без всяких оснований.
    double outlier_sigmas = 3.0;
    /// Подпиксельная погрешность определения кромки.
    double edge_sigma_px = 0.2;
    /// Нижняя граница погрешности, отражающая волнение поверхности и неточность
    /// калибровки. Без неё оптическая σ выходит порядка сотых миллиметра, любое
    /// реальное расхождение выглядит как выброс в сотни сигм, и отбраковка
    /// срабатывает от малейшего возмущения.
    double systematic_sigma_mm = 0.5;
};

/// Вклад одного участника.
struct Participant {
    std::string name;
    double level_mm = 0.0;
    double sigma_mm = 0.0;    ///< собственная погрешность
    double weight = 0.0;      ///< 1/sigma^2 после отбраковки
    bool used = false;
};

struct LevelResult {
    std::optional<double> level_mm;
    double sigma_mm = 0.0;
    LevelState state = LevelState::Empty;
    std::vector<Participant> participants;
    double disagreement_mm = 0.0;   ///< разброс между участниками
    bool camera_shift_suspected = false;
};

/// Сводит показания участников в одну оценку уровня.
///
/// Веса берутся обратно пропорционально квадрату собственной погрешности
/// каждого участника, а она вычисляется из геометрической чувствительности,
/// а не задаётся вручную. Благодаря этому перераспределение веса между
/// шириной и центром происходит автоматически по ходу изменения уровня:
/// внизу диапазона ширина чувствительнее втрое, вверху втрое уступает.
class LevelEstimator {
public:
    LevelEstimator(LauncherGeometry geometry = {}, LevelConfig config = {})
        : geometry_(geometry), config_(config) {}

    LevelResult Estimate(const EdgeObservation& observation,
                         const CameraCalibration& calibration,
                         double center_reference_px,
                         bool camera_shifted = false) const {
        LevelResult result;
        result.camera_shift_suspected = camera_shifted;

        if (!calibration.valid) return result;

        // Состояние сцены определяется до всяких вычислений.
        if (!observation.near_found && !observation.far_found) {
            result.state = observation.brightness > config_.melt_present_brightness
                               ? LevelState::BelowVisible
                               : LevelState::Empty;
            return result;
        }

        const bool sharp_near = observation.near_sharpness >= config_.min_sharpness;
        const bool sharp_far = observation.far_sharpness >= config_.min_sharpness;
        if (!sharp_near && !sharp_far) {
            result.state = LevelState::Unreliable;
            return result;
        }

        if (!observation.near_found || !observation.far_found) {
            result.state = LevelState::SingleEdge;
            return result;   // одна кромка без второй уровень не даёт
        }

        // --- участник «ширина» ---
        AddWidthParticipant(observation, calibration, result);

        // --- участник «центр» ---
        // Смещение камеры двигает центр так же, как уровень, поэтому при
        // подозрении на сдвиг этот участник исключается: его показание
        // неотличимо от артефакта.
        if (!camera_shifted) {
            AddCenterParticipant(observation, calibration, center_reference_px, result);
        }

        Combine(result);
        return result;
    }

private:
    void AddWidthParticipant(const EdgeObservation& observation,
                             const CameraCalibration& calibration,
                             LevelResult& result) const {
        const double sin_beta = calibration.SinBeta();
        if (sin_beta < 1e-6) return;

        // Видимая ширина сжата перспективой: b_видимая = b·sin β.
        const double width_mm =
            observation.WidthPx() * calibration.scale_mm_per_px / sin_beta;
        const auto level = geometry_.LevelFromWidth(width_mm);
        if (!level) return;

        // Чувствительность: сколько пикселей ширины даёт миллиметр уровня.
        const double slope_px_per_mm =
            2.0 * geometry_.HalfWidthSlope(*level) * sin_beta /
            calibration.scale_mm_per_px;
        if (slope_px_per_mm < 1e-6) return;   // стенка вертикальна, ширина слепа

        Participant participant;
        participant.name = "width";
        participant.level_mm = *level;
        // Ширина — разность двух кромок, погрешности складываются квадратично.
        const double optical =
            config_.edge_sigma_px * std::sqrt(2.0) / slope_px_per_mm;
        participant.sigma_mm =
            std::hypot(optical, config_.systematic_sigma_mm);
        result.participants.push_back(participant);
    }

    void AddCenterParticipant(const EdgeObservation& observation,
                              const CameraCalibration& calibration,
                              double center_reference_px,
                              LevelResult& result) const {
        const double cos_beta = calibration.CosBeta();
        if (cos_beta < 1e-6) return;

        // Центр поднимается вместе с уровнем: p_центр = h·cos β / s + const.
        // Опорное значение соответствует нулевому уровню и задаётся при наладке.
        const double slope_px_per_mm = cos_beta / calibration.scale_mm_per_px;
        const double level =
            (observation.CenterPx() - center_reference_px) / slope_px_per_mm;
        if (level < 0.0 || level > geometry_.max_level_mm * 1.5) return;

        Participant participant;
        participant.name = "center";
        participant.level_mm = level;
        // Центр — полусумма двух кромок, погрешность вдвое меньше разности.
        const double optical =
            config_.edge_sigma_px / (std::sqrt(2.0) * slope_px_per_mm);
        participant.sigma_mm =
            std::hypot(optical, config_.systematic_sigma_mm);
        result.participants.push_back(participant);
    }

    /// Отбраковка выбросов и взвешенное объединение.
    void Combine(LevelResult& result) const {
        if (result.participants.empty()) return;

        // Расхождение считается по ВСЕМ участникам до всякой отбраковки.
        // Иначе показатель обнулялся бы именно тогда, когда он нужен: при
        // зарастании кромки один участник отбрасывается, и конфликт,
        // который следовало бы показать, исчезает из отчёта.
        double max_gap = 0.0;
        for (const Participant& a : result.participants) {
            for (const Participant& b : result.participants) {
                max_gap = std::max(max_gap, std::fabs(a.level_mm - b.level_mm));
            }
        }
        result.disagreement_mm = max_gap;

        // Отбраковка осмысленна только при трёх и более участниках.
        const bool may_reject = result.participants.size() >= 3;
        double median = 0.0;
        if (may_reject) {
            std::vector<double> levels;
            for (const Participant& p : result.participants) {
                levels.push_back(p.level_mm);
            }
            std::sort(levels.begin(), levels.end());
            median = levels[levels.size() / 2];
        }

        double sum_weight = 0.0, sum_value = 0.0;
        for (Participant& p : result.participants) {
            if (may_reject && p.sigma_mm > 1e-9 &&
                std::fabs(p.level_mm - median) > config_.outlier_sigmas * p.sigma_mm) {
                continue;   // выброс
            }
            p.weight = 1.0 / (p.sigma_mm * p.sigma_mm);
            p.used = true;
            sum_weight += p.weight;
            sum_value += p.weight * p.level_mm;
        }
        if (sum_weight < 1e-12) return;

        result.level_mm = sum_value / sum_weight;
        // Если участники расходятся сильнее собственных погрешностей, значит
        // модель их не описывает, и формальная σ занижена. Расширяем её до
        // фактического расхождения — иначе потребитель получит уверенное
        // на вид число при внутреннем конфликте измерений.
        const double formal = std::sqrt(1.0 / sum_weight);
        result.sigma_mm = std::max(formal, result.disagreement_mm / 2.0);
        result.state = LevelState::Measured;
    }

    LauncherGeometry geometry_;
    LevelConfig config_;
};

}  // namespace magma

#endif  // MAGMA_LEVEL_HPP
