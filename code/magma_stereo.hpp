// magma_stereo.hpp — измерение уровня триангуляцией по паре камер.
//
// НЕЗАВИСИМОЕ ДОПОЛНЕНИЕ. Ничего из существующих файлов не меняет и не
// требует менять — подключает magma_level.hpp только чтобы переиспользовать
// уже готовые типы StripDefinition и EdgeObservation (кромки по-прежнему
// ищет старый, нетронутый EdgeFinder — по одному разу на каждую из двух
// камер). Не подключён ни к magma_vision.cpp, ни к какому-либо другому
// файлу — включение в общий процесс зрения оставлено на следующий шаг.
//
// МОНТАЖ: БАЗА ВДОЛЬ ПОТОКА. Камеры разнесены вдоль оси жёлоба на 125 мм, а
// не поперёк неё. Из-за этого локальная «вертикальная» ось системы координат
// рига НЕ совпадает с истинной вертикалью — она повёрнута относительно неё
// на угол наклона камеры к плоскости расплава (elevation_deg). Уровень и
// ширина восстанавливаются поворотом триангулированной точки на этот угол;
// подробности и явная проверка — в StereoRigConfig и StereoLevelEstimator.
//
// СИНХРОНИЗАЦИЯ ДВУХ КАМЕР — по-разному важна для уровня и для скорости.
// Глобальный затвор устраняет перекос ВНУТРИ одного кадра, но не гарантирует,
// что два физически разных модуля сняли кадр в один и тот же момент — без
// общего аппаратного триггера рассинхрон может достигать периода кадра
// (единицы-десятки миллисекунд).
//
// Для УРОВНЯ (см. StereoLevelEstimator) это не критично: граница
// расплав-стенка — почти неподвижная деталь, смещающаяся со скоростью
// ИЗМЕНЕНИЯ УРОВНЯ (секунды), а не со скоростью потока.
//
// Для СКОРОСТИ (см. StereoSpeedEstimator ниже) — наоборот, разница времён
// между кадрами двух камер не мешающий фактор, а РАБОЧИЙ ИНГРЕДИЕНТ метода:
// без неё разделить параллакс и движение невозможно (см. раздел 6). Здесь
// нужно не отсутствие рассинхрона, а точное ЗНАНИЕ его величины.
//
// ЗАЧЕМ ЭТО ВООБЩЕ НУЖНО, ЕСЛИ ТОЧНОСТЬ ХУЖЕ. Действующий одиночный метод
// (magma_level.hpp) вычисляет уровень из ширины и центра зеркала, ПРЕДПОЛАГАЯ
// точную форму сечения жёлоба — дугу заданного радиуса. Если сечение на
// самом деле не идеальная дуга (неравномерный износ футеровки, асимметричный
// нарост шлака, ошибка в паспортном радиусе), одиночный метод унаследует эту
// ошибку молча — она нигде не проявится. Триангуляция по паре камер вычисляет
// АБСОЛЮТНЫЕ координаты точек контакта расплава со стенкой напрямую, без
// всякого предположения о форме сечения между ними. Отсюда и назначение:
// не замена, а независимая проверка — расхождение между вычисленной шириной
// и паспортной (по известному радиусу и вычисленному уровню) прямо укажет
// на реальную форму жёлоба.
//
// ОЦЕНКА ТОЧНОСТИ (расчёт приведён в сопроводительном сообщении). При базе
// 125 мм и дистанции 4 м угол параллакса — около 1.8°, что для стереопары
// очень мало (типичный подбор базы под такую дальность — 400-1000 мм).
// Отсюда ожидаемая точность по глубине — единицы миллиметров в зависимости
// от оптики, то есть СОПОСТАВИМА С РЕАЛЬНОЙ (ограниченной волнением
// поверхности) точностью одиночного метода, но заметно хуже его оптического
// предела. Короткая база — принципиальное ограничение, не лечится
// разрешением камеры.

#ifndef MAGMA_STEREO_HPP
#define MAGMA_STEREO_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <numeric>
#include <optional>
#include <vector>

#include "magma_edge.hpp"    // переиспользуем StripDefinition, EdgeObservation
#include "magma_level.hpp"   // переиспользуем CalibrationMark
#include "magma_signal.hpp"  // переиспользуем GccPhat, Fft, ParabolicInterpolation (ядро, не меняется)

namespace magma {

// =============================================================================
// Простая линейная алгебра — без внешних библиотек, только то, что нужно
// =============================================================================

using Vec3 = std::array<double, 3>;

inline Vec3 Add(const Vec3& a, const Vec3& b) { return {a[0]+b[0], a[1]+b[1], a[2]+b[2]}; }
inline Vec3 Sub(const Vec3& a, const Vec3& b) { return {a[0]-b[0], a[1]-b[1], a[2]-b[2]}; }
inline Vec3 Scale(const Vec3& a, double k) { return {a[0]*k, a[1]*k, a[2]*k}; }
inline double Dot(const Vec3& a, const Vec3& b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
inline double Norm(const Vec3& a) { return std::sqrt(Dot(a, a)); }
inline Vec3 Normalized(const Vec3& a) {
    const double n = Norm(a);
    return n > 1e-12 ? Scale(a, 1.0 / n) : Vec3{0.0, 0.0, 1.0};
}

// =============================================================================
// 1. Геометрия рига: две камеры, известная база, известный угол схождения
// =============================================================================

/// Параметры жёсткого узла из двух камер.
///
/// Система координат рига: X — вдоль базы, Z — общее номинальное направление
/// взгляда (биссектриса осей камер), Y = Z × X.
///
/// МОНТАЖ — БАЗА ВДОЛЬ ПОТОКА. Камеры разнесены вдоль оси жёлоба (не поперёк
/// неё), а направление взгляда, как и у одиночной камеры, лежит в плоскости
/// «поперёк потока — вертикаль» и наклонено к горизонту на угол elevation_deg.
/// При таком монтаже ось Y рига НЕ совпадает с истинной вертикалью — она
/// повёрнута относительно неё ровно на elevation_deg в плоскости «поперёк —
/// вертикаль» (проверено явным вычислением: Y риг = (0, sinβ, cosβ) в мировых
/// координатах (вдоль потока, поперёк, вертикаль) — обе составляющие не равны
/// нулю сразу при любом β, отличном от 0 или 90°). Наивное «уровень = Y рига»
/// поэтому неверно и даёт смесь уровня с поперечным смещением; правильные
/// формулы — в StereoLevelEstimator::Estimate, они поворачивают триангулированную
/// точку на угол elevation_deg перед тем, как разделить её на уровень и ширину.
///
/// Если риг когда-нибудь перемонтируют базой ПОПЕРЁК потока, elevation_deg
/// в этих формулах становится ненужным (поворот вырождается в тождество) —
/// но пересчитывать формулы обратно тогда придётся отдельно, это не текущий
/// случай.
struct StereoRigConfig {
    double baseline_mm = 125.0;
    /// Угол схождения осей камер, градусов, ПОЛНЫЙ (между осями), а не
    /// половинный. Каждая камера довёрнута на половину этого угла к центру.
    /// Малое положительное значение (единицы градусов) — «схождение»,
    /// ноль — параллельные оси. Само по себе схождение почти не меняет
    /// точность по глубине — угол мал, а поправка на него в расчёте точная
    /// (не приближение), поэтому подбирать его стоит по качеству картинки
    /// (чтобы обе камеры видели рабочую область ближе к центру кадра, где
    /// меньше искажения объектива), а не по формуле точности.
    double convergence_deg = 2.0;

    /// Угол камеры к плоскости расплава (тот же β, что и в CameraCalibration
    /// одиночного метода) — НУЖЕН именно из-за монтажа базы вдоль потока:
    /// без него нельзя разделить триангулированную точку на уровень и ширину.
    double elevation_deg = 31.0;

    /// Оптика — предполагается одинаковой для обеих камер.
    /// Пересчитано с 50 мм на 30 мм 2026-09-14 (см. calibration.scale_mm_per_px
    /// в magma_vision.conf) — если объектив снова сменится, поправьте вместе.
    double focal_length_mm = 30.0;
    double pixel_size_um = 1.55;

    /// Точка нулевого уровня (мм, в уже спроецированной, «истинной»
    /// вертикальной шкале — см. Estimate). Играет ту же роль, что
    /// center_reference_px в одиночном методе: определяется однократно по
    /// кадру с известным уровнем.
    double level_reference_mm = 0.0;

    /// Знак соответствия «спроецированная высота растёт» -> «уровень растёт».
    /// Определяется на месте тем же простым опытом, что и раньше.
    double level_sign = 1.0;

    double FocalLengthPx() const { return focal_length_mm * 1000.0 / pixel_size_um; }
};

/// Положение и ориентация одной камеры в системе координат рига.
struct CameraExtrinsics {
    Vec3 position{0.0, 0.0, 0.0};
    double yaw_rad = 0.0;   // поворот вокруг оси Y (схождение)
};

struct StereoExtrinsics {
    CameraExtrinsics left;
    CameraExtrinsics right;
};

/// Построить внешние параметры обеих камер из параметров рига.
///
/// Левая камера довёрнута на +half к центру (к оси +X), правая — на -half:
/// при таком выборе знаков оси камер сходятся впереди рига, а не расходятся.
inline StereoExtrinsics BuildExtrinsics(const StereoRigConfig& rig) {
    const double half = rig.convergence_deg * M_PI / 180.0 / 2.0;
    StereoExtrinsics result;
    result.left.position = {-rig.baseline_mm / 2.0, 0.0, 0.0};
    result.left.yaw_rad = half;
    result.right.position = {rig.baseline_mm / 2.0, 0.0, 0.0};
    result.right.yaw_rad = -half;
    return result;
}

// =============================================================================
// 2. Луч из пикселя и пересечение двух лучей
// =============================================================================

struct Ray {
    Vec3 origin;
    Vec3 direction;   // нормирован
};

/// Внутренние параметры камеры (принцип камеры-обскуры, дисторсия не
/// учитывается — при небольшом угле обзора, который даёт длиннофокусный
/// объектив на такой дистанции, она пренебрежимо мала).
struct CameraIntrinsics {
    double focal_length_px = 0.0;
    double principal_x_px = 0.0;   // как правило — центр кадра
    double principal_y_px = 0.0;
};

/// Построить луч из пикселя изображения одной камеры, в системе координат рига.
///
/// Ось Y изображения растёт вниз (обычное соглашение для пикселей), а ось Y
/// рига выбрана растущей вверх (см. StereoRigConfig) — отсюда знак минус
/// при вертикальной координате направления.
inline Ray CameraRay(const CameraExtrinsics& camera, const CameraIntrinsics& intrinsics,
                     double pixel_x, double pixel_y) {
    const double dx = pixel_x - intrinsics.principal_x_px;
    const double dy = -(pixel_y - intrinsics.principal_y_px);
    const Vec3 direction_in_camera{dx, dy, intrinsics.focal_length_px};

    // Поворот вокруг оси Y рига (схождение — только рысканье, без крена
    // и тангажа: обе камеры смотрят из одной горизонтальной плоскости).
    const double c = std::cos(camera.yaw_rad), s = std::sin(camera.yaw_rad);
    const Vec3 rotated{
        c * direction_in_camera[0] + s * direction_in_camera[2],
        direction_in_camera[1],
        -s * direction_in_camera[0] + c * direction_in_camera[2],
    };

    return {camera.position, Normalized(rotated)};
}

/// Результат пересечения двух лучей — точка минимального расстояния между
/// ними (сами лучи из-за ошибок измерения почти никогда не пересекаются
/// точно) и величина этого расстояния как мера согласия.
struct TriangulationResult {
    Vec3 point{0.0, 0.0, 0.0};
    /// Расстояние между ближайшими точками двух лучей, мм. Ноль означало бы
    /// точное пересечение; практически — несколько сотых при верной
    /// калибровке и хорошем соответствии кромок. Резкий рост означает, что
    /// либо калибровка рига разъехалась, либо кромки на двух кадрах
    /// перепутаны местами (например, из-за иного освещения одна из камер
    /// нашла не тот перепад яркости).
    double ray_gap_mm = 0.0;
    bool valid = false;
};

/// Классическая формула ближайших точек двух скрещивающихся прямых в
/// пространстве. Направления считаются уже нормированными.
inline TriangulationResult Triangulate(const Ray& first, const Ray& second) {
    TriangulationResult result;

    const Vec3 w0 = Sub(first.origin, second.origin);
    const double a = 1.0;  // Dot(d1,d1), направления нормированы
    const double b = Dot(first.direction, second.direction);
    const double c = 1.0;  // Dot(d2,d2)
    const double d = Dot(first.direction, w0);
    const double e = Dot(second.direction, w0);
    const double denominator = a * c - b * b;

    // Знаменатель близок к нулю, когда лучи почти параллельны — при базе
    // 125 мм и дистанции метры такого не бывает при разумной геометрии,
    // но проверка защищает от деления на ноль при ошибке в конфигурации.
    if (std::fabs(denominator) < 1e-9) return result;

    const double sc = (b * e - c * d) / denominator;
    const double tc = (a * e - b * d) / denominator;

    // Отрицательный параметр означает точку позади камеры — верный признак
    // того, что кромки на двух изображениях не соответствуют друг другу.
    if (sc < 0.0 || tc < 0.0) return result;

    const Vec3 p1 = Add(first.origin, Scale(first.direction, sc));
    const Vec3 p2 = Add(second.origin, Scale(second.direction, tc));

    result.point = Scale(Add(p1, p2), 0.5);
    result.ray_gap_mm = Norm(Sub(p1, p2));
    result.valid = true;
    return result;
}

// =============================================================================
// 3. Восстановление 2D-координаты кромки из EdgeObservation
// =============================================================================

/// EdgeObservation хранит положение кромки как индекс в профиле (см.
/// magma_edge.hpp), а не как готовые координаты пикселя. Чтобы получить
/// точку на изображении для луча, нужно повторить ту же формулу, что строит
/// профиль в EdgeFinder::ExtractProfile — центр полосы плюс смещение вдоль
/// направления «поперёк потока».
///
/// ДОПУЩЕНИЕ: используется только центральная линия полосы (без учёта
/// усреднения поперёк неё), и предполагается, что strip.samples примерно
/// равно strip.length_px — то же самое допущение (шаг ~1 пиксель на
/// отсчёт), что уже неявно принято в существующих файлах конфигурации, где
/// эти два числа всегда заданы равными.
inline std::array<double, 2> StripPointPx(const StripDefinition& strip,
                                          double profile_position) {
    const double angle = strip.across_angle_deg * M_PI / 180.0;
    const double step = strip.samples > 1
        ? strip.length_px / static_cast<double>(strip.samples - 1) : 1.0;
    const double offset = -strip.length_px / 2.0 + step * profile_position;
    return {strip.center_x_px + std::cos(angle) * offset,
            strip.center_y_px + std::sin(angle) * offset};
}

// =============================================================================
// 4. Уровень по паре камер
// =============================================================================

struct StereoLevelResult {
    std::optional<double> level_mm;
    std::optional<double> width_mm;
    /// Разность координаты «вдоль потока» между ближней и дальней
    /// триангулированными точками. При базе вдоль потока эта координата
    /// восстанавливается напрямую (без всякого поворота — см. Estimate) и в
    /// норме должна быть близка к нулю: обе точки лежат на одном поперечном
    /// сечении, которое наблюдают обе камеры. Заметное отклонение от нуля —
    /// признак того, что полосы выборки на двух камерах на самом деле смотрят
    /// на разные сечения жёлоба, а не на общий диагностический бонус —
    /// такой проверки не было бы при монтаже базы поперёк потока.
    double along_flow_mismatch_mm = 0.0;
    double near_gap_mm = 0.0;   ///< согласие лучей на ближней кромке
    double far_gap_mm = 0.0;    ///< согласие лучей на дальней кромке
    /// Расхождение уровня, полученного отдельно по ближней и по дальней
    /// точке (обе видят один и тот же уровень зеркала, поэтому в норме
    /// должны совпадать) — самостоятельная проверка на манер той, что
    /// сравнивает участников «ширина» и «центр» в одиночном методе.
    double edge_disagreement_mm = 0.0;
};

/// Оценивает уровень триангуляцией по кромкам, найденным независимо на двух
/// изображениях. Кромки ищет обычный, нетронутый EdgeFinder — по одному разу
/// на каждую камеру; сюда передаются уже готовые результаты.
class StereoLevelEstimator {
public:
    explicit StereoLevelEstimator(StereoRigConfig rig) : rig_(rig) {
        extrinsics_ = BuildExtrinsics(rig_);
    }

    /// left_strip/right_strip — где на СВОЁМ изображении каждая камера ищет
    /// кромки; эти координаты, в отличие от одиночного метода, не обязаны
    /// совпадать между камерами, поскольку изображения не совмещены пиксель
    /// в пиксель.
    StereoLevelResult Estimate(const EdgeObservation& left, const EdgeObservation& right,
                               const StripDefinition& left_strip,
                               const StripDefinition& right_strip,
                               std::size_t image_width_px,
                               std::size_t image_height_px) const {
        StereoLevelResult result;
        if (!left.near_found || !left.far_found) return result;
        if (!right.near_found || !right.far_found) return result;

        CameraIntrinsics intrinsics;
        intrinsics.focal_length_px = rig_.FocalLengthPx();
        intrinsics.principal_x_px = static_cast<double>(image_width_px) / 2.0;
        intrinsics.principal_y_px = static_cast<double>(image_height_px) / 2.0;

        const auto left_near_px = StripPointPx(left_strip, left.near_px);
        const auto left_far_px = StripPointPx(left_strip, left.far_px);
        const auto right_near_px = StripPointPx(right_strip, right.near_px);
        const auto right_far_px = StripPointPx(right_strip, right.far_px);

        const Ray left_near_ray =
            CameraRay(extrinsics_.left, intrinsics, left_near_px[0], left_near_px[1]);
        const Ray right_near_ray =
            CameraRay(extrinsics_.right, intrinsics, right_near_px[0], right_near_px[1]);
        const Ray left_far_ray =
            CameraRay(extrinsics_.left, intrinsics, left_far_px[0], left_far_px[1]);
        const Ray right_far_ray =
            CameraRay(extrinsics_.right, intrinsics, right_far_px[0], right_far_px[1]);

        const TriangulationResult near_point = Triangulate(left_near_ray, right_near_ray);
        const TriangulationResult far_point = Triangulate(left_far_ray, right_far_ray);
        result.near_gap_mm = near_point.ray_gap_mm;
        result.far_gap_mm = far_point.ray_gap_mm;
        if (!near_point.valid || !far_point.valid) return result;

        // Координата X триангулированной точки — это координата «вдоль
        // потока» НАПРЯМУЮ (ось X рига построена вдоль базы, а база вдоль
        // потока), поворота не требует. Обе точки должны лежать на одном и
        // том же сечении — заметная разность выдаёт рассогласование полос
        // выборки на двух камерах.
        result.along_flow_mismatch_mm = std::fabs(far_point.point[0] - near_point.point[0]);

        // А вот (Y, Z) рига НЕ являются (вертикаль, что угодно) при базе
        // вдоль потока: ось Y повёрнута относительно истинной вертикали на
        // угол elevation_deg (см. обоснование в StereoRigConfig). Уровень и
        // поперечное положение получаются поворотом на этот угол — той же
        // операцией, что и проекция одиночной камеры (CameraCalibration::
        // ProjectAcross в magma_level.hpp), только в явном виде и без
        // деления на масштаб, поскольку триангуляция уже даёт метры, а не
        // пиксели.
        const double sin_b = std::sin(rig_.elevation_deg * M_PI / 180.0);
        const double cos_b = std::cos(rig_.elevation_deg * M_PI / 180.0);

        const double near_level =
            rig_.level_sign * (near_point.point[1] * cos_b + near_point.point[2] * sin_b -
                               rig_.level_reference_mm);
        const double far_level =
            rig_.level_sign * (far_point.point[1] * cos_b + far_point.point[2] * sin_b -
                               rig_.level_reference_mm);

        result.level_mm = (near_level + far_level) / 2.0;
        result.edge_disagreement_mm = std::fabs(near_level - far_level);

        // Ширина — из проекции на ось «поперёк потока» той же точки, а не
        // из координаты X (та отдана под «вдоль потока», см. выше).
        const double near_across = near_point.point[1] * sin_b - near_point.point[2] * cos_b;
        const double far_across = far_point.point[1] * sin_b - far_point.point[2] * cos_b;
        result.width_mm = std::fabs(far_across - near_across);

        return result;
    }

private:
    StereoRigConfig rig_;
    StereoExtrinsics extrinsics_;
};

// =============================================================================
// 5. Теоретическая точность — вспомогательный расчёт для подбора параметров
// =============================================================================

/// Ожидаемое стандартное отклонение оценки глубины при заданных параметрах
/// рига, дистанции до цели и повторяемости поиска кромки ОДНОЙ камерой.
///
/// Формула приближённая (предполагает параллельные оси), но при малых углах
/// схождения, какие здесь и предполагаются, погрешность приближения
/// пренебрежимо мала по сравнению с самой оцениваемой величиной. Полезна на
/// этапе выбора оптики и базы — до всякого макета, из одной формулы видно,
/// оправдана ли короткая база выбранным объективом.
inline double EstimateDepthSigmaMm(const StereoRigConfig& rig, double distance_mm,
                                   double edge_repeatability_px) {
    const double disparity_sigma_px = edge_repeatability_px * std::sqrt(2.0);
    const double focal_px = rig.FocalLengthPx();
    if (focal_px < 1e-9 || rig.baseline_mm < 1e-9) return 0.0;
    return (distance_mm * distance_mm) / (focal_px * rig.baseline_mm) * disparity_sigma_px;
}

// =============================================================================
// 6. Скорость по паре камер: диспаратность движущейся текстуры с поправкой
//    на неодновременность съёмки
// =============================================================================
//
// ПОЧЕМУ ЭТО ВООБЩЕ НЕТРИВИАЛЬНО. Диспаратность (сдвиг узора между левым и
// правым кадром) движущейся текстуры содержит СРАЗУ ДВЕ причины: параллакс,
// зависящий от глубины точки, и физическое смещение расплава за время Δt
// между моментами съёмки двух камер (общего затвора между ними нет — см.
// обсуждение синхронизации в шапке файла). Одно наблюдение — одно уравнение
// с двумя неизвестными, разделить их по нему одному нельзя.
//
// КАК РАЗДЕЛЯЕТСЯ. Калибровочные метки на бортах жёлоба неподвижны, поэтому
// их диспаратность в ТОЙ ЖЕ паре кадров — чистый параллакс, без всякой
// примеси скорости. Текстура расплава лежит примерно на той же глубине, что
// и метки (уровень известен, метки рядом с зеркалом), поэтому параллакс у
// неё почти такой же. Вычитание убирает общую (неизвестную, но одинаковую)
// глубинную составляющую и оставляет только вклад движения:
//
//   диспаратность(текстура) − диспаратность(меток) ≈ v · Δt / масштаб
//
// Отсюда v = [диспаратность(текстура) − диспаратность(меток)] · mm_per_px / Δt.
//
// ЭТО НЕ ПОЛНАЯ 3D-ТРИАНГУЛЯЦИЯ ТОЧКИ, а работа напрямую с пиксельными
// диспаратностями — сознательный выбор, а не упрощение по недосмотру.
// Приближение «текстура и метки на одной глубине» не хуже, чем неизбежная
// погрешность самой триангуляции при короткой базе (см. оценку точности
// выше), а лишний шаг через 3D-координаты добавил бы риск ещё одной
// геометрической ошибки того рода, что уже дважды находилась в этом файле.
//
// ТРЕБОВАНИЕ К Δt: ДОЛЖНО БЫТЬ ИЗВЕСТНО ТОЧНО И НЕ БЛИЗКО К НУЛЮ. Метод
// работает от джиттера рассинхронизации, а не вопреки ему — при Δt→0 формула
// вырождается в 0/0, и если камеры когда-нибудь получат общий аппаратный
// триггер (см. обсуждение выше), этот способ измерения скорости перестанет
// работать и его придётся либо убрать, либо намеренно сдвигать триггер одной
// из камер на небольшую фиксированную задержку. Источник Δt — метка времени
// самого кадра (в идеале аппаратная, из буфера V4L2, а не момент, когда её
// забрал процессор); передаётся вызывающей стороной, этот файл её не измеряет.

struct StereoSpeedConfig {
    double mm_per_px = 0.4135;      ///< масштаб вдоль потока (см. CameraCalibration; объектив 30 мм)

    /// Минимальный по модулю Δt, при котором оценка ещё считается надёжной.
    /// Меньшие значения дают деление на почти-ноль и разброс, а не точность.
    double min_delta_t_s = 0.001;

    /// Границы поиска диспаратности текстуры, отсчётов профиля (≈пиксели).
    /// Ограничивает перебор физически правдоподобным диапазоном и защищает
    /// от ложного пика вдалеке от истинного при периодичной текстуре —
    /// тот же приём, что и в остальных корреляционных оценках проекта.
    int max_disparity_px = 200;
    int snr_exclusion_px = 10;
    double min_snr = 4.0;

    /// Разумные пределы скорости — отбрасывают то, что вышло за пределы
    /// физически ожидаемого потока (означает всплеск, а не измерение).
    double min_speed_ms = 0.1;
    double max_speed_ms = 6.0;

    std::size_t min_observations = 5;   ///< сколько пар набрать перед оценкой
    std::size_t history_size = 60;
};

struct StereoSpeedResult {
    std::optional<double> speed_ms;
    double spread_ms = 0.0;         ///< межквартильный разброс — мера доверия
    std::size_t accepted = 0;
    std::size_t attempted = 0;
};

/// Копит оценки скорости по последовательным парам кадров внутри серии и
/// сводит их в одно значение. Каждая пара обрабатывается независимо от
/// остальных — метод не требует прослеживать один и тот же элемент текстуры
/// через разные пары, только различие диспаратностей ВНУТРИ одной пары.
class StereoSpeedEstimator {
public:
    explicit StereoSpeedEstimator(StereoSpeedConfig config = {}) : config_(config) {}

    void Reset() { history_.clear(); attempted_ = 0; }

    /// Обработать одну пару кадров.
    ///
    /// left_profile/right_profile — профили яркости вдоль потока (тот же вид
    /// данных, что ProfileExtractor уже готовит для одиночного метода), из
    /// левой и правой камеры, снятые примерно в один момент.
    /// left_timestamp_s/right_timestamp_s — метки времени этих кадров, в
    /// одной временной шкале (например, от аппаратного штампа V4L2).
    /// left_marks/right_marks — калибровочные метки, увиденные каждой
    /// камерой на этой же паре кадров (сопоставляются по имени).
    void AddObservation(const std::vector<double>& left_profile,
                        const std::vector<double>& right_profile,
                        double left_timestamp_s, double right_timestamp_s,
                        const std::vector<CalibrationMark>& left_marks,
                        const std::vector<CalibrationMark>& right_marks) {
        ++attempted_;

        const double delta_t = right_timestamp_s - left_timestamp_s;
        if (std::fabs(delta_t) < config_.min_delta_t_s) return;

        const auto marks_disparity = MatchedMarksDisparity(left_marks, right_marks);
        if (!marks_disparity) return;   // нет ни одной общей метки — не с чем сравнивать

        if (left_profile.size() < 32 || left_profile.size() != right_profile.size()) return;

        const auto texture_disparity = TextureDisparity(left_profile, right_profile);
        if (!texture_disparity) return;   // пик ниже порога достоверности

        const double speed_ms =
            (*texture_disparity - *marks_disparity) * config_.mm_per_px / delta_t / 1000.0;

        if (std::fabs(speed_ms) < config_.min_speed_ms ||
            std::fabs(speed_ms) > config_.max_speed_ms) {
            return;   // вне физически правдоподобного диапазона — похоже на выброс
        }

        history_.push_back(speed_ms);
        if (history_.size() > config_.history_size) history_.pop_front();
    }

    /// Свести накопленные пары в одну оценку — медиана устойчива к единичным
    /// выбросам ровно по той же причине, что и в остальных частях проекта.
    StereoSpeedResult Estimate() const {
        StereoSpeedResult result;
        result.attempted = attempted_;
        result.accepted = history_.size();
        if (history_.size() < config_.min_observations) return result;

        std::vector<double> sorted(history_.begin(), history_.end());
        std::sort(sorted.begin(), sorted.end());
        const std::size_t middle = sorted.size() / 2;
        result.speed_ms = sorted.size() % 2 == 1
            ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2.0;
        result.spread_ms = sorted[sorted.size() * 3 / 4] - sorted[sorted.size() / 4];
        return result;
    }

private:
    /// Средняя диспаратность (право минус лево, в пикселях) по меткам,
    /// присутствующим в ОБОИХ списках (сопоставление по имени). Метки не
    /// движутся, поэтому их диспаратность в этой же паре кадров — чистый
    /// параллакс, без примеси скорости, — то, ради чего они здесь и нужны.
    static std::optional<double> MatchedMarksDisparity(
        const std::vector<CalibrationMark>& left_marks,
        const std::vector<CalibrationMark>& right_marks) {
        double sum = 0.0;
        std::size_t count = 0;
        for (const CalibrationMark& left : left_marks) {
            for (const CalibrationMark& right : right_marks) {
                if (left.name != right.name) continue;
                sum += right.image_x_px - left.image_x_px;
                ++count;
                break;
            }
        }
        if (count == 0) return std::nullopt;
        return sum / static_cast<double>(count);
    }

    /// Диспаратность движущейся текстуры: положение пика взаимной корреляции
    /// профилей левой и правой камеры, в отсчётах (≈пикселях), с проверкой
    /// достоверности и субпиксельным уточнением — та же машина, что и в
    /// одиночном методе (magma_flow.hpp), применённая между камерами вместо
    /// разнесённых во времени кадров одной камеры.
    std::optional<double> TextureDisparity(const std::vector<double>& left,
                                           const std::vector<double>& right) const {
        std::vector<double> a = left, b = right;
        // Center() из ядра (magma_signal.hpp) центрирует вектор И возвращает
        // его СКО за один проход — тем самым избавляет от отдельного вызова
        // Deviation() по уже центрированным данным, то есть от двух лишних
        // проходов по 900-элементному профилю на каждую пару кадров.
        const double sigma_a = Center(a);
        const double sigma_b = Center(b);
        if (sigma_a < 1e-9 || sigma_b < 1e-9) return std::nullopt;

        // Порядок аргументов важен: GccPhat(a, b) отдаёт положение пика как
        // сдвиг A относительно B, а нужен сдвиг ПРАВОГО кадра относительно
        // ЛЕВОГО («право минус лево», та же система знаков, что и у меток
        // выше). Перепутанный порядок был найден именно на этом прогоне —
        // синтетика с известной скоростью 1.5 м/с возвращала корректную по
        // модулю, но противоположную по знаку диспаратность.
        const std::vector<double> correlation = GccPhat(b, a);
        const std::size_t zero = a.size() - 1;
        const int bound = std::min(config_.max_disparity_px,
                                   static_cast<int>(zero));

        std::size_t peak = zero;
        double best = -1e300;
        for (int shift = -bound; shift <= bound; ++shift) {
            // Индекс собирается через ЗНАКОВУЮ арифметику и лишь затем
            // переводится в std::size_t: приведение отрицательного int к
            // size_t напрямую даёт огромное беззнаковое число, а не «минус
            // столько-то», и «zero + это» улетает далеко за пределы массива.
            // Именно эта ошибка обнаружилась на первом же прогоне синтетики
            // — результат уходил в область, не имеющую отношения к сигналу.
            const std::size_t index =
                static_cast<std::size_t>(static_cast<long>(zero) + shift);
            if (correlation[index] > best) { best = correlation[index]; peak = index; }
        }

        std::vector<double> noise;
        noise.reserve(correlation.size());
        for (std::size_t i = 0; i < correlation.size(); ++i) {
            const long distance =
                std::labs(static_cast<long>(i) - static_cast<long>(peak));
            if (distance > config_.snr_exclusion_px) noise.push_back(correlation[i]);
        }
        if (noise.size() < 10) return std::nullopt;

        const double mean =
            std::accumulate(noise.begin(), noise.end(), 0.0) /
            static_cast<double>(noise.size());
        double variance = 0.0;
        for (double value : noise) variance += (value - mean) * (value - mean);
        const double deviation = std::sqrt(variance / static_cast<double>(noise.size()));
        if (deviation < 1e-12) return std::nullopt;

        const double snr = (correlation[peak] - mean) / deviation;
        if (snr < config_.min_snr) return std::nullopt;

        const double offset = ParabolicInterpolation(correlation, peak);
        return static_cast<double>(static_cast<long>(peak) - static_cast<long>(zero)) + offset;
    }

    StereoSpeedConfig config_;
    std::deque<double> history_;
    std::size_t attempted_ = 0;
};

}  // namespace magma

#endif  // MAGMA_STEREO_HPP
