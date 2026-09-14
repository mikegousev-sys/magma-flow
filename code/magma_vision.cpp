// magma_vision.cpp — процесс машинного зрения: скорость по камере и уровень.
//
// САМОСТОЯТЕЛЬНЫЙ ПРОЦЕСС, СВОЙ ПРОТОКОЛ, СВОЙ ПОРТ. Отказ камеры или её
// драйвера не должен ронять измерение скорости по фотодиодам — это основная
// функция системы, зрение вспомогательная. Камере нужны заголовки V4L2 и
// скрипт производителя, тогда как защищённое ядро (magma_server.cpp) собирается
// без единой внешней зависимости — смешивать это незачем. Наконец, формат
// вывода зрения и его параметры будут меняться при наладке заметно чаще,
// чем формат ядра, поэтому у зрения СВОЙ протокол (magma_vision_protocol.hpp)
// и СВОЙ порт (по умолчанию 9100): ядро не затронуто ни в одной точке, и
// изменения здесь никогда не потребуют трогать защищённые файлы.
//
// Камеру может открыть только один процесс, поэтому скорость и уровень
// считаются здесь оба — из одной серии кадров получаются обе величины.
//
// НАСТРОЙКА НА МЕСТЕ, БЕЗ ПЕРЕЗАПУСКА. Угол камеры, масштаб, положение полос
// выборки, координаты калибровочных меток невозможно знать заранее — они
// подбираются итеративно на установленном оборудовании. Все параметры такого
// рода читаются из текстового файла настроек (magma_config.hpp) и
// перечитываются между сериями: правка файла применяется к следующей серии
// без перезапуска процесса. Не перечитываются на ходу только сетевые
// параметры вещания (адрес и порт) — перепривязка слушающего сокета без
// остановки процесса требует отдельной синхронизации с принимающим потоком
// и является редкой, заведомо разовой операцией; см. ApplySettings().
//
// Сборка:
//   g++ -std=c++17 -O2 -pthread magma_vision.cpp -o magma_vision
// Запуск:
//   ./magma_vision [путь_к_файлу_настроек]   (по умолчанию ./magma_vision.conf)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "magma_broadcast.hpp"
#include "magma_capture.hpp"
#include "magma_config.hpp"
#include "magma_edge.hpp"
#include "magma_exposure.hpp"
#include "magma_flow.hpp"
#include "magma_level.hpp"
#include "magma_veye.hpp"
#include "magma_vision_protocol.hpp"

namespace magma {

inline double MonotonicSeconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// =============================================================================
// Диагностика: сохранение кадра и текстовой сводки для наладки на месте
// =============================================================================

/// Записать кадр в формате PGM (простейший растровый формат с текстовым
/// заголовком, который открывает любой просмотрщик изображений без
/// дополнительных библиотек на стороне процесса).
///
/// Копирование построчно учитывает шаг строки кадра (stride): драйвер может
/// выравнивать строки шире фактической ширины изображения, и построчная
/// запись только первых width байт исключает попадание этого выравнивания
/// в файл.
inline bool SavePgm(const std::string& path, const GrayImage& image) {
    std::FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) return false;
    std::fprintf(file, "P5\n%zu %zu\n255\n", image.width, image.height);
    const std::size_t stride = image.Stride();
    for (std::size_t y = 0; y < image.height; ++y) {
        std::fwrite(image.data + y * stride, 1, image.width, file);
    }
    std::fclose(file);
    return true;
}

// =============================================================================
// Конфигурация процесса — то, что не подбирается на месте и не хранится
// в файле настроек (путь к файлу, порт вещания начального запуска).
// =============================================================================

struct StartupConfig {
    std::string config_path = "./magma_vision.conf";
};

// =============================================================================
// Процесс
// =============================================================================

/// Связывает камеру, расчёт скорости и уровня, адаптивную выдержку,
/// горячую перезагрузку настроек и раздачу результата по сети.
class VisionProcess {
public:
    explicit VisionProcess(StartupConfig startup) : startup_(std::move(startup)) {}

    void RequestStop() { stop_.store(true); }

    int Run() {
        if (!config_.Load(startup_.config_path)) {
            std::printf("[VISION] Файл настроек %s не найден — использую значения "
                        "по умолчанию (см. пример magma_vision.conf)\n",
                        startup_.config_path.c_str());
        }
        ApplySettings(/*is_reload=*/false);

        std::thread broadcast_thread([this] { broadcaster_->ServeForever(); });

        double next_burst = MonotonicSeconds();
        while (!stop_.load()) {
            if (config_.ReloadIfChanged()) {
                std::printf("[VISION] Файл настроек изменился, применяю новые значения\n");
                ApplySettings(/*is_reload=*/true);
            }

            RunBurst();

            // Планирование от РАСЧЁТНОГО времени предыдущего такта, а не от
            // фактического: иначе длительность серии и её обработки
            // прибавлялась бы к периоду, и расписание постепенно отставало
            // бы от заданного интервала.
            next_burst += schedule_.burst_interval_s;
            const double now = MonotonicSeconds();
            if (next_burst < now) next_burst = now + schedule_.burst_interval_s;
            SleepUntil(next_burst);
        }

        stop_.store(true);
        if (camera_) camera_->Close();
        if (camera2_) camera2_->Close();
        broadcaster_->CloseAll();
        if (broadcast_thread.joinable()) broadcast_thread.join();
        std::printf("[VISION] Завершение\n");
        return 0;
    }

private:
    // -------------------------------------------------------------------
    // Расписание и второстепенные настройки, хранятся как простые поля
    // -------------------------------------------------------------------
    struct Schedule {
        double burst_interval_s = 10.0;
        std::size_t burst_frames = 120;
        std::size_t level_frames = 9;
        std::size_t warmup_frames = 3;
    };

    void SleepUntil(double moment) const {
        while (!stop_.load() && MonotonicSeconds() < moment) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    // -------------------------------------------------------------------
    // Применение настроек: читает файл, обновляет все узлы. Вызывается один
    // раз при старте и затем всякий раз, когда ConfigFile сообщает, что
    // файл на диске изменился.
    // -------------------------------------------------------------------
    void ApplySettings(bool is_reload) {
        ApplyCaptureAndVeye();
        ApplySchedule();
        ApplyGeometryAndThresholds();
        ApplyStrips();
        ApplyCalibration();
        ApplyExposure();
        ApplySecondCamera();
        ApplyOutput(is_reload);

        config_version_ = VisionIsoTimestamp();
        std::printf("[VISION] Настройки применены (%s)%s\n", config_version_.c_str(),
                    is_reload ? ", без перезапуска процесса" : "");
    }

    /// Параметры камеры и скрипта управления. Устройство/разрешение/частота
    /// кадров — структурные параметры: их изменение требует переоткрыть
    /// камеру, поэтому сравниваются с прежними через простую текстовую
    /// «подпись» перед тем, как решить, пересоздавать ли Camera.
    void ApplyCaptureAndVeye() {
        CaptureConfig next_capture;
        next_capture.device = config_.GetString("camera.device", "/dev/video0");
        next_capture.width = config_.GetSize("camera.width", 1280);
        next_capture.height = config_.GetSize("camera.height", 1024);
        next_capture.frame_rate = config_.GetDouble("camera.frame_rate", 60.0);
        next_capture.buffer_count = config_.GetSize("camera.buffer_count", 6);
        next_capture.timeout_ms = config_.GetInt("camera.timeout_ms", 2000);

        const std::string signature =
            next_capture.device + "|" + std::to_string(next_capture.width) + "|" +
            std::to_string(next_capture.height) + "|" +
            std::to_string(next_capture.frame_rate);

        capture_config_ = next_capture;
        if (signature != capture_signature_ || !camera_) {
            capture_signature_ = signature;
            camera_needs_reopen_ = true;   // фактическое переоткрытие — в RunBurst
        }

        VeyeConfig next_veye;
        next_veye.script = config_.GetString("veye.script",
                                             "/home/mike/veye/mv_mipi_i2c_new.sh");
        next_veye.i2c_bus = config_.GetInt("veye.i2c_bus", 10);

        // Если поменялась цель управления (другой скрипт или другая шина),
        // ручной режим и частота кадров должны быть отправлены заново —
        // сброс флага делает это в начале следующей серии. Без этого правка
        // veye.script в файле настроек молча продолжала бы считаться
        // «уже настроенной» и ничего бы не переслала новому получателю.
        if (next_veye.script != veye_config_.script ||
            next_veye.i2c_bus != veye_config_.i2c_bus) {
            veye_camera_configured_once_ = false;
        }
        veye_config_ = next_veye;
        veye_ = VeyeControl(veye_config_);   // объект лёгкий, пересоздаём без раздумий
    }

    void ApplySchedule() {
        schedule_.burst_interval_s = config_.GetDouble("schedule.burst_interval_s", 10.0);
        schedule_.burst_frames = config_.GetSize("schedule.burst_frames", 120);
        schedule_.level_frames = config_.GetSize("schedule.level_frames", 9);
        schedule_.warmup_frames = config_.GetSize("schedule.warmup_frames", 3);
    }

    void ApplyGeometryAndThresholds() {
        geometry_.channel_radius_mm = config_.GetDouble("geometry.channel_radius_mm", 150.0);
        geometry_.max_level_mm = config_.GetDouble("geometry.max_level_mm", 105.0);

        level_config_.min_sharpness = config_.GetDouble("level.min_sharpness", 5.0);
        level_config_.melt_present_brightness =
            config_.GetDouble("level.melt_present_brightness", 40.0);
        level_config_.outlier_sigmas = config_.GetDouble("level.outlier_sigmas", 3.0);
        level_config_.edge_sigma_px = config_.GetDouble("level.edge_sigma_px", 0.2);
        level_config_.systematic_sigma_mm =
            config_.GetDouble("level.systematic_sigma_mm", 0.5);
        center_reference_px_ = config_.GetDouble("level.center_reference_px", 0.0);

        edge_config_.smooth_radius = config_.GetSize("edge.smooth_radius", 3);
        edge_config_.min_gradient = config_.GetDouble("edge.min_gradient", 3.0);
        edge_config_.margin_fraction = config_.GetDouble("edge.margin_fraction", 0.05);
        edge_config_.min_separation = config_.GetSize("edge.min_separation", 20);

        flow_config_.min_speed_ms = config_.GetDouble("flow.min_speed_ms", 0.4);
        flow_config_.max_speed_ms = config_.GetDouble("flow.max_speed_ms", 5.0);
        flow_config_.velocity_steps = config_.GetSize("flow.velocity_steps", 400);
        flow_config_.max_interval_frames = config_.GetSize("flow.max_interval_frames", 10);
        flow_config_.background_tau_s = config_.GetDouble("flow.background_tau_s", 3.0);
        flow_config_.min_snr = config_.GetDouble("flow.min_snr", 4.0);
        flow_config_.snr_exclusion = config_.GetInt("flow.snr_exclusion", 8);
        flow_config_.min_profiles = config_.GetSize("flow.min_profiles", 20);
    }

    /// Полосы выборки: где именно в кадре искать кромки жёлоба (поперёк
    /// потока) и профиль для скорости (вдоль потока). Единственный способ
    /// узнать правильные координаты — посмотреть на кадр с работающей
    /// камеры, поэтому именно эти числа наладчик будет подбирать чаще всего;
    /// см. diagnostics.dir для получения снимка с текущими координатами.
    void ApplyStrips() {
        level_strip_.center_x_px = config_.GetDouble("level.strip.center_x_px", 400.0);
        level_strip_.center_y_px = config_.GetDouble("level.strip.center_y_px", 512.0);
        level_strip_.across_angle_deg = config_.GetDouble("level.strip.angle_deg", 90.0);
        level_strip_.length_px = config_.GetDouble("level.strip.length_px", 500.0);
        level_strip_.average_px = config_.GetDouble("level.strip.average_px", 40.0);
        level_strip_.samples = config_.GetSize("level.strip.samples", 500);
        level_strip_.average_lines = config_.GetSize("level.strip.average_lines", 9);

        flow_strip_.center_x_px = config_.GetDouble("flow.strip.center_x_px", 640.0);
        flow_strip_.center_y_px = config_.GetDouble("flow.strip.center_y_px", 512.0);
        flow_strip_.along_angle_deg = config_.GetDouble("flow.strip.angle_deg", 0.0);
        flow_strip_.length_px = config_.GetDouble("flow.strip.length_px", 900.0);
        flow_strip_.average_px = config_.GetDouble("flow.strip.average_px", 60.0);
        flow_strip_.samples = config_.GetSize("flow.strip.samples", 900);
        flow_strip_.average_lines = config_.GetSize("flow.strip.average_lines", 15);
    }

    /// Калибровка — угол камеры и масштаб. Основной путь: три и более
    /// меток с известными мировыми координатами, из которых угол и масштаб
    /// вычисляются (см. magma_level.hpp, CalibrationSolver). Запасной путь
    /// на время, пока метки ещё не размечены: ручные значения угла и
    /// масштаба прямо в файле настроек.
    void ApplyCalibration() {
        const std::string signature = config_.Signature("calibration.mark");
        const std::vector<std::string> raw_marks = config_.GetList("calibration.mark");

        CalibrationSolver solver;
        for (const std::string& raw : raw_marks) {
            const auto fields = ParseFields(raw);
            CalibrationMark mark;
            mark.name = FieldString(fields, "name");
            mark.world_u_mm = FieldDouble(fields, "u");
            mark.world_v_mm = FieldDouble(fields, "v");
            mark.image_x_px = FieldDouble(fields, "x");
            mark.image_y_px = FieldDouble(fields, "y");
            solver.AddMark(mark);
        }

        const bool marks_changed = signature != marks_signature_;
        const bool manual_requested = config_.GetBool("calibration.manual", false);

        if (solver.MarkCount() >= 3 && !manual_requested) {
            const CameraCalibration candidate = solver.Solve();
            if (candidate.valid) {
                if (marks_changed) {
                    // Метки поменялись по воле оператора — это осмысленное
                    // обновление, а не случайный дрейф изображения (поиска
                    // меток на самом кадре автоматически здесь нет). Сравнение
                    // с прежней калибровкой — подстраховка от опечатки:
                    // резкий скачок угла или масштаба чаще следствие неверно
                    // вписанной координаты, чем реального перемонтажа камеры.
                    if (calibration_.valid) {
                        const auto drift = mark_tracker_.Compare(candidate);
                        if (drift.shifted) {
                            Warn("Новые метки дают заметно другую калибровку "
                                "(Δβ=" + std::to_string(drift.angle_deg) +
                                "°, Δмасштаб=" + std::to_string(drift.scale_percent) +
                                "%) — проверьте координаты меток в файле настроек");
                        }
                    }
                    mark_tracker_.SetReference(candidate);
                    marks_signature_ = signature;
                    Warn("Калибровка обновлена по " +
                        std::to_string(solver.MarkCount()) + " меткам: угол " +
                        std::to_string(candidate.elevation_deg) + "°, масштаб " +
                        std::to_string(candidate.scale_mm_per_px) + " мм/пикс");
                }
                calibration_ = candidate;
                calibration_manual_ = false;
                return;
            }
            if (marks_changed) {
                Warn("Метки не дают решения (возможно, лежат на одной прямой) — "
                    "калибровка НЕ обновлена, используется прежняя");
                marks_signature_ = signature;
            }
            return;   // прежняя calibration_ остаётся в силе
        }

        // Ручной путь: угол и масштаб заданы явно. Актуален на самом раннем
        // этапе наладки, пока метки ещё физически не закреплены на жёлобе.
        CameraCalibration manual;
        manual.elevation_deg = config_.GetDouble("calibration.beta_deg", 31.0);
        manual.scale_mm_per_px = config_.GetDouble("calibration.scale_mm_per_px", 0.4135);
        manual.flow_angle_deg = config_.GetDouble("calibration.flow_angle_deg", 0.0);
        manual.residual_px = 0.0;
        manual.valid = true;
        calibration_ = manual;
        calibration_manual_ = true;
        marks_signature_ = signature;
    }

    void ApplyExposure() {
        exposure_config_.line_time_us = config_.GetDouble("exposure.line_time_us", 4.6);
        exposure_config_.min_exposure_us = config_.GetInt("exposure.min_exposure_us", 5);
        exposure_config_.frame_rate = capture_config_.frame_rate;
        exposure_config_.min_gain_db = config_.GetDouble("exposure.min_gain_db", 0.0);
        exposure_config_.max_gain_db = config_.GetDouble("exposure.max_gain_db", 40.0);
        exposure_config_.gain_step_db = config_.GetDouble("exposure.gain_step_db", 0.1);
        exposure_config_.gain_increment_db =
            config_.GetDouble("exposure.gain_increment_db", 2.0);
        exposure_config_.max_blur_px = config_.GetDouble("exposure.max_blur_px", 3.0);
        exposure_config_.target_level = config_.GetDouble("exposure.target_level", 210.0);
        exposure_config_.tolerance = config_.GetDouble("exposure.tolerance", 15.0);
        exposure_config_.max_clipped_fraction =
            config_.GetDouble("exposure.max_clipped_fraction", 0.002);
        exposure_config_.damping = config_.GetDouble("exposure.damping", 0.5);
        exposure_config_.fallback_speed_ms =
            config_.GetDouble("exposure.fallback_speed_ms", 1.5);

        // Масштаб для расчёта смаза берётся из действующей калибровки: она
        // точнее, чем любое значение, вписанное в файл настроек вручную под
        // эту секцию, и без такой синхронизации предел по смазу считался бы
        // по устаревшему масштабу.
        exposure_config_.mm_per_px =
            calibration_.valid ? calibration_.scale_mm_per_px : exposure_config_.mm_per_px;

        exposure_.UpdateConfig(exposure_config_);
    }

    /// ВТОРАЯ КАМЕРА (задел под стереопару, см. magma_stereo.hpp).
    ///
    /// Полностью отключена по умолчанию (camera2.device пуст) — ничего не
    /// открывается и не логируется, пока явно не задан путь к устройству.
    /// Экспозиция и калибровка — ПОЛНОСТЬЮ НЕЗАВИСИМЫЕ от первой камеры
    /// наборы параметров (свои veye2.*/calibration2.*/exposure2.*), а не
    /// зеркалирование настроек первой: камеры физически идентичны, но на
    /// них может стоять разная оптика (объектив по умолчанию — 30 мм,
    /// F/6, как у первой камеры, но может быть другим).
    ///
    /// ВАЖНО: сюда переносится только детект/подключение/автовыдержка —
    /// сам расчёт скорости/уровня по-прежнему ведётся ТОЛЬКО по первой
    /// камере. Подключение второго потока к общему расчёту (триангуляция
    /// из magma_stereo.hpp) — отдельная, более крупная задача, сознательно
    /// не делается здесь заодно (см. PROJECT_STATE.md, п.7): она требует
    /// проверки на реальной паре камер, которой на момент этой правки нет.
    void ApplySecondCamera() {
        CaptureConfig next_capture;
        next_capture.device = config_.GetString("camera2.device", "");
        camera2_enabled_ = !next_capture.device.empty();
        if (!camera2_enabled_) return;

        next_capture.width = config_.GetSize("camera2.width", 1280);
        next_capture.height = config_.GetSize("camera2.height", 1024);
        next_capture.frame_rate = config_.GetDouble("camera2.frame_rate", 60.0);
        next_capture.buffer_count = config_.GetSize("camera2.buffer_count", 6);
        next_capture.timeout_ms = config_.GetInt("camera2.timeout_ms", 2000);

        const std::string signature =
            next_capture.device + "|" + std::to_string(next_capture.width) + "|" +
            std::to_string(next_capture.height) + "|" +
            std::to_string(next_capture.frame_rate);
        capture_config2_ = next_capture;
        if (signature != capture_signature2_ || !camera2_) {
            capture_signature2_ = signature;
            camera2_needs_reopen_ = true;
        }

        VeyeConfig next_veye;
        next_veye.script = config_.GetString("veye2.script",
                                             "/home/mike/veye/mv_mipi_i2c_new.sh");
        next_veye.i2c_bus = config_.GetInt("veye2.i2c_bus", 11);
        if (next_veye.script != veye_config2_.script ||
            next_veye.i2c_bus != veye_config2_.i2c_bus) {
            veye2_configured_once_ = false;
        }
        veye_config2_ = next_veye;
        veye2_ = VeyeControl(veye_config2_);

        // Калибровка второй камеры — только ручной путь (без меток): для
        // одиночного зрения основной путь — метки на борту жёлоба, но для
        // задела под стереопару, ещё не установленную физически, размечать
        // метки не по чему. Когда стереопара будет смонтирована, сюда
        // естественно добавится тот же путь через CalibrationSolver, что и
        // у первой камеры.
        calibration2_.elevation_deg = config_.GetDouble("calibration2.beta_deg", 31.0);
        calibration2_.scale_mm_per_px =
            config_.GetDouble("calibration2.scale_mm_per_px", 0.4135);
        calibration2_.flow_angle_deg =
            config_.GetDouble("calibration2.flow_angle_deg", 0.0);
        calibration2_.residual_px = 0.0;
        calibration2_.valid = true;

        exposure_config2_.line_time_us = config_.GetDouble("exposure2.line_time_us", 4.6);
        exposure_config2_.min_exposure_us = config_.GetInt("exposure2.min_exposure_us", 5);
        exposure_config2_.frame_rate = capture_config2_.frame_rate;
        exposure_config2_.min_gain_db = config_.GetDouble("exposure2.min_gain_db", 0.0);
        exposure_config2_.max_gain_db = config_.GetDouble("exposure2.max_gain_db", 40.0);
        exposure_config2_.gain_step_db = config_.GetDouble("exposure2.gain_step_db", 0.1);
        exposure_config2_.gain_increment_db =
            config_.GetDouble("exposure2.gain_increment_db", 2.0);
        exposure_config2_.max_blur_px = config_.GetDouble("exposure2.max_blur_px", 3.0);
        exposure_config2_.target_level = config_.GetDouble("exposure2.target_level", 210.0);
        exposure_config2_.tolerance = config_.GetDouble("exposure2.tolerance", 15.0);
        exposure_config2_.max_clipped_fraction =
            config_.GetDouble("exposure2.max_clipped_fraction", 0.002);
        exposure_config2_.damping = config_.GetDouble("exposure2.damping", 0.5);
        exposure_config2_.fallback_speed_ms =
            config_.GetDouble("exposure2.fallback_speed_ms", 1.5);
        exposure_config2_.mm_per_px = calibration2_.scale_mm_per_px;
        exposure2_.UpdateConfig(exposure_config2_);
    }

    /// Сетевые параметры вещания. В отличие от всего перечисленного выше,
    /// адрес и порт СЧИТЫВАЮТСЯ ТОЛЬКО ПРИ ПЕРВОМ ЗАПУСКЕ: перепривязка уже
    /// слушающего сокета требует остановки и повторного запуска потока
    /// приёма подключений, то есть по сути краткого перезапуска именно этой
    /// части процесса. Такая операция настолько редкая (адрес порта решается
    /// один раз при развёртывании), что она сознательно вынесена за рамки
    /// горячей перезагрузки — вместо этого при изменении порта в файле
    /// процесс печатает предупреждение и продолжает слушать прежний.
    void ApplyOutput(bool is_reload) {
        print_json_ = config_.GetBool("output.print_json", true);
        diagnostics_dir_ = config_.GetString("diagnostics.dir", "");

        BroadcastConfig next;
        next.host = config_.GetString("output.host", "0.0.0.0");
        next.port = static_cast<uint16_t>(config_.GetInt("output.port", 9100));

        if (!broadcaster_) {
            broadcast_config_ = next;
            broadcaster_ = std::make_unique<LineBroadcaster>(stop_, broadcast_config_);
        } else if (is_reload && (next.host != broadcast_config_.host ||
                                 next.port != broadcast_config_.port)) {
            Warn("Изменение output.host/output.port требует перезапуска процесса "
                "зрения — продолжаю слушать " + broadcast_config_.host + ":" +
                std::to_string(broadcast_config_.port));
        }
    }

    // -------------------------------------------------------------------
    // Работа с камерой
    // -------------------------------------------------------------------

    /// Открыть камеру заново, если структурные параметры изменились или она
    /// ещё не была открыта. Неудача не останавливает процесс: следующая
    /// попытка будет предпринята перед следующей серией — это позволяет
    /// запустить magma_vision раньше, чем физически подключена камера, и
    /// пережить её временное отключение без вмешательства оператора.
    bool EnsureCameraOpen() {
        if (camera_ && camera_->IsOpen() && !camera_needs_reopen_) return true;

        camera_ = std::make_unique<Camera>(capture_config_);
        camera_needs_reopen_ = false;
        if (camera_->Open()) {
            std::printf("[VISION] Камера открыта: %s, %zux%zu @ %.0f к/с\n",
                        capture_config_.device.c_str(), capture_config_.width,
                        capture_config_.height, capture_config_.frame_rate);
            ApplyExposureToCamera(exposure_.Current());
            return true;
        }

        Warn("Камера не открыта (" + camera_->LastError() + "), попробую перед "
            "следующей серией");
        camera_.reset();
        return false;
    }

    /// Записать выдержку и усиление, прочитав фактически установленное.
    /// Обязательно читать обратно: сенсор квантует выдержку временем строки,
    /// и расхождение заданного с фактическим накапливалось бы в расчёте
    /// смаза и в следующем шаге регулятора.
    void ApplyExposureToCamera(const ExposureSetting& setting) {
        if (!veye_.Available()) return;
        const auto applied = veye_.Apply(setting.exposure_us, setting.gain_db);
        if (applied.ok) {
            actual_exposure_us_ = applied.exposure_us;
            actual_gain_db_ = applied.gain_db;
        }
    }

    // ---- Вторая камера: та же логика открытия/выдержки, независимо ----

    bool EnsureCamera2Open() {
        if (camera2_ && camera2_->IsOpen() && !camera2_needs_reopen_) return true;

        camera2_ = std::make_unique<Camera>(capture_config2_);
        camera2_needs_reopen_ = false;
        if (camera2_->Open()) {
            std::printf("[VISION] Камера 2 открыта: %s, %zux%zu @ %.0f к/с\n",
                        capture_config2_.device.c_str(), capture_config2_.width,
                        capture_config2_.height, capture_config2_.frame_rate);
            ApplyExposureToCamera2(exposure2_.Current());
            return true;
        }

        Warn("Камера 2 не открыта (" + camera2_->LastError() + "), попробую перед "
            "следующей серией");
        camera2_.reset();
        return false;
    }

    void ApplyExposureToCamera2(const ExposureSetting& setting) {
        if (!veye2_.Available()) return;
        const auto applied = veye2_.Apply(setting.exposure_us, setting.gain_db);
        if (applied.ok) {
            actual_exposure_us2_ = applied.exposure_us;
            actual_gain_db2_ = applied.gain_db;
        }
    }

    /// Один цикл камеры 2: захват серии только для измерения яркости и
    /// подстройки выдержки — расчёт скорости/уровня по ней пока не ведётся
    /// (см. комментарий в ApplySecondCamera). Захват полного кадра нужен
    /// всё равно: FrameStats считается по реальным пикселям, а не по
    /// метаданным.
    void RunSecondCameraBurst() {
        if (!camera2_enabled_) return;
        if (!EnsureCamera2Open()) return;

        if (!veye2_configured_once_) {
            if (veye2_.Available()) {
                if (!veye2_.SetManualMode()) {
                    Warn("Камера 2: не удалось перевести в ручной режим: " +
                        veye2_.LastError());
                }
                veye2_.SetFrameRate(capture_config2_.frame_rate);
            } else {
                Warn("Камера 2: скрипт управления (" + veye_config2_.script +
                    ") недоступен — выдержка останется той, что задана извне");
            }
            veye2_configured_once_ = true;
        }

        FrameStats accumulated_stats;
        bool have_stats = false;
        const CaptureStats capture = camera2_->CaptureBurst(
            schedule_.burst_frames,
            [&](const GrayImage& image, std::size_t) {
                if (!have_stats) {
                    accumulated_stats =
                        FrameStats::Measure(image.data, image.width * image.height);
                    have_stats = true;
                }
            },
            schedule_.warmup_frames, &stop_);

        if (!capture.Ok()) {
            Warn("Камера 2: серия не удалась: принято " +
                std::to_string(capture.received) + ", ошибок " +
                std::to_string(capture.errors));
            camera2_needs_reopen_ = capture.errors > 0;
            return;
        }

        const ExposureSetting next_exposure =
            exposure2_.Update(accumulated_stats, /*speed_hint=*/-1.0);
        if (next_exposure.changed) ApplyExposureToCamera2(next_exposure);
    }

    // -------------------------------------------------------------------
    // Одна серия: захват, расчёт обеих величин, подстройка выдержки, отчёт
    // -------------------------------------------------------------------
    void RunBurst() {
        RunSecondCameraBurst();   // независимо от первой; см. комментарий выше

        if (!EnsureCameraOpen()) {
            SendReport(FlowResult{}, LevelResult{}, CaptureStats{});
            return;
        }

        if (veye_camera_configured_once_ == false) {
            // Ручной режим и частота кадров задаются один раз после открытия
            // камеры/скрипта, а не на каждой серии: это управляющие команды,
            // а не данные, и незачем слать их 8640 раз в сутки.
            if (veye_.Available()) {
                if (!veye_.SetManualMode()) {
                    Warn("Не удалось перевести камеру в ручной режим: " +
                        veye_.LastError());
                }
                veye_.SetFrameRate(capture_config_.frame_rate);
            } else {
                Warn("Скрипт управления камерой (" + capture_config_.device +
                    ") недоступен — выдержка останется той, что задана извне");
            }
            veye_camera_configured_once_ = true;
        }

        FlowEstimator flow(flow_config_);
        EdgeFinder edges(edge_config_);

        std::vector<EdgeObservation> level_observations;
        level_observations.reserve(schedule_.level_frames + 1);

        const std::size_t level_every =
            std::max<std::size_t>(1, schedule_.burst_frames /
                                     std::max<std::size_t>(1, schedule_.level_frames));

        FrameStats accumulated_stats;
        bool have_stats = false;
        const bool want_diagnostics = !diagnostics_dir_.empty();
        std::vector<uint8_t> diagnostic_frame;
        std::size_t diagnostic_width = 0, diagnostic_height = 0;
        std::optional<EdgeObservation> diagnostic_edges;

        const double burst_started = MonotonicSeconds();

        // Кадры НЕ НАКАПЛИВАЮТСЯ: из каждого сразу извлекается профиль и,
        // на части кадров, положение кромок, после чего буфер возвращается
        // драйверу. Серия из ста двадцати кадров полного разрешения заняла
        // бы более ста мегабайт; профили той же серии — единицы мегабайт.
        const CaptureStats capture = camera_->CaptureBurst(
            schedule_.burst_frames,
            [&](const GrayImage& image, std::size_t index) {
                const double timestamp = MonotonicSeconds() - burst_started;
                flow.AddProfile(ProfileExtractor::Extract(image, flow_strip_), timestamp);

                if (index % level_every == 0) {
                    level_observations.push_back(edges.Find(image, level_strip_));
                }
                if (!have_stats) {
                    accumulated_stats =
                        FrameStats::Measure(image.data, image.width * image.height);
                    have_stats = true;
                }
                if (index == 0) {
                    if (!diagnostic_edges) diagnostic_edges = edges.Find(image, level_strip_);
                    if (want_diagnostics) {
                        diagnostic_width = image.width;
                        diagnostic_height = image.height;
                        diagnostic_frame.assign(image.width * image.height, 0);
                        const std::size_t stride = image.Stride();
                        for (std::size_t y = 0; y < image.height; ++y) {
                            std::memcpy(diagnostic_frame.data() + y * image.width,
                                       image.data + y * stride, image.width);
                        }
                    }
                }
            },
            schedule_.warmup_frames, &stop_);

        if (!capture.Ok()) {
            Warn("Серия не удалась: принято " + std::to_string(capture.received) +
                ", таймаутов " + std::to_string(capture.timeouts) + ", ошибок " +
                std::to_string(capture.errors));
            camera_needs_reopen_ = capture.errors > 0;   // возможен обрыв устройства
            SendReport(FlowResult{}, LevelResult{}, capture);
            return;
        }

        const FlowResult flow_result = flow.Estimate();
        const LevelResult level_result = CombineLevelObservations(level_observations);

        // Выдержка подстраивается МЕЖДУ сериями, а не внутри: изменение
        // посреди серии дало бы ступеньку яркости между кадрами, а расчёт
        // скорости сравнивает кадры друг с другом — ступенька внесла бы
        // ложный сигнал.
        const double speed_hint = flow_result.speed_ms.value_or(-1.0);
        const ExposureSetting next_exposure = exposure_.Update(accumulated_stats, speed_hint);
        if (next_exposure.changed) ApplyExposureToCamera(next_exposure);

        SendReport(flow_result, level_result, capture);

        if (want_diagnostics) {
            WriteDiagnostics(diagnostic_frame, diagnostic_width, diagnostic_height,
                            diagnostic_edges, flow_result, level_result);
        }
    }

    /// Свести наблюдения кромок нескольких кадров серии в одну оценку уровня.
    /// Берётся медиана значений уровня по кадрам, а не среднее: единичный
    /// всплеск или брызги сдвинули бы среднее, тогда как медиана их
    /// игнорирует. Прочие поля результата (состояние, участники, сигма)
    /// берутся от того кадра, чья оценка ближе всего к медиане, — это
    /// честное представление «типичного» результата серии, а не искусственно
    /// усреднённые характеристики от разных наблюдений.
    LevelResult CombineLevelObservations(
        const std::vector<EdgeObservation>& observations) const {
        // Строится заново из ДЕЙСТВУЮЩИХ geometry_/level_config_, а не хранится
        // постоянным полем: LevelEstimator снимает собственную копию настроек в
        // момент конструирования, и если бы объект жил постоянно, перезагрузка
        // файла настроек молча переставала бы на него влиять. Конструирование
        // не требует ничего, кроме копирования двух небольших структур, — цена
        // пренебрежимая по сравнению с самим захватом кадров.
        const LevelEstimator level(geometry_, level_config_);

        std::vector<LevelResult> per_frame;
        per_frame.reserve(observations.size());
        for (const EdgeObservation& observation : observations) {
            per_frame.push_back(level.Estimate(observation, calibration_,
                                               center_reference_px_,
                                               /*camera_shifted=*/false));
        }

        std::vector<std::pair<double, std::size_t>> ranked;
        for (std::size_t i = 0; i < per_frame.size(); ++i) {
            if (per_frame[i].level_mm) ranked.push_back({*per_frame[i].level_mm, i});
        }
        if (ranked.empty()) {
            // Ни один кадр не дал уровня — возвращаем результат первого
            // наблюдения как представителя (несёт причину отказа в поле state).
            return per_frame.empty() ? LevelResult{} : per_frame.front();
        }

        std::sort(ranked.begin(), ranked.end());
        const double median = ranked[ranked.size() / 2].first;

        std::size_t closest = ranked.front().second;
        double best_gap = 1e300;
        for (const auto& [value, index] : ranked) {
            const double gap = std::fabs(value - median);
            if (gap < best_gap) { best_gap = gap; closest = index; }
        }

        LevelResult result = per_frame[closest];
        result.level_mm = median;   // сама оценка — устойчивая медиана
        return result;
    }

    // -------------------------------------------------------------------
    // Вывод: сетевая рассылка и, при необходимости, диагностика на диск
    // -------------------------------------------------------------------

    void SendReport(const FlowResult& flow, const LevelResult& level,
                    const CaptureStats& capture) {
        VisionReport report;

        report.speed_valid = static_cast<bool>(flow.speed_ms);
        report.speed_ms = flow.speed_ms.value_or(0.0);
        report.speed_snr = flow.snr;
        report.speed_profiles = flow.profiles_used;
        report.speed_intervals = flow.intervals_used;
        report.speed_shift_px_per_frame = flow.shift_px_per_frame;

        report.level_valid = static_cast<bool>(level.level_mm);
        report.level_mm = level.level_mm.value_or(0.0);
        report.level_sigma_mm = level.sigma_mm;
        report.level_state = LevelStateName(level.state);
        report.level_disagreement_mm = level.disagreement_mm;
        for (const Participant& participant : level.participants) {
            if (participant.used) report.level_sources.push_back(participant.name);
        }
        report.level_frames = 0;   // заполняется ниже, если наблюдения были

        report.camera_frames = capture.received;
        report.camera_fps = capture.ActualFrameRate();
        report.camera_exposure_us = actual_exposure_us_;
        report.camera_gain_db = actual_gain_db_;
        report.camera_timeouts = capture.timeouts;
        report.camera_errors = capture.errors;

        report.calibration_valid = calibration_.valid;
        report.calibration_beta_deg = calibration_.elevation_deg;
        report.calibration_scale_mm_per_px = calibration_.scale_mm_per_px;
        report.calibration_residual_px = calibration_.residual_px;
        report.calibration_marks =
            calibration_manual_ ? 0 : config_.GetList("calibration.mark").size();
        report.calibration_manual = calibration_manual_;

        report.config_version = config_version_;

        const std::string line = FormatVisionReport(report);
        if (print_json_) std::printf("%s\n", line.c_str());
        broadcaster_->Broadcast(line);
    }

    void Warn(const std::string& text) {
        std::printf("[VISION] %s\n", text.c_str());
        if (broadcaster_) broadcaster_->Broadcast(FormatVisionWarning(text));
    }

    static const char* LevelStateName(LevelState state) {
        switch (state) {
            case LevelState::Measured:     return "measured";
            case LevelState::Empty:        return "empty";
            case LevelState::BelowVisible: return "below_visible";
            case LevelState::SingleEdge:   return "single_edge";
            case LevelState::Unreliable:   return "unreliable";
        }
        return "unknown";
    }

    /// Сохранить кадр и текстовую сводку того, что программа в нём увидела.
    ///
    /// Это единственный способ подобрать координаты полос выборки и меток,
    /// не имея доступа к самому изображению иначе как через файл: технику
    /// достаточно открыть сохранённый кадр в любом просмотрщике, найти
    /// нужные пиксельные координаты и вписать их в файл настроек — без
    /// пересборки и без перезапуска.
    ///
    /// Файлы не удаляются автоматически: диагностика — временный инструмент
    /// наладки, а не постоянный журнал, и каталог назначения (diagnostics.dir)
    /// следует очищать вручную или отключать пустым значением после того,
    /// как параметры подобраны — иначе, в отличие от всего остального в этом
    /// проекте, здесь место на диске будет расходоваться безостановочно.
    void WriteDiagnostics(const std::vector<uint8_t>& frame, std::size_t width,
                          std::size_t height, const std::optional<EdgeObservation>& edges,
                          const FlowResult& flow, const LevelResult& level) const {
        const std::string stamp = VisionIsoTimestamp();
        std::string safe_stamp = stamp;
        for (char& symbol : safe_stamp) {
            if (symbol == ':' || symbol == '.') symbol = '-';
        }

        if (!frame.empty()) {
            GrayImage image{frame.data(), width, height, width};
            SavePgm(diagnostics_dir_ + "/" + safe_stamp + ".pgm", image);
        }

        std::FILE* file = std::fopen((diagnostics_dir_ + "/" + safe_stamp + ".txt").c_str(),
                                     "w");
        if (file == nullptr) return;

        std::fprintf(file, "Снимок наладки: %s\n\n", stamp.c_str());
        std::fprintf(file, "-- Калибровка --\n");
        std::fprintf(file, "  действительна: %s (%s)\n", calibration_.valid ? "да" : "нет",
                    calibration_manual_ ? "вручную" : "по меткам");
        std::fprintf(file, "  угол: %.2f град\n", calibration_.elevation_deg);
        std::fprintf(file, "  масштаб: %.4f мм/пикс\n", calibration_.scale_mm_per_px);
        std::fprintf(file, "  невязка меток: %.3f пикс\n\n", calibration_.residual_px);

        std::fprintf(file, "-- Полоса уровня (level.strip.*) --\n");
        std::fprintf(file, "  центр: (%.1f, %.1f), угол %.1f, длина %.0f, "
                    "усреднение %.0f, отсчётов %zu\n\n",
                    level_strip_.center_x_px, level_strip_.center_y_px,
                    level_strip_.across_angle_deg, level_strip_.length_px,
                    level_strip_.average_px, level_strip_.samples);

        std::fprintf(file, "-- Полоса скорости (flow.strip.*) --\n");
        std::fprintf(file, "  центр: (%.1f, %.1f), угол %.1f, длина %.0f, "
                    "усреднение %.0f, отсчётов %zu\n\n",
                    flow_strip_.center_x_px, flow_strip_.center_y_px,
                    flow_strip_.along_angle_deg, flow_strip_.length_px,
                    flow_strip_.average_px, flow_strip_.samples);

        std::fprintf(file, "-- Кромки на первом кадре серии --\n");
        if (edges) {
            std::fprintf(file, "  ближняя: %s, позиция %.2f, резкость %.2f\n",
                        edges->near_found ? "найдена" : "НЕ найдена",
                        edges->near_px, edges->near_sharpness);
            std::fprintf(file, "  дальняя: %s, позиция %.2f, резкость %.2f\n",
                        edges->far_found ? "найдена" : "НЕ найдена",
                        edges->far_px, edges->far_sharpness);
            std::fprintf(file, "  яркость полосы: %.1f\n",
                        edges->brightness);
            if (edges->near_found && edges->far_found) {
                std::fprintf(file, "  ширина: %.2f пикс, центр: %.2f пикс\n",
                            edges->WidthPx(), edges->CenterPx());
            }
        } else {
            std::fprintf(file, "  (не вычислены)\n");
        }

        std::fprintf(file, "\n-- Результат серии --\n");
        std::fprintf(file, "  скорость по камере: %s\n",
                    flow.speed_ms ? (std::to_string(*flow.speed_ms) + " м/с, snr " +
                                    std::to_string(flow.snr)).c_str()
                                 : "не определена");
        std::fprintf(file, "  уровень: %s\n",
                    level.level_mm ? (std::to_string(*level.level_mm) + " мм, состояние " +
                                     LevelStateName(level.state)).c_str()
                                  : (std::string("не определён, состояние ") +
                                     LevelStateName(level.state)).c_str());

        std::fclose(file);
    }

    // -------------------------------------------------------------------
    // Поля
    // -------------------------------------------------------------------
    StartupConfig startup_;
    ConfigFile config_;
    std::string config_version_;

    // сеть
    BroadcastConfig broadcast_config_;
    std::unique_ptr<LineBroadcaster> broadcaster_;
    bool print_json_ = true;
    std::string diagnostics_dir_;

    // расписание
    Schedule schedule_;

    // камера и управление
    CaptureConfig capture_config_;
    std::string capture_signature_;
    bool camera_needs_reopen_ = true;
    std::unique_ptr<Camera> camera_;
    VeyeConfig veye_config_;
    VeyeControl veye_{VeyeConfig{}};
    bool veye_camera_configured_once_ = false;
    double actual_exposure_us_ = 0.0;
    double actual_gain_db_ = 0.0;

    // расчёт
    LauncherGeometry geometry_;
    LevelConfig level_config_;
    EdgeConfig edge_config_;
    FlowConfig flow_config_;
    FlowStrip flow_strip_;
    StripDefinition level_strip_;
    double center_reference_px_ = 0.0;

    // калибровка
    CameraCalibration calibration_;
    bool calibration_manual_ = true;
    std::string marks_signature_;
    MarkTracker mark_tracker_;

    // выдержка
    ExposureConfig exposure_config_;
    ExposureController exposure_{ExposureConfig{}};

    // вторая камера (задел под стереопару — см. ApplySecondCamera)
    bool camera2_enabled_ = false;
    CaptureConfig capture_config2_;
    std::string capture_signature2_;
    bool camera2_needs_reopen_ = true;
    std::unique_ptr<Camera> camera2_;
    VeyeConfig veye_config2_;
    VeyeControl veye2_{VeyeConfig{}};
    bool veye2_configured_once_ = false;
    double actual_exposure_us2_ = 0.0;
    double actual_gain_db2_ = 0.0;
    CameraCalibration calibration2_;
    ExposureConfig exposure_config2_;
    ExposureController exposure2_{ExposureConfig{}};

    std::atomic<bool> stop_{false};
};

}  // namespace magma

namespace {
magma::VisionProcess* g_process = nullptr;
void HandleSignal(int) { if (g_process != nullptr) g_process->RequestStop(); }
}  // namespace

int main(int argc, char** argv) {
    magma::StartupConfig startup;
    if (argc > 1) startup.config_path = argv[1];

    magma::VisionProcess process(startup);
    g_process = &process;
    std::signal(SIGINT, HandleSignal);
    std::signal(SIGTERM, HandleSignal);
    return process.Run();
}
