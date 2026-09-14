// magma_vision_protocol.hpp — формат строки, которую отдаёт процесс зрения.
//
// ЗАЧЕМ ОТДЕЛЬНЫЙ ПРОТОКОЛ, А НЕ РАСШИРЕНИЕ magma_protocol.hpp.
// В защищённом ядре формат строки сервера — «единый источник истины» именно
// для скорости по фотодиодам, которую сервер измеряет раз в секунду. Данные
// зрения — другой физический канал (камера, не фотодиоды), с другим тактом
// (серия раз в 10 секунд, а не секундная сводка) и с параметрами, которые
// будут меняться на ходу при наладке заметно чаще, чем формат ядра. Дописывать
// это в защищённый файл значило бы либо трогать ядро ради вещи, которая его
// не касается, либо смешать два независимых такта в одной структуре. Поэтому
// у зрения свой протокол, свой порт и свой независимый разбор — ядро не
// затронуто ни в одной точке.
//
// Формат — NDJSON (одна JSON-строка на сообщение, разделитель '\n'), тот же
// принцип, что принят в ядре после перевода протокола на JSON 04.09.2026:
//   {"type":"report","ts":"...", "speed_camera":{...}, "level":{...},
//    "camera":{...}, "calibration":{...}, "config_version":"..."}
// Предупреждения:
//   {"type":"warning","ts":"...","text":"..."}
//
// Скорость по камере и уровень идут РАЗДЕЛЬНО, каждая со своим полем valid, и
// никогда не сводятся с показаниями фотодиодов в одно число: расхождение двух
// независимых измерений скорости — самая ценная диагностика во всей системе.

#ifndef MAGMA_VISION_PROTOCOL_HPP
#define MAGMA_VISION_PROTOCOL_HPP

#include <chrono>
#include <cstdio>
#include <ctime>
#include <optional>
#include <regex>
#include <string>
#include <vector>

namespace magma {

/// Метка времени вида 2026-09-10T14:03:11.204.
///
/// Реализация совпадает с IsoTimestamp() из ядра, но код продублирован
/// намеренно: подключать защищённый заголовок ради одной функции значило бы
/// создать зависимость дополнения от ядра там, где в этом нет необходимости.
inline std::string VisionIsoTimestamp() {
    using namespace std::chrono;
    const auto now = system_clock::now();
    const auto time = system_clock::to_time_t(now);
    const auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    std::tm parts{};
    ::localtime_r(&time, &parts);
    char date[32];
    std::strftime(date, sizeof(date), "%Y-%m-%dT%H:%M:%S", &parts);
    char result[48];
    std::snprintf(result, sizeof(result), "%s.%03d", date, static_cast<int>(ms.count()));
    return result;
}

/// Экранирование строки для вставки в JSON.
inline std::string VisionJsonEscape(const std::string& text) {
    std::string result;
    result.reserve(text.size() + 8);
    for (unsigned char symbol : text) {
        switch (symbol) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (symbol < 0x20) {
                    char buffer[8];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", symbol);
                    result += buffer;
                } else {
                    result += static_cast<char>(symbol);
                }
        }
    }
    return result;
}

// =============================================================================
// Сводка серии
// =============================================================================

/// Что отправляется по итогам одной серии кадров.
struct VisionReport {
    // --- скорость по камере ---
    bool speed_valid = false;
    double speed_ms = 0.0;
    double speed_snr = 0.0;
    std::size_t speed_profiles = 0;
    std::size_t speed_intervals = 0;
    double speed_shift_px_per_frame = 0.0;

    // --- уровень ---
    bool level_valid = false;
    double level_mm = 0.0;
    double level_sigma_mm = 0.0;
    std::string level_state = "unknown";
    double level_disagreement_mm = 0.0;
    std::vector<std::string> level_sources;
    std::size_t level_frames = 0;

    // --- состояние камеры за серию ---
    std::size_t camera_frames = 0;
    double camera_fps = 0.0;
    double camera_exposure_us = 0.0;
    double camera_gain_db = 0.0;
    std::size_t camera_timeouts = 0;
    std::size_t camera_errors = 0;

    // --- действующая калибровка ---
    bool calibration_valid = false;
    double calibration_beta_deg = 0.0;
    double calibration_scale_mm_per_px = 0.0;
    double calibration_residual_px = 0.0;
    std::size_t calibration_marks = 0;
    bool calibration_manual = false;

    /// Метка времени успешной (пере)загрузки файла настроек — по ней
    /// оператор видит в самом потоке данных, что правка файла подхватилась,
    /// не заглядывая в журнал процесса.
    std::string config_version;
};

/// Дописать поле "ключ":число, отформатированное по fmt, в конец строки JSON.
/// Заменяет собой повторяющуюся тройку «snprintf во временный буфер, затем
/// конкатенация трёх строковых литералов» — 13 раз в FormatVisionReport ниже.
inline void AppendNumberField(std::string& json, const char* key, const char* fmt,
                              double value) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), fmt, value);
    json += ',';
    json += '"';
    json += key;
    json += "\":";
    json += buffer;
}

/// Собрать строку отчёта.
inline std::string FormatVisionReport(const VisionReport& report) {
    std::string json;
    json.reserve(640);
    json += "{\"type\":\"report\",\"ts\":\"" + VisionIsoTimestamp() + "\",";

    json += "\"speed_camera\":{\"valid\":";
    json += report.speed_valid ? "true" : "false";
    if (report.speed_valid) {
        AppendNumberField(json, "value", "%.4f", report.speed_ms);
        json += ",\"unit\":\"m/s\"";
    }
    AppendNumberField(json, "snr", "%.2f", report.speed_snr);
    json += ",\"profiles\":" + std::to_string(report.speed_profiles);
    json += ",\"intervals\":" + std::to_string(report.speed_intervals);
    AppendNumberField(json, "shift_px_per_frame", "%.2f", report.speed_shift_px_per_frame);
    json += "},";

    json += "\"level\":{\"valid\":";
    json += report.level_valid ? "true" : "false";
    if (report.level_valid) {
        AppendNumberField(json, "value", "%.2f", report.level_mm);
        json += ",\"unit\":\"mm\"";
    }
    AppendNumberField(json, "sigma", "%.3f", report.level_sigma_mm);
    json += ",\"state\":\"" + VisionJsonEscape(report.level_state) + "\"";
    AppendNumberField(json, "disagreement", "%.3f", report.level_disagreement_mm);
    json += ",\"sources\":[";
    for (std::size_t i = 0; i < report.level_sources.size(); ++i) {
        if (i > 0) json += ",";
        json += "\"" + VisionJsonEscape(report.level_sources[i]) + "\"";
    }
    json += "],\"frames\":" + std::to_string(report.level_frames) + "},";

    json += "\"camera\":{\"frames\":" + std::to_string(report.camera_frames);
    AppendNumberField(json, "fps", "%.1f", report.camera_fps);
    AppendNumberField(json, "exposure_us", "%.0f", report.camera_exposure_us);
    AppendNumberField(json, "gain_db", "%.1f", report.camera_gain_db);
    json += ",\"timeouts\":" + std::to_string(report.camera_timeouts);
    json += ",\"errors\":" + std::to_string(report.camera_errors) + "},";

    json += "\"calibration\":{\"valid\":";
    json += report.calibration_valid ? "true" : "false";
    AppendNumberField(json, "beta_deg", "%.2f", report.calibration_beta_deg);
    AppendNumberField(json, "scale_mm_per_px", "%.4f", report.calibration_scale_mm_per_px);
    AppendNumberField(json, "residual_px", "%.3f", report.calibration_residual_px);
    json += ",\"marks\":" + std::to_string(report.calibration_marks);
    json += std::string(",\"manual\":") + (report.calibration_manual ? "true" : "false") + "},";

    json += "\"config_version\":\"" + VisionJsonEscape(report.config_version) + "\"}";
    return json;
}

/// Служебная строка (предупреждение о состоянии процесса зрения).
inline std::string FormatVisionWarning(const std::string& text) {
    return "{\"type\":\"warning\",\"ts\":\"" + VisionIsoTimestamp() + "\",\"text\":\"" +
           VisionJsonEscape(text) + "\"}";
}

// =============================================================================
// Разбор (для клиента, читающего поток зрения)
// =============================================================================

struct VisionMeasurement {
    bool speed_valid = false;
    double speed_ms = 0.0;
    double speed_snr = 0.0;

    bool level_valid = false;
    double level_mm = 0.0;
    double level_sigma_mm = 0.0;
    std::string level_state;

    double camera_exposure_us = 0.0;
    double camera_gain_db = 0.0;
    std::size_t camera_frames = 0;

    double calibration_beta_deg = 0.0;
    double calibration_scale_mm_per_px = 0.0;
};

/// Разбирает строки процесса зрения.
///
/// Числовые преобразования обёрнуты в try/catch — тот же приём, что уже
/// защитил основной клиент ядра от аварийного завершения на строке с
/// числом, слишком длинным для типа данных. Опыт показал, что регулярное
/// выражение проверяет форму записи, но не гарантирует, что значение
/// поместится в число, поэтому проверка нужна и здесь.
class VisionLineParser {
public:
    static bool IsMarker(const std::string& line) {
        return line.find("\"type\":\"warning\"") != std::string::npos;
    }

    static std::string MarkerText(const std::string& line) {
        std::smatch match;
        if (std::regex_search(line, match, kText)) return match[1].str();
        return line;
    }

    std::optional<VisionMeasurement> Parse(const std::string& line) const {
        try {
            return ParseUnchecked(line);
        } catch (const std::exception&) {
            return std::nullopt;
        }
    }

private:
    std::optional<VisionMeasurement> ParseUnchecked(const std::string& line) const {
        if (line.find("\"type\":\"report\"") == std::string::npos) return std::nullopt;

        VisionMeasurement result;

        std::smatch match;
        if (std::regex_search(line, match, kSpeedBlock)) {
            result.speed_valid = match[1].str() == "true";
            if (const auto value = FindIn(match[2].str(), kNumber)) result.speed_ms = *value;
        }
        if (const auto value = Find(line, kSpeedSnr)) result.speed_snr = *value;

        if (std::regex_search(line, match, kLevelBlock)) {
            result.level_valid = match[1].str() == "true";
            if (const auto value = FindIn(match[2].str(), kNumber)) result.level_mm = *value;
        }
        if (const auto value = Find(line, kLevelSigma)) result.level_sigma_mm = *value;
        if (std::regex_search(line, match, kLevelState)) result.level_state = match[1].str();

        if (const auto value = Find(line, kExposure)) result.camera_exposure_us = *value;
        if (const auto value = Find(line, kGain)) result.camera_gain_db = *value;
        if (std::regex_search(line, match, kFrames)) {
            result.camera_frames = static_cast<std::size_t>(std::stoull(match[1].str()));
        }

        if (const auto value = Find(line, kBeta)) result.calibration_beta_deg = *value;
        if (const auto value = Find(line, kScale)) result.calibration_scale_mm_per_px = *value;

        return result;
    }

    static std::optional<double> Find(const std::string& line, const std::regex& pattern) {
        std::smatch match;
        if (!std::regex_search(line, match, pattern)) return std::nullopt;
        return std::stod(match[1].str());
    }

    static std::optional<double> FindIn(const std::string& text, const std::regex& pattern) {
        std::smatch match;
        if (!std::regex_search(text, match, pattern)) return std::nullopt;
        return std::stod(match[1].str());
    }

    static inline const std::regex kNumber{R"(([\d.]+))"};
    static inline const std::regex kSpeedBlock{
        R"("speed_camera":\{"valid":(true|false)(,"value":([\d.]+))?)"};
    static inline const std::regex kSpeedSnr{R"("speed_camera"[^}]*"snr":([\d.]+))"};
    static inline const std::regex kLevelBlock{
        R"("level":\{"valid":(true|false)(,"value":([\d.]+))?)"};
    static inline const std::regex kLevelSigma{R"("level"[^}]*"sigma":([\d.]+))"};
    static inline const std::regex kLevelState{R"re("level"[^}]*"state":"(\w+)")re"};
    static inline const std::regex kExposure{R"("exposure_us":([\d.]+))"};
    static inline const std::regex kGain{R"("gain_db":([\d.]+))"};
    static inline const std::regex kFrames{R"("camera":\{"frames":(\d+))"};
    static inline const std::regex kBeta{R"("beta_deg":([\d.]+))"};
    static inline const std::regex kScale{R"("scale_mm_per_px":([\d.]+))"};
    static inline const std::regex kText{R"re("text":"((?:[^"\\]|\\.)*)")re"};
};

}  // namespace magma

#endif  // MAGMA_VISION_PROTOCOL_HPP
