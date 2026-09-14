// magma_protocol.hpp — формат строки обмена между сервером и клиентом.
//
// ЕДИНЫЙ ИСТОЧНИК ИСТИНЫ: заголовок используется и сервером (формирование),
// и клиентом (разбор). Когда формат и парсер жили в разных файлах, добавление
// полей молча сломало разбор статуса — клиент подхватывал имя поля вместо
// OK/REJECT, без ошибки и без сообщения. Здесь такое расхождение невозможно.
//
// Наружу уходит одна строка в секунду. Скорость внутри сервера оценивается
// чаще (порядка 20 раз в секунду), а в строку попадает медиана за секунду:
// она устойчива к единичным выбросам и не требует передавать промежуточные
// значения.
//
// Формат (переведён на JSON 04.09.2026, целевой формат контура НМЗ; одна
// JSON-строка на сообщение, разделитель '\n' — NDJSON):
//   {"type":"report","ts":"<ISO8601.мс>","v":X,"v_med":M,"accepted":K,
//    "attempted":N,"snr":S,"baselines":B,"adc":[a,b,c,d],"frames":F,
//    "lost":L,"crc":C,"overruns":O,"ok":true|false}
// Служебные предупреждения:
//   {"type":"warning","ts":"<ISO8601.мс>","text":"..."}
// JSON собирается и разбирается здесь же, без внешних библиотек (принцип
// сборки одним g++ сохраняется). Разбор понимает и ПРЕЖНИЙ текстовый формат
// («... | v: X м/с | ... | OK»), чтобы читать архивные записи.
//
// Поля кадров, потерь и ошибок контрольной суммы заменяют собой лог: без
// записи на диске это единственный способ заметить деградацию канала.

#ifndef MAGMA_PROTOCOL_HPP
#define MAGMA_PROTOCOL_HPP

#include <array>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <optional>
#include <regex>
#include <string>

namespace magma {

/// Метка времени вида 2026-08-20T10:08:08.669 (ISO 8601 с миллисекундами).
inline std::string IsoTimestamp() {
    using namespace std::chrono;
    const auto now = system_clock::now();
    const auto time = system_clock::to_time_t(now);
    const auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    std::tm parts{};
#ifdef _WIN32
    localtime_s(&parts, &time);
#else
    localtime_r(&time, &parts);
#endif
    char date[32];
    std::strftime(date, sizeof(date), "%Y-%m-%dT%H:%M:%S", &parts);
    char result[48];
    std::snprintf(result, sizeof(result), "%s.%03d", date, static_cast<int>(ms.count()));
    return result;
}

/// Сводка за одну секунду.
struct Report {
    double speed_ms = 0.0;         ///< средняя скорость за секунду
    double speed_median_ms = 0.0;  ///< медиана за секунду, устойчива к выбросам
    std::size_t accepted = 0;      ///< оценок, прошедших порог достоверности
    std::size_t attempted = 0;     ///< всего попыток за секунду
    double snr = 0.0;              ///< средний SNR принятых оценок
    std::size_t baselines = 0;     ///< сколько баз дало вклад
    std::array<double, 4> adc{};   ///< средний уровень каналов за секунду
    unsigned long long frames = 0; ///< принято кадров с Arduino за секунду
    unsigned long long lost = 0;   ///< потеряно кадров (по счётчику в кадре)
    unsigned long long crc = 0;    ///< кадров отброшено по контрольной сумме
    unsigned long long overruns = 0; ///< кадров, где Arduino сообщил о перегрузке
    bool is_valid = false;         ///< была ли за секунду хоть одна оценка
};

/// Экранировать строку для вставки в JSON (кавычки, бэкслеш, управляющие).
/// Тексты предупреждений фиксированные, но экранирование дешёвое и снимает
/// класс ошибок навсегда.
inline std::string JsonEscape(const std::string& text) {
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
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", symbol);
                    result += buf;
                } else {
                    result += static_cast<char>(symbol);
                }
        }
    }
    return result;
}

/// Собрать строку — единый формат для передачи клиентам (JSON, NDJSON).
inline std::string FormatLine(const Report& report) {
    char buffer[512];
    std::snprintf(buffer, sizeof(buffer),
                  "{\"type\":\"report\",\"ts\":\"%s\",\"v\":%.3f,\"v_med\":%.3f,"
                  "\"accepted\":%zu,\"attempted\":%zu,\"snr\":%.2f,\"baselines\":%zu,"
                  "\"adc\":[%.0f,%.0f,%.0f,%.0f],\"frames\":%llu,\"lost\":%llu,"
                  "\"crc\":%llu,\"overruns\":%llu,\"ok\":%s}",
                  IsoTimestamp().c_str(), report.speed_ms, report.speed_median_ms,
                  report.accepted, report.attempted, report.snr, report.baselines,
                  report.adc[0], report.adc[1], report.adc[2], report.adc[3],
                  report.frames, report.lost, report.crc, report.overruns,
                  report.is_valid ? "true" : "false");
    return buffer;
}

/// Служебная строка сервера (предупреждения о состоянии) — тоже JSON.
inline std::string FormatMarker(const std::string& text) {
    return "{\"type\":\"warning\",\"ts\":\"" + IsoTimestamp() + "\",\"text\":\"" +
           JsonEscape(text) + "\"}";
}

/// Разобранная сводка.
struct Measurement {
    double speed_ms = 0.0;
    double speed_median_ms = 0.0;
    std::size_t accepted = 0;
    std::size_t attempted = 0;
    double snr = 0.0;
    std::size_t baselines = 0;
    std::array<double, 4> adc{};
    unsigned long long frames = 0, lost = 0, crc = 0, overruns = 0;
    bool is_valid = false;
};

/// Разбирает строки сервера.
///
/// Статус берётся якорем с конца строки: поиск статуса как первого слова
/// после '|' однажды уже привёл к тому, что парсер подхватывал имя поля.
class LineParser {
public:
    /// Служебная строка (предупреждение), а не данные.
    /// JSON-вариант — {"type":"warning",...}; старый текстовый — '# ...'.
    static bool IsMarker(const std::string& line) {
        if (!line.empty() && line.front() == '#') return true;
        return line.find("\"type\":\"warning\"") != std::string::npos;
    }

    /// Текст предупреждения без служебной обвязки (для печати в консоль).
    static std::string MarkerText(const std::string& line) {
        if (!line.empty() && line.front() == '#') {
            return line.size() > 2 ? line.substr(2) : line;
        }
        std::smatch match;
        if (std::regex_search(line, match, kJsonText)) return match[1].str();
        return line;
    }

    /// Разобрать строку; вернуть пусто, если формат не распознан.
    ///
    /// Числовые преобразования обёрнуты в try: регулярное выражение проверяет
    /// только форму записи, но не диапазон. Строка с числом из сорока цифр
    /// формально проходит шаблон и роняла разбор исключением out_of_range,
    /// а вместе с ним и весь клиент.
    std::optional<Measurement> Parse(const std::string& line) const {
        try {
            // Новый формат — JSON-объект; прежний текстовый (« | v: … | OK»)
            // разбирается старой веткой, чтобы архивные записи читались.
            const auto first = line.find_first_not_of(" \t");
            if (first != std::string::npos && line[first] == '{') {
                return ParseJson(line);
            }
            return ParseUnchecked(line);
        } catch (const std::exception&) {
            return std::nullopt;   // повреждённая строка — не повод падать
        }
    }

private:
    /// Разбор JSON-строки. Ключи фиксированы этим же заголовком, поэтому
    /// достаточно поиска по ключам (регулярные выражения, как и в текстовой
    /// ветке) — полноценный JSON-парсер тянуть незачем.
    std::optional<Measurement> ParseJson(const std::string& line) const {
        std::smatch adc, tail;
        if (!std::regex_search(line, tail, kJsonOk)) return std::nullopt;
        if (!std::regex_search(line, adc, kJsonAdc)) return std::nullopt;
        const auto speed = Find(line, kJsonSpeed);
        if (!speed) return std::nullopt;

        Measurement result;
        result.speed_ms = *speed;
        result.speed_median_ms = Find(line, kJsonMedian).value_or(*speed);
        result.accepted = static_cast<std::size_t>(Count(line, kJsonAccepted));
        result.attempted = static_cast<std::size_t>(Count(line, kJsonAttempted));
        result.snr = Find(line, kJsonSnr).value_or(0.0);
        result.baselines = static_cast<std::size_t>(Count(line, kJsonBaselines));
        for (int i = 0; i < 4; ++i) {
            result.adc[static_cast<std::size_t>(i)] = std::stod(adc[i + 1].str());
        }
        result.frames = Count(line, kJsonFrames);
        result.lost = Count(line, kJsonLost);
        result.crc = Count(line, kJsonCrc);
        result.overruns = Count(line, kJsonOverruns);
        result.is_valid = tail[1].str() == "true";
        return result;
    }

    std::optional<Measurement> ParseUnchecked(const std::string& line) const {
        std::smatch adc, counts, tail;
        if (!std::regex_search(line, tail, kTail)) return std::nullopt;
        if (!std::regex_search(line, adc, kAdc)) return std::nullopt;
        if (!std::regex_search(line, counts, kCounts)) return std::nullopt;
        const auto speed = Find(line, kSpeed);
        if (!speed) return std::nullopt;

        Measurement result;
        result.speed_ms = *speed;
        result.speed_median_ms = Find(line, kMedian).value_or(*speed);
        result.accepted = static_cast<std::size_t>(std::stoul(counts[1].str()));
        result.attempted = static_cast<std::size_t>(std::stoul(counts[2].str()));
        result.snr = Find(line, kSnr).value_or(0.0);
        result.baselines = static_cast<std::size_t>(Find(line, kBaselines).value_or(0.0));
        for (int i = 0; i < 4; ++i) {
            result.adc[static_cast<std::size_t>(i)] = std::stod(adc[i + 1].str());
        }
        result.frames = Count(line, kFrames);
        result.lost = Count(line, kLost);
        result.crc = Count(line, kCrc);
        result.overruns = Count(line, kOverruns);
        result.is_valid = tail[1].str() == "OK";
        return result;
    }

    static std::optional<double> Find(const std::string& line, const std::regex& pattern) {
        std::smatch match;
        if (!std::regex_search(line, match, pattern)) return std::nullopt;
        return std::stod(match[1].str());
    }

    static unsigned long long Count(const std::string& line, const std::regex& pattern) {
        std::smatch match;
        if (!std::regex_search(line, match, pattern)) return 0;
        return std::stoull(match[1].str());
    }

    // Шаблоны статические: компиляция регулярных выражений дорога.
    // JSON-ветка: ключ в кавычках с двоеточием — "v": не совпадает с
    // "v_med": (закрывающая кавычка ключа исключает частичные совпадения).
    static inline const std::regex kJsonSpeed{R"("v":([\d.]+))"};
    static inline const std::regex kJsonMedian{R"("v_med":([\d.]+))"};
    static inline const std::regex kJsonAccepted{R"("accepted":(\d+))"};
    static inline const std::regex kJsonAttempted{R"("attempted":(\d+))"};
    static inline const std::regex kJsonSnr{R"("snr":([\d.]+))"};
    static inline const std::regex kJsonBaselines{R"("baselines":(\d+))"};
    static inline const std::regex kJsonAdc{
        R"("adc":\[([\d.]+),([\d.]+),([\d.]+),([\d.]+)\])"};
    static inline const std::regex kJsonFrames{R"("frames":(\d+))"};
    static inline const std::regex kJsonLost{R"("lost":(\d+))"};
    static inline const std::regex kJsonCrc{R"("crc":(\d+))"};
    static inline const std::regex kJsonOverruns{R"("overruns":(\d+))"};
    static inline const std::regex kJsonOk{R"("ok":(true|false))"};
    static inline const std::regex kJsonText{R"re("text":"((?:[^"\\]|\\.)*)")re"};

    static inline const std::regex kSpeed{R"(\|\s*v:\s*([\d.]+))"};
    static inline const std::regex kMedian{R"(\|\s*v_med:\s*([\d.]+))"};
    static inline const std::regex kCounts{R"(оценок:\s*(\d+)/(\d+))"};
    static inline const std::regex kSnr{R"(snr:\s*([\d.]+))"};
    static inline const std::regex kBaselines{R"(баз:\s*(\d+))"};
    static inline const std::regex kAdc{
        R"(АЦП:\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+))"};
    static inline const std::regex kFrames{R"(кадров:\s*(\d+))"};
    static inline const std::regex kLost{R"(потеряно:\s*(\d+))"};
    static inline const std::regex kCrc{R"(crc:\s*(\d+))"};
    static inline const std::regex kOverruns{R"(перегрузок:\s*(\d+))"};
    static inline const std::regex kTail{R"(\|\s*(OK|REJECT)\s*$)"};
};

}  // namespace magma

#endif  // MAGMA_PROTOCOL_HPP
