// magma_config.hpp — файл настроек «ключ = значение» с горячей перезагрузкой.
//
// НАЗНАЧЕНИЕ. Угол камеры, масштаб, положение полос выборки, координаты
// калибровочных меток — всё это известно только после того, как камера
// установлена и наведена на жёлоб, и подбирается итеративно, на месте.
// Требовать пересборки программы на каждую пробу немыслимо. Поэтому такие
// параметры читаются из текстового файла, а не зашиты в код, и файл
// перечитывается на ходу: технику достаточно сохранить изменённый файл,
// и новые значения применятся к следующей серии кадров без перезапуска
// процесса.
//
// ФОРМАТ. Простейший из возможных: одна пара «ключ = значение» на строку,
// пустые строки и строки, начинающиеся с '#', игнорируются. Внешних
// библиотек не требует — то же решение, что уже принято для остальных
// частей проекта: минимум зависимостей, максимум прозрачности при чтении
// исходников.
//
// ПОВТОРЯЮЩИЕСЯ КЛЮЧИ. Одна и та же строка ключа может встречаться в файле
// несколько раз — так задаются списки переменной длины, в первую очередь
// калибровочные метки: одна строка "mark = ..." на каждую метку.

#ifndef MAGMA_CONFIG_HPP
#define MAGMA_CONFIG_HPP

#include <cctype>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace magma {

/// Файл настроек вида "ключ = значение" с отслеживанием изменений на диске.
class ConfigFile {
public:
    /// Загрузить файл. При неудаче возвращает false и НЕ ТРОГАЕТ уже
    /// загруженные значения: временная недоступность файла (например,
    /// редактор на мгновение удаляет его перед пересозданием при сохранении)
    /// не должна обнулять уже работающие настройки процесса.
    bool Load(const std::string& path) {
        std::ifstream file(path);
        if (!file) return false;

        std::map<std::string, std::vector<std::string>> next;
        std::string line;
        while (std::getline(file, line)) {
            const std::string trimmed = Trim(line);
            if (trimmed.empty() || trimmed.front() == '#') continue;
            const std::size_t eq = trimmed.find('=');
            if (eq == std::string::npos) continue;
            const std::string key = Trim(trimmed.substr(0, eq));
            const std::string value = Trim(trimmed.substr(eq + 1));
            if (key.empty()) continue;
            next[key].push_back(value);
        }

        values_ = std::move(next);
        path_ = path;

        // Метка времени запоминается уже здесь, а не только в ReloadIfChanged():
        // иначе первый же вызов ReloadIfChanged() после явного Load() увидел бы
        // «пусто» в last_write_ и сообщил бы о мнимом изменении файла, который
        // на самом деле только что был прочитан этим же вызовом. Именно это и
        // проявилось при пробном запуске: строка «файл настроек изменился»
        // печаталась сразу после старта, хотя файл никто не трогал.
        std::error_code error;
        const auto stamp = std::filesystem::last_write_time(path_, error);
        if (!error) last_write_ = stamp;
        return true;
    }

    /// Перечитать файл, если он изменился с последней успешной загрузки.
    /// Возвращает true, если содержимое действительно обновилось — по этому
    /// признаку вызывающая сторона решает, нужно ли пересчитывать то, что
    /// зависит от настроек.
    bool ReloadIfChanged() {
        if (path_.empty()) return false;
        std::error_code error;
        const auto stamp = std::filesystem::last_write_time(path_, error);
        if (error) return false;  // файл временно недоступен — не в счёт
        if (last_write_ && *last_write_ == stamp) return false;
        last_write_ = stamp;
        return Load(path_);
    }

    /// Последнее значение ключа (при повторе побеждает последняя строка).
    std::string GetString(const std::string& key, std::string fallback = {}) const {
        const auto found = values_.find(key);
        if (found == values_.end() || found->second.empty()) return fallback;
        return found->second.back();
    }

    double GetDouble(const std::string& key, double fallback) const {
        const std::string text = GetString(key);
        if (text.empty()) return fallback;
        try { return std::stod(text); } catch (...) { return fallback; }
    }

    int GetInt(const std::string& key, int fallback) const {
        const std::string text = GetString(key);
        if (text.empty()) return fallback;
        try { return std::stoi(text); } catch (...) { return fallback; }
    }

    std::size_t GetSize(const std::string& key, std::size_t fallback) const {
        const int value = GetInt(key, static_cast<int>(fallback));
        return value >= 0 ? static_cast<std::size_t>(value) : fallback;
    }

    bool GetBool(const std::string& key, bool fallback) const {
        const std::string text = GetString(key);
        if (text.empty()) return fallback;
        return text == "1" || text == "true" || text == "yes" || text == "on";
    }

    /// Все значения повторяющегося ключа в порядке появления в файле.
    /// Применяется там, где список имеет переменную длину — прежде всего
    /// для калибровочных меток, которых может быть три, четыре или больше.
    std::vector<std::string> GetList(const std::string& key) const {
        const auto found = values_.find(key);
        if (found == values_.end()) return {};
        return found->second;
    }

    /// Все значения повторяющегося ключа одной строкой — дешёвая «подпись»
    /// списка, по которой можно обнаружить, что он ИЗМЕНИЛСЯ между
    /// перезагрузками, не сравнивая поля по отдельности.
    std::string Signature(const std::string& key) const {
        std::string result;
        for (const std::string& value : GetList(key)) result += value + ';';
        return result;
    }

private:
    static std::string Trim(const std::string& text) {
        std::size_t begin = 0, end = text.size();
        while (begin < end && std::isspace(static_cast<unsigned char>(text[begin]))) ++begin;
        while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) --end;
        return text.substr(begin, end - begin);
    }

    std::map<std::string, std::vector<std::string>> values_;
    std::string path_;
    std::optional<std::filesystem::file_time_type> last_write_;
};

/// Разобрать строку вида "u=0 v=0 x=120.5 y=340.2 name=near-left" на поля.
/// Свой маленький разбор внутри значения одного ключа — так координаты
/// метки умещаются в одну строку файла и не требуют вложенных секций,
/// которых общий формат "ключ = значение" не поддерживает.
inline std::map<std::string, std::string> ParseFields(const std::string& text) {
    std::map<std::string, std::string> fields;
    std::istringstream stream(text);
    std::string token;
    while (stream >> token) {
        const std::size_t eq = token.find('=');
        if (eq == std::string::npos) continue;
        fields[token.substr(0, eq)] = token.substr(eq + 1);
    }
    return fields;
}

/// Число с плавающей точкой из разобранных полей, с запасным значением.
inline double FieldDouble(const std::map<std::string, std::string>& fields,
                          const std::string& key, double fallback = 0.0) {
    const auto found = fields.find(key);
    if (found == fields.end()) return fallback;
    try { return std::stod(found->second); } catch (...) { return fallback; }
}

inline std::string FieldString(const std::map<std::string, std::string>& fields,
                               const std::string& key, std::string fallback = {}) {
    const auto found = fields.find(key);
    return found != fields.end() ? found->second : fallback;
}

}  // namespace magma

#endif  // MAGMA_CONFIG_HPP
