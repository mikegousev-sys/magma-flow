// magma_veye.hpp — управление камерой MV-MIPI-SC130M.
//
// ПОЧЕМУ НЕ ЧЕРЕЗ V4L2.
// Стандартные органы управления V4L2 для этой камеры не подходят по двум
// причинам. Во-первых, камеры VEYE управляются через регистры по шине I2C, а
// не через ioctl видеоустройства. Во-вторых, даже если бы стандартный путь
// работал, он непригоден по разрешающей способности: единица измерения
// V4L2_CID_EXPOSURE_ABSOLUTE равна 100 микросекундам, тогда как расчётные
// выдержки для расплава лежат в диапазоне от 37 до 740 микросекунд. Значения
// короче 100 микросекунд просто обнулились бы, а остальные округлились бы до
// сотен — при том, что сенсор способен на шаг в одну строку, около 4.6 мкс.
//
// Поэтому управление ведётся штатным скриптом производителя mv_mipi_i2c_new.sh,
// который документирован и поставляется вместе с драйвером.
//
// ОБЯЗАТЕЛЬНОЕ ЧТЕНИЕ ОБРАТНО.
// Документация прямо предупреждает: сенсор не поддерживает точность выдержки
// до микросекунды, поэтому после записи значение следует прочитать для
// подтверждения. Фактическая выдержка нужна не для отчётности, а для расчёта:
// по ней вычисляется ожидаемый смаз и по ней же регулятор делает следующий
// шаг. Расхождение заданного и фактического значений тут накапливалось бы.

#ifndef MAGMA_VEYE_HPP
#define MAGMA_VEYE_HPP

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>

namespace magma {

/// Где лежит скрипт и на какой шине камера.
struct VeyeConfig {
    /// Путь к скрипту из комплекта драйвера. В свежих версиях он называется
    /// mv_mipi_i2c_new.sh; в старых — mv_mipi_i2c.sh.
    std::string script = "/home/mike/veye/mv_mipi_i2c_new.sh";
    /// Номер шины I2C. На Raspberry Pi зависит от разъёма и модели платы,
    /// определяется командой i2cdetect -l.
    int i2c_bus = 10;
    bool verbose = false;
};

/// Управление экспозицией, усилением и частотой кадров камеры VEYE MV-серии.
class VeyeControl {
public:
    explicit VeyeControl(VeyeConfig config = {}) : config_(std::move(config)) {}

    bool Available() const {
        return std::system((std::string("test -x ") + config_.script +
                            " >/dev/null 2>&1").c_str()) == 0;
    }

    const std::string& LastError() const { return error_; }

    /// Перевести камеру в ручной режим.
    ///
    /// Собственная автоматика камеры сводит к цели СРЕДНЮЮ яркость выбранной
    /// области. Для нашей задачи это неверный критерий: средняя яркость
    /// зависит от того, какую долю кадра занимает расплав, и менялась бы при
    /// изменении уровня сама по себе, заставляя камеру подстраиваться под
    /// геометрию вместо освещённости. Управление ведёт наш регулятор по
    /// высокому процентилю, поэтому автоматику камеры нужно отключить.
    bool SetManualMode() {
        return Write("expmode", "0") && Write("gainmode", "0");
    }

    /// Частота кадров. Задаётся ДО выдержки: паспортный предел выдержки равен
    /// периоду кадра, и при последующем снижении частоты выдержка не изменится
    /// сама, а при повышении — может быть молча урезана камерой.
    bool SetFrameRate(double fps) {
        char buffer[32];
        std::snprintf(buffer, sizeof(buffer), "%.2f", fps);
        return Write("fps", buffer);
    }

    /// Выдержка в микросекундах.
    bool SetExposureUs(int exposure_us) {
        char buffer[32];
        std::snprintf(buffer, sizeof(buffer), "%d", exposure_us);
        return Write("metime", buffer);
    }

    /// Усиление в децибелах. Диапазон сенсора SC130GS — от 0 до 40 дБ.
    bool SetGainDb(double gain_db) {
        char buffer[32];
        std::snprintf(buffer, sizeof(buffer), "%.1f", gain_db);
        return Write("mgain", buffer);
    }

    /// Фактическая выдержка, установленная камерой.
    std::optional<double> ReadExposureUs() { return ReadNumber("exptime"); }

    /// Фактическое усиление.
    std::optional<double> ReadGainDb() { return ReadNumber("curgain"); }

    /// Максимальная частота кадров при текущей области интереса.
    std::optional<double> ReadMaxFrameRate() { return ReadNumber("maxfps"); }

    /// Допустимый диапазон выдержки, сообщаемый камерой.
    ///
    /// Запрашивается вместо того, чтобы вычислять его по формуле: пределы
    /// зависят от области интереса и частоты кадров, и камера знает их точно.
    struct ExposureRange { double min_us = 0.0; double max_us = 0.0; };
    std::optional<ExposureRange> ReadExposureRange() {
        const std::string output = Read("exptime_range");
        if (output.empty()) return std::nullopt;
        double low = 0.0, high = 0.0;
        if (ParseTwoNumbers(output, low, high)) return ExposureRange{low, high};
        return std::nullopt;
    }

    /// Задать выдержку и усиление, затем прочитать, что получилось.
    ///
    /// Возвращает фактические значения — именно они должны идти в расчёт
    /// смаза и в следующий шаг регулятора.
    struct Applied {
        double exposure_us = 0.0;
        double gain_db = 0.0;
        bool ok = false;
    };
    Applied Apply(int exposure_us, double gain_db) {
        Applied applied;
        if (!SetExposureUs(exposure_us)) return applied;
        if (!SetGainDb(gain_db)) return applied;

        const auto actual_exposure = ReadExposureUs();
        const auto actual_gain = ReadGainDb();
        if (!actual_exposure) return applied;

        applied.exposure_us = *actual_exposure;
        applied.gain_db = actual_gain.value_or(gain_db);
        applied.ok = true;

        if (config_.verbose) {
            std::printf("[VEYE] выдержка задана %d мкс, установлена %.0f мкс; "
                        "усиление %.1f дБ\n",
                        exposure_us, applied.exposure_us, applied.gain_db);
        }
        return applied;
    }

    /// Однократная автонастройка с сохранением результата как ручного значения.
    ///
    /// Полезна при первичной наладке: даёт разумную стартовую точку, от которой
    /// дальше работает наш регулятор. В рабочем режиме не применяется.
    bool AutoOnce() { return Write("aeag_run_once_save", "1"); }

private:
    std::string Command(const std::string& tail) const {
        return config_.script + " " + tail + " -b " +
               std::to_string(config_.i2c_bus) + " 2>&1";
    }

    bool Write(const std::string& field, const std::string& value) {
        const std::string command = Command("-w " + field + " " + value);
        FILE* pipe = ::popen(command.c_str(), "r");
        if (pipe == nullptr) { error_ = "не запустить " + config_.script; return false; }
        std::string output;
        char buffer[256];
        while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) output += buffer;
        const int status = ::pclose(pipe);
        if (status != 0) {
            error_ = field + ": " + output;
            return false;
        }
        return true;
    }

    std::string Read(const std::string& field) {
        const std::string command = Command("-r " + field);
        FILE* pipe = ::popen(command.c_str(), "r");
        if (pipe == nullptr) { error_ = "не запустить " + config_.script; return {}; }
        std::string output;
        char buffer[256];
        while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) output += buffer;
        if (::pclose(pipe) != 0) { error_ = field + ": " + output; return {}; }
        return output;
    }

    /// Из вывода скрипта извлекается первое число. Формат вывода между
    /// версиями скрипта менялся, поэтому разбор намеренно нестрогий: ищется
    /// число, а не фиксированный шаблон строки.
    std::optional<double> ReadNumber(const std::string& field) {
        const std::string output = Read(field);
        if (output.empty()) return std::nullopt;
        double value = 0.0;
        if (ParseFirstNumber(output, value)) return value;
        return std::nullopt;
    }

    static bool ParseFirstNumber(const std::string& text, double& value) {
        for (std::size_t i = 0; i < text.size(); ++i) {
            if (!std::isdigit(static_cast<unsigned char>(text[i]))) continue;
            std::size_t start = i;
            if (start > 0 && (text[start - 1] == '-' || text[start - 1] == '.')) {
                --start;
            }
            value = std::strtod(text.c_str() + start, nullptr);
            return true;
        }
        return false;
    }

    static bool ParseTwoNumbers(const std::string& text, double& first, double& second) {
        const char* p = text.c_str();
        char* end = nullptr;
        int found = 0;
        while (*p != '\0' && found < 2) {
            if (std::isdigit(static_cast<unsigned char>(*p))) {
                const double value = std::strtod(p, &end);
                (found == 0 ? first : second) = value;
                ++found;
                p = end;
            } else {
                ++p;
            }
        }
        return found == 2;
    }

    VeyeConfig config_;
    std::string error_;
};

}  // namespace magma

#endif  // MAGMA_VEYE_HPP
