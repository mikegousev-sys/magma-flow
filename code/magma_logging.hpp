// magma_logging.hpp — запись логов с ротацией и автоочисткой.
//
// Используется только клиентом с логированием: на Raspberry Pi запись
// отключена совсем. Сводки приходят раз в секунду по ~200 байт, то есть
// около 17 МБ в сутки — триггер по уровню сигнала здесь не нужен, но
// ротация, периодический сброс на диск и автоудаление старых файлов
// сохранены, чтобы каталог не рос бесконечно.

#ifndef MAGMA_LOGGING_HPP
#define MAGMA_LOGGING_HPP

#include <chrono>
#include <cstdio>
#include <ctime>
#include <deque>
#include <filesystem>
#include <mutex>
#include <optional>
#include <regex>
#include <string>
#include <vector>

#include "magma_protocol.hpp"

namespace magma {

// =============================================================================
// Конфигурация
// =============================================================================

struct LogConfig {
    std::string directory = "/home/mike/magma-logs";
    std::string name_prefix = "speed";
    std::size_t max_lines_per_file = 1'000'000;

    double retention_days = 1.0;
    double cleanup_interval_s = 600.0;
    double min_free_disk_mb = 500.0;

    std::size_t flush_lines = 500;
    double flush_interval_s = 2.0;
    double stats_interval_s = 60.0;
};

// =============================================================================
// Ротация файлов
// =============================================================================

/// Лог-файл с именем по времени старта и ротацией по числу строк.
///
/// Периодический сброс на диск обязателен: без него последние строки терялись
/// при аварийном завершении процесса.
class RotatingLogger {
public:
    RotatingLogger(std::string directory, std::string prefix, std::size_t max_lines,
                   std::size_t flush_lines, double flush_interval_s)
        : directory_(std::move(directory)),
          prefix_(std::move(prefix)),
          max_lines_(max_lines),
          flush_lines_(flush_lines),
          flush_interval_s_(flush_interval_s),
          base_name_(MakeBaseName(prefix_)) {
        OpenCurrent();
    }

    ~RotatingLogger() { Close(); }

    RotatingLogger(const RotatingLogger&) = delete;
    RotatingLogger& operator=(const RotatingLogger&) = delete;

    /// Путь к активному файлу (нужен автоочистке, чтобы его не удалить).
    std::string Path() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return path_;
    }

    /// Записать строку; после Close() вызов игнорируется.
    void Write(const std::string& line) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_ || file_ == nullptr) return;

        std::fputs(line.c_str(), file_);
        std::fputc('\n', file_);
        ++lines_;
        ++lines_since_flush_;

        const double now = MonotonicSeconds();
        if (lines_since_flush_ >= flush_lines_ ||
            now - last_flush_ >= flush_interval_s_) {
            std::fflush(file_);
            lines_since_flush_ = 0;
            last_flush_ = now;
        }

        if (lines_ >= max_lines_) Rotate();
    }

    /// Сбросить данные на диск и закрыть файл. Идемпотентно.
    void Close() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_ != nullptr && !closed_) {
            std::fflush(file_);
            std::fclose(file_);
        }
        file_ = nullptr;
        closed_ = true;
    }

private:
    static std::string MakeBaseName(const std::string& prefix) {
        const std::string stamp = IsoTimestamp();  // 2026-08-20T10:08:08.669
        std::string compact;
        for (char symbol : stamp) {
            if (symbol == '.') break;
            if (symbol != '-' && symbol != ':') compact.push_back(symbol);
        }
        compact[8] = '_';  // YYYYMMDDTHHMMSS -> YYYYMMDD_HHMMSS
        return prefix + "_" + compact;
    }

    /// Открыть очередную часть (вызывается под уже взятым локом).
    void OpenCurrent() {
        const std::string suffix = part_ == 1 ? "" : "_" + std::to_string(part_);
        path_ = directory_ + "/" + base_name_ + suffix + ".log";
        file_ = std::fopen(path_.c_str(), "a");
        lines_ = 0;
        lines_since_flush_ = 0;
        last_flush_ = MonotonicSeconds();
        std::printf("[LOG] Файл: %s\n", path_.c_str());
    }

    /// Начать следующую часть (вызывается под уже взятым локом).
    void Rotate() {
        std::fflush(file_);
        std::fclose(file_);
        ++part_;
        OpenCurrent();
    }

    static double MonotonicSeconds() {
        using namespace std::chrono;
        return duration<double>(steady_clock::now().time_since_epoch()).count();
    }

    std::string directory_;
    std::string prefix_;
    std::size_t max_lines_;
    std::size_t flush_lines_;
    double flush_interval_s_;
    std::string base_name_;
    std::string path_;

    mutable std::mutex mutex_;
    std::FILE* file_ = nullptr;
    bool closed_ = false;
    std::size_t part_ = 1;
    std::size_t lines_ = 0;
    std::size_t lines_since_flush_ = 0;
    double last_flush_ = 0.0;
};

// =============================================================================
// Запись по триггеру
// =============================================================================

// =============================================================================
// Автоочистка
// =============================================================================

/// Удаляет устаревшие лог-файлы, освобождая место на карте памяти.
///
/// Трогает только файлы, подходящие под строгий шаблон имени, и никогда —
/// файл, открытый на запись.
class LogCleaner {
public:
    explicit LogCleaner(LogConfig config)
        : config_(std::move(config)),
          pattern_("^" + config_.name_prefix + R"(_\d{8}_\d{6}(?:_\d+)?\.log$)") {}

    struct Result {
        std::size_t removed = 0;
        std::uintmax_t freed_bytes = 0;
    };

    /// Свободное место в каталоге логов, МБ (пусто — определить не удалось).
    std::optional<double> FreeSpaceMb() const {
        std::error_code error;
        const auto space = std::filesystem::space(config_.directory, error);
        if (error) return std::nullopt;
        return static_cast<double>(space.available) / 1024.0 / 1024.0;
    }

    /// Очистить по возрасту и, при нехватке места, по размеру.
    Result Run(const std::string& protect_path = {}, bool verbose = true) {
        Result result;
        auto entries = Collect(protect_path);

        const auto cutoff = std::filesystem::file_time_type::clock::now() -
                            std::chrono::seconds(
                                static_cast<long long>(config_.retention_days * 86400));

        std::vector<Entry> survivors;
        for (const Entry& entry : entries) {
            if (entry.modified < cutoff && Remove(entry, "устарел", verbose)) {
                ++result.removed;
                result.freed_bytes += entry.size;
            } else {
                survivors.push_back(entry);
            }
        }

        auto free_mb = FreeSpaceMb();
        if (!free_mb || *free_mb >= config_.min_free_disk_mb) return result;

        if (verbose) {
            std::printf("[CLEANUP] Мало места: %.0f МБ < %.0f МБ — удаляю самые старые логи\n",
                        *free_mb, config_.min_free_disk_mb);
        }
        for (const Entry& entry : survivors) {
            if (*free_mb >= config_.min_free_disk_mb) break;
            if (Remove(entry, "нехватка места", verbose)) {
                ++result.removed;
                result.freed_bytes += entry.size;
                *free_mb += static_cast<double>(entry.size) / 1024.0 / 1024.0;
            }
        }
        return result;
    }

private:
    struct Entry {
        std::filesystem::file_time_type modified;
        std::uintmax_t size = 0;
        std::string path;
    };

    /// Кандидаты на удаление, самые старые первыми.
    std::vector<Entry> Collect(const std::string& protect_path) const {
        std::vector<Entry> entries;
        std::error_code error;

        std::filesystem::directory_iterator iterator(config_.directory, error);
        if (error) {
            std::printf("[CLEANUP] Не удалось прочитать %s: %s\n",
                        config_.directory.c_str(), error.message().c_str());
            return entries;
        }

        const std::string protect =
            protect_path.empty() ? "" : std::filesystem::absolute(protect_path).string();

        for (const auto& item : iterator) {
            if (!item.is_regular_file(error)) continue;
            const std::string name = item.path().filename().string();
            if (!std::regex_match(name, pattern_)) continue;  // чужие файлы не трогаем

            const std::string absolute = std::filesystem::absolute(item.path()).string();
            if (!protect.empty() && absolute == protect) continue;

            Entry entry;
            entry.modified = item.last_write_time(error);
            if (error) continue;
            entry.size = item.file_size(error);
            if (error) continue;
            entry.path = item.path().string();
            entries.push_back(entry);
        }

        std::sort(entries.begin(), entries.end(),
                  [](const Entry& left, const Entry& right) {
                      return left.modified < right.modified;
                  });
        return entries;
    }

    static bool Remove(const Entry& entry, const char* reason, bool verbose) {
        std::error_code error;
        if (!std::filesystem::remove(entry.path, error) || error) {
            if (verbose) {
                std::printf("[CLEANUP] Не удалось удалить %s: %s\n",
                            entry.path.c_str(), error.message().c_str());
            }
            return false;
        }
        if (verbose) {
            std::printf("[CLEANUP] Удалён (%s): %s (%.0f МБ)\n", reason,
                        std::filesystem::path(entry.path).filename().string().c_str(),
                        static_cast<double>(entry.size) / 1024.0 / 1024.0);
        }
        return true;
    }

    LogConfig config_;
    std::regex pattern_;
};

}  // namespace magma

#endif  // MAGMA_LOGGING_HPP
