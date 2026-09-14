// magma_client_logging.cpp — клиент с записью сводок на диск.
//
// Полностью повторяет обычный клиент (magma_client.cpp): те же подключение,
// разбор, история в памяти и вывод. Отличие одно — принятые сводки пишутся
// в файл с ротацией и автоочисткой.
//
// На Raspberry Pi запись отключена совсем, поэтому вести архив имеет смысл
// именно здесь: на рабочей машине места достаточно, а при одной сводке в
// секунду по ~200 байт это около 17 МБ в сутки против 17 ГБ у прежней версии.
// Архив нужен для разбора инцидентов: все ошибки, найденные в этом проекте,
// были найдены по записям.
//
// О ГРАФИКАХ: у matplotlib нет прямого аналога в стандартном C++, а тянуть Qt
// ради двух графиков усложняет сборку. Здесь сделан потокобезопасный слой
// данных (History) и текстовый вывод; к этому слою подключается любая
// библиотека отрисовки.
//
// Разделы: 1) конфигурация, 2) история, 3) сеть, 4) приложение.
//
// Сборка (Linux):   g++ -std=c++17 -O2 -pthread magma_client_logging.cpp -o magma_client_logging
// Сборка (Windows): g++ -std=c++17 -O2 magma_client_logging.cpp -o magma_client_logging.exe -lws2_32

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
using socket_t = SOCKET;
constexpr socket_t kInvalidSocket = INVALID_SOCKET;
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
using socket_t = int;
constexpr socket_t kInvalidSocket = -1;
#endif

#include <array>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "magma_logging.hpp"
#include "magma_protocol.hpp"

namespace magma {

// =============================================================================
// 1. КОНФИГУРАЦИЯ
// =============================================================================

struct ConnectionConfig {
    std::string host = "127.0.0.1";
    uint16_t port = 9000;
    double recv_timeout_s = 1.0;
    double reconnect_delay_s = 2.0;
    std::size_t recv_chunk_bytes = 4096;
    /// Потолок приёмного буфера: защита от роста при потоке без '\n'.
    std::size_t max_buffer_bytes = 1u << 20;
};

struct ClientLogConfig {
#ifdef _WIN32
    std::string directory = "C:/magma-logs";
#else
    std::string directory = "./magma-logs";
#endif
    std::string name_prefix = "client";
    std::size_t max_lines_per_file = 1'000'000;
    std::size_t flush_lines = 10;      // при 1 Гц сбрасываем часто: потерять
    double flush_interval_s = 5.0;     // последние строки при аварии недопустимо
};

struct DisplayConfig {
    std::size_t history_points = 3600;  // при 1 Гц это последний час
    bool print_each_report = true;      // печатать каждую сводку в консоль
};

constexpr std::size_t kParseWarnEvery = 10;

inline double MonotonicSeconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

/// Флаг остановки с ожиданием по таймауту.
class StopFlag {
public:
    void Set() {
        { std::lock_guard<std::mutex> lock(mutex_); stopped_ = true; }
        condition_.notify_all();
    }
    bool IsSet() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stopped_;
    }
    void Wait(double seconds) const {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait_for(lock, std::chrono::duration<double>(seconds),
                            [this] { return stopped_; });
    }

private:
    mutable std::mutex mutex_;
    mutable std::condition_variable condition_;
    bool stopped_ = false;
};

// =============================================================================
// 2. ИСТОРИЯ
// =============================================================================

/// Кольцевая история сводок для отрисовки и текущего состояния.
///
/// Доступ потокобезопасен: наполняет сетевой поток, читает поток отображения.
class History {
public:
    explicit History(std::size_t capacity) : capacity_(capacity) {}

    void Add(const Measurement& measurement) {
        std::lock_guard<std::mutex> lock(mutex_);
        records_.push_back(measurement);
        if (records_.size() > capacity_) records_.pop_front();
        ++received_;
    }

    /// Копия истории для отрисовки.
    std::vector<Measurement> Snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return {records_.begin(), records_.end()};
    }

    std::optional<Measurement> Latest() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (records_.empty()) return std::nullopt;
        return records_.back();
    }

    std::size_t Received() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return received_;
    }

private:
    mutable std::mutex mutex_;
    std::deque<Measurement> records_;
    std::size_t capacity_;
    std::size_t received_ = 0;
};

/// Статистика разбора с периодическим предупреждением.
///
/// Молчаливое отбрасывание строк однажды уже скрыло несовместимость форматов
/// сервера и клиента, поэтому о неудачах сообщаем явно. Порог мал: при одной
/// строке в секунду десять подряд нераспознанных — уже десять секунд вслепую.
class ParseStats {
public:
    void NoteSuccess() { ++parsed_; }

    void NoteFailure(const std::string& sample) {
        ++failed_;
        if (failed_ - last_warn_ < kParseWarnEvery) return;
        last_warn_ = failed_;
        std::printf("[PARSE] Не распознано строк: %zu (успешно: %zu). Пример: %.120s\n",
                    failed_, parsed_, sample.c_str());
        std::printf("[PARSE] Возможно, формат сервера изменился — проверьте версии.\n");
    }

    std::string Summary() const {
        char buffer[128];
        std::snprintf(buffer, sizeof(buffer), "Разобрано сводок: %zu, не распознано: %zu",
                      parsed_, failed_);
        return buffer;
    }

private:
    std::size_t parsed_ = 0, failed_ = 0, last_warn_ = 0;
};

// =============================================================================
// 3. СЕТЬ
// =============================================================================

/// Принимает сводки от сервера и наполняет историю.
///
/// При обрыве связи переподключается с паузой; при потоке без переводов
/// строки сбрасывает буфер, не давая ему расти бесконечно.
class ServerConnection {
public:
    ServerConnection(RotatingLogger& logger, ParseStats& stats, History& history,
                     StopFlag& stop, ConnectionConfig config, bool print_each)
        : logger_(logger), stats_(stats), history_(history), stop_(stop),
          config_(std::move(config)), print_each_(print_each) {}

    void Run() {
        while (!stop_.IsSet()) {
            const socket_t connection = Connect();
            if (connection == kInvalidSocket) continue;
            ReceiveLoop(connection);
            Close(connection);
            std::printf("[SOCKET] Соединение закрыто\n");
        }
    }

private:
    static void Close(socket_t connection) {
#ifdef _WIN32
        ::closesocket(connection);
#else
        ::close(connection);
#endif
    }

    socket_t Connect() {
        std::printf("[SOCKET] Подключение к %s:%u...\n", config_.host.c_str(), config_.port);
        const socket_t connection = ::socket(AF_INET, SOCK_STREAM, 0);
        if (connection == kInvalidSocket) {
            stop_.Wait(config_.reconnect_delay_s);
            return kInvalidSocket;
        }
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(config_.port);
        ::inet_pton(AF_INET, config_.host.c_str(), &address.sin_addr);

        if (::connect(connection, reinterpret_cast<sockaddr*>(&address),
                      sizeof(address)) < 0) {
            std::printf("[SOCKET] Не подключиться. Повтор через %.0f с\n",
                        config_.reconnect_delay_s);
            Close(connection);
            stop_.Wait(config_.reconnect_delay_s);
            return kInvalidSocket;
        }
#ifdef _WIN32
        DWORD timeout = static_cast<DWORD>(config_.recv_timeout_s * 1000);
        ::setsockopt(connection, SOL_SOCKET, SO_RCVTIMEO,
                     reinterpret_cast<const char*>(&timeout), sizeof(timeout));
#else
        timeval timeout{static_cast<time_t>(config_.recv_timeout_s), 0};
        ::setsockopt(connection, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
#endif
        std::printf("[SOCKET] Подключено: %s:%u\n", config_.host.c_str(), config_.port);
        return connection;
    }

    void HandleLine(const std::string& text) {
        // Пишем сырую строку до разбора: даже нераспознанную сводку полезно
        // сохранить, иначе при расхождении версий в архиве будет пусто.
        logger_.Write(text);

        if (LineParser::IsMarker(text)) {
            std::printf("[SERVER] %s\n", LineParser::MarkerText(text).c_str());
            return;
        }
        const auto measurement = parser_.Parse(text);
        if (!measurement) { stats_.NoteFailure(text); return; }
        stats_.NoteSuccess();
        history_.Add(*measurement);

        if (print_each_) {
            std::printf("v %.3f м/с (мед %.3f) | оценок %zu/%zu | snr %.1f | "
                        "АЦП %.0f %.0f %.0f %.0f | кадров %llu",
                        measurement->speed_ms, measurement->speed_median_ms,
                        measurement->accepted, measurement->attempted, measurement->snr,
                        measurement->adc[0], measurement->adc[1], measurement->adc[2],
                        measurement->adc[3], measurement->frames);
            if (measurement->lost > 0 || measurement->crc > 0 ||
                measurement->overruns > 0) {
                std::printf(" | ПОТЕРИ %llu, CRC %llu, ПЕРЕГРУЗОК %llu",
                            measurement->lost, measurement->crc, measurement->overruns);
            }
            std::printf(" | %s\n", measurement->is_valid ? "OK" : "REJECT");
        }
    }

    /// Обработать завершённые строки; вернуть неполный остаток.
    std::string Drain(std::string buffer) {
        std::size_t start = 0, position;
        while ((position = buffer.find('\n', start)) != std::string::npos) {
            std::string text = buffer.substr(start, position - start);
            while (!text.empty() && (text.back() == '\r' || text.back() == ' ')) {
                text.pop_back();
            }
            if (!text.empty()) HandleLine(text);
            start = position + 1;
        }
        return buffer.substr(start);
    }

    void ReceiveLoop(socket_t connection) {
        std::string buffer;
        std::vector<char> chunk(config_.recv_chunk_bytes);
        while (!stop_.IsSet()) {
            const auto received =
                ::recv(connection, chunk.data(), static_cast<int>(chunk.size()), 0);
            if (received < 0) {
#ifdef _WIN32
                if (WSAGetLastError() == WSAETIMEDOUT) continue;
#else
                if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) continue;
#endif
                break;
            }
            if (received == 0) break;

            buffer.append(chunk.data(), static_cast<std::size_t>(received));
            if (buffer.size() > config_.max_buffer_bytes) {
                std::printf("[SOCKET] Буфер превысил %zu байт без перевода строки — "
                            "сбрасываю\n", config_.max_buffer_bytes);
                buffer.clear();
                continue;
            }
            buffer = Drain(std::move(buffer));
        }
    }

    RotatingLogger& logger_;
    ParseStats& stats_;
    History& history_;
    StopFlag& stop_;
    ConnectionConfig config_;
    bool print_each_;
    LineParser parser_;
};

// =============================================================================
// 4. ПРИЛОЖЕНИЕ
// =============================================================================

/// Связывает сеть и отображение, обеспечивая чистое завершение.
class MagmaClient {
public:
    MagmaClient() : history_(display_.history_points) {}

    void Run() {
        std::error_code error;
        std::filesystem::create_directories(log_.directory, error);
        logger_ = std::make_unique<RotatingLogger>(
            log_.directory, log_.name_prefix, log_.max_lines_per_file,
            log_.flush_lines, log_.flush_interval_s);
        cleaner_ = std::make_unique<LogCleaner>(MakeLogConfig());

        std::printf("[ДИСК] Запись включена: %s, история в памяти %zu записей\n",
                    log_.directory.c_str(), display_.history_points);

        ServerConnection connection(*logger_, stats_, history_, stop_, connection_,
                                    display_.print_each_report);
        std::thread worker([&connection] { connection.Run(); });
        std::thread cleanup([this] { CleanupWorker(); });

        while (!stop_.IsSet()) stop_.Wait(0.5);

        // Поток приёма может быть убит при выходе, поэтому лог закрывает
        // главный поток — иначе последние строки остались бы в буфере.
        worker.join();
        cleanup.join();
        logger_->Close();
        std::printf("[STOP] %s\n", stats_.Summary().c_str());
    }

    void RequestStop() { stop_.Set(); }

private:
    /// Настройки очистки каталога: имена файлов и сроки хранения.
    LogConfig MakeLogConfig() const {
        LogConfig config;
        config.directory = log_.directory;
        config.name_prefix = log_.name_prefix;
        config.retention_days = 30.0;   // сводки компактны, храним месяц
        config.min_free_disk_mb = 500.0;
        return config;
    }

    /// Периодическая очистка каталога от устаревших файлов.
    void CleanupWorker() {
        while (!stop_.IsSet()) {
            const auto result = cleaner_->Run(logger_->Path(), false);
            if (result.removed > 0) {
                std::printf("[CLEANUP] Удалено файлов: %zu, освобождено %.1f МБ\n",
                            result.removed,
                            static_cast<double>(result.freed_bytes) / 1024.0 / 1024.0);
            }
            stop_.Wait(3600.0);   // раз в час: при 17 МБ/сутки чаще незачем
        }
    }

    ConnectionConfig connection_;
    ClientLogConfig log_;
    DisplayConfig display_;

    StopFlag stop_;
    History history_;
    ParseStats stats_;
    std::unique_ptr<RotatingLogger> logger_;
    std::unique_ptr<LogCleaner> cleaner_;
};

}  // namespace magma

namespace {
magma::MagmaClient* g_client = nullptr;
void HandleSignal(int) { if (g_client) g_client->RequestStop(); }
}  // namespace

int main() {
#ifdef _WIN32
    WSADATA wsa_data;
    if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) return 1;
#endif
    magma::MagmaClient client;
    g_client = &client;
    std::signal(SIGINT, HandleSignal);
    std::signal(SIGTERM, HandleSignal);
    client.Run();
#ifdef _WIN32
    WSACleanup();
#endif
    std::printf("\n[STOP] Завершение...\n");
    return 0;
}
