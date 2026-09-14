// magma_vision_client.cpp — клиент процесса зрения (скорость по камере, уровень).
//
// Аналог magma_client.cpp из защищённого ядра, но для отдельного протокола
// и порта зрения (magma_vision_protocol.hpp, порт 9100 по умолчанию) —
// это разные потоки данных с разным тактом, и объединять их в одном клиенте
// значило бы либо трогать ядро, либо усложнять его разбор чужими полями.
//
// Ведение архива НЕОБЯЗАТЕЛЬНО и включается ключом командной строки: сам
// процесс зрения на диск ничего не пишет (как и защищённый сервер), а если
// история измерений всё же нужна, эта программа использует RotatingLogger и
// LogCleaner из защищённого magma_logging.hpp В НЕИЗМЕНЁННОМ ВИДЕ — простое
// подключение заголовка, а не редактирование его кода.
//
// Сборка:
//   g++ -std=c++17 -O2 -pthread magma_vision_client.cpp -o magma_vision_client
// Запуск:
//   ./magma_vision_client [хост] [порт] [--log[=каталог]]

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

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

#include "magma_logging.hpp"          // ядро, подключается, но не изменяется
#include "magma_vision_protocol.hpp"  // дополнение

namespace magma {

struct ClientConfig {
    std::string host = "127.0.0.1";
    uint16_t port = 9100;
    double reconnect_delay_s = 2.0;
    double recv_timeout_s = 1.0;
    std::size_t recv_chunk_bytes = 4096;
    std::size_t max_buffer_bytes = 1u << 20;
};

/// Флаг остановки с ожиданием по таймауту — тот же приём, что и в клиентах
/// ядра; код продублирован, а не подключён из ядра, поскольку ядро не
/// экспортирует эту небольшую вспомогательную часть отдельно от файла
/// целиком, а трогать защищённый файл ради экспорта одного класса не стоит.
class StopFlag {
public:
    void Set() { { std::lock_guard<std::mutex> lock(mutex_); stopped_ = true; }
                cv_.notify_all(); }
    bool IsSet() const { std::lock_guard<std::mutex> lock(mutex_); return stopped_; }
    void Wait(double seconds) const {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait_for(lock, std::chrono::duration<double>(seconds), [this] { return stopped_; });
    }

private:
    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    bool stopped_ = false;
};

/// Принимает строки от процесса зрения, печатает их и, если включено,
/// передаёт логгеру.
class VisionConnection {
public:
    VisionConnection(StopFlag& stop, ClientConfig config, RotatingLogger* logger)
        : stop_(stop), config_(std::move(config)), logger_(logger) {}

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
        if (connection == kInvalidSocket) { stop_.Wait(config_.reconnect_delay_s); return kInvalidSocket; }
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(config_.port);
        ::inet_pton(AF_INET, config_.host.c_str(), &address.sin_addr);
        if (::connect(connection, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
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
        if (logger_ != nullptr) logger_->Write(text);   // сначала архив, потом разбор

        if (VisionLineParser::IsMarker(text)) {
            std::printf("[ЗРЕНИЕ] %s\n", VisionLineParser::MarkerText(text).c_str());
            return;
        }
        const auto measurement = parser_.Parse(text);
        if (!measurement) {
            std::printf("[PARSE] Не распознана строка: %.160s\n", text.c_str());
            return;
        }

        std::printf("скорость(камера) ");
        if (measurement->speed_valid) {
            std::printf("%.3f м/с (snr %.1f)", measurement->speed_ms, measurement->speed_snr);
        } else {
            std::printf("нет данных");
        }
        std::printf(" | уровень ");
        if (measurement->level_valid) {
            std::printf("%.2f мм (σ %.2f, %s)", measurement->level_mm,
                        measurement->level_sigma_mm, measurement->level_state.c_str());
        } else {
            std::printf("нет данных (%s)", measurement->level_state.c_str());
        }
        std::printf(" | выдержка %.0f мкс, усиление %.1f дБ, кадров %zu\n",
                    measurement->camera_exposure_us, measurement->camera_gain_db,
                    measurement->camera_frames);
    }

    std::string Drain(std::string buffer) {
        std::size_t start = 0, position;
        while ((position = buffer.find('\n', start)) != std::string::npos) {
            std::string text = buffer.substr(start, position - start);
            while (!text.empty() && (text.back() == '\r' || text.back() == ' ')) text.pop_back();
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

    StopFlag& stop_;
    ClientConfig config_;
    RotatingLogger* logger_;
    VisionLineParser parser_;
};

}  // namespace magma

namespace {
magma::StopFlag* g_stop = nullptr;
void HandleSignal(int) { if (g_stop != nullptr) g_stop->Set(); }
}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    WSADATA wsa_data;
    if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) return 1;
#endif

    magma::ClientConfig config;
    std::string log_directory;
    bool logging_enabled = false;

    // Простой разбор аргументов: [хост] [порт] [--log[=каталог]] в любом
    // порядке после первых двух позиционных.
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument.rfind("--log", 0) == 0) {
            logging_enabled = true;
            const std::size_t eq = argument.find('=');
            if (eq != std::string::npos) log_directory = argument.substr(eq + 1);
        } else {
            positional.push_back(argument);
        }
    }
    if (positional.size() > 0) config.host = positional[0];
    if (positional.size() > 1) config.port = static_cast<uint16_t>(std::stoi(positional[1]));
    if (log_directory.empty()) log_directory = "./vision-logs";

    magma::StopFlag stop;
    g_stop = &stop;
    std::signal(SIGINT, HandleSignal);
    std::signal(SIGTERM, HandleSignal);

    std::unique_ptr<magma::RotatingLogger> logger;
    std::unique_ptr<magma::LogCleaner> cleaner;
    std::thread cleanup_thread;

    if (logging_enabled) {
        std::error_code error;
        std::filesystem::create_directories(log_directory, error);
        logger = std::make_unique<magma::RotatingLogger>(log_directory, "vision",
                                                          1'000'000, 10, 5.0);
        magma::LogConfig cleanup_config;
        cleanup_config.directory = log_directory;
        cleanup_config.name_prefix = "vision";
        cleanup_config.retention_days = 30.0;
        cleaner = std::make_unique<magma::LogCleaner>(cleanup_config);
        std::printf("[ЗРЕНИЕ] Архив включён: %s\n", log_directory.c_str());

        cleanup_thread = std::thread([&] {
            while (!stop.IsSet()) {
                cleaner->Run(logger->Path(), false);
                stop.Wait(3600.0);
            }
        });
    } else {
        std::printf("[ЗРЕНИЕ] Архив выключен (запустите с --log, чтобы включить)\n");
    }

    magma::VisionConnection connection(stop, config, logger.get());
    std::thread worker([&connection] { connection.Run(); });

    while (!stop.IsSet()) stop.Wait(0.5);

    worker.join();
    if (cleanup_thread.joinable()) cleanup_thread.join();
    if (logger) logger->Close();

#ifdef _WIN32
    WSACleanup();
#endif
    std::printf("\n[STOP] Завершение...\n");
    return 0;
}
