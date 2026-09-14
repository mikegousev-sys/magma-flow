// magma_server.cpp — скоростемер потока никелевого расплава (Raspberry Pi 5).
//
// Принимает кадры с Arduino (4 фотодиода) по UART, оценивает скорость потока
// согласованно по всем шести базам между диодами и раз в секунду отдаёт
// сводку подключённым клиентам по TCP.
//
// НА ДИСК НИЧЕГО НЕ ПИШЕТСЯ. Прежняя версия писала около 17 ГБ в сутки на
// карту памяти и не удаляла старые файлы. Вместо этого в каждую секундную
// сводку включены счётчики принятых, потерянных и отброшенных по контрольной
// сумме кадров: без записи это единственный способ заметить деградацию канала.
//
// Внутренний темп расчёта (около 20 Гц) отделён от темпа выдачи (1 Гц):
// окно корреляции длиной 500 отсчётов сдвигается за такт лишь на доли
// процента, поэтому чаще считать бессмысленно, а наружу уходит медиана за
// секунду — она устойчива к единичным выбросам.
//
// Разделы: 1) конфигурация, 2) синхронизация, 3) сбор данных, 4) агрегация,
// 5) сеть, 6) приложение.
//
// Сборка: g++ -std=c++17 -O2 -pthread magma_server.cpp -o magma_server

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <termios.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <deque>
#include <mutex>
#include <numeric>
#include <thread>
#include <vector>

#include "magma_protocol.hpp"
#include "magma_signal.hpp"

namespace magma {

// =============================================================================
// 1. КОНФИГУРАЦИЯ
// =============================================================================

struct SerialConfig {
    std::string port = "/dev/ttyUSB0";  // через CH340G; GPIO-UART = /dev/ttyAMA0
    // 500000 бод делится нацело и от 16 МГц у ATmega328P, и от 12 МГц у CH340G,
    // поэтому погрешности нет ни на одной стороне канала. Загрузка линии 24%.
    speed_t baudrate = B500000;
    std::size_t num_channels = 4;
    double sample_rate_nominal = 1000.0;
    bool use_frame_sync = true;         // false — старая прошивка без маркера
    std::size_t read_batch_frames = 32;
    double stale_data_timeout_s = 2.0;
    // CH340G дёргает линию DTR при открытии порта, и Arduino от этого
    // перезагружается. Загрузчик держит управление около секунды, прежде чем
    // запустить скетч, поэтому сразу после открытия данных не будет. Ждём и
    // очищаем приёмный буфер от мусора, который загрузчик мог оставить.
    double reset_delay_s = 2.0;

    std::size_t PayloadSize() const { return num_channels * 2; }
    /// Кадр: маркер(2) + счётчик(1) + данные(8) + контрольная сумма(1).
    std::size_t FrameSize() const { return 2 + 1 + PayloadSize() + 1; }
};

struct OutputConfig {
    double report_interval_s = 1.0;   // как часто уходит сводка клиентам
    double compute_interval_s = 0.05; // как часто пересчитывается скорость
};

struct NetworkConfig {
    std::string host = "0.0.0.0";
    uint16_t port = 9000;
    double accept_timeout_s = 1.0;
    int backlog = 5;
};

constexpr unsigned char kFrameSync[2] = {0xAA, 0x55};

// =============================================================================
// 2. СИНХРОНИЗАЦИЯ
// =============================================================================

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

inline double MonotonicSeconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// =============================================================================
// 3. СБОР ДАННЫХ
// =============================================================================

/// Счётчики состояния канала — заменяют собой диагностику по логам.
struct LinkStats {
    std::atomic<unsigned long long> frames{0};
    std::atomic<unsigned long long> lost{0};
    std::atomic<unsigned long long> crc{0};
    /// Кадры, при которых Arduino сообщил о пропуске такта или переполнении
    /// буфера передачи, — признак того, что плата не успевает.
    std::atomic<unsigned long long> overruns{0};
};

/// Потокобезопасные кольцевые буферы каналов и учёт частоты кадров.
///
/// Пишет поток чтения UART, читает главный цикл, поэтому снимок всех каналов
/// берётся под общим локом: иначе каналы оказались бы сдвинуты во времени
/// друг относительно друга, что для корреляционного метода недопустимо.
class ChannelBuffers {
public:
    ChannelBuffers(std::size_t channels, std::size_t window, double nominal_rate)
        : buffers_(channels), window_(window), nominal_rate_(nominal_rate) {}

    void PushFrame(const std::vector<uint16_t>& values) {
        const double now = MonotonicSeconds();
        std::lock_guard<std::mutex> lock(mutex_);
        for (std::size_t i = 0; i < buffers_.size() && i < values.size(); ++i) {
            buffers_[i].push_back(static_cast<double>(values[i]));
            if (buffers_[i].size() > window_) buffers_[i].pop_front();
        }
        timestamps_.push_back(now);
        if (timestamps_.size() > 2000) timestamps_.pop_front();
        last_frame_at_ = now;
    }

    std::vector<std::vector<double>> Snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::vector<double>> result(buffers_.size());
        for (std::size_t i = 0; i < buffers_.size(); ++i) {
            result[i].assign(buffers_[i].begin(), buffers_[i].end());
        }
        return result;
    }

    /// Фактическая частота кадров; при нехватке данных — номинальная.
    /// Измеряется, а не берётся из константы: прежняя версия делила лаг на
    /// зашитые 1000 Гц, не проверяя, столько ли кадров реально приходит.
    double EffectiveSampleRate() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (timestamps_.size() < 20) return nominal_rate_;
        const double span = timestamps_.back() - timestamps_.front();
        return span > 0.0 ? static_cast<double>(timestamps_.size() - 1) / span
                          : nominal_rate_;
    }

    std::optional<double> SecondsSinceLastFrame() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (last_frame_at_ < 0.0) return std::nullopt;
        return MonotonicSeconds() - last_frame_at_;
    }

private:
    mutable std::mutex mutex_;
    std::vector<std::deque<double>> buffers_;
    std::deque<double> timestamps_;
    std::size_t window_;
    double nominal_rate_;
    double last_frame_at_ = -1.0;
};

/// Читает кадры Arduino в фоне и складывает их в буферы каналов.
///
/// Принятые байты накапливаются в буфере и никогда не отбрасываются частями.
/// Прежняя версия при неполном чтении делала continue и теряла принятое —
/// один такой случай сдвигал разбор всех последующих кадров навсегда.
class UartReader {
public:
    UartReader(ChannelBuffers& buffers, LinkStats& stats, StopFlag& stop,
               SerialConfig config)
        : buffers_(buffers), stats_(stats), stop_(stop), config_(std::move(config)) {}

    void Run() {
        while (!stop_.IsSet()) {
            const int port = OpenPort();
            if (port < 0) { stop_.Wait(2.0); continue; }

            const std::size_t unit =
                config_.use_frame_sync ? config_.FrameSize() : config_.PayloadSize();
            std::vector<unsigned char> chunk(unit * config_.read_batch_frames);
            std::vector<unsigned char> buffer;

            while (!stop_.IsSet()) {
                const ssize_t received = ::read(port, chunk.data(), chunk.size());
                if (received < 0) {
                    if (errno == EAGAIN || errno == EINTR) continue;
                    std::printf("[UART] Обрыв: %s. Переподключение...\n",
                                std::strerror(errno));
                    break;
                }
                if (received == 0) continue;
                buffer.insert(buffer.end(), chunk.begin(), chunk.begin() + received);
                buffer = config_.use_frame_sync ? ConsumeSynced(std::move(buffer))
                                                : ConsumeRaw(std::move(buffer));
            }
            ::close(port);
        }
    }

private:
    static unsigned char Checksum(const unsigned char* data, std::size_t size) {
        unsigned char checksum = 0;
        for (std::size_t i = 0; i < size; ++i) checksum ^= data[i];
        return checksum;
    }

    int OpenPort() {
        const int port = ::open(config_.port.c_str(), O_RDWR | O_NOCTTY);
        if (port < 0) {
            std::printf("[UART] Не открыть %s: %s. Повтор через 2 с\n",
                        config_.port.c_str(), std::strerror(errno));
            return -1;
        }
        termios options{};
        if (::tcgetattr(port, &options) != 0) { ::close(port); return -1; }
        ::cfmakeraw(&options);  // данные бинарные, обработка символов недопустима
        ::cfsetispeed(&options, config_.baudrate);
        ::cfsetospeed(&options, config_.baudrate);
        options.c_cflag |= (CLOCAL | CREAD);
        options.c_cflag &= ~CRTSCTS;
        options.c_cc[VMIN] = 0;
        options.c_cc[VTIME] = 10;
        if (::tcsetattr(port, TCSANOW, &options) != 0) { ::close(port); return -1; }
        ::tcflush(port, TCIFLUSH);

        // Пауза на перезагрузку Arduino, вызванную открытием порта.
        std::printf("[UART] Открыт %s, жду перезапуска платы (%.0f с)...\n",
                    config_.port.c_str(), config_.reset_delay_s);
        std::this_thread::sleep_for(
            std::chrono::duration<double>(config_.reset_delay_s));
        ::tcflush(port, TCIFLUSH);
        std::printf("[UART] Готов к приёму\n");
        return port;
    }

    /// Разобрать кадры с маркером и контрольной суммой; вернуть остаток.
    std::vector<unsigned char> ConsumeSynced(std::vector<unsigned char> buffer) {
        const std::size_t frame = config_.FrameSize();
        const std::size_t payload = config_.PayloadSize();
        std::size_t offset = 0;

        while (buffer.size() - offset >= frame) {
            const auto begin = buffer.begin() + static_cast<long>(offset);
            const auto found = std::search(begin, buffer.end(), kFrameSync, kFrameSync + 2);
            if (found == buffer.end()) {
                // маркер мог быть разорван границей чтения — сохраняем хвост
                return {buffer.end() - std::min<long>(1, buffer.size()), buffer.end()};
            }
            offset = static_cast<std::size_t>(found - buffer.begin());
            if (buffer.size() - offset < frame) break;

            const unsigned char* checked = buffer.data() + offset + 2;
            // Бит 7 — признак сбоя такта на Arduino, биты 0..6 — счётчик.
            const unsigned char raw_sequence = checked[0];
            const unsigned char sequence = raw_sequence & 0x7F;
            if (raw_sequence & 0x80) ++stats_.overruns;
            const unsigned char* payload_ptr = checked + 1;
            const bool valid = Checksum(checked, 1 + payload) == buffer[offset + 3 + payload];
            offset += frame;
            if (!valid) { ++stats_.crc; continue; }

            // Счётчик семибитный; разрыв означает потерю кадров между Arduino
            // и Pi — без логов это единственный способ о них узнать.
            if (has_sequence_) {
                const unsigned char expected = (last_sequence_ + 1) & 0x7F;
                if (sequence != expected) stats_.lost += (sequence - expected) & 0x7F;
            }
            last_sequence_ = sequence;
            has_sequence_ = true;
            ++stats_.frames;

            std::vector<uint16_t> values(config_.num_channels);
            std::memcpy(values.data(), payload_ptr, payload);  // little-endian
            buffers_.PushFrame(values);
        }
        return {buffer.begin() + static_cast<long>(offset), buffer.end()};
    }

    /// Разобрать кадры без маркера (режим совместимости со старой прошивкой).
    std::vector<unsigned char> ConsumeRaw(std::vector<unsigned char> buffer) {
        const std::size_t payload = config_.PayloadSize();
        std::size_t offset = 0;
        while (buffer.size() - offset >= payload) {
            std::vector<uint16_t> values(config_.num_channels);
            std::memcpy(values.data(), buffer.data() + offset, payload);
            buffers_.PushFrame(values);
            ++stats_.frames;
            offset += payload;
        }
        return {buffer.begin() + static_cast<long>(offset), buffer.end()};
    }

    ChannelBuffers& buffers_;
    LinkStats& stats_;
    StopFlag& stop_;
    SerialConfig config_;
    unsigned char last_sequence_ = 0;
    bool has_sequence_ = false;
};

// =============================================================================
// 4. АГРЕГАЦИЯ
// =============================================================================

/// Копит оценки за секунду и сводит их в одну строку.
///
/// Наружу отдаётся медиана: одна ошибочная оценка из двадцати не сдвинет её,
/// тогда как среднее она бы заметно исказила. Среднее тоже передаётся —
/// расхождение среднего и медианы само по себе указывает на выбросы.
class SecondAggregator {
public:
    void Add(const SpeedResult& result, const std::vector<std::vector<double>>& snapshot) {
        ++attempted_;
        if (result.speed_ms) {
            speeds_.push_back(*result.speed_ms);
            snr_sum_ += result.snr;
            baselines_ = result.baselines_used;
        }
        // Счётчик увеличивается один раз за такт, а не по разу на канал:
        // при пустом канале деление на «пары канал-такт» давало бы неверное
        // среднее по остальным каналам.
        bool any = false;
        for (std::size_t i = 0; i < adc_sum_.size() && i < snapshot.size(); ++i) {
            if (snapshot[i].empty()) continue;
            adc_sum_[i] += snapshot[i].back();
            any = true;
        }
        if (any) ++adc_count_;
    }

    /// Сформировать сводку и начать новую секунду.
    Report Flush(const LinkStats& stats) {
        Report report;
        report.attempted = attempted_;
        report.accepted = speeds_.size();
        report.baselines = baselines_;

        if (!speeds_.empty()) {
            report.speed_ms = std::accumulate(speeds_.begin(), speeds_.end(), 0.0) /
                              static_cast<double>(speeds_.size());
            std::vector<double> sorted = speeds_;
            std::sort(sorted.begin(), sorted.end());
            const std::size_t middle = sorted.size() / 2;
            report.speed_median_ms = sorted.size() % 2 == 1
                ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2.0;
            report.snr = snr_sum_ / static_cast<double>(speeds_.size());
            report.is_valid = true;
        }

        const double samples = adc_count_ > 0 ? static_cast<double>(adc_count_) : 1.0;
        for (std::size_t i = 0; i < adc_sum_.size(); ++i) report.adc[i] = adc_sum_[i] / samples;

        // Счётчики канала — разность с прошлой секундой, а не с запуска.
        const unsigned long long frames = stats.frames.load();
        const unsigned long long lost = stats.lost.load();
        const unsigned long long crc = stats.crc.load();
        const unsigned long long overruns = stats.overruns.load();
        report.frames = frames - last_frames_;
        report.lost = lost - last_lost_;
        report.crc = crc - last_crc_;
        report.overruns = overruns - last_overruns_;
        last_frames_ = frames;
        last_lost_ = lost;
        last_crc_ = crc;
        last_overruns_ = overruns;

        speeds_.clear();
        attempted_ = 0;
        snr_sum_ = 0.0;
        adc_sum_.fill(0.0);
        adc_count_ = 0;
        return report;
    }

private:
    std::vector<double> speeds_;
    std::size_t attempted_ = 0, baselines_ = 0, adc_count_ = 0;
    double snr_sum_ = 0.0;
    std::array<double, 4> adc_sum_{};
    unsigned long long last_frames_ = 0, last_lost_ = 0, last_crc_ = 0, last_overruns_ = 0;
};

// =============================================================================
// 5. СЕТЬ
// =============================================================================

/// Принимает TCP-клиентов и рассылает им секундные сводки.
class ClientBroadcaster {
public:
    ClientBroadcaster(StopFlag& stop, NetworkConfig config)
        : stop_(stop), config_(std::move(config)) {}
    ~ClientBroadcaster() { CloseAll(); }

    void ServeForever() {
        const int server = ::socket(AF_INET, SOCK_STREAM, 0);
        if (server < 0) return;
        int reuse = 1;
        ::setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
        timeval timeout{static_cast<time_t>(config_.accept_timeout_s), 0};
        ::setsockopt(server, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(config_.port);
        address.sin_addr.s_addr = ::inet_addr(config_.host.c_str());
        if (::bind(server, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0 ||
            ::listen(server, config_.backlog) < 0) {
            std::printf("[SOCKET] bind/listen: %s\n", std::strerror(errno));
            ::close(server);
            return;
        }
        std::printf("[SOCKET] Сервер на %s:%u\n", config_.host.c_str(), config_.port);

        while (!stop_.IsSet()) {
            sockaddr_in peer{};
            socklen_t size = sizeof(peer);
            const int client = ::accept(server, reinterpret_cast<sockaddr*>(&peer), &size);
            if (client < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) continue;
                break;
            }
            std::printf("[SOCKET] Клиент подключён: %s\n", ::inet_ntoa(peer.sin_addr));
            std::lock_guard<std::mutex> lock(mutex_);
            clients_.push_back(client);
        }
        ::close(server);
    }

    void Broadcast(const std::string& line) {
        const std::string payload = line + "\n";
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = clients_.begin(); it != clients_.end();) {
            if (::send(*it, payload.data(), payload.size(), MSG_NOSIGNAL) < 0) {
                ::close(*it);
                it = clients_.erase(it);
            } else {
                ++it;
            }
        }
    }

    void CloseAll() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (int client : clients_) ::close(client);
        clients_.clear();
    }

private:
    StopFlag& stop_;
    NetworkConfig config_;
    std::mutex mutex_;
    std::vector<int> clients_;
};

// =============================================================================
// 6. ПРИЛОЖЕНИЕ
// =============================================================================

/// Связывает сбор данных, расчёт скорости и раздачу сводок клиентам.
class MagmaServer {
public:
    MagmaServer()
        : buffers_(serial_.num_channels, detection_.window_size,
                   serial_.sample_rate_nominal),
          estimator_(detection_, geometry_),
          broadcaster_(stop_, network_) {}

    void Run() {
        std::printf("[РАСЧЁТ] Скорость по %zu базам, диапазон %.1f..%.1f м/с\n",
                    geometry_.Baselines().size(), detection_.min_speed_ms,
                    detection_.max_speed_ms);
        std::printf("[ВЫДАЧА] Сводка раз в %.0f с, внутренний темп расчёта %.0f Гц\n",
                    output_.report_interval_s, 1.0 / output_.compute_interval_s);
        std::printf("[ДИСК] Запись логов отключена\n");

        UartReader reader(buffers_, stats_, stop_, serial_);
        std::thread accept_thread([this] { broadcaster_.ServeForever(); });
        std::thread uart_thread([&reader] { reader.Run(); });
        std::this_thread::sleep_for(std::chrono::seconds(1));  // буферы наполняются

        double compute_at = MonotonicSeconds();
        double report_at = compute_at + output_.report_interval_s;

        while (!stop_.IsSet()) {
            stop_.Wait(output_.compute_interval_s);
            if (stop_.IsSet()) break;

            const double now = MonotonicSeconds();
            if (now >= compute_at && !DataIsStale()) {
                // Планируем от РАСЧЁТНОГО времени, а не от фактического: иначе
                // задержка каждого такта прибавлялась бы к следующему, и темп
                // уходил бы примерно на 2.5% — около 36 минут «недостающих»
                // сводок за сутки.
                compute_at += output_.compute_interval_s;
                if (compute_at < now) compute_at = now + output_.compute_interval_s;
                const auto snapshot = buffers_.Snapshot();
                aggregator_.Add(estimator_.Estimate(snapshot, buffers_.EffectiveSampleRate()),
                                snapshot);
            }

            if (now >= report_at) {
                report_at += output_.report_interval_s;
                if (report_at < now) report_at = now + output_.report_interval_s;
                const Report report = aggregator_.Flush(stats_);
                broadcaster_.Broadcast(FormatLine(report));
                if (report.overruns > 0) {
                    broadcaster_.Broadcast(FormatMarker(
                        "ВНИМАНИЕ: Arduino не успевает, часть тактов пропущена"));
                }
                if (report.lost > 0 || report.crc > 0) {
                    broadcaster_.Broadcast(FormatMarker(
                        "ВНИМАНИЕ: канал теряет кадры, проверьте соединение с Arduino"));
                }
            }
        }

        stop_.Set();
        broadcaster_.CloseAll();
        accept_thread.join();
        uart_thread.join();
    }

    void RequestStop() { stop_.Set(); }

private:
    /// Проверить, не прекратился ли поток кадров с Arduino.
    bool DataIsStale() {
        const auto idle = buffers_.SecondsSinceLastFrame();
        if (idle && *idle > serial_.stale_data_timeout_s) {
            if (!warned_stale_) {
                std::printf("[WARN] Нет данных с UART больше %.0f с\n",
                            serial_.stale_data_timeout_s);
                broadcaster_.Broadcast(FormatMarker("ВНИМАНИЕ: нет данных с Arduino"));
                warned_stale_ = true;
            }
            return true;
        }
        warned_stale_ = false;
        return false;
    }

    SerialConfig serial_;
    DetectionConfig detection_;
    SensorGeometry geometry_;
    OutputConfig output_;
    NetworkConfig network_;

    StopFlag stop_;
    LinkStats stats_;
    ChannelBuffers buffers_;
    SpeedEstimator estimator_;
    SecondAggregator aggregator_;
    ClientBroadcaster broadcaster_;
    bool warned_stale_ = false;
};

}  // namespace magma

namespace {
magma::MagmaServer* g_server = nullptr;
void HandleSignal(int) { if (g_server) g_server->RequestStop(); }
}  // namespace

int main() {
    magma::MagmaServer server;
    g_server = &server;
    std::signal(SIGINT, HandleSignal);
    std::signal(SIGTERM, HandleSignal);
    server.Run();
    std::printf("\n[STOP] Завершение...\n");
    return 0;
}
