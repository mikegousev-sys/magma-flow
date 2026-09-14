// magma_broadcast.hpp — приём TCP-подключений и рассылка текстовых строк.
//
// В magma_server.cpp (защищённое ядро) уже есть класс с точно такой же
// задачей — ClientBroadcaster. Здесь та же логика вынесена в отдельный,
// самостоятельный заголовок, чтобы ею мог пользоваться процесс зрения, не
// заглядывая в защищённые файлы ядра и не завися от них. Дублирование
// небольшого класса — сознательная цена за то, что ядро и дополнение
// эволюционируют независимо: правка формата вещания зрения никогда не
// потребует трогать сервер, и наоборот.

#ifndef MAGMA_BROADCAST_HPP
#define MAGMA_BROADCAST_HPP

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace magma {

struct BroadcastConfig {
    std::string host = "0.0.0.0";
    uint16_t port = 9100;
    double accept_timeout_s = 1.0;
    int backlog = 5;
};

/// Принимает подключения и рассылает строки всем подключённым клиентам.
///
/// Флаг остановки передаётся ссылкой на атомарную переменную, а не через
/// отдельный тип StopFlag (как в ядре): это избавляет заголовок от
/// собственного примитива синхронизации и позволяет использовать любой
/// существующий в вызывающем коде атомарный флаг без переходников.
class LineBroadcaster {
public:
    LineBroadcaster(const std::atomic<bool>& stop, BroadcastConfig config)
        : stop_(stop), config_(std::move(config)) {}
    ~LineBroadcaster() { CloseAll(); }

    LineBroadcaster(const LineBroadcaster&) = delete;
    LineBroadcaster& operator=(const LineBroadcaster&) = delete;

    /// Цикл приёма подключений — предназначен для отдельного потока.
    void ServeForever() {
        const int server = ::socket(AF_INET, SOCK_STREAM, 0);
        if (server < 0) {
            std::printf("[BROADCAST] Не создать сокет: %s\n", std::strerror(errno));
            return;
        }
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
            std::printf("[BROADCAST] bind/listen на %s:%u: %s\n",
                        config_.host.c_str(), config_.port, std::strerror(errno));
            ::close(server);
            return;
        }
        std::printf("[BROADCAST] Слушаю %s:%u\n", config_.host.c_str(), config_.port);

        while (!stop_.load()) {
            sockaddr_in peer{};
            socklen_t size = sizeof(peer);
            const int client = ::accept(server, reinterpret_cast<sockaddr*>(&peer), &size);
            if (client < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) continue;
                break;
            }
            std::printf("[BROADCAST] Клиент подключён: %s\n", ::inet_ntoa(peer.sin_addr));
            std::lock_guard<std::mutex> lock(mutex_);
            clients_.push_back(client);
        }
        ::close(server);
    }

    /// Отправить строку всем клиентам, отцепив отвалившихся.
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

    std::size_t ClientCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return clients_.size();
    }

    void CloseAll() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (int client : clients_) ::close(client);
        clients_.clear();
    }

private:
    const std::atomic<bool>& stop_;
    BroadcastConfig config_;
    mutable std::mutex mutex_;
    std::vector<int> clients_;
};

}  // namespace magma

#endif  // MAGMA_BROADCAST_HPP
