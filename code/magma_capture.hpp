// magma_capture.hpp — захват кадров с камеры MV-MIPI-SC130M через V4L2.
//
// Камера подключается к Raspberry Pi через адаптер ADP-MV1 и появляется в
// системе как обычное устройство /dev/videoN. Работа ведётся напрямую через
// V4L2 — это не требует ни libcamera, ни OpenCV, и сохраняет свойство проекта
// собираться одной командой компилятора.
//
// СЕРИЙНЫЙ ЗАХВАТ. Кадры берутся не постоянно, а сериями: примерно раз в
// десять секунд снимается около сотни кадров подряд, после чего камера
// простаивает. Такой режим выбран потому, что уровень достаточно измерять
// редко, а скорости для проверки нужна короткая непрерывная серия — из одной
// серии получаются обе величины, и камера захватывается один раз.
//
// КАДРЫ НЕ НАКАПЛИВАЮТСЯ. Серия из ста кадров разрешением 1280x1024 заняла бы
// 130 мегабайт. Вместо хранения каждый кадр отдаётся обработчику сразу по
// приходу, а после возврата из обработчика буфер возвращается драйверу.
// Обработчик извлекает то немногое, что нужно — профиль яркости или положение
// кромок, — и кадр забывается. Память при этом не растёт с длиной серии.

#ifndef MAGMA_CAPTURE_HPP
#define MAGMA_CAPTURE_HPP

#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <functional>
#include <string>
#include <vector>

#include "magma_edge.hpp"

namespace magma {

// =============================================================================
// Конфигурация
// =============================================================================

struct CaptureConfig {
    std::string device = "/dev/video0";
    std::size_t width = 1280;
    std::size_t height = 1024;

    /// Формат кадра. Камера монохромная, поэтому запрашивается восьмибитная
    /// градация серого: она приходит без всякого преобразования и не требует
    /// ни распаковки, ни демозаика.
    uint32_t pixel_format = V4L2_PIX_FMT_GREY;

    /// Управлять ли выдержкой средствами V4L2.
    ///
    /// Для камер VEYE это НЕ РАБОТАЕТ и по умолчанию отключено. Управление
    /// ведётся отдельно, через magma_veye.hpp, по двум причинам: камера
    /// настраивается регистрами по шине I2C, а не через ioctl видеоустройства,
    /// и вдобавок единица V4L2_CID_EXPOSURE_ABSOLUTE равна 100 микросекундам —
    /// при рабочих выдержках в десятки и сотни микросекунд значения короче
    /// 100 мкс обнулялись бы, а прочие округлялись до сотен.
    ///
    /// Флаг оставлен для камер, где стандартный путь всё же работает.
    bool set_exposure_via_v4l2 = false;
    int exposure_us = 200;
    int gain = 0;

    double frame_rate = 60.0;     ///< кадров в секунду в серии
    std::size_t buffer_count = 6; ///< буферов в очереди драйвера
    int timeout_ms = 2000;        ///< ожидание кадра
};

/// Что вернул захват серии.
struct CaptureStats {
    std::size_t requested = 0;
    std::size_t received = 0;
    std::size_t timeouts = 0;
    std::size_t errors = 0;
    double duration_s = 0.0;

    double ActualFrameRate() const {
        return duration_s > 0.0 ? static_cast<double>(received) / duration_s : 0.0;
    }
    bool Ok() const { return received > 0 && errors == 0; }
};

// =============================================================================
// Захват
// =============================================================================

/// Обработчик кадра. Вызывается по мере поступления; после возврата буфер
/// снова уходит драйверу, поэтому сохранять указатель нельзя — только
/// извлечь нужное.
using FrameHandler = std::function<void(const GrayImage&, std::size_t index)>;

/// Захват серий кадров через V4L2 с отображением буферов в память.
///
/// Отображение в память выбрано вместо чтения через read(): драйвер пишет
/// прямо в разделяемые буферы, и лишнего копирования кадра не происходит.
class Camera {
public:
    explicit Camera(CaptureConfig config = {}) : config_(std::move(config)) {}
    ~Camera() { Close(); }

    Camera(const Camera&) = delete;
    Camera& operator=(const Camera&) = delete;

    bool IsOpen() const { return fd_ >= 0; }
    const std::string& LastError() const { return error_; }

    /// Открыть устройство, задать формат и подготовить буферы.
    bool Open() {
        Close();
        fd_ = ::open(config_.device.c_str(), O_RDWR | O_NONBLOCK);
        if (fd_ < 0) return Fail("не открыть " + config_.device);

        v4l2_capability capability{};
        if (Ioctl(VIDIOC_QUERYCAP, &capability) < 0) return Fail("VIDIOC_QUERYCAP");
        if (!(capability.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
            return Fail("устройство не поддерживает захват видео");
        }
        if (!(capability.capabilities & V4L2_CAP_STREAMING)) {
            return Fail("устройство не поддерживает потоковый захват");
        }

        if (!SetFormat()) return false;
        if (config_.set_exposure_via_v4l2 && !SetExposure()) {
            // Не критично: часть драйверов не даёт управлять выдержкой через
            // стандартные органы. Сообщаем, но работу продолжаем.
            std::printf("[CAM] Предупреждение: выдержка через V4L2 не задана (%s)\n",
                        error_.c_str());
            error_.clear();
        }
        SetFrameRate();
        return AllocateBuffers();
    }

    void Close() {
        if (streaming_) StopStreaming();
        for (Buffer& buffer : buffers_) {
            if (buffer.start != nullptr && buffer.start != MAP_FAILED) {
                ::munmap(buffer.start, buffer.length);
            }
        }
        buffers_.clear();
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    }

    /// Снять серию кадров, передавая каждый обработчику.
    ///
    /// Первые кадры после запуска потока часто приходят с неустановившейся
    /// выдержкой, поэтому заданное число кадров отбрасывается в начале —
    /// иначе они внесли бы в измерение яркий выброс.
    CaptureStats CaptureBurst(std::size_t frame_count, const FrameHandler& handler,
                              std::size_t warmup_frames = 3,
                              const std::atomic<bool>* stop = nullptr) {
        CaptureStats stats;
        stats.requested = frame_count;
        if (!IsOpen()) { Fail("камера не открыта"); ++stats.errors; return stats; }
        if (!StartStreaming()) { ++stats.errors; return stats; }

        const double started = Now();
        std::size_t taken = 0, skipped = 0;

        // Серия занимает секунды (например, 2 с при 120 кадрах на 60 к/с), и
        // без проверки внешнего флага остановки процесс не реагировал бы на
        // Ctrl+C или системный stop до конца текущей серии. Параметр
        // необязателен и по умолчанию сохраняет прежнее поведение.
        while (taken < frame_count) {
            if (stop != nullptr && stop->load()) break;
            v4l2_buffer buffer{};
            buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buffer.memory = V4L2_MEMORY_MMAP;

            if (!WaitFrame()) { ++stats.timeouts; break; }

            if (Ioctl(VIDIOC_DQBUF, &buffer) < 0) {
                if (errno == EAGAIN) continue;
                Fail("VIDIOC_DQBUF");
                ++stats.errors;
                break;
            }

            if (skipped < warmup_frames) {
                ++skipped;
            } else {
                GrayImage image;
                image.data = static_cast<const uint8_t*>(buffers_[buffer.index].start);
                image.width = config_.width;
                image.height = config_.height;
                image.stride = stride_;
                handler(image, taken);
                ++taken;
                ++stats.received;
            }

            // Буфер возвращается драйверу сразу: обработчик уже забрал нужное.
            if (Ioctl(VIDIOC_QBUF, &buffer) < 0) {
                Fail("VIDIOC_QBUF");
                ++stats.errors;
                break;
            }
        }

        stats.duration_s = Now() - started;
        StopStreaming();
        return stats;
    }

private:
    struct Buffer {
        void* start = nullptr;
        std::size_t length = 0;
    };

    int Ioctl(unsigned long request, void* argument) const {
        int result;
        do { result = ::ioctl(fd_, request, argument); }
        while (result < 0 && errno == EINTR);   // повтор при прерывании сигналом
        return result;
    }

    bool Fail(const std::string& what) {
        error_ = what + (errno != 0 ? std::string(": ") + std::strerror(errno) : "");
        return false;
    }

    static double Now() {
        timespec ts{};
        ::clock_gettime(CLOCK_MONOTONIC, &ts);
        return static_cast<double>(ts.tv_sec) + ts.tv_nsec * 1e-9;
    }

    bool SetFormat() {
        v4l2_format format{};
        format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        format.fmt.pix.width = static_cast<uint32_t>(config_.width);
        format.fmt.pix.height = static_cast<uint32_t>(config_.height);
        format.fmt.pix.pixelformat = config_.pixel_format;
        format.fmt.pix.field = V4L2_FIELD_NONE;
        if (Ioctl(VIDIOC_S_FMT, &format) < 0) return Fail("VIDIOC_S_FMT");

        // Драйвер вправе выдать не то, что запрошено, — принимаем как есть,
        // иначе кадр будет разобран с неверной геометрией.
        config_.width = format.fmt.pix.width;
        config_.height = format.fmt.pix.height;
        stride_ = format.fmt.pix.bytesperline > 0
                      ? format.fmt.pix.bytesperline
                      : config_.width;
        if (format.fmt.pix.pixelformat != config_.pixel_format) {
            std::printf("[CAM] Предупреждение: драйвер выбрал другой формат кадра\n");
            config_.pixel_format = format.fmt.pix.pixelformat;
        }
        return true;
    }

    bool SetExposure() {
        v4l2_control control{};
        control.id = V4L2_CID_EXPOSURE_AUTO;
        control.value = V4L2_EXPOSURE_MANUAL;
        Ioctl(VIDIOC_S_CTRL, &control);   // может отсутствовать, не критично

        // Единица этого органа управления — 100 мкс, поэтому округление
        // ведётся к ближайшему, а не отбрасыванием: при выдержке 78 мкс
        // деление нацело дало бы ноль, то есть потерю значения целиком.
        control.id = V4L2_CID_EXPOSURE_ABSOLUTE;
        control.value = std::max(1, (config_.exposure_us + 50) / 100);
        if (Ioctl(VIDIOC_S_CTRL, &control) < 0) {
            control.id = V4L2_CID_EXPOSURE;          // запасной орган управления
            control.value = config_.exposure_us;
            if (Ioctl(VIDIOC_S_CTRL, &control) < 0) return Fail("выдержка");
        }

        if (config_.gain > 0) {
            control.id = V4L2_CID_GAIN;
            control.value = config_.gain;
            Ioctl(VIDIOC_S_CTRL, &control);
        }
        return true;
    }

    void SetFrameRate() {
        v4l2_streamparm parameters{};
        parameters.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (Ioctl(VIDIOC_G_PARM, &parameters) < 0) return;
        if (!(parameters.parm.capture.capability & V4L2_CAP_TIMEPERFRAME)) return;
        parameters.parm.capture.timeperframe.numerator = 1;
        parameters.parm.capture.timeperframe.denominator =
            static_cast<uint32_t>(config_.frame_rate);
        Ioctl(VIDIOC_S_PARM, &parameters);   // фактическую частоту проверяем по факту
    }

    bool AllocateBuffers() {
        v4l2_requestbuffers request{};
        request.count = static_cast<uint32_t>(config_.buffer_count);
        request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        request.memory = V4L2_MEMORY_MMAP;
        if (Ioctl(VIDIOC_REQBUFS, &request) < 0) return Fail("VIDIOC_REQBUFS");
        if (request.count < 2) return Fail("драйвер выделил меньше двух буферов");

        buffers_.resize(request.count);
        for (std::size_t i = 0; i < buffers_.size(); ++i) {
            v4l2_buffer buffer{};
            buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buffer.memory = V4L2_MEMORY_MMAP;
            buffer.index = static_cast<uint32_t>(i);
            if (Ioctl(VIDIOC_QUERYBUF, &buffer) < 0) return Fail("VIDIOC_QUERYBUF");

            buffers_[i].length = buffer.length;
            buffers_[i].start = ::mmap(nullptr, buffer.length, PROT_READ | PROT_WRITE,
                                       MAP_SHARED, fd_, buffer.m.offset);
            if (buffers_[i].start == MAP_FAILED) return Fail("mmap");
        }
        return true;
    }

    bool StartStreaming() {
        for (std::size_t i = 0; i < buffers_.size(); ++i) {
            v4l2_buffer buffer{};
            buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buffer.memory = V4L2_MEMORY_MMAP;
            buffer.index = static_cast<uint32_t>(i);
            if (Ioctl(VIDIOC_QBUF, &buffer) < 0) return Fail("VIDIOC_QBUF при старте");
        }
        int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (Ioctl(VIDIOC_STREAMON, &type) < 0) return Fail("VIDIOC_STREAMON");
        streaming_ = true;
        return true;
    }

    void StopStreaming() {
        int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        Ioctl(VIDIOC_STREAMOFF, &type);
        streaming_ = false;
    }

    /// Ожидание готового кадра. Устройство открыто в неблокирующем режиме,
    /// поэтому без ожидания цикл крутился бы вхолостую, занимая ядро.
    bool WaitFrame() const {
        fd_set set;
        FD_ZERO(&set);
        FD_SET(fd_, &set);
        timeval timeout{};
        timeout.tv_sec = config_.timeout_ms / 1000;
        timeout.tv_usec = (config_.timeout_ms % 1000) * 1000;
        int result;
        do { result = ::select(fd_ + 1, &set, nullptr, nullptr, &timeout); }
        while (result < 0 && errno == EINTR);
        return result > 0;
    }

    CaptureConfig config_;
    int fd_ = -1;
    bool streaming_ = false;
    std::size_t stride_ = 0;
    std::string error_;
    std::vector<Buffer> buffers_;
};

}  // namespace magma

#endif  // MAGMA_CAPTURE_HPP
