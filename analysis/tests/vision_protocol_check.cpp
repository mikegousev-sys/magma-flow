// Проверка совместимости вывода vision_series.py с C++ протоколом magma_vision.
//
// Читает NDJSON-строки со stdin и разбирает их тем же VisionLineParser, что использует
// magma_vision_client (code/magma_vision_protocol.hpp, файл не изменяется). Для каждой строки
// report сравнивает разобранные значения с числами, которые печатает Python рядом
// (формат строки проверки: "<ожидаемая скорость> <ожидаемый уровень> <валидность скорости>
// <валидность уровня>\t<строка JSON>"), и считает расхождения.
//
//   g++ -std=c++17 -O2 -I../../code vision_protocol_check.cpp -o vision_protocol_check
//   python3 ../vision_series.py ... --out out.ndjson && python3 check_lines.py out.ndjson | ./vision_protocol_check

#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>

#include "magma_vision_protocol.hpp"

int main() {
    magma::VisionLineParser parser;
    std::string line;
    std::size_t reports = 0, warnings = 0, failed = 0, mismatched = 0;
    while (std::getline(std::cin, line)) {
        const std::size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        std::istringstream expect(line.substr(0, tab));
        double speed = 0, level = 0;
        int speed_valid = 0, level_valid = 0;
        expect >> speed >> level >> speed_valid >> level_valid;
        const std::string json = line.substr(tab + 1);

        if (magma::VisionLineParser::IsMarker(json)) {
            ++warnings;
            continue;
        }
        const auto m = parser.Parse(json);
        if (!m) {
            ++failed;
            std::printf("НЕ РАЗОБРАНА: %.120s...\n", json.c_str());
            continue;
        }
        ++reports;
        const bool ok = m->speed_valid == static_cast<bool>(speed_valid) &&
                        m->level_valid == static_cast<bool>(level_valid) &&
                        (!speed_valid || std::fabs(m->speed_ms - speed) < 1e-3) &&
                        (!level_valid || std::fabs(m->level_mm - level) < 1e-2) &&
                        m->camera_frames > 0 && m->calibration_scale_mm_per_px > 0;
        if (!ok) {
            ++mismatched;
            if (mismatched <= 5) {
                std::printf("РАСХОЖДЕНИЕ: ожидалось v=%.4f(%d) h=%.2f(%d), разобрано v=%.4f(%d) h=%.2f(%d) "
                            "state=%s frames=%zu scale=%.4f\n",
                            speed, speed_valid, level, level_valid, m->speed_ms, m->speed_valid,
                            m->level_mm, m->level_valid, m->level_state.c_str(), m->camera_frames,
                            m->calibration_scale_mm_per_px);
            }
        }
    }
    std::printf("VisionLineParser: отчётов разобрано %zu, предупреждений %zu, не разобрано %zu, "
                "расхождений со значениями Python %zu\n",
                reports, warnings, failed, mismatched);
    return (failed || mismatched) ? 1 : 0;
}
