#!/bin/bash
# Инициализация MV-MIPI-SC130M через media-ctl/V4L2 с повторами.
#
# Причина retry: I2C-пробинг сенсора после подачи питания завершается не
# мгновенно, и камера может физически стоять на CAM0 ИЛИ CAM1 (наблюдалось
# оба варианта на практике — см. PROJECT_STATE.md, инцидент 2026-09-15).
# Скрипт media_setting_rpi5.sh сам не повторяет попытку и не знает, на каком
# порту камера физически сейчас — поэтому пробуем оба порта (-c 2) несколько
# раз с паузой, пока не увидим по выводу, что хотя бы один найден.

SCRIPT=/home/mike/raspberrypi_v4l2/rpi5_scripts/media_setting_rpi5.sh
ARGS="mvcam -fmt RAW8 -w 1280 -h 1024 -c 2"
MAX_ATTEMPTS=15
DELAY_S=3

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    echo "[camera-init] попытка $attempt/$MAX_ATTEMPTS"
    output=$("$SCRIPT" $ARGS 2>&1)
    echo "$output"

    if echo "$output" | grep -q "set CAM. finish"; then
        echo "[camera-init] камера найдена и настроена"
        exit 0
    fi

    echo "[camera-init] камера не найдена ни на одном порту, жду ${DELAY_S}с"
    sleep "$DELAY_S"
done

echo "[camera-init] камера не найдена после $MAX_ATTEMPTS попыток — "
echo "[camera-init] magma-vision запустится и сам будет повторять открытие /dev/video0"
exit 0
