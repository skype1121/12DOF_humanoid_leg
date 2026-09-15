#!/bin/bash
# PCAN(can1) 브링업 — 2026-07-27 모터 12/12 통신 성공 당시 설정 그대로.
#
# 배경: 젯슨의 사설 pcan 드라이버(9.1.0)가 어댑터를 가로채면 RX가 깨져
# 모터 무응답이 된다. 메인라인 peak_usb(~/peak_build/peak_usb.ko, 수동 빌드)로
# 잡아야 하며, 재부팅하면 pcan이 다시 자동 로드되므로 부팅 후 이 스크립트를
# 한 번 실행할 것. (영구화 = /lib/modules 설치 + blacklist 스왑은 승인 대기)
#
# 설정 근거: 노트북(juho)에서 12모터 검증된 값과 동일 —
#   bitrate 1M / restart-ms 100(버스오프 자동복구) / txqueuelen 1000(TX 폭주 방지)
set -u
sudo rmmod pcan 2>/dev/null && echo "[can_up] pcan 사설 드라이버 제거"
if ! lsmod | grep -q '^peak_usb'; then
    sudo insmod /home/mama/peak_build/peak_usb.ko && echo "[can_up] peak_usb 로드"
fi
sleep 1
# 0915: 재부팅 후 내장 mttcan 이 can1 을 선점하고 PCAN 이 can0 으로 뜬 사고(모터 12/12 무응답).
#       '이름'이 아니라 '드라이버(peak_usb)'로 PCAN 을 찾아 항상 can1 로 고정한다(내장 CAN 은 tegracan0 로 이동).
#       영구화는 /etc/systemd/network/10-pcan-can1.link · 10-tegra-mttcan.link (0915) — 여기는 이중 방어.
pcan=""
for c in /sys/class/net/*; do
    [ -e "$c/device/driver" ] || continue
    if [ "$(basename "$(readlink "$c/device/driver")")" = "peak_usb" ]; then pcan=$(basename "$c"); break; fi
done
if [ -z "$pcan" ]; then
    echo "[can_up] PCAN(peak_usb) 인터페이스 없음 — PCAN USB를 뽑았다 다시 꽂은 뒤 재실행하세요" >&2
    exit 1
fi
if [ "$pcan" != "can1" ]; then
    if [ -e /sys/class/net/can1 ]; then
        echo "[can_up] can1 이 PCAN 이 아님(driver=$(basename "$(readlink /sys/class/net/can1/device/driver 2>/dev/null)")) → tegracan0 로 이동"
        sudo ip link set can1 down 2>/dev/null
        sudo ip link set can1 name tegracan0
    fi
    echo "[can_up] PCAN 인터페이스 $pcan → can1 로 개명"
    sudo ip link set "$pcan" down 2>/dev/null
    sudo ip link set "$pcan" name can1
fi
sudo ip link set can1 down 2>/dev/null
sudo ip link set can1 type can bitrate 1000000 restart-ms 100
sudo ip link set can1 txqueuelen 1000
sudo ip link set can1 up
ip -details link show can1 | grep -E 'can state|bitrate'
echo "[can_up] 완료 — 프로브: python3 ~/rl_calib/zero_torque_read.py probe --id 1 --channel can1"
