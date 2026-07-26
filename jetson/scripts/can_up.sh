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
if [ ! -e /sys/class/net/can1 ]; then
    echo "[can_up] can1 없음 — PCAN USB를 뽑았다 다시 꽂은 뒤 재실행하세요" >&2
    exit 1
fi
sudo ip link set can1 down 2>/dev/null
sudo ip link set can1 type can bitrate 1000000 restart-ms 100
sudo ip link set can1 txqueuelen 1000
sudo ip link set can1 up
ip -details link show can1 | grep -E 'can state|bitrate'
echo "[can_up] 완료 — 프로브: python3 ~/rl_calib/zero_torque_read.py probe --id 1 --channel can1"
