"""가짜 AK 모터 12개 (vcan 전용). 명령 프레임 위치를 그대로 응답 — 실물 버스(can1)는 쓰지 않는다."""
import can, json, sys, time
CH = sys.argv[1]; OUT = sys.argv[2]; DUR = float(sys.argv[3])
assert CH.startswith("can") and CH != "can1" and CH != "can0", "실물 채널 금지"
INIT = {1: 35715, 2: 32968, 3: 33088, 4: 35231, 5: 35943, 6: 32923, 7: 31611, 8: 33159, 9: 34246, 10: 31698, 11: 32405, 12: 33180}
pos = dict(INIT)
bus = can.Bus(interface="socketcan", channel=CH)
log = []
t_end = time.time() + DUR
SPECIAL = {bytes([0xFF] * 7 + [x]) for x in (0xFC, 0xFD, 0xFE)}
while time.time() < t_end:
    m = bus.recv(timeout=0.1)
    if m is None: continue
    mid = m.arbitration_id; d = bytes(m.data)
    if not (1 <= mid <= 12) or len(d) != 8: continue
    if d[0] == mid and d[7] < 0x10 and d not in SPECIAL:
        continue                      # (안전) 다른 응답 프레임은 무시
    if d not in SPECIAL:
        pos[mid] = (d[0] << 8) | d[1]
        log.append((m.timestamp, mid, pos[mid]))
    p = pos[mid]
    bus.send(can.Message(arbitration_id=mid, is_extended_id=False,
                         data=[mid, p >> 8, p & 0xFF, 0x7F, 0xF7, 0xFF, 30, 0]))
json.dump(log, open(OUT, "w"))
bus.shutdown()
