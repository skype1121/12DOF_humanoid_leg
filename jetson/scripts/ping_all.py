#!/usr/bin/env python3
"""12모터 CAN 응답 일괄 점검 (robotctl motors 백엔드).

0토크 MIT 폴링으로 모터를 절대 움직이지 않고 응답만 확인한다.
버스오프가 나면 무암호 sudo(can1 down/up)로 스스로 복구하며 계속 훑는다.
전원/커넥터 문제로 죽은 모터를 콕 집어준다 — 특히 버스를 끌어내리는 범인 격리용.
"""
import argparse
import subprocess
import sys
import time

sys.path.insert(0, "/home/mama/rl_calib")
import can  # noqa: E402
from ak_mit_command import pack_ak_mit_command  # noqa: E402
from ak_mit_decoder import decode_ak_mit_feedback  # noqa: E402

ENTER_MIT = bytes([0xFF] * 7 + [0xFC])
EXIT_MIT = bytes([0xFF] * 7 + [0xFD])
import json
JOINT = {}
try:
    _hw = json.load(open("/home/mama/rl_calib/robot_12dof_hardware_map.json"))
    for _j, _c in (_hw.get("joints") or _hw).items():
        if isinstance(_c, dict) and "motor_id" in _c:
            JOINT[int(_c["motor_id"])] = _c.get("ui_label") or _j
except Exception:
    pass


def bus_state(ch):
    try:
        out = subprocess.run(["ip", "-d", "link", "show", ch],
                             capture_output=True, text=True, timeout=3).stdout
        for tok in ("BUS-OFF", "ERROR-PASSIVE", "ERROR-WARNING", "ERROR-ACTIVE", "STOPPED"):
            if tok in out:
                return tok
    except Exception:
        pass
    return "?"


def recover(ch):
    """버스오프/정지 시 무암호 sudo 로 재기동 (sudoers 허용 범위)."""
    subprocess.run(["sudo", "-n", "ip", "link", "set", ch, "down"],
                   capture_output=True)
    subprocess.run(["sudo", "-n", "ip", "link", "set", ch, "type", "can",
                    "bitrate", "1000000", "restart-ms", "100"], capture_output=True)
    subprocess.run(["sudo", "-n", "ip", "link", "set", ch, "up"], capture_output=True)
    time.sleep(0.15)


def open_bus(ch):
    return can.Bus(channel=ch, interface="socketcan")


def probe(bus, ch, mid, timeout=0.2):
    """한 모터 응답 확인. (fb, note) 반환. 송신 실패=버스다운이면 note에 표시."""
    try:
        bus.send(can.Message(arbitration_id=mid, data=ENTER_MIT, is_extended_id=False))
        time.sleep(0.03)
        bus.send(can.Message(arbitration_id=mid,
                             data=pack_ak_mit_command(0.0, 0.0, 0.0, 0.0, 0.0),
                             is_extended_id=False))
    except can.CanError:
        return None, "BUSDOWN"
    t_end = time.time() + timeout
    fb = None
    while time.time() < t_end:
        msg = bus.recv(timeout=max(0.0, t_end - time.time()))
        if msg is None:
            break
        if len(msg.data) >= 6 and msg.data[0] == mid:
            fb = decode_ak_mit_feedback(bytes(msg.data))
            break
    try:
        bus.send(can.Message(arbitration_id=mid, data=EXIT_MIT, is_extended_id=False))
    except can.CanError:
        pass
    return fb, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="can1")
    ap.add_argument("--ids", default="1-12")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    ids = []
    for part in a.ids.split(","):
        if "-" in part:
            x, y = part.split("-"); ids += list(range(int(x), int(y) + 1))
        else:
            ids.append(int(part))

    st0 = bus_state(a.channel)
    if st0 in ("BUS-OFF", "STOPPED", "?"):
        recover(a.channel)
    try:
        bus = open_bus(a.channel)
    except OSError as e:
        print(f"✗ CAN {a.channel} 열기 실패: {e} — can_up 먼저", flush=True)
        return 3

    ok, dead, busoff_hits = [], [], 0
    rows = []
    for mid in ids:
        fb, note = probe(bus, a.channel, mid)
        if note == "BUSDOWN" or bus_state(a.channel) == "BUS-OFF":
            busoff_hits += 1
            try:
                bus.shutdown()
            except Exception:
                pass
            recover(a.channel)
            bus = open_bus(a.channel)
            # 복구 후 1회 재시도
            fb, note = probe(bus, a.channel, mid)
        label = JOINT.get(mid, "?")
        if fb is None:
            dead.append(mid)
            rows.append((mid, label, None))
        else:
            ok.append(mid)
            rows.append((mid, label, fb))

    if a.json:
        print(json.dumps({
            "ok": ok, "dead": dead, "busoff_recover": busoff_hits,
            "bus_state": bus_state(a.channel),
            "motors": {str(m): (None if fb is None else {
                "pos_deg": round(fb.position_deg, 2), "temp": fb.temperature_raw,
                "err": ("OK" if fb.error_ok else fb.error_raw)}) for m, _, fb in rows}},
            ensure_ascii=False))
    else:
        print(f"모터 CAN 응답 점검 ({a.channel})   버스: {bus_state(a.channel)}"
              + (f"   ⚠ 버스오프복구 {busoff_hits}회" if busoff_hits else ""))
        print("-" * 60)
        for m, label, fb in rows:
            if fb is None:
                print(f"  {m:>2} {label:12s}  ✗ 무응답")
            else:
                e = "OK" if fb.error_ok else f"ERR{fb.error_raw}"
                print(f"  {m:>2} {label:12s}  ✓ pos {fb.position_deg:+7.2f}°  "
                      f"온도 {fb.temperature_raw:>2}  {e}")
        print("-" * 60)
        print(f"응답 {len(ok)}/{len(ids)}"
              + (f"   ✗ 무응답: {dead}" if dead else "   ✓ 전원·통신 정상"))
    try:
        bus.shutdown()
    except Exception:
        pass
    return 0 if not dead else 1


if __name__ == "__main__":
    raise SystemExit(main())
