"""stage8 코어의 상태 직렬화·명령 처리 비용 측정 (ROS·CAN 없이 순수 파이썬). 젯슨/PC 공용."""
import sys, time, json, os
sys.path.insert(0, os.path.expanduser(sys.argv[1]))
import stage8_12axis_mit_control_node as s8
core = s8.Stage8TwelveAxisMitControlCore()
# 무장 상태와 비슷하게 관절별 필드를 채운다 (피드백·베이스라인·목표)
now = time.time()
for j in s8.JOINT_NAMES_12DOF:
    for attr in ("baseline_deg_by_joint", "desired_relative_deg_by_joint", "commanded_relative_deg_by_joint"):
        d = getattr(core, attr, None)
        if isinstance(d, dict): d[j] = 1.234
def bench(fn, n=300):
    fn()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter() - t0) / n * 1000
st = core.get_status()
js = json.dumps(st, ensure_ascii=False, sort_keys=True)
print(f"상태 JSON 크기 {len(js)} B, 키 {len(st)}")
a = bench(core.get_status)
b = bench(lambda: json.dumps(core.get_status(), ensure_ascii=False, sort_keys=True))
cmd = json.dumps({"command": "SET_JOINT_TARGETS", "targets": {j: 1.0 for j in s8.JOINT_NAMES_12DOF}})
c = bench(lambda: core.handle_command_data(cmd))
print(f"get_status {a:.3f} ms | get_status+dumps {b:.3f} ms | handle_command(SET_JOINT_TARGETS, 비무장) {c:.3f} ms")
print(f"초당 부하 추정: 상태타이머 100Hz = {100*b:.0f} ms/s, 명령마다 상태발행 ~50Hz = {50*b:.0f} ms/s, 명령처리 50Hz = {50*c:.0f} ms/s")
