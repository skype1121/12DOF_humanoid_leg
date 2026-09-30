"""stage8 사본 4종: v0 원본 / v1 명령 구독 깊이 1 / v2 상태 타이머 50 Hz + 명령마다 상태발행 제거 / v3 = v1+v2"""
import sys, os
src = open(sys.argv[1]).read(); outdir = sys.argv[2]
SUB = """            COMMAND_TOPIC,
            self._on_command_message,
            10,"""
assert src.count(SUB) == 1
PUB = """        now = time.time()
        if now - self._last_cmd_status_pub >= 0.005:"""
assert src.count(PUB) == 1
TMR = "STATUS_PERIOD_SEC = 0.01"
assert src.count(TMR) == 1
def v1(s): return s.replace(SUB, SUB.replace("10,", "1,"))
def v2(s): return s.replace(TMR, "STATUS_PERIOD_SEC = 0.02").replace(PUB, """        now = time.time()
        if message.data.find('"SET_JOINT_TARGETS"') < 0 and now - self._last_cmd_status_pub >= 0.005:""")
for name, s in (("v0", src), ("v1", v1(src)), ("v2", v2(src)), ("v3", v2(v1(src)))):
    open(os.path.join(outdir, f"stage8_bench_{name}.py"), "w").write(s)
print("ok")
