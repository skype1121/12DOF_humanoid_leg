"""브리지 블랙박스 비동기 기록 검증 (0929) — 디스크가 막혀도 제어 틱(_bb_write)은 막히지 않아야 한다."""
import ast, json, os, tempfile, textwrap, threading, time, types

SRC = open("/home/ryu/humanoid_leg_test1/jetson/scripts/rl_bridge_node.py").read()
tree = ast.parse(SRC)
fns = {}
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name in ("_bb_write", "_bb_writer", "_bb_close"):
        fns[node.name] = textwrap.dedent(ast.get_source_segment(SRC, node))
assert set(fns) == {"_bb_write", "_bb_writer", "_bb_close"}, fns.keys()


class SlowFile:
    """write 마다 delay 초 막히는 가짜 파일 (디스크 정체 모사)."""
    def __init__(self, path, delay):
        self.f = open(path, "w"); self.delay = delay
    def write(self, s):
        time.sleep(self.delay); return self.f.write(s)
    def flush(self): self.f.flush()
    def close(self): self.f.close()


def make(delay, qmax=20000):
    import queue
    o = types.SimpleNamespace()
    ns = {"json": json}
    for name, code in fns.items():
        exec(code, ns); setattr(o, name, types.MethodType(ns[name], o))
    tmp = tempfile.mktemp(suffix=".jsonl")
    o.bb = SlowFile(tmp, delay); o._bbq = queue.Queue(maxsize=qmax); o._bb_dropped = 0
    o.get_logger = lambda: types.SimpleNamespace(error=print)
    o._bbt = threading.Thread(target=o._bb_writer, daemon=True); o._bbt.start()
    return o, tmp


def test_tick_not_blocked_by_slow_disk():
    o, tmp = make(delay=0.2)                 # 쓰기 한 번에 200 ms 막히는 디스크
    t0 = time.perf_counter(); worst = 0.0
    for i in range(50):                      # 50 Hz 1 초 분량
        a = time.perf_counter(); o._bb_write({"i": i}); worst = max(worst, time.perf_counter() - a)
    assert worst < 0.005, f"제어 틱이 {worst*1000:.1f} ms 막혔다"
    o._bb_close()


def test_all_lines_flushed_on_close():
    o, tmp = make(delay=0.0)
    for i in range(300): o._bb_write({"i": i})
    o._bb_close()
    lines = [json.loads(l) for l in open(tmp)]
    assert [l["i"] for l in lines] == list(range(300)), len(lines)


def test_queue_full_drops_and_counts_not_blocks():
    o, tmp = make(delay=0.5, qmax=5)
    a = time.perf_counter()
    for i in range(50): o._bb_write({"i": i})
    assert time.perf_counter() - a < 0.05, "큐가 가득 차도 막히면 안 된다"
    assert o._bb_dropped > 0


if __name__ == "__main__":
    for fn in (test_tick_not_blocked_by_slow_disk, test_all_lines_flushed_on_close, test_queue_full_drops_and_counts_not_blocks):
        fn(); print("PASS", fn.__name__)
