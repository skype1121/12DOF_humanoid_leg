"""브리지 CHECKPOINTS / CKPT_GAIT_FREQ 에 세대 등록 (멱등).
python3 register_bridge.py <name> <run> [bridge_path]
"""
import sys

name, run = sys.argv[1], sys.argv[2]
p = sys.argv[3] if len(sys.argv) > 3 else "jetson/scripts/rl_bridge_node.py"
s = open(p, encoding="utf-8").read()
if f'"{name}"' in s:
    print(f"  브리지 이미 등록: {name}")
    sys.exit(0)
i = s.index("CHECKPOINTS = {")
j = s.index("\n}\n", i)                      # CHECKPOINTS 닫는 괄호 직전
entry = (f"\n    # {name} (0905): 정정 env(실질량 17.62kg·명목 노이즈 정정, 새 축 없음) — 판정 reports/gen_*/README.md\n"
         f'    "{name}": os.path.join(\n        REPO, "logs/rsl_rl/biped12_stage4/{run}/exported"),')
s = s[:j] + entry + s[j:]
k = s.index("CKPT_GAIT_FREQ = {")
m = s.index("}", k)
s = s[:m] + f', "{name}": 1.1' + s[m:]
open(p, "w", encoding="utf-8").write(s)
print(f"  브리지 등록: {name} → {run}")
