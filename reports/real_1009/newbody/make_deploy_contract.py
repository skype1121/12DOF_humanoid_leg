#!/usr/bin/env python3
"""새 몸 학습 run 의 배포 계약 폴더를 만든다(1010): <run>/deploy_<N>/{exported/, params/env.yaml}.
왜: 브리지(jetson/scripts/rl_bridge_node.py _parse_env_contract)는 <exported 의 부모>/params/env.yaml 의 observations.policy.<항>.func 를
줄 단위로 읽어 관측 필터 계약(act EMA·jvel LPF β·jvel_ema v2)을 정한다. 새 몸 학습 태스크는 joint_pos·joint_vel·joint_vel_ema 를 감싸개
nb_obs_fixed_zero 로 싸서 그 줄이 감싸개 이름이 된다 → joint_vel β 를 못 읽고(조용히 원시값) joint_vel_ema 는 '알 수 없는 함수'로 거부된다.
이 도구는 env.yaml 사본에서 감싸개 항만 **안쪽 함수·안쪽 인자** 로 풀어 쓴다(러너가 계산할 관측 = 안쪽 함수; 굳은 관절 열 0 은 stage8 합성
피드백이 만든다). 그 밖의 줄은 그대로. 원본 params/env.yaml·exported_<N> 은 건드리지 않는다.
사용: make_deploy_contract.py <run 디렉토리> <N>   → 계약 대조(--ref run 의 env.yaml 과 브리지 파서 결과 비교)까지 출력."""
import os, shutil, sys

run, n = os.path.abspath(sys.argv[1]), sys.argv[2]
ref = sys.argv[3] if len(sys.argv) > 3 else None
src = open(os.path.join(run, "params", "env.yaml"), encoding="utf-8").read().split("\n")
out = []; i = 0; changed = []
while i < len(src):
    ln = src[i]
    if ln.strip() == "func: rl_walking.newbody_1009:nb_obs_fixed_zero" and ln.startswith("      func:"):
        term = out[-1].strip().rstrip(":")
        assert src[i + 1] == "      params:" and src[i + 2].startswith("        func: "), (term, src[i + 1:i + 3])
        inner = src[i + 2].strip()[len("func: "):]
        j = i + 3; inner_params = []
        if src[j].startswith("        params:"):
            if src[j].strip() == "params:":
                j += 1
                while src[j].startswith("          "):
                    inner_params.append(src[j][2:]); j += 1      # 들여쓰기 10 → 8
            else:
                j += 1                                           # 'params: {}' 한 줄
        while src[j].startswith("        "):                     # joint_names·noise(감싸개 전용) 건너뜀
            j += 1
        out.append(f"      func: {inner}")
        out.append("      params:" + ("" if inner_params else " {}"))
        out += inner_params
        changed.append((term, inner, [p.strip() for p in inner_params])); i = j; continue
    out.append(ln); i += 1
assert [c[0] for c in changed] == ["joint_pos", "joint_vel", "joint_vel_ema"], changed
dst = os.path.join(run, f"deploy_{n}"); os.makedirs(os.path.join(dst, "params"), exist_ok=True)
open(os.path.join(dst, "params", "env.yaml"), "w", encoding="utf-8").write("\n".join(out))
ex = os.path.join(dst, "exported")
if os.path.isdir(ex):
    shutil.rmtree(ex)
shutil.copytree(os.path.join(run, f"exported_{n}"), ex)
open(os.path.join(dst, "README.txt"), "w", encoding="utf-8").write(
    f"배포 계약 폴더(1010, make_deploy_contract.py). exported/ = ../exported_{n} 사본. params/env.yaml = ../params/env.yaml 에서 관측 감싸개\n"
    "nb_obs_fixed_zero 세 항(joint_pos·joint_vel·joint_vel_ema)을 안쪽 함수로 풀어 쓴 사본 — 브리지 계약 파서용. 굳은 관절(hip_r) 관측 0 은\n"
    "러너가 아니라 stage8 의 기구 고정 합성 피드백(mech_fixed)이 만든다: 이 정책은 하드웨어맵에 mech_fixed 선언이 있는 구성에서만 쓴다.\n")
print("wrote", dst); [print("  풀어 쓴 항:", c) for c in changed]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "jetson", "scripts"))
import rl_bridge_node as B  # noqa: E402
c = B._parse_env_contract(os.path.join(dst, "params", "env.yaml")); print("  계약:", {k: v for k, v in c.items() if k != "joint_pos"}, "필터", B.obs_filter_contract(c), "jvel_ema", B._jvel_ema_contract(c.get("jvel_ema_func")))
if ref:
    r = B._parse_env_contract(os.path.join(os.path.abspath(ref), "params", "env.yaml"))
    print("  기준 run 대비 계약 차이:", {k: (r.get(k), c.get(k)) for k in set(r) | set(c) if r.get(k) != c.get(k)} or "없음")
