"""1007 parity 감사: W3S3d-15800 exported policy.onnx(젯슨 onnxruntime 경로) = policy.pt(PC torch 경로) 동치 — 기록된 시뮬 관측 320 전 틱으로.

onnxruntime 이 없는 PC 에서도 돌도록 onnx.reference.ReferenceEvaluator(순수 numpy, onnx ≥ 1.13)를 쓴다 — Isaac 파이썬에 onnx 가 있다:
  /home/ryu/IsaacLab/_isaac_sim/python.sh rl_walking/deploy/selftest_w3s3d_onnx.py
허용: |Δ| ≤ 1e-5 (float32 연산 순서). 시뮬 행동(GPU)과의 차도 함께 보고.
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXP = os.path.join(REPO, "logs/rsl_rl/biped12_stepb0/2026-10-07_00-11-12_w3S3d_from_s2d/exported")
NPZ = os.path.join(REPO, "log_picture/w3parity_rec_w3s3d_15800_d3.npz")
if len(sys.argv) == 3:      # 1010: selftest_w3s3d_onnx.py <exported 폴더> <기록 npz> — 다른 W3 키(새 몸 종종걸음 등) 대조. 인자 없으면 종전 그대로
    EXP, NPZ = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])


def main():
    try:
        import onnx
        from onnx.reference import ReferenceEvaluator
    except ImportError as e:
        print(f"[SKIP] onnx 없음({e}) — Isaac python.sh 로 실행")
        return 0
    import torch
    if not (os.path.isfile(NPZ) and os.path.isdir(EXP)):
        print("[SKIP] 기록·export 없음")
        return 0
    z = np.load(NPZ)
    obs = z["obs"].reshape(-1, 320).astype(np.float32)
    act_sim = z["act"].reshape(-1, 12).astype(np.float32)
    m = onnx.load(os.path.join(EXP, "policy.onnx"))
    ev = ReferenceEvaluator(m)
    iname = m.graph.input[0].name
    shp = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
    pt = torch.jit.load(os.path.join(EXP, "policy.pt"), map_location="cpu").eval()
    idx = np.arange(0, obs.shape[0], 7)          # 965 표본(전 구간 고르게)
    a_onnx = np.concatenate([ev.run(None, {iname: obs[i:i + 1]})[0] for i in idx], 0)
    with torch.no_grad():
        a_pt = pt(torch.as_tensor(obs[idx])).numpy()
    d1 = float(np.abs(a_onnx - a_pt).max()); d2 = float(np.abs(a_onnx - act_sim[idx]).max())
    ok = d1 <= 1e-5 and d2 <= 2e-4
    print(("[PASS]" if ok else "[FAIL]") + f" onnx(입력 {iname} {shp}) vs policy.pt: {len(idx)} 표본 최대 |Δ| {d1:.2e} · onnx vs 시뮬 행동(GPU) {d2:.2e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
