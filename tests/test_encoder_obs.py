"""엔코더측 관측(유격 관절) 단위 검사 — torch 만."""
import sys, types, unittest, math
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
# isaaclab 의존 없이 mdp_hang 의 해당 함수만 로드
src = open(ROOT / "rl_walking/mdp_hang.py", encoding="utf-8").read()
start = src.index("def _play_by_joint"); code = "import torch\n" + src[start:]
ns = {}; exec(compile(code, "enc_obs", "exec"), ns)

class Act:  # DeadzoneImplicit 흉내
    def __init__(self, play, idx): self._play = play; self.joint_indices = idx
class Data:
    def __init__(self, q, d): self.joint_pos = q; self.default_joint_pos = d
class Asset:
    def __init__(self, q, d, acts): self.data = Data(q, d); self.actuators = acts
class Term:
    def __init__(self, ref): self.processed_actions = ref
class AM:
    def __init__(self, ref): self.t = Term(ref)
    def get_term(self, n): return self.t
class Env:
    def __init__(self, asset, ref, N):
        self.scene = {"robot": asset}; self.action_manager = AM(ref); self.episode_length_buf = torch.full((N,), 5); self.step_dt = 0.02

class T(unittest.TestCase):
    def setUp(self):
        self.N, self.J = 3, 12
        self.q = torch.zeros(self.N, self.J); self.d = torch.zeros(self.N, self.J)
        self.play = torch.full((self.N, 2), math.radians(4.0))
        self.acts = {"ankle": Act(self.play, [8, 9])}
    def env(self, ref): return Env(Asset(self.q, self.d, self.acts), ref, self.N)
    def test_no_play_equals_joint_pos(self):
        ref = torch.zeros(self.N, self.J); ref[:, 0] = 0.3; self.q[:, 0] = 0.1
        out = ns["encoder_joint_pos_rel"](self.env(ref), types.SimpleNamespace(name="robot"))
        self.assertAlmostEqual(float(out[0, 0]), 0.1, places=6)        # 유격 없는 관절: 물리각 그대로
    def test_inside_play_reads_target(self):
        ref = torch.zeros(self.N, self.J); ref[:, 8] = math.radians(2.0)
        out = ns["encoder_joint_pos_rel"](self.env(ref), types.SimpleNamespace(name="robot"))
        self.assertAlmostEqual(float(out[0, 8]), math.radians(2.0), places=6)   # |e|<play → θ_enc = θ_ref
    def test_outside_play_offset_by_play(self):
        ref = torch.zeros(self.N, self.J); ref[:, 9] = math.radians(10.0)
        out = ns["encoder_joint_pos_rel"](self.env(ref), types.SimpleNamespace(name="robot"))
        self.assertAlmostEqual(float(out[0, 9]), math.radians(4.0), places=6)   # θ_j + play
    def test_velocity_and_reset(self):
        ref = torch.zeros(self.N, self.J); e = self.env(ref)
        v0 = ns["encoder_joint_vel"](e, types.SimpleNamespace(name="robot")); self.assertEqual(float(v0.abs().max()), 0.0)
        ref[:, 8] = math.radians(1.0)
        v1 = ns["encoder_joint_vel"](e, types.SimpleNamespace(name="robot"))
        self.assertAlmostEqual(float(v1[0, 8]), math.radians(1.0) / 0.02, places=4)
        e.episode_length_buf[:] = 0; ref[:, 8] = math.radians(3.0)
        v2 = ns["encoder_joint_vel"](e, types.SimpleNamespace(name="robot")); self.assertEqual(float(v2[0, 8]), 0.0)   # 리셋 직후 0

if __name__ == "__main__":
    unittest.main()
