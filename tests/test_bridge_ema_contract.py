"""1001 관측 joint_vel_ema 계약 모드 — 브리지 판별·거부 경로 (ROS·CAN 없이).

G4~G15 는 학습 중 joint_vel_ema 가 0+잡음(mdp_hang 구버전 결함)이라 'ema0' 이 학습 계약이고,
v2 로 학습한 정책(G16~)에 ema0 을 쓰면 반대 방향의 같은 결함이 된다 → 기동 거부."""
import importlib.util
import os
import sys
import types

import pytest

REPO = "/home/ryu/humanoid_leg_test1"
G12_RUN = f"{REPO}/logs/rsl_rl/biped12_stage4/2026-09-29_14-32-50_g12"
G16O_RUN = f"{REPO}/logs/rsl_rl/biped12_stage4/2026-10-01_01-46-29_g16o"


def _load():
    for name in ("rclpy", "rclpy.node", "rclpy.qos", "std_msgs", "std_msgs.msg",
                 "sensor_msgs", "sensor_msgs.msg", "geometry_msgs", "geometry_msgs.msg"):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = []
            sys.modules[name] = m
    sys.modules["rclpy.node"].Node = type("Node", (), {"__init__": lambda self, *a, **k: None})
    for mod, cls in (("std_msgs.msg", "String"), ("sensor_msgs.msg", "Imu"),
                     ("sensor_msgs.msg", "JointState"), ("geometry_msgs.msg", "Vector3")):
        setattr(sys.modules[mod], cls, type(cls, (), {"__init__": lambda self, *a, **k: None}))
    sys.modules["rclpy.qos"].QoSProfile = type("QoSProfile", (), {"__init__": lambda self, *a, **k: None})
    sys.modules["rclpy.qos"].ReliabilityPolicy = types.SimpleNamespace(BEST_EFFORT=0, RELIABLE=1)
    sys.modules["rclpy.qos"].HistoryPolicy = types.SimpleNamespace(KEEP_LAST=0)
    sys.modules["rclpy.qos"].DurabilityPolicy = types.SimpleNamespace(VOLATILE=0)
    spec = importlib.util.spec_from_file_location("rlbn_ema", REPO + "/jetson/scripts/rl_bridge_node.py")
    m = importlib.util.module_from_spec(spec)
    argv = sys.argv
    sys.argv = ["rl_bridge_node.py", "--selftest"]
    try:
        spec.loader.exec_module(m)
    except SystemExit:
        pass
    finally:
        sys.argv = argv
    return m


M = _load()


def test_contract_classification():
    assert M._jvel_ema_contract("rl_walking.mdp_hang:encoder_joint_vel_ema") == "legacy"
    assert M._jvel_ema_contract("rl_walking.mdp_hang:encoder_joint_vel_ema_v2") == "v2"
    assert M._jvel_ema_contract("rl_walking.mdp_hang:joint_vel_ema") == "v2"      # S8a: 물리 관절속도 EMA (결함 없음)
    assert M._jvel_ema_contract(None) is None
    with pytest.raises(ValueError):
        M._jvel_ema_contract("rl_walking.mdp_hang:something_else")


@pytest.mark.skipif(not os.path.isfile(f"{G12_RUN}/params/env.yaml"), reason="G12 env.yaml 없음")
def test_parse_env_yaml_func():
    assert M._parse_env_contract(f"{G12_RUN}/params/env.yaml")["jvel_ema_func"] == "rl_walking.mdp_hang:encoder_joint_vel_ema"


def test_registry_modes():
    assert M.CKPT_JVEL_EMA_MODE.get("stand_crouch_g12", "v2") == "v2"          # 이름 그대로 = 0929 과 같은 동작
    assert M.ckpt_jvel_ema_mode("stand_crouch_g12_ema0") == "ema0"             # legacy 계약 확인 후 허용
    assert M.CHECKPOINTS["stand_crouch_g12_ema0"] == M.CHECKPOINTS["stand_crouch_g12"]


@pytest.mark.skipif(not os.path.isfile(f"{G16O_RUN}/params/env.yaml"), reason="G16o env.yaml 없음")
def test_ema0_on_v2_policy_refused(monkeypatch):
    monkeypatch.setitem(M.CHECKPOINTS, "stand_test_v2", f"{G16O_RUN}/exported")
    monkeypatch.setitem(M.CKPT_JVEL_EMA_MODE, "stand_test_v2", "ema0")
    with pytest.raises(ValueError, match="반대 방향"):
        M.ckpt_jvel_ema_mode("stand_test_v2")


def test_ema0_without_env_yaml_refused(monkeypatch, tmp_path):
    monkeypatch.setitem(M.CHECKPOINTS, "stand_test_noyaml", str(tmp_path / "run" / "exported"))
    monkeypatch.setitem(M.CKPT_JVEL_EMA_MODE, "stand_test_noyaml", "ema0")
    with pytest.raises(ValueError, match="env.yaml 없음"):
        M.ckpt_jvel_ema_mode("stand_test_noyaml")
