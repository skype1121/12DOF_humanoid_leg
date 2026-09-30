"""dof12_mapping 파싱 캐시 (0930): 같은 값·파일 변경 즉시 반영·호출자 수정이 캐시를 오염시키지 않음."""
import json
import os
import shutil
import time

from robot_runtime import dof12_mapping as m


def _copy_map(tmp_path):
    dst = tmp_path / "map.json"
    shutil.copy(m.default_config_path(), dst)
    return dst


def test_cached_equals_fresh_parse():
    fresh = json.load(open(m.default_config_path(), encoding="utf-8"))
    assert m.load_12dof_config() == fresh
    for j in m.get_joint_names():
        assert m.get_joint_config(j) == fresh["joints"][j]
        assert m.get_motor_id(j) == int(fresh["joints"][j]["motor_id"])


def test_reload_after_file_change(tmp_path):
    p = _copy_map(tmp_path)
    j = m.get_joint_names()[0]
    before = m.load_12dof_config(p)["joints"][j]["kp"]
    data = json.load(open(p, encoding="utf-8"))
    data["joints"][j]["kp"] = float(before) + 7.0
    time.sleep(0.01)
    tmp = tmp_path / "map.json.tmp"
    json.dump(data, open(tmp, "w", encoding="utf-8"))
    os.replace(tmp, p)                                   # set_pitch_comp 식 원자 교체
    assert m.load_12dof_config(p)["joints"][j]["kp"] == float(before) + 7.0
    # 같은 inode 에 덮어쓰기(크기 변화)도 반영
    data["joints"][j]["kp"] = 1.5
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    assert m.load_12dof_config(p)["joints"][j]["kp"] == 1.5


def test_caller_mutation_does_not_poison_cache():
    j = m.get_joint_names()[0]
    cfg = m.load_12dof_config()
    cfg["joints"][j]["kp"] = -999
    jc = m.get_joint_config(j)
    jc["kp"] = -999
    if isinstance(jc.get("isaac_dof_aliases"), list):
        jc["isaac_dof_aliases"].append("poison")
    fresh = json.load(open(m.default_config_path(), encoding="utf-8"))
    assert m.get_joint_config(j) == fresh["joints"][j]


def test_lookup_is_fast():
    j = m.get_joint_names()[0]
    m.get_motor_id(j)
    t0 = time.perf_counter()
    for _ in range(2000):
        m.get_motor_id(j)
    assert (time.perf_counter() - t0) / 2000 < 2e-4      # 조회당 < 0.2 ms (종전: 파싱 ~0.8 ms/회 젯슨)
