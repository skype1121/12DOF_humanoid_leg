"""DofMapping 계층.

CAN ID, URDF/Isaac DOF 이름, UI row index를 joint name 기준으로 연결한다.
어느 계층도 1~12 또는 DOF index를 직접 하드코딩하지 않도록 이 클래스를 통한다.
"""

from robot_runtime.robot_model import RobotModel


class DofMapping:
    def __init__(self, robot_model=None):
        self.robot_model = robot_model or RobotModel.load_12dof()

    @property
    def joint_names(self):
        return self.robot_model.joint_names

    def can_id_for_joint(self, joint_name):
        return self.robot_model.get_joint(joint_name).can_id

    def joint_for_can_id(self, can_id):
        return self.robot_model.get_by_can_id(can_id).joint_name

    def isaac_dof_index_for_joint(self, joint_name):
        return self.robot_model.get_joint(joint_name).isaac_dof_index

    def joint_for_isaac_dof_index(self, dof_index):
        return self.robot_model.get_by_isaac_dof_index(dof_index).joint_name

    def ui_index_for_joint(self, joint_name):
        return self.joint_names.index(joint_name)

    def build_isaac_name_map(self, dof_names):
        """Isaac에서 받은 DOF 이름 목록을 RobotModel joint와 이름 기반 매칭한다."""
        dof_names = tuple(str(name) for name in dof_names)
        dof_name_to_index = {name: index for index, name in enumerate(dof_names)}
        matched = {}
        missing = []
        expected_isaac_names = set()

        for spec in self.robot_model.joint_specs:
            aliases = tuple(dict.fromkeys((spec.isaac_joint_name,) + tuple(spec.isaac_dof_aliases)))
            expected_isaac_names.update(aliases)
            matched_name = next(
                (alias for alias in aliases if alias in dof_name_to_index),
                None,
            )
            if matched_name is not None:
                matched[spec.joint_name] = {
                    "dof_name": matched_name,
                    "dof_index": dof_name_to_index[matched_name],
                }
            else:
                missing.append(spec.joint_name)

        extra = [name for name in dof_names if name not in expected_isaac_names]
        return {
            "matched": matched,
            "missing": missing,
            "extra": extra,
            "dof_count": len(dof_names),
            "dof_names": list(dof_names),
        }

    def snapshot(self):
        return {
            "joint_names": list(self.joint_names),
            "can_id_by_joint": self.robot_model.can_id_by_joint(),
            "joint_by_can_id": self.robot_model.joint_by_can_id(),
            "isaac_dof_index_by_joint": self.robot_model.isaac_dof_index_by_joint(),
            "joint_by_isaac_dof_index": self.robot_model.joint_by_isaac_dof_index(),
        }
