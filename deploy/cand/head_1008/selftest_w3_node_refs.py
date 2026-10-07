#!/usr/bin/env python3
"""selftest_w3_bridge_node.py 를 **git 없는 저장소 사본(젯슨 ~/humanoid_repo)** 에서 돌리기 위한 얇은 실행기.

배경: selftest_w3_bridge_node.py 의 비트 비교 3 건([E] 플래그 꺼짐·[G] W3-A·[K1] 합성 꺼짐)은 기준 커밋의 브리지를
`git show <커밋>:jetson/scripts/rl_bridge_node.py` 로 읽는다. 젯슨 사본은 git 저장소가 아니라서 그 3 건이 CalledProcessError 로
FAIL 한다(다른 항목은 PASS) → 배포 게이트가 항상 실패한다.
이 실행기는 원본 selftest 를 고치지 않고 `_load_ref` 만 "파일에서 읽기"로 바꿔 끼운다. 기준 브리지 원문은
deploy/make_head_manifest.py 가 git 에서 뽑아 `ref_bridge/rl_bridge_node.<커밋>.ref` 로 두고 md5 를 매니페스트에 적는다.
기준 파일이 없으면 건너뛰지 않고 그 항목이 FAIL 한다(조용한 통과 없음).

  python3 deploy/cand/head_1008/selftest_w3_node_refs.py        # 저장소 루트에서. ROS·CAN·로봇 불필요(가짜 rclpy·가짜 시계)

마지막 줄이 `w3 node selftest: ALL PASS` 이고 종료 코드 0 이면 통과. 계약·순서 검증이다 — 폐루프 성능이나 실물 안전을 증명하지 않는다."""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SCRIPTS = os.path.join(REPO, "jetson", "scripts")
REF_DIR = os.path.join(HERE, "ref_bridge")


def main():
    sys.path.insert(0, SCRIPTS)
    import selftest_w3_bridge_node as st

    def load_ref_from_file(rev):
        """원본 _load_ref 와 같은 방식(같은 __file__ 경로로 실행한 별도 모듈) — 원문만 git 대신 ref_bridge/ 에서 읽는다."""
        path = os.path.join(REF_DIR, f"rl_bridge_node.{rev}.ref")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"기준 브리지 원문 없음: {path} (deploy/make_head_manifest.py 로 다시 만들 것)")
        with open(path, "rb") as f:
            src = f.read()
        mod = types.ModuleType("rl_bridge_node_ref")
        mod.__file__ = os.path.join(st.HERE, "rl_bridge_node.py")
        sys.modules["rl_bridge_node_ref"] = mod
        exec(compile(src, mod.__file__ + f"@{rev}", "exec"), mod.__dict__)
        return mod

    if not callable(getattr(st, "_load_ref", None)):
        print("[FAIL] selftest_w3_bridge_node._load_ref 가 없다 — 원본 selftest 구조가 바뀜(이 실행기 갱신 필요)")
        return 1
    st._load_ref = load_ref_from_file
    refs = sorted(f for f in os.listdir(REF_DIR) if f.endswith(".ref")) if os.path.isdir(REF_DIR) else []
    print(f"[refs] 기준 브리지 원문은 파일에서 읽는다({os.path.relpath(REF_DIR, REPO)}): {', '.join(refs) or '없음'}")
    return st.main()


if __name__ == "__main__":
    raise SystemExit(main())
