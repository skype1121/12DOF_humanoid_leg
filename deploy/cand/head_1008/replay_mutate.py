#!/usr/bin/env python3
"""replay_compare.py 의 --new-bb-keys 옵션용 변이 대조군 — 덤프 사본 두 개를 만든다(로컬 시험 전용, 로봇·ROS 불필요).

  python3 replay_mutate.py <덤프.json> <블랙박스 변이.json> <발행 변이.json> <새 필드 키(쉼표)>

① 블랙박스 변이: 새 필드가 **아닌** 블랙박스 키 하나의 값을 바꾼다 → 같은 옵션으로 비교하면 '다름'이어야 한다(옵션이 다른 필드의 차이를 가리지 않는지).
② 발행 변이: 새 필드 이름의 키를 **발행**(토픽 메시지)에 끼운다 → '다름'이어야 한다(옵션은 블랙박스 줄에만 적용되고 발행은 전부 비교되는지).
"""
import copy
import json
import sys


def mutate(v):
    if isinstance(v, bool):
        return not v
    if isinstance(v, (int, float)):
        return v + 1
    return str(v) + "x"


def main():
    src, out_bb, out_pub, keys = sys.argv[1], sys.argv[2], sys.argv[3], set(k for k in sys.argv[4].split(",") if k)
    D = json.load(open(src, encoding="utf-8"))

    B, done = copy.deepcopy(D), None
    for r in B["runs"]:
        for i, row in enumerate(r["bb"]):
            try:
                d = json.loads(row)
            except (ValueError, TypeError):
                continue
            k = next((k for k in d if k not in keys), None) if isinstance(d, dict) else None
            if k is not None and i > 5:
                d[k] = mutate(d[k])
                r["bb"][i] = json.dumps(d, ensure_ascii=False)
                done = (r["key"], r["scenario"], i, k)
                break
        if done:
            break
    if not done:
        print("바꿀 블랙박스 줄이 없음")
        return 1
    json.dump(B, open(out_bb, "w", encoding="utf-8"), ensure_ascii=False)
    print("블랙박스 변이(키·대본·줄·바꾼 키):", done)

    P, done = copy.deepcopy(D), None
    for r in P["runs"]:
        for i, (topic, payload) in enumerate(r["pubs"]):
            try:
                d = json.loads(payload)
            except (ValueError, TypeError):
                continue
            if isinstance(d, dict) and i > 5:
                d[sorted(keys)[0] if keys else "mutated"] = 1
                r["pubs"][i] = [topic, json.dumps(d, ensure_ascii=False)]
                done = (r["key"], r["scenario"], i, topic)
                break
        if done:
            break
    if not done:
        print("바꿀 발행이 없음")
        return 1
    json.dump(P, open(out_pub, "w", encoding="utf-8"), ensure_ascii=False)
    print("발행 변이(키·대본·순번·토픽):", done)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
