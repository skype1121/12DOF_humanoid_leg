# 시도 분석 20260913_1918 (2026-09-13 20:13:40)

## 1. 구간·순서·운영자 이벤트 (analyze_hang2stand_trial)
```
Traceback (most recent call last):
  File "/home/ryu/humanoid_leg_test1/jetson/scripts/analyze_hang2stand_trial.py", line 530, in <module>
    main()
  File "/home/ryu/humanoid_leg_test1/jetson/scripts/analyze_hang2stand_trial.py", line 526, in main
    report(a.bb, a.timeline, a.pads, ref=a.ref)
  File "/home/ryu/humanoid_leg_test1/jetson/scripts/analyze_hang2stand_trial.py", line 302, in report
    ticks, events = load_bb(bb_path)
  File "/home/ryu/humanoid_leg_test1/jetson/scripts/analyze_hang2stand_trial.py", line 32, in load_bb
    for l in open(path, encoding="utf-8"):
FileNotFoundError: [Errno 2] No such file or directory: 'reports/real_20260913/20260913_1918/rl_bridge_20260913_1918.jsonl'
```
## 2. 로그 재생 계약 검증 (replay_bridge_log — 출력·계약, 폐루프 성능 아님)
```
    sys.exit(main())
  File "/home/ryu/humanoid_leg_test1/rl_walking/deploy/replay_bridge_log.py", line 53, in main
    ticks, stops = load(a.bb)
  File "/home/ryu/humanoid_leg_test1/rl_walking/deploy/replay_bridge_log.py", line 31, in load
    for l in open(path, encoding="utf-8"):
FileNotFoundError: [Errno 2] No such file or directory: 'reports/real_20260913/20260913_1918/rl_bridge_20260913_1918.jsonl'
```
## 3. 낙하 등급 (analyze_fall — 성공 판정 아님)
> ⚠ 이 등급은 지지 줄·보조 여부를 모른다. 0910 trial1(윈치에 매달린 채 260 s 안정)도 'A급 완전 성공'으로 나온다. 낙하 여부·자세 진단으로만 읽고, 자립 성공은 §끝의 규칙(모든 줄 느슨·무보조 확인 이후 관측 + 영상)으로만 판정한다.
```
Traceback (most recent call last):
  File "/home/ryu/humanoid_leg_test1/jetson/scripts/analyze_fall.py", line 681, in <module>
    sys.exit(main())
  File "/home/ryu/humanoid_leg_test1/jetson/scripts/analyze_fall.py", line 666, in main
    a = analyze(args.jsonl, args.foot_scale)
  File "/home/ryu/humanoid_leg_test1/jetson/scripts/analyze_fall.py", line 323, in analyze
    ticks, events, bad = load(path)
  File "/home/ryu/humanoid_leg_test1/jetson/scripts/analyze_fall.py", line 137, in load
    with open(path, "r") as f:
FileNotFoundError: [Errno 2] No such file or directory: 'reports/real_20260913/20260913_1918/rl_bridge_20260913_1918.jsonl'
```
## 4. 패드 진단 (analyze_cop_transfer — 미보정, 원인·성공 판정 금지)
```
Traceback (most recent call last):
  File "/home/ryu/humanoid_leg_test1/jetson/scripts/analyze_cop_transfer.py", line 155, in <module>
    main()
  File "/home/ryu/humanoid_leg_test1/jetson/scripts/analyze_cop_transfer.py", line 119, in main
    pads = load_pads(a.pads); bb = load_bb(a.bb) if a.bb else []
  File "/home/ryu/humanoid_leg_test1/jetson/scripts/analyze_cop_transfer.py", line 70, in load_bb
    for l in open(p):
FileNotFoundError: [Errno 2] No such file or directory: 'reports/real_20260913/20260913_1918/rl_bridge_20260913_1918.jsonl'
```

판정 규칙: 자립 성공은 운영자 '모든 줄 느슨·무보조 확인' 이후 관측 시간 + 영상. 가설 판별표: reports/real_prep_0911/hypotheses_and_field_checks.md §B.
