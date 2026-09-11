# 콘솔 모의 검증 (--demo, 2026-09-11)

## 1. 초기 선행조건 게이팅
- 초기 next_action=can_up, stand 허용=False (거짓이어야) → PASS
- handoff 허용=False (거짓이어야) → PASS
- 순서위반(stand 조기) 응답: ok=False '{"ok": false, "message": "먼저 CAN 준비 단계를 완료하세요"}' → PASS(거부)

## 2. 정상 접지 절차 순차 진행
- can_up: 요청ok=True 완료=True phase='준비 확인' → PASS
- sensors: 요청ok=True 완료=True phase='준비 확인' → PASS
- boot_pose: 요청ok=True 완료=True phase='준비 확인' → PASS
- selftest: 요청ok=True 완료=True phase='준비 확인' → PASS
- stage8: 요청ok=True 완료=True phase='자세 유지 준비' → PASS
- hang_stand: 요청ok=True 완료=True phase='공중 자세 유지' → PASS
- gain_ramp: 요청ok=True 완료=True phase='공중 자세 유지' → PASS
- op_touchdown(착지 모사): 접촉=[True, True]
- ground: 요청ok=True 완료=True phase='공중 자세 유지' → PASS
- handoff: 요청ok=True 완료=True phase='정책 인계' → PASS
- baseline: 요청ok=True 완료=True phase='정책 인계' → PASS
- arm: 요청ok=True 완료=True phase='정책 시작 준비' → PASS
- bridge: 요청ok=True 완료=True phase='정책 시작 준비' → PASS
- 정책 선택=stand_hang_v5 로드=stand_hang_v5 → PASS(일치)
- stand: 완료=True phase='정책 제어 · 지지 제거' → PASS

## 3. 측정 게이팅 (모든 줄 느슨 전엔 관측 시작 안 함)
- stand 직후 measurement.state='waiting' (waiting 이어야) → PASS
- op_slack_done 후 measurement.state='measuring' (measuring 이어야) → PASS

## 4. 실패(비상정지) 처리
- estop 후 stop=True measurement.state='interrupted' (interrupted 기대) → PASS

## 5. 회수·새 시험
- support_confirm: 요청ok=True 완료=False
- stack_down: 요청ok=True 완료=True
- release: 요청ok=True 완료=True
- new_trial 후: number=2 completed=[] next=can_up → PASS(초기화)
