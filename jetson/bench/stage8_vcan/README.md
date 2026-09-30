# stage8 가상 CAN 벤치 (0930)

실물 버스(can1)·모터·콘솔과 무관하게 stage8 의 명령 처리율과 '발행 → CAN 송신' 지연을 잰다.
- 가상 CAN `can9`(vcan) + 가짜 AK 모터 12개(`responder.py`, 명령 위치를 그대로 응답) + 격리 ROS 도메인 77.
- `driver.py`: 베이스라인 → ARM → 50 Hz SET_JOINT_TARGETS(2° 0.7 Hz 사인) 25 s → STOP_ALL. 상태 토픽 기록.
- `make_variants.py`: stage8 사본 v0(원본)·v1(명령 구독 깊이 1)·v2(상태 50 Hz·명령마다 상태발행 제거)·v3(v1+v2).
- `run_bench.sh`(젯슨 `/tmp/s8bench` 에 복사해 실행) → `analyze.py <out> v0 v1 …`.
- `prof_run.sh`: cProfile. `s8_microbench.py`: ROS 없이 상태 직렬화 비용.
실행 전 확인: 제어 스택이 내려가 있을 것(ps 로 stage8/hang_stand/rl_bridge 없음). 끝나면 `sudo ip link del can9`.
