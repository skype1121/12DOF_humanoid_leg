# urdf/ — SolidWorks 익스포트 URDF 패키지 계보 (한곳 정리, 2026-09-05)

학습·평가에 실제로 쓰이는 파생 자산은 `rl_walking/assets/`(URDF)와 `rl_walking/assets/usd_*/`(USD)이고,
이 폴더는 그 **원본 소스**들이다. 파생 절차: `make_base_urdf.py --src <여기 urdf> --out rl_walking/assets/biped12_X.urdf`
→ `convert_urdf.py` → `postprocess_usd.py`(리밋 주입) → (`patch_usd_filter_pairs.py`, base↔hip_a 충돌필터).

| 패키지 | 익스포트 | 총질량 | 상태 | 비고 |
|---|---|---|---|---|
| `URDF12DOF/` | 2026-07-18 | 9.94 kg | 구판 | 초기 자산. `sim_walking/`·`isaacsim_scene/newURDF12DOF_isaac.urdf` 가 참조 |
| `12URDF0725/` | 2026-07-25 | 10.49 kg | **V6~V13 학습 자산 소스** (`biped12_base.urdf`/`usd/`) | 질량 배분만 갱신. 실물 실측(17.62) 대비 −7.1kg — 모터·배터리·전장 목업 저밀도 |
| `final1_urdf/` | 2026-09-03 | 10.84 kg | 감사용 | 압력패드(RA30P) 8개 링크 첫 도입. 결함 5종(right_hip_r 프레임 붕괴·리밋 잠김·축부호·D455 질량0·링크명) — `fix_final1_frames.py` |
| `final2_urdf/` | 2026-09-03 | 15.54 kg | 감사용 | 어셈블리 레벨 질량 재지정 일부만 반영(골반 서브어셈 힙모터 2개·D455 누락) |
| `final3_urdf/` | 2026-09-04 | **17.63 kg** | **현행** (`biped12_final3.urdf`/`usd_final3`, 플러시 `biped12_final3f.urdf`/`usd_final3f`) | 저울 실측 17.62 정합. right_hip_r 메시 프레임만 잔존 → `fix_final3_mesh.py` (`urdf/final3_urdf_fixed.urdf`), 패드 플러시판 `final3_urdf_flush.urdf` |

익스포터 고질병(재발 시 참고): left_hip_f 축부호 반전(0725·final1·final3 모두), 리밋 전부 0, 어셈블리 레벨 질량 재지정 무시(파트 레벨로 넣어야 반영),
right_hip_r 기준좌표계 오지정(final1: 관절·메시 전부 무릎 프레임 / final3: 메시만).
