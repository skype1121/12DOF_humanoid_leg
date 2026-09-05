# rl_walking/assets/archive/ — 더 이상 학습·판정에 쓰지 않는 파생 자산 (2026-09-05 정리)

| 자산 | 만든 날 | 용도였던 것 | 폐기 사유 | 재사용 가치 |
|---|---|---|---|---|
| `biped12_payload.urdf` | 08-02 | 결손 1.51kg 을 뒤150·위100mm 에 고정 링크로 주입 | 실물 17.62kg 저울 확정 + final3 가 실질량 내장 → payload 개념 소멸 | 없음 |
| `biped12_footvar.urdf` + `usd_footvar/` | 08-26 | "실측 발바닥 박스"(앞85/뒤65) 발 콜리전 | 실접촉은 패드 8점(앞73/뒤74)로 판명 — 오모델 | 감사 기록용 |
| `biped12_symbox.urdf` + `usd_symbox/` | 08-26 | 헐과 같은 풋프린트의 대칭 박스(접촉형식 대조군) | footvar 가설과 함께 퇴역 | 접촉형식 대조군으로 재활용 가능 |
| `biped12_final1.urdf` + `usd_final1/` | 09-03 | 패드 8점 첫 자산(10.94kg) | final3(17.63kg) 로 대체. base↔hip_a 충돌필터가 처음 적용된 자산 | 없음 |

현행 자산은 상위 폴더: `biped12_base.urdf`/`usd/`(V6~V13 재현용), `biped12_final3.urdf`/`usd_final3/`(V14~), `biped12_final3f.urdf`/`usd_final3f/`(패드 플러시).
