# STEP-I0 config diff (vs STEP-B1)
- events.sbi = sbi_step(ref_path=reports/gen_step/ref_step_primitive.npz) — sb0 뒤 실행, 같은 위상(23틱/반주기, 오른발 반주기 = 관절 미러·y 반전).
- rewards: load_transfer(r_asym) 2.0 → 1.0 · + joint_pose_reference 1.5 (σq 0.10 rad, 가중: 좌우 방향·무릎 1, hip_f·ankle_f 0.3 — 이륙 전 값 고정)
  · + foot_position_reference 2.0 (σp 3 cm, 스윙발 골반 기준 상대 궤적: z 최대 +1.9 cm, x +5 cm smoothstep) · + contact_phase_reference 2.0.
- curriculum: sbi_anneal_curriculum (배율 1.0/0.75/0.5/0.25/0.05, 블록 ≥ 20000 예정 반주기·체류 ≥ 50 반복·하중 수용 ≥ 0.5).
- 기준 생성: reports/gen_step/build_ref.py ← extract_ref.py(G22R2-13099, 뒤 18 N 충격 유효 스텝 489, 오른발 사건 미러).
