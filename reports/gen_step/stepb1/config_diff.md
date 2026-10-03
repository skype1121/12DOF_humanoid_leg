# STEP-B1 config diff (vs STEP-B0)
- rewards.load_transfer: sb0_value("r_transfer" = ramp(Fz_st, 0.5→0.85)·ramp(Fz_sw, 0.5→0.175)) → sb0_value("r_asym" = clamp(Fz_st − Fz_sw, 0, 0.65) [mg]), weight 2.0 동일, 전환 구간·스텝 모드 게이트 동일.
- 그 밖: 없음 (Biped12HangStandStepB1EnvCfg ← StepB0, 실험 디렉터리 biped12_stepb0 공유, run 이름 stepb1_*).
