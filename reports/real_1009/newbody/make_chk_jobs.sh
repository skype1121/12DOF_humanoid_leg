#!/bin/bash
# 1010 실물 전 점검 작업 목록(NEWBODY_CHECKS.md). 줄 형식은 run_chk.sh 머리말.
HERE=$(cd "$(dirname "$0")" && pwd)/chk; L=/home/ryu/humanoid_leg_test1/logs/rsl_rl
X2=$L/biped12_stepb0/2026-10-10_03-05-58_w3s3dnbj_s2_from_15800/deploy_16299/exported
RN=$L/biped12_stepb0/2026-10-10_00-46-17_g24ranb_s1_from_21793/deploy_23292/exported
J="walk@2:0,0,0;stop@7;walk@12:0.06,0,0;stop@18;walk@23:0.06,0,0;stop@29;walk@34:0,0,0;stop@39"
NW="walk@900:0,0,0"
W4="walk@2:0,0,0;stop@10;walk@14:0.06,0,0;stop@22"
j() { printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$@"; }
# 동일성 확인(128 env·10 s): 원 평가기 대 점검 사본(옵션 꺼짐)
j chkeq_a 128 2 1 10 $X2 - "walk@2:0.06,0,0;stop@6" nb - > $HERE/jobs_eq_a.tsv
j chkeq_b 128 2 1 10 $X2 - "walk@2:0.06,0,0;stop@6" nb - > $HERE/jobs_eq_b.tsv
{ # P1
j p1_nb_e10_s1 256 2 1 45 $X2 - "$J" nb "--trace_qd"
j p1_nb_e08_s1 256 2 1 45 $X2 - "$J" nb "--trace_qd --kw_effort_scales 0.8"
j p1_ob_e10_s1 256 2 1 45 - - "$J" ob "--trace_qd"
j p1_nb_e08_s2 256 2 2 45 $X2 - "$J" nb "--trace_qd --kw_effort_scales 0.8"
j p1_ob_e08_s1 256 2 1 45 - - "$J" ob "--trace_qd --kw_effort_scales 0.8"
} > $HERE/jobs_p1.tsv
{ # P7 접촉 관측 의존도(seed 1). (a) 기준 = p1_nb_e10_s1
j p7_B1_all 256 2 1 45 $X2 - "$J" nb "--kw_contact_fix B:1"
j p7_B0_all 256 2 1 45 $X2 - "$J" nb "--kw_contact_fix B:0"
j p7_L0_all 256 2 1 45 $X2 - "$J" nb "--kw_contact_fix L:0"
j p7_d2f2 256 2 1 45 $X2 - "$J" nb "--contact_pert delay:2;flip:0.02"
j p7_B0_w3pol 256 2 1 45 $X2 - "$J" nb "--kw_contact_fix B:0 --kw_contact_layer w3pol"
j p7_B0_kepol 256 2 1 45 $X2 - "$J" nb "--kw_contact_fix B:0 --kw_contact_layer kepol"
j p7_B0_sup 256 2 1 45 $X2 - "$J" nb "--kw_contact_fix B:0 --kw_contact_layer sup"
j p7_B1_w3pol 256 2 1 45 $X2 - "$J" nb "--kw_contact_fix B:1 --kw_contact_layer w3pol"
j p7_B1_kepol 256 2 1 45 $X2 - "$J" nb "--kw_contact_fix B:1 --kw_contact_layer kepol"
j p7_B1_sup 256 2 1 45 $X2 - "$J" nb "--kw_contact_fix B:1 --kw_contact_layer sup"
} > $HERE/jobs_p7.tsv
{ # P2 고정각 오차(왼,오른 관절 좌표 부호)
for o in "2,2:pp" "2,-2:pm" "-2,2:mp" "-2,-2:mm"; do j p2_${o#*:}_s1 256 2 1 45 $X2 - "$J" nb "--lock_offset_deg=${o%:*}"; done
} > $HERE/jobs_p2.tsv
{ # P3 질량 오차: 왕복(seed 1) + 서기 뒤 18 N(seed 1·2)
for m in "base:0.5@bp" "base:-0.5@bm" "left_hip_a_joint:0.1,right_hip_a_joint:0.1@hp" "left_hip_a_joint:-0.1,right_hip_a_joint:-0.1@hm"; do
  j p3_${m#*@}_J_s1 256 2 1 45 $X2 - "$J" nb "--mass_delta=${m%@*}"; done
for s in 1 2; do j p6_cur_b18_s$s 256 2 $s 8 $X2 - "$NW" nb "--dist push:b:2:18:0.3"; done
for m in "base:0.5@bp" "base:-0.5@bm" "left_hip_a_joint:0.1,right_hip_a_joint:0.1@hp" "left_hip_a_joint:-0.1,right_hip_a_joint:-0.1@hm"; do
  for s in 1 2; do j p3_${m#*@}_b18_s$s 256 2 $s 8 $X2 - "$NW" nb "--mass_delta=${m%@*} --dist push:b:2:18:0.3"; done; done
} > $HERE/jobs_p3.tsv
{ # P6 회복 정책 교체(합성 러너 경로, 서기 중). 현 회복 뒤 18 N 은 P3 목록에서 이미 돈다(건너뜀)
for s in 1 2; do
  j p6_cur_b18_s$s 256 2 $s 8 $X2 - "$NW" nb "--dist push:b:2:18:0.3"
  j p6_nbr_b18_s$s 256 2 $s 8 $X2 $RN "$NW" nb "--dist push:b:2:18:0.3"
  j p6_cur_sf15_s$s 256 2 $s 8 $X2 - "$NW" nb "--dist ramp:f:2:15:0.5:1.0"
  j p6_nbr_sf15_s$s 256 2 $s 8 $X2 $RN "$NW" nb "--dist ramp:f:2:15:0.5:1.0"
done; } > $HERE/jobs_p6.tsv
{ # P4 걷는 중 12 N × 0.2 s: 제자리(6 s) → 정지 → 앞 0.06(18 s) → 정지, 켜는 시각 8 위상 분산
for d in f b l r; do j p4_nb_$d 256 2 1 28 $X2 - "$W4" nb "--on_phases 8 --dist push:$d:6:12:0.2;push:$d:18:12:0.2"; done
} > $HERE/jobs_p4.tsv
{ # P5 지연
j p5_d3_s1 256 3 1 45 $X2 - "$J" nb -
j p5_d5_s1 256 5 1 45 $X2 - "$J" nb -
j p5_d4_s2 256 4 2 45 $X2 - "$J" nb -
} > $HERE/jobs_p5.tsv
wc -l $HERE/jobs_*.tsv
