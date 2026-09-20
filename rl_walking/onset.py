"""온셋 인계 가중치 (Isaac 없이 테스트 가능하도록 분리, 0921)."""
import torch


def onset_alpha(t, hold_s, blend_s):
    """(N,) 온셋 가중치: t<hold → 0, hold≤t<hold+blend → 스무스스텝 0→1, 그 뒤 1. blend=0 이면 hold 뒤 즉시 1.
    브리지 deploy_chain.blend_alpha(스무스스텝) 와 같은 식 — 정책이 학습 안에서 홀드→정책 크로스페이드를 본다."""
    a = ((t - hold_s) / blend_s.clamp(min=1e-6)).clamp(0.0, 1.0)
    a = torch.where(blend_s <= 0.0, (t >= hold_s).to(t.dtype), a)
    return a * a * (3.0 - 2.0 * a)
