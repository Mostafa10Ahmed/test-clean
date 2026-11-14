"""بروفايلات التبييض. بتتبعت كـ config للسيرفر مع كل صفحة."""
from __future__ import annotations

from typing import Any

PROFILE_CHOICES = ("perfect", "fast")

# perfect: أعلى جودة (CTD للكشف، LaMa large، الحلقات الصغيرة بتتعالج كل واحدة لوحدها، والبالونات السادة بتتملى بلون الخلفية)
# fast:    أسرع (كشف وInpaint بحجم 1024، والموديل الأخف lama_mpe)
_PROFILES: dict[str, dict[str, Any]] = {
    "perfect": {
        "detector": {"detector": "ctd", "detection_size": 2048},
        "inpainter": {
            "inpainter": "lama_large",
            "inpainting_size": 2048,
            "inpainting_precision": "bf16",
            "solid_fill_pure_bubbles": True,
            "per_block_inpainting": True,
        },
    },
    "fast": {
        "detector": {"detector": "ctd", "detection_size": 1024},
        "inpainter": {
            "inpainter": "lama_mpe",
            "inpainting_size": 1024,
            "inpainting_precision": "bf16",
            "solid_fill_pure_bubbles": True,
            "per_block_inpainting": True,
        },
    },
}


def build_config(profile: str) -> dict[str, Any]:
    if profile not in _PROFILES:
        raise ValueError(f"Unknown profile: {profile}")
    # نسخة جديدة كل مرة عشان ما حدش يعدل الـ dict الأصلي
    return {section: dict(values) for section, values in _PROFILES[profile].items()}
