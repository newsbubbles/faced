"""Independent emotion judge for the read-vs-drive study.

Scores text on the seven `faced` axes with a pretrained GoEmotions classifier
(`SamLowe/roberta-base-go_emotions`) — a DIFFERENT model from the one being
steered — so "drive" (does steering change the output's emotion?) is measured
independently of the direction we steer. This is what keeps the test from being
circular.

Confidence is the bipolar axis, scored as a signed composite. An optional
OpenRouter LLM cross-check is available for it (open models only, per project rule).
"""
from __future__ import annotations

import functools

# axis -> (positive GoEmotions labels, negative labels for bipolar axes)
AXIS_LABELS = {
    "surprise":    (["surprise", "realization"], []),
    "confidence":  (["pride", "optimism", "approval"], ["nervousness", "confusion"]),
    "curiosity":   (["curiosity"], []),
    "confusion":   (["confusion"], []),
    "frustration": (["annoyance", "disappointment", "anger"], []),
    "fear":        (["fear", "nervousness"], []),
    "warmth":      (["love", "caring", "gratitude"], []),
}


@functools.lru_cache(maxsize=1)
def _clf():
    from transformers import pipeline
    return pipeline("text-classification", model="SamLowe/roberta-base-go_emotions",
                    top_k=None, truncation=True, max_length=256, device=-1)


def _axis_scores(probs: dict) -> dict:
    return {axis: sum(probs.get(l, 0.0) for l in pos) - sum(probs.get(l, 0.0) for l in neg)
            for axis, (pos, neg) in AXIS_LABELS.items()}


def score_batch(texts: list[str]) -> list[dict]:
    """Per-axis judge score for each text (signed for the bipolar confidence axis)."""
    if not texts:
        return []
    results = _clf()([t[:1000] if t.strip() else "." for t in texts])
    return [_axis_scores({d["label"]: d["score"] for d in res}) for res in results]


def score_text(text: str) -> dict:
    return score_batch([text])[0]


def validate(axes=None) -> dict:
    """Judge-validation gate. For each axis, the judge score must separate that axis's
    POSITIVE probe prompts from its NEUTRAL ones (does GoEmotions actually see the
    emotion in text?). Returns per-axis judge AUC; low AUC = the judge can't measure
    that axis and the drive result for it is untrustworthy."""
    import json
    from pathlib import Path

    import numpy as np
    from sklearn.metrics import roc_auc_score

    PD = Path(__file__).resolve().parent.parent / "data" / "prompts"
    axes = axes or list(AXIS_LABELS)
    out = {}
    for a in axes:
        recs = [json.loads(l) for l in open(PD / f"{a}.jsonl", encoding="utf-8")]
        y = np.array([int(r["label"]) for r in recs])
        s = np.array([sc[a] for sc in score_batch([r["text"] for r in recs])])
        try:
            out[a] = round(float(roc_auc_score(y, s)), 3)
        except ValueError:
            out[a] = float("nan")
    return out


if __name__ == "__main__":  # quick self-test / validation gate
    import json
    print("judge-validation (judge AUC separating pos vs neutral probe text):")
    v = validate()
    for a, auc in v.items():
        flag = "" if (auc == auc and auc >= 0.75) else "   <-- WEAK, cross-check needed"
        print(f"  {a:12s} {auc}{flag}")
    print(json.dumps(v))
