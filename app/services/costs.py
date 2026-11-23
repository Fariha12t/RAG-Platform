"""Token + cost tracking. Prices are USD per 1M tokens (input, output) and are editable:
set PRICING_JSON='{"my-model": [3.0, 15.0]}' to override. Verify against your provider's price list."""
from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import UsageLog

DEFAULT_PRICES = {
    "claude-sonnet-5-5": (3.0, 15.0),
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.0),
    "extractive": (0.0, 0.0),
}


def estimate_tokens(text: str) -> int:
    """~4 characters per token heuristic (used when a provider doesn't report usage)."""
    return max(1, len(text) // 4)


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = {**DEFAULT_PRICES, **{k: tuple(v) for k, v in get_settings().pricing_overrides.items()}}
    p_in, p_out = prices.get(model, (0.0, 0.0))
    return round((input_tokens * p_in + output_tokens * p_out) / 1_000_000, 6)


def log_usage(db: Session, user_id: int, kind: str, model: str, input_tokens: int, output_tokens: int) -> float:
    cost = cost_usd(model, input_tokens, output_tokens)
    db.add(UsageLog(user_id=user_id, kind=kind, model=model, input_tokens=input_tokens,
                    output_tokens=output_tokens, cost_usd=cost))
    return cost


def usage_summary(db: Session, user_id: int | None = None) -> dict:
    q = select(UsageLog.model, UsageLog.kind, func.count(), func.sum(UsageLog.input_tokens),
               func.sum(UsageLog.output_tokens), func.sum(UsageLog.cost_usd)).group_by(UsageLog.model, UsageLog.kind)
    if user_id is not None:
        q = q.where(UsageLog.user_id == user_id)
    rows = db.execute(q).all()
    by_model = [{"model": m, "kind": k, "requests": n, "input_tokens": int(i or 0),
                 "output_tokens": int(o or 0), "cost_usd": round(float(c or 0), 6)} for m, k, n, i, o, c in rows]
    return {"total_requests": sum(r["requests"] for r in by_model),
            "total_input_tokens": sum(r["input_tokens"] for r in by_model),
            "total_output_tokens": sum(r["output_tokens"] for r in by_model),
            "total_cost_usd": round(sum(r["cost_usd"] for r in by_model), 6),
            "by_model": by_model}
