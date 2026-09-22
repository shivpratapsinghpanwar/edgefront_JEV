"""Cost model.

Hosted cost is arithmetic on a published price. Local cost is an estimate built
from assumptions that reasonable people will argue about - so every assumption
is carried into the report and printed next to the number, rather than being
buried in a constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Jev, as published 2026-09: input $0.042 / M tokens, output free.
HOSTED_PRICES_USD_PER_MTOK = {"jev": 0.042}


@dataclass(frozen=True)
class CostModel:
    """Assumptions behind a local cost-per-call figure.

    The defaults describe a modest always-on CPU box. They are deliberately
    conservative: if local still wins under pessimistic assumptions, the finding
    survives argument.
    """

    hardware_usd: float = 600.0
    amortise_years: float = 3.0
    watts: float = 45.0
    electricity_usd_per_kwh: float = 0.12
    utilisation: float = 0.25  # fraction of wall time actually serving

    def to_dict(self) -> dict:
        return {
            "hardware_usd": self.hardware_usd,
            "amortise_years": self.amortise_years,
            "watts": self.watts,
            "electricity_usd_per_kwh": self.electricity_usd_per_kwh,
            "utilisation": self.utilisation,
        }

    def local_usd_per_call(self, p50_ms: float) -> float:
        """Amortised hardware + power attributable to one call."""
        if p50_ms <= 0:
            return 0.0
        seconds = self.amortise_years * 365 * 24 * 3600 * self.utilisation
        if seconds <= 0:
            return 0.0
        calls_over_life = seconds / (p50_ms / 1000.0)
        hardware_share = self.hardware_usd / calls_over_life
        kwh = (self.watts / 1000.0) * (p50_ms / 1000.0 / 3600.0)
        return hardware_share + kwh * self.electricity_usd_per_kwh


@dataclass
class CostReport:
    usd_per_call: float
    usd_per_million_calls: float
    basis: str
    assumptions: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "usd_per_call": self.usd_per_call,
            "usd_per_million_calls": round(self.usd_per_million_calls, 4),
            "basis": self.basis,
            "assumptions": self.assumptions,
        }


def hosted_cost(
    total_input_tokens: int, n_calls: int, price_per_mtok: float
) -> CostReport:
    if n_calls <= 0:
        return CostReport(0.0, 0.0, "hosted: no calls", {})
    per_call = (total_input_tokens / n_calls) * price_per_mtok / 1_000_000
    return CostReport(
        usd_per_call=per_call,
        usd_per_million_calls=per_call * 1_000_000,
        basis=(
            f"{total_input_tokens / n_calls:.0f} input tok/call "
            f"x ${price_per_mtok}/Mtok, output free"
        ),
        assumptions={"price_usd_per_mtok": price_per_mtok},
    )


def cascade_cost(
    local_p50_ms: float,
    cost_model: CostModel,
    hosted_avg_tokens_per_call: float,
    hosted_price_per_mtok: float,
    escalation_rate: float,
) -> CostReport:
    """Cost of a confidence-gated local->hosted cascade.

    The local backend runs on every call - it always pays its (amortised)
    local cost. The hosted backend runs only on the fraction of calls that
    escalated. So the blended per-call cost is the local cost plus the hosted
    cost scaled by how often escalation actually happened - arithmetic on the
    two existing, unchanged cost functions above, not a new cost model.
    """
    local_component = local_cost(local_p50_ms, cost_model)
    hosted_component = hosted_cost(
        total_input_tokens=round(hosted_avg_tokens_per_call),
        n_calls=1,
        price_per_mtok=hosted_price_per_mtok,
    )
    per_call = (
        local_component.usd_per_call + escalation_rate * hosted_component.usd_per_call
    )
    return CostReport(
        usd_per_call=per_call,
        usd_per_million_calls=per_call * 1_000_000,
        basis=(
            f"local always runs ({local_component.basis}) "
            f"plus hosted on {escalation_rate:.1%} of calls "
            f"({hosted_component.basis})"
        ),
        assumptions={
            "local": local_component.assumptions,
            "hosted": hosted_component.assumptions,
            "escalation_rate": escalation_rate,
        },
    )


def local_cost(p50_ms: float, model: CostModel) -> CostReport:
    per_call = model.local_usd_per_call(p50_ms)
    return CostReport(
        usd_per_call=per_call,
        usd_per_million_calls=per_call * 1_000_000,
        basis=(
            f"amortised ${model.hardware_usd:.0f} over {model.amortise_years:g}y "
            f"at {model.utilisation:.0%} utilisation, plus {model.watts:g}W "
            f"at ${model.electricity_usd_per_kwh}/kWh"
        ),
        assumptions=model.to_dict(),
    )
