"""Chart drawing for edgefront's accuracy/latency frontier plots.

matplotlib is imported inside each function, not at module level, so
`pip install edgefront` (the core, stdlib-only install) never gains a hard
dependency just because this module got imported - only actually calling one
of these functions does, and every caller in this project
(`docs/make_charts.py`, `edgefront cascade --plot`) already wraps that call in
a try/except ImportError pointing at `pip install 'edgefront[plot]'`.

`plot_frontier` is the generic scatter+annotate drawer `docs/make_charts.py`
originally implemented inline for its own two README charts (model
comparison: baseline/local-fp32/local-int8/hosted); it moved here so a second
chart type does not need a second copy of the same matplotlib boilerplate.
`draw_cascade` is a thin wrapper for the new cost-accuracy sweep chart: a
handful of swept threshold points connected by a line between a local-only
and a hosted-only anchor.

Palette: the project's validated categorical slots 1-3 - see
`docs/make_charts.py`'s original note, `node scripts/validate_palette.js
"#2a78d6,#eb6834,#1baf7a" --mode light --pairs all` -> ALL CHECKS PASS (and
the same three also pass --mode dark). Slots past the third do NOT validate
all-pairs alongside these three in either mode - that is a documented property
of the reference palette itself ("the first three slots validate all-pairs...
past three, fold to Other or facet"), confirmed again here rather than assumed:
adding a plausible 4th/5th hue (violet, yellow, magenta, green, red) to this
same three-color set was tried and each one fails the all-pairs floor in at
least one of light/dark mode. So every chart in this project, including the
new cascade one, is built to show at most these three categorical hues at
once and never introduces a fourth - see `draw_cascade` for how the cascade
chart's three "kinds" reuse them.
"""

from __future__ import annotations

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a86"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2dc"

COLOR = {"baseline": GRAY, "local-fp32": ORANGE, "local-int8": AQUA, "hosted": BLUE}
KIND_LABEL = {
    "baseline": "keyword baseline",
    "local-fp32": "local model, fp32",
    "local-int8": "local model, int8",
    "hosted": "hosted (Jev)",
}


def plot_frontier(
    points: list[tuple],
    title: str,
    subtitle: str,
    out_path,
    curve_kind: str | None = None,
    colors: dict[str, str] | None = None,
    kind_labels: dict[str, str] | None = None,
) -> None:
    """Accuracy vs. p50-latency scatter, cost as marker area (sqrt-scaled).

    ``points``: ``(label, kind, accuracy, p50_ms, cost_per_million_usd,
    label_anchor)``. ``label_anchor`` is either an ``(ha, va, dx, dy)`` tuple
    that hand-places one annotation next to the point - the convention the
    original 4-point comparison charts use, where every point gets one and no
    collision avoidance is needed because there are only ever a handful of
    them - or ``None`` to leave the point unlabeled, which is what a sweep of
    more than a few points needs instead (see ``draw_cascade``: only its two
    endpoints get a label_anchor).

    ``colors``/``kind_labels`` extend the module-level ``COLOR``/
    ``KIND_LABEL`` defaults for this call only, so a caller can introduce an
    ad hoc ``kind`` (``draw_cascade``'s local/hosted/curve kinds) without
    registering it globally.

    ``curve_kind``, if given, draws a line through every point of that kind,
    sorted by p50 latency (the x-axis), underneath the scatter markers - so a
    sweep reads as an actual frontier curve rather than a handful of
    unrelated dots.
    """
    import matplotlib.pyplot as plt

    color_map = {**COLOR, **(colors or {})}
    label_map = {**KIND_LABEL, **(kind_labels or {})}

    fig = plt.figure(figsize=(8.2, 5.2), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    # Reserve the top 22% of the figure for title+subtitle as their own
    # fig.text calls, in figure coordinates - so they can never collide with
    # each other or with ax.set_title's internal padding.
    ax = fig.add_axes((0.09, 0.13, 0.87, 0.65))
    ax.set_facecolor("#fcfcfb")

    fig.text(0.045, 0.95, title, color=INK, fontsize=13, fontweight="bold", va="top")
    fig.text(0.045, 0.885, subtitle, color=MUTED, fontsize=9, va="top")

    # cost -> marker area, sqrt-scaled so area (not radius) tracks cost
    costs = [row[4] for row in points]
    lo, hi = min(costs), max(costs)

    def size(c: float) -> float:
        if hi <= lo:
            return 260.0
        t = ((c - lo) / (hi - lo)) ** 0.5
        return 90.0 + t * 520.0

    if curve_kind is not None:
        curve_points = sorted(
            (row for row in points if row[1] == curve_kind), key=lambda row: row[3]
        )
        if len(curve_points) > 1:
            ax.plot(
                [row[3] for row in curve_points],
                [row[2] for row in curve_points],
                color=color_map[curve_kind],
                linewidth=1.6,
                alpha=0.75,
                zorder=2,
            )

    seen_kinds: list[str] = []
    for label, kind, acc, p50, cost, anchor in points:
        color = color_map[kind]
        ax.scatter(
            [p50], [acc], s=[size(cost)], color=color, alpha=0.85,
            edgecolors="white", linewidths=1.2, zorder=3,
        )
        if anchor is not None:
            ha, va, dx, dy = anchor
            ax.annotate(
                f"{label}\nacc {acc:.0%} · ${cost:g}/1M",
                (p50, acc), xytext=(dx, dy), textcoords="offset points",
                fontsize=8.3, color=INK, va=va, ha=ha, fontfamily="monospace",
            )
        if kind not in seen_kinds:
            seen_kinds.append(kind)

    ax.set_xscale("log")
    ax.set_xlabel(
        "p50 latency, ms (log scale) — further right is slower",
        color=MUTED, fontsize=9,
    )
    ax.set_ylabel("accuracy", color=MUTED, fontsize=9)
    ax.set_ylim(-0.02, 1.08)
    xs = [row[3] for row in points]
    ax.set_xlim(min(xs) * 0.2, max(xs) * 6)

    ax.grid(True, which="both", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)

    handles = [
        plt.Line2D(
            [0], [0], marker="o", linestyle="", color=color_map[k],
            markersize=8, label=label_map.get(k, k),
        )
        for k in seen_kinds
    ]
    ax.legend(
        handles=handles, loc="upper left", frameon=False, fontsize=8.3,
        labelcolor=INK, handletextpad=0.5,
    )
    fig.text(
        0.955, 0.035,
        "marker area ∝ cost per million calls — bigger circle costs more",
        ha="right", fontsize=7.6, color=MUTED, style="italic",
    )

    fig.savefig(
        out_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.15
    )
    plt.close(fig)


def draw_cascade(
    local_label: str,
    local_acc: float,
    local_p50: float,
    local_cost: float,
    hosted_label: str,
    hosted_acc: float,
    hosted_p50: float,
    hosted_cost: float,
    curve: list[tuple[float, float, float]],
    title: str,
    subtitle: str,
    out_path,
) -> None:
    """The cost-accuracy sweep chart for ``edgefront cascade --plot``.

    Only three "kinds" ever appear on this chart at once: the hosted-only
    anchor, the local-only anchor, and the swept cascade curve between them -
    so it reuses the module's three already-validated categorical hues
    (``BLUE``/``ORANGE``/``AQUA``) rather than adding a fourth (see this
    module's docstring for why a fourth does not validate). Hosted keeps its
    usual meaning (``BLUE``, matching the README's comparison charts); local
    gets ``ORANGE``; the curve reuses the one remaining validated slot,
    ``AQUA`` - purely as a color to borrow for a line, not because the curve
    "is" an int8 local model in the `docs/make_charts.py` sense. Local,
    hosted and curve are given their own ad hoc kind names
    (``cascade-local``/``cascade-hosted``/``cascade-curve``) via
    ``plot_frontier``'s per-call ``colors``/``kind_labels`` override, so this
    reuse never touches the module-level ``COLOR``/``KIND_LABEL`` dicts that
    the unrelated comparison charts also read.

    ``curve``: ``(accuracy, p50_ms, cost_per_million_usd)`` per swept
    threshold, any order - re-sorted by latency before drawing. Unlike the
    4-point comparison charts, a sweep can have many points, so the
    hand-anchored label convention does not scale (its own comment says so
    explicitly) - every curve point except the two endpoints is drawn
    unlabeled, which is why only ``local_label``/``hosted_label`` (not the
    curve) take a name at all.
    """
    points = [
        (
            local_label, "cascade-local", local_acc, local_p50, local_cost,
            ("right", "top", -12, -10),
        ),
        (
            hosted_label, "cascade-hosted", hosted_acc, hosted_p50, hosted_cost,
            ("left", "center", 12, 0),
        ),
    ]
    for i, (acc, p50, cost) in enumerate(curve):
        points.append((f"_sweep_{i}", "cascade-curve", acc, p50, cost, None))

    plot_frontier(
        points,
        title,
        subtitle,
        out_path,
        curve_kind="cascade-curve",
        colors={
            "cascade-local": ORANGE,
            "cascade-hosted": BLUE,
            "cascade-curve": AQUA,
        },
        kind_labels={
            "cascade-local": "local only",
            "cascade-hosted": "hosted only",
            "cascade-curve": "cascade sweep (threshold varies)",
        },
    )
