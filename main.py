"""Forecast GTA VI's launch-day Twitch viewers hour by hour.

Shape model, fitted at run time from the case's own comparables.

Idea
----
Every big launch's Twitch curve has the same skeleton: hour 1 is already a
real launch print (some regions unlocked at rolling midnight), viewers dip
in the small hours, then climb as the US unlocks (05:00/08:00 UTC) into an
evening peak around hours 19-22 UTC, and fade overnight.  So for each hour
h = 2..24 we take the comparables' log-ratios ln(v_h / v_1), and forecast

    q_tau(h) = p1 * exp(median_h + z_tau * ALPHA * sd_h)

where p1 is the case's hour-1 print, median_h / sd_h are the median and the
population standard deviation of the log-ratios across the comparables that
survive the filter below, and z_tau is the standard normal quantile at each
scored level.  The median is the shape; the spread is the honest
cross-launch dispersion, so a launch that behaves like a typical comparable
lands inside the 90% band about nine hours in ten, and a flatter or steeper
launch still lands inside the wide tails.

Which comparables count
------------------------
Not every past launch is a rolling-midnight launch like GTA VI, and the
wrong ones poison the shape.  A launch whose hour-1 print is nearly zero
because it unlocked at one single global time (e.g. 10:00 UTC+8, or 14:00
UTC) has ratios of 100x-2000x by the evening; a launch with 72 h of paid
early access has its peak before release day even starts.  Both are a
different animal from a rolling-midnight console launch, where hour 1
already holds New Zealand / Australia / Japan / the UK.  So a comparable
counts only when it is a clean rolling-midnight analogue:

  * its unlock note carries "Tier: comparable" (the kit's own label), and
  * its release-day peak is at most 5x its hour-1 print -- hour 1 was
    already a substantial launch print, not a pre-unlock trickle.

The 5x cut is a property of the comparable's own fully-known curve, never
of the case being forecast.  Leave-one-out validation on the kit's 38
comparables (each one predicted from the rest) gives this filtered model a
mean pinball loss of ~0.23, against ~0.27 for the house persistence
baseline on the same launches, and ~0.93 for the baseline when the
unfiltered pool's simultaneous-unlock outliers are allowed to dominate.
The two dropped tiers are early-access-contaminated launches and weak /
free-to-play launches; the case's own game (the GTA VI pre-release
category) has no release-day hour 1 and is excluded by construction.

Fallbacks, in order, if the filter leaves too few comparables (< 8):
relax the peak cut to 12x, then keep tier "comparable" only, then use
every comparable with a usable hour 1, then persistence at hour 1 with
the house spread.  The program never crashes on a thin case: a missing
or zero hour is skipped for that hour, and an hour with no usable
comparable at all falls back to persistence.

Runtime: standard library only, deterministic (sorted inputs, no
randomness, no wall-clock use), well inside the 8 s cap.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import.  This entry is a single file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Spread calibration: the raw cross-launch sd is slightly too narrow for a
# proper pinball score (a too-narrow band pays at every hour outside it),
# so the log-spread is widened by this factor.  Tuned on leave-one-out
# pinball over the kit's comparables (1.0 -> 0.229, 1.15 -> 0.229,
# 1.2 -> 0.229; 1.15 sits on the flat optimum, not on a knife edge).
ALPHA = 1.15
# The per-hour log-sd is floored and capped: with only ~15 comparables a
# single quiet hour must not collapse the band, nor a noisy one explode it.
SD_FLOOR = 0.20
SD_CAP = 0.65
# A comparable counts only if its day peak is at most this many times its
# hour-1 print (see the module docstring for why).
PEAK_RATIO_MAX = 5.0
PEAK_RATIO_RELAXED = 12.0
MIN_POOL = 8
MIN_HOUR1 = 1000.0

_TIER_RE = re.compile(r"Tier:\s*([a-z][a-z\-]*)")


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as the starter rounds it."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def _parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _hour_of(day_start: datetime, when: str) -> int | None:
    try:
        delta = _parse_time(when) - day_start
    except (TypeError, ValueError):
        return None
    hour = int(delta.total_seconds() // 3600) + 1
    return hour if 1 <= hour <= 24 else None


def _day_curve(comp: dict) -> dict[int, float]:
    """Release-day hourly viewers {hour: viewers} for one comparable."""
    try:
        day_start = _parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    hours: dict[int, float] = {}
    for row in comp.get("hourly") or []:
        hour = _hour_of(day_start, row.get("from", ""))
        if hour is None:
            continue
        try:
            viewers = float(row["viewers"])
        except (KeyError, TypeError, ValueError):
            continue
        if viewers > 0:
            hours[hour] = viewers
    return hours


def _tier(comp: dict) -> str:
    match = _TIER_RE.search(comp.get("unlock", "") or "")
    return match.group(1) if match else "unknown"


def _comparable_pool(case: dict, game_name: str) -> list[dict[int, float]]:
    """Filtered release-day curves, with the fallback ladder of the docstring."""
    curves: list[tuple[dict[int, float], str, float]] = []
    for comp in case.get("comparables") or []:
        if comp.get("name") == game_name:
            continue
        hours = _day_curve(comp)
        hour1 = hours.get(1)
        if not hour1 or hour1 < MIN_HOUR1:
            continue
        peak = max(hours.values())
        curves.append((hours, _tier(comp), peak / hour1))

    def _pick(tier: str | None, peak_max: float | None) -> list[dict[int, float]]:
        out = []
        for hours, t, ratio in curves:
            if tier is not None and t != tier:
                continue
            if peak_max is not None and ratio > peak_max:
                continue
            out.append(hours)
        return out

    pool = _pick("comparable", PEAK_RATIO_MAX)
    if len(pool) < MIN_POOL:
        pool = _pick("comparable", PEAK_RATIO_RELAXED)
    if len(pool) < MIN_POOL:
        pool = _pick("comparable", None)
    if len(pool) < MIN_POOL:
        pool = _pick(None, None)
    return pool


def _quantile(sorted_vals: list[float], p: float) -> float:
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    idx = p * (n - 1)
    lo = int(math.floor(idx))
    hi = min(n - 1, lo + 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _median_and_sd(log_ratios: list[float]) -> tuple[float, float]:
    ordered = sorted(log_ratios)
    median = _quantile(ordered, 0.5)
    mean = sum(ordered) / len(ordered)
    var = sum((v - mean) ** 2 for v in ordered) / len(ordered)
    sd = math.sqrt(var)
    return median, min(SD_CAP, max(SD_FLOOR, sd))


def _smooth_medians(medians: dict[int, float]) -> dict[int, float]:
    """Light 3-hour smoothing (0.25/0.5/0.25) of the median log-ratios.

    The per-hour medians come from only ~15 comparables, so a single
    hour's median can jump on sampling noise; launch curves themselves
    are smooth in log space.  Smoothing is neutral on leave-one-out
    pinball (0.2289 vs 0.2290 unsmoothed) and removes jagged steps
    between neighbouring hours.
    """
    out: dict[int, float] = {}
    for hour in range(2, 25):
        if hour not in medians:
            continue
        total = 0.0
        weight_sum = 0.0
        for neighbour, weight in ((hour - 1, 0.25), (hour, 0.5), (hour + 1, 0.25)):
            if neighbour in medians:
                total += weight * medians[neighbour]
                weight_sum += weight
        out[hour] = total / weight_sum if weight_sum else medians[hour]
    return out


def _reads_extra_sd(reads: int | None) -> float:
    """Extra log-sd for a noisy hour-1 print, added in quadrature.

    The final's hour-1 print is the mean of ~60 minute reads, so it is
    precise and adds nothing.  A practice case's print rests on a single
    SullyGnome point (reads = 1), and scaling every hour by a noisy p1
    shifts the whole forecast; 0.10 log units covers that.  Between the
    two, 0.05.
    """
    if reads is None or reads >= 30:
        return 0.0
    if reads >= 10:
        return 0.05
    return 0.10


def _hour_quantiles(p1: float, median_log: float, sd: float) -> list[float | int]:
    out: list[float | int] = []
    prev = 0.0
    for z in Z:
        value = round2(p1 * math.exp(median_log + z * ALPHA * sd))
        value = max(value, prev)
        out.append(value)
        prev = float(value)
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    try:
        reads = int(first.get("reads", 0) or 0)
    except (TypeError, ValueError):
        reads = 0
    extra_sd = _reads_extra_sd(reads)
    game_name = (case.get("game") or {}).get("name", "")

    pool = _comparable_pool(case, game_name)

    # Per-hour log-ratios across the pool, then median + sd per hour.
    medians: dict[int, float] = {}
    sds: dict[int, float] = {}
    for hour in range(2, 25):
        logs = [
            math.log(hours[hour] / hours[1])
            for hours in pool
            if hour in hours and 1 in hours
        ]
        if logs:
            medians[hour], sds[hour] = _median_and_sd(logs)
    medians = _smooth_medians(medians)
    stats = {
        hour: (medians[hour], math.sqrt(sds[hour] ** 2 + extra_sd ** 2))
        for hour in medians
    }

    hours_out = []
    for row in case["forecast"]:
        hour = int(row["hour"])
        if hour in stats:
            median_log, sd = stats[hour]
        else:
            # No comparable covers this hour: persistence with exactly
            # the house spread (sd chosen so ALPHA * sd = 0.15*sqrt(h-1)).
            median_log = 0.0
            sd = 0.15 * math.sqrt(hour - 1) / ALPHA
            sd = min(SD_CAP, max(SD_FLOOR, sd))
        hours_out.append({"hour": hour, "q": _hour_quantiles(p1, median_log, sd)})
    hours_out.sort(key=lambda r: r["hour"])
    return {"hours": hours_out}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
