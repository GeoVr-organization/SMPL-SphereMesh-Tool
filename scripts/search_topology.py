"""Search the design parameters of ``build_topology`` instead of hand-tuning them.

    .venv/Scripts/python.exe scripts/search_topology.py --rounds 2 --candidates 24
    .venv/Scripts/python.exe scripts/search_topology.py --workers 6 --out data/generated/search

The handoff's single biggest regret was using the validator as a report rather than as
an objective: two thirds of that session went into edit-a-constant, re-run, interpret.
This script closes the loop. It defines one scalar score over the validator's numbers,
evaluates a population of ``Design`` variants in parallel processes (quick pipeline: no
poses, short fits), keeps the best, and repeats with the population re-centred on it.
Every evaluation is logged to a JSON lines file so the argument for every constant is on
disk, and the winner is written as a design file ``build_topology.py --design`` accepts.

The score is deliberately simple and stated here rather than hidden:

    score = 1.0 * gap_fraction_pct        (% of vertices further than 15 mm off)
          + 0.6 * max_gap_mm / 10
          + 1.0 * protrusion_fraction_pct (% of vertices inside by more than 5 mm)
          + 0.8 * max_protrusion_mm / 10
          + 1.5 * walk_break_pct
          + 0.4 * rms_mm
          + 0.05 * spheres

Lower is better. Weights express the brief: protrusion and walk breaks are the things the
solver pays for; gaps are visible; sphere count is a mild tie-breaker under the budget.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_topology import Design, build  # noqa: E402

#: Parameter -> (kind, low, high). "scalar" perturbs one float; "tuple_shift" shifts every
#: entry of a tuple by the same amount; "tuple_scale" scales the spacing about the mean.
SEARCH_SPACE: dict[str, tuple[str, float, float]] = {
    "protrusion_weight": ("scalar", 3.0, 12.0),
    "max_offset": ("scalar", 0.006, 0.020),
    "lower_rows": ("tuple_shift", -0.03, 0.03),
    "lower_rows_spacing": ("tuple_scale", 0.85, 1.15),
    "upper_rows": ("tuple_shift", -0.02, 0.02),
    "lower_columns": ("column", 0.5, 0.9),
    "upper_columns": ("column", 0.55, 0.9),
    "deltoid": ("scalar", 0.55, 0.85),
    "clavicle": ("scalar", 0.10, 0.35),
    "thigh": ("tuple_shift", -0.03, 0.03),
    "shin": ("tuple_shift", -0.03, 0.03),
    "upper_arm": ("tuple_shift", -0.03, 0.03),
    "forearm": ("tuple_shift", -0.03, 0.03),
    "gap_threshold": ("scalar", 0.010, 0.015),
}


def score_of(stats: dict, sphere_count: int) -> float:
    rest = stats["rest"]
    walk = stats["walk"]
    return (
        1.0 * rest["gap_fraction"] * 100.0
        + 0.6 * rest["max_gap"] * 100.0
        + 1.0 * rest["protrusion_fraction"] * 100.0
        + 0.8 * rest["max_protrusion"] * 100.0
        + 1.5 * walk["fraction"] * 100.0
        + 0.4 * rest["rms"] * 1000.0
        + 0.05 * sphere_count
    )


def perturb(base: Design, rng: np.random.Generator, strength: float) -> Design:
    """One random neighbour of ``base``; ``strength`` in (0, 1] scales every step."""
    values = base.to_dict()
    chosen = rng.choice(list(SEARCH_SPACE), size=int(rng.integers(1, 4)), replace=False)
    for key in chosen:
        kind, low, high = SEARCH_SPACE[key]
        if kind == "scalar":
            current = float(values[key])
            step = (high - low) * strength * rng.normal()
            values[key] = float(np.clip(current + step, low, high))
        elif kind == "tuple_shift":
            shift = (high - low) * 0.5 * strength * rng.normal()
            values[key] = tuple(float(entry + shift) for entry in values[key])
        elif kind == "tuple_scale":
            target = key.replace("_spacing", "")
            entries = np.asarray(values[target], dtype=float)
            factor = float(np.clip(1.0 + (high - low) * 0.5 * strength * rng.normal(), low, high))
            values[target] = tuple((entries.mean() + (entries - entries.mean()) * factor).tolist())
        elif kind == "column":
            entries = list(values[key])
            entries[-1] = float(np.clip(entries[-1] + (high - low) * strength * rng.normal(), low, high))
            values[key] = tuple(entries)
    # Station parameters must stay inside (0, 1) and away from the joints.
    for key in ("thigh", "shin", "upper_arm", "forearm"):
        values[key] = tuple(float(np.clip(entry, 0.04, 0.96)) for entry in values[key])
    return Design.from_dict(values)


def evaluate(payload: tuple[int, dict, str]) -> dict:
    """Worker: build with the quick pipeline and return the score and key numbers."""
    index, design_values, gender = payload
    import torch

    torch.set_num_threads(2)
    from src.topology.anatomy import load_body

    started = time.perf_counter()
    try:
        body = load_body(gender)
        result = build(body, Design.from_dict(design_values), poses=None, quick=True, verbose=False)
        stats = result.report.stats
        return {
            "index": index, "design": design_values, "ok": True,
            "score": score_of(stats, len(result.centers)),
            "spheres": int(len(result.centers)),
            "rest": stats["rest"], "walk_fraction": stats["walk"]["fraction"],
            "passed": int(sum(1 for check in result.report.checks if check.passed)),
            "checks": int(len(result.report.checks)),
            "seconds": round(time.perf_counter() - started, 1),
        }
    except Exception as error:  # noqa: BLE001 - a failed candidate is a data point
        return {
            "index": index, "design": design_values, "ok": False,
            "error": repr(error), "score": float("inf"),
            "seconds": round(time.perf_counter() - started, 1),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gender", default="male")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=16, help="per round")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strength", type=float, default=0.6,
                        help="perturbation size in round 1; halves every round")
    parser.add_argument("--start", type=Path, default=None, help="design JSON to start from")
    parser.add_argument("--out", type=Path, default=Path("data/generated/search"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "evaluations.jsonl"
    rng = np.random.default_rng(args.seed)
    base = Design() if args.start is None else Design.from_dict(
        json.loads(args.start.read_text(encoding="utf-8"))
    )

    best: dict | None = None
    strength = args.strength
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for round_index in range(args.rounds):
            candidates = [base.to_dict()] if round_index == 0 else []
            while len(candidates) < args.candidates:
                candidates.append(perturb(base, rng, strength).to_dict())
            jobs = [(round_index * 1000 + index, values, args.gender)
                    for index, values in enumerate(candidates)]
            print(f"round {round_index}: {len(jobs)} candidates, strength {strength:.2f}")
            started = time.perf_counter()
            with open(log_path, "a", encoding="utf-8") as handle:
                for outcome in pool.map(evaluate, jobs):
                    handle.write(json.dumps(outcome, default=float) + "\n")
                    handle.flush()
                    if outcome["ok"]:
                        rest = outcome["rest"]
                        print(
                            f"  #{outcome['index']:<5} score {outcome['score']:7.2f}  "
                            f"gap>15mm {rest['gap_fraction'] * 100:5.2f}%  "
                            f"maxgap {rest['max_gap'] * 1000:5.1f}  "
                            f"protr {rest['max_protrusion'] * 1000:5.1f} "
                            f"({rest['protrusion_fraction'] * 100:4.2f}%)  "
                            f"walk {outcome['walk_fraction'] * 100:4.2f}%  "
                            f"rms {rest['rms'] * 1000:4.1f}  n={outcome['spheres']}  "
                            f"{outcome['seconds']:.0f}s"
                        )
                    else:
                        print(f"  #{outcome['index']:<5} FAILED {outcome['error'][:100]}")
                    if best is None or outcome["score"] < best["score"]:
                        best = outcome
            print(f"  round took {time.perf_counter() - started:.0f} s; best score {best['score']:.2f}")
            base = Design.from_dict(best["design"])
            # Persist only the parameters the search actually varied, so the file composes
            # with the current defaults of everything else. A full snapshot silently froze
            # fit settings (the pose-term weight) that were later changed by default, and
            # every run made from it inherited the stale values.
            defaults = Design().to_dict()
            searched = {
                key: value for key, value in best["design"].items()
                if value != defaults.get(key)
            }
            (args.out / "best_design.json").write_text(
                json.dumps(searched, indent=1), encoding="utf-8"
            )
            strength *= 0.5
    print(f"best: {json.dumps(best, default=float)[:600]}")
    print(f"wrote {args.out / 'best_design.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
