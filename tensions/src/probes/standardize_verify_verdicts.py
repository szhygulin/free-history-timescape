#!/usr/bin/env python3
"""Standardize the paper-3 verify_*.json artifacts onto ONE top-level verdict key.

The verify_*.json artifacts historically carry their overall verdict under
inconsistent keys ("overall" here, "verdict" there; sometimes a bare string,
sometimes a nested dict). This adds a uniform top-level string key
`verdict_of_verification` to EACH artifact, set to that artifact's existing overall
verdict token (SURVIVES / SURVIVES_WITH_CAVEATS / ...), PRESERVING all existing keys.

Extraction rule (per file):
  * if a top-level "overall" or "verdict" value is a bare STRING, copy it verbatim;
  * else (the token lives in a nested dict) use the explicit VERDICT_OVERRIDE below,
    each derived by inspection of that artifact's own verdict block.

Idempotent: re-running re-derives the same token from the SAME source fields
(never from verdict_of_verification itself) and rewrites the same value.
One script -> N artifacts (a light JSON merge, no numbers recomputed).
"""
import os
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.dirname(_SRC)
_POUT = os.path.join(_REPO, "probes_out")

FILES = [
    "verify_bpred.json",
    "verify_bpred_survey.json",
    "verify_wpb_integrability.json",
    "verify_hb_catalog_forced.json",
    "verify_wpc_sound_horizon.json",
]

# Explicit tokens for artifacts whose overall verdict is a NESTED dict (no bare
# top-level string to copy). Each token reflects that artifact's own verdict block:
#   verify_bpred.json         overall.per_check all PASS + prose "The COMPUTATION
#                             survives" but flagged "HONEST but exploitable / fragile"
#                             -> survives WITH framing caveats.
#   verify_wpb_integrability  verdict.all_checks_pass=true with one disclosed,
#                             documented gap (the 81% figure) -> survives WITH a caveat.
VERDICT_OVERRIDE = {
    "verify_bpred.json": "SURVIVES_WITH_CAVEATS",
    "verify_wpb_integrability.json": "SURVIVES_WITH_CAVEATS",
}


def extract_token(d, fname):
    for key in ("overall", "verdict"):
        v = d.get(key)
        if isinstance(v, str):
            return v
    if fname in VERDICT_OVERRIDE:
        return VERDICT_OVERRIDE[fname]
    raise ValueError(f"{fname}: no top-level string verdict and no override")


def main():
    for fname in FILES:
        path = os.path.join(_POUT, fname)
        with open(path) as f:
            d = json.load(f)
        token = extract_token(d, fname)
        d["verdict_of_verification"] = token
        with open(path, "w") as f:
            json.dump(d, f, indent=1)
        print(f"  {fname:34s} verdict_of_verification = {token}")


if __name__ == "__main__":
    main()
