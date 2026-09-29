"""
mock_models.py - FAKE data for design work. Only used when MOCK_MODELS=1.

Nothing in here is a real prediction. It lets us build and preview the website
on a laptop that has no TensorFlow / PyTorch and no trained model files.

It provides three things:
  1. make_samples()  - 20 made-up "test transformers" (gas readings + true answers)
  2. fake_analyse()  - a pretend model that returns the same kind of answer as the real one
  3. a few constants (threshold, error radius, cap) that the real app reads from artifacts/
"""
import hashlib
import numpy as np

GAS_COLUMNS = ["H2", "CO", "C2H4", "C2H2"]
N_ROWS = 420                 # one reading every 12 hours, about 210 days

# Fake settings. In real mode these come from the model files, never from here.
FDD_THRESHOLD = 0.70         # below this confidence -> "needs manual review"
RUL_RADIUS = 95.0            # +/- range in days
RUL_CAP_DAYS = 500.0         # longest remaining life seen in (fake) training data

# --------------------------------------------------------------------------
# 1. Fake gas curves. Each fault class gets a different "signature":
#    (start value, end value, how sharply it curves upward) for H2, CO, C2H4, C2H2
# --------------------------------------------------------------------------
SHAPES = {
    1: [(40, 55, 1), (200, 230, 1), (12, 16, 1), (0.5, 1.0, 1)],     # Normal: flat and low
    2: [(60, 520, 2.5), (210, 260, 1), (12, 25, 1), (0.5, 1.5, 1)],  # Partial discharge: H2 climbs
    3: [(50, 160, 2), (200, 300, 1), (15, 40, 1), (1, 70, 3)],       # Low-energy discharge: C2H2 climbs late
    4: [(50, 120, 1.5), (220, 520, 2), (14, 260, 2.5), (0.5, 3, 1)], # Low-temp overheating: C2H4 climbs
}

# Typical remaining life (days) per predicted class, used by the fake model
RUL_RANGES = {1: (280, 540), 2: (120, 300), 3: (60, 200), 4: (40, 160)}


def make_curves(fault_class, rng):
    """Return a (420, 4) array of fake gas readings in the style of `fault_class`."""
    t = np.linspace(0, 1, N_ROWS)
    columns = []
    for start, end, power in SHAPES[fault_class]:
        curve = start + (end - start) * t ** power
        noise = rng.normal(0, 0.03 * max(start, end), N_ROWS)
        columns.append(np.clip(curve + noise, 0, None))
    # Round to 3 decimals so a CSV download and re-upload gives exactly the same numbers
    return np.round(np.column_stack(columns), 3)


# --------------------------------------------------------------------------
# 2. The pretend model
# --------------------------------------------------------------------------
_SAMPLE_PLAN = {}   # fingerprint of a sample's readings -> (class the fake model will say, confidence)


def _fingerprint(raw):
    return hashlib.sha256(np.ascontiguousarray(raw, dtype=float).tobytes()).hexdigest()


def _guess_class(raw):
    """Crude rule for uploaded files: look at which gas is high in the last 30 days."""
    h2, co, c2h4, c2h2 = raw[-60:].mean(axis=0)
    if c2h2 > 25:
        return 3
    if c2h4 > 100:
        return 4
    if h2 > 250:
        return 2
    return 1


def fake_analyse(raw):
    """Same output shape as analyse() in the original app.py:
    {'fdd_class', 'confidence', 'probs', 'rul_days'}"""
    key = _fingerprint(raw)
    rng = np.random.default_rng(int(key[:8], 16))   # same input -> same fake answer, every time
    if key in _SAMPLE_PLAN:
        fdd_class, confidence = _SAMPLE_PLAN[key]
    else:
        fdd_class, confidence = _guess_class(raw), float(rng.uniform(0.60, 0.99))

    # Spread the leftover probability over the other three classes
    others = [c for c in (1, 2, 3, 4) if c != fdd_class]
    leftover = rng.dirichlet(np.ones(3)) * (1 - confidence)
    probs = {fdd_class: confidence, **{c: float(p) for c, p in zip(others, leftover)}}

    low, high = RUL_RANGES[fdd_class]
    return {"fdd_class": fdd_class, "confidence": confidence,
            "probs": dict(sorted(probs.items())), "rul_days": float(rng.uniform(low, high))}


# --------------------------------------------------------------------------
# 3. Twenty fake test transformers
#    (true class, class the fake model predicts, forced confidence or None)
#    Two of them are deliberately wrong AND low-confidence, so the
#    "needs manual review" warning can be seen in the design.
# --------------------------------------------------------------------------
SAMPLE_PLAN = [
    (1, 1, None), (1, 1, None), (4, 4, None), (1, 1, None), (2, 2, None),
    (1, 1, None), (3, 2, 0.52), (1, 1, None), (4, 4, None), (1, 1, None),
    (2, 2, None), (1, 1, None), (1, 1, None), (3, 3, 0.58), (1, 1, None),
    (4, 4, None), (1, 1, None), (2, 2, None), (1, 1, None), (3, 3, None),
]


def make_samples():
    """Return {sample_id: {'raw', 'true_fdd', 'true_rul_days'}} for the 20 fake transformers."""
    rng = np.random.default_rng(2026)
    samples = {}
    for i, (true_cls, pred_cls, forced_conf) in enumerate(SAMPLE_PLAN, start=1):
        sample_id = f"transformer_{i:02d}.csv"
        raw = make_curves(true_cls, rng)
        confidence = forced_conf if forced_conf else float(rng.uniform(0.85, 0.99))
        _SAMPLE_PLAN[_fingerprint(raw)] = (pred_cls, confidence)

        # "True" remaining life = the fake prediction plus some random error
        predicted_rul = fake_analyse(raw)["rul_days"]
        true_rul = max(0.0, predicted_rul + float(rng.normal(0, 45)))
        samples[sample_id] = {"raw": raw, "true_fdd": true_cls, "true_rul_days": true_rul}
    return samples
