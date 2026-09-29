"""
Transformer Health Check - web server.

What this file does, in plain words:
  1. Starts a small web server (FastAPI + uvicorn) on the port Azure gives us.
  2. Loads the two models in the BACKGROUND, so the server answers immediately
     ("waking up") instead of Azure killing us during the 1-2 minute model load.
  3. Offers a JSON API (/api/...) that the web page in static/ talks to.
  4. Serves the web page itself (static/index.html and friends).

Run locally with fake results (no TensorFlow/PyTorch needed):
    MOCK_MODELS=1 python app.py          (Windows PowerShell:  $env:MOCK_MODELS=1; python app.py)
Run for real (Azure does this):
    python app.py
"""
import asyncio
import io
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"

MOCK = os.environ.get("MOCK_MODELS", "").strip().lower() in ("1", "true", "yes")
MAX_UPLOAD_BYTES = 2 * 1024 * 1024        # uploads bigger than 2 MB are refused
STEP_TO_DAYS = 0.5                        # 1 reading every 12 hours (same as the original app)

# Same class names as the original app
FDD_CLASS_NAMES = {1: "Normal", 2: "Partial discharge", 3: "Low-energy discharge",
                   4: "Low-temperature overheating"}

# Test-set results shown on the "Model performance" page.
# TODO(Step 3): the student must confirm these numbers before they are displayed.
PERFORMANCE = {
    "fdd": {"accuracy": 0.95, "n_test": 900, "weakest_class": 4, "n_flagged_for_review": 45},
    "rul": {"mae_days": 40.3, "rmse_days": 58.0, "r2": 0.76, "coverage_within_range": 0.89},
}


class InputError(Exception):
    """The uploaded file is not usable. The message is a friendly sentence shown to the user."""


# =====================================================================
# The "backend": everything that differs between mock and real mode.
# Both modes fill in the same fields, so the API code below doesn't care which one it got.
#   gas_columns, n_rows          what the CSV must contain
#   threshold, radius, cap_days  FDD review threshold, RUL +/- range, longest life in training
#   validate(df, name) -> array  checks a parsed CSV, returns (n_rows, 4) numbers or raises InputError
#   analyse(raw) -> dict         {'fdd_class', 'confidence', 'probs', 'rul_days'}  (as in the original app)
#   sample_ids                   list of the 20 test transformer file names
#   sample_raw(id), sample_csv(id), true_fdd(id), true_rul_days(id)
# =====================================================================
backend = None                            # filled in by the background loader
state = {"ready": False, "error": None}   # what /api/health reports
MODEL_LOCK = threading.Lock()             # only one prediction touches the models at a time


def load_mock():
    """Fake backend: no models, made-up but realistic answers."""
    import mock_models as mm
    time.sleep(float(os.environ.get("MOCK_LOAD_SECONDS", "4")))   # pretend loading, so the "waking up" screen can be seen
    samples = mm.make_samples()

    def validate(df, name):
        return validate_readings(df, mm.GAS_COLUMNS, mm.N_ROWS)

    def sample_csv(sample_id):
        return pd.DataFrame(samples[sample_id]["raw"], columns=mm.GAS_COLUMNS).to_csv(index=False).encode()

    return SimpleNamespace(
        gas_columns=mm.GAS_COLUMNS, n_rows=mm.N_ROWS,
        threshold=mm.FDD_THRESHOLD, radius=mm.RUL_RADIUS, cap_days=mm.RUL_CAP_DAYS,
        validate=validate, analyse=mm.fake_analyse,
        sample_ids=list(samples),
        sample_raw=lambda i: samples[i]["raw"], sample_csv=sample_csv,
        true_fdd=lambda i: samples[i]["true_fdd"],
        true_rul_days=lambda i: samples[i]["true_rul_days"],
    )


def load_real():
    """Real backend: the PyTorch + Keras models. Comes in Step 4."""
    raise RuntimeError("Real mode is not wired up yet. Start with MOCK_MODELS=1.")


def load_backend_in_background():
    """Runs in a separate thread so the web server can answer requests while models load."""
    global backend
    try:
        backend = load_mock() if MOCK else load_real()
        state["ready"] = True
    except Exception as e:                # show the problem on /api/health instead of crashing silently
        state["error"] = f"{type(e).__name__}: {e}"


# =====================================================================
# Reading and checking an uploaded CSV
# =====================================================================
def parse_csv(data: bytes) -> pd.DataFrame:
    """Bytes -> table of text cells (dtype=str, like the original app, so we can check values ourselves)."""
    try:
        return pd.read_csv(io.BytesIO(data), dtype=str, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        raise InputError("The file is empty. Please upload a CSV with the gas readings.")
    except (pd.errors.ParserError, ValueError):
        raise InputError("The file could not be read as a CSV. Please export it as a plain "
                         "comma-separated .csv file with a header row.")


def validate_readings(df, gas_columns, n_rows):
    """Checks used in MOCK mode. (Real mode uses the project's own tx_preprocessing checks.)
    Returns an (n_rows, 4) float array, or raises InputError with a friendly reason."""
    needed = ", ".join(gas_columns)
    missing = [g for g in gas_columns if g not in df.columns]
    if missing:
        raise InputError(f"The file is missing the gas column(s): {', '.join(missing)}. "
                         f"The CSV needs a header row with these columns: {needed}.")
    if len(df) != n_rows:
        raise InputError(f"The file has {len(df)} readings, but exactly {n_rows} are needed "
                         f"(one reading every 12 hours, about {n_rows // 2} days).")
    values = {}
    for gas in gas_columns:
        numbers = pd.to_numeric(df[gas], errors="coerce")
        bad = numbers.isna() | np.isinf(numbers)
        if bad.any():
            i = int(bad.idxmax())                       # first bad row
            what = "is empty" if pd.isna(df[gas][i]) else f"is not a number ('{df[gas][i]}')"
            raise InputError(f"Column {gas} has a value that {what} on line {i + 2} of the file "
                             f"(line 1 is the header). Every reading must be a number.")
        if (numbers < 0).any():
            i = int((numbers < 0).idxmax())
            raise InputError(f"Column {gas} has a negative value ({numbers[i]:g}) on line {i + 2} of the file. "
                             "Gas concentrations cannot be negative.")
        values[gas] = numbers.to_numpy(dtype=float)
    return np.column_stack([values[g] for g in gas_columns])


# =====================================================================
# Turning the model's answer into the JSON the web page needs
# (the numbers and rules below are copied from build_report() in the original app)
# =====================================================================
def build_analysis(raw, result, name, sample_id=None):
    b = backend
    cls = result["fdd_class"]
    days = result["rul_days"]
    analysis = {
        "name": name,
        "sample_id": sample_id,
        "fdd": {
            "class": cls,
            "name": FDD_CLASS_NAMES[cls],
            "confidence": result["confidence"],
            "probabilities": [{"class": c, "name": FDD_CLASS_NAMES[c], "probability": p}
                              for c, p in sorted(result["probs"].items())],
            "needs_review": result["confidence"] < b.threshold,
            "threshold": b.threshold,
        },
        "rul": {
            "days": days,
            "months": days / 30.4,
            "low": max(0.0, days - b.radius),
            "high": days + b.radius,
            "radius": b.radius,
            "near_cap": days >= 0.9 * b.cap_days,     # "at least this long" note
            "cap_days": b.cap_days,
        },
        # x axis = reading number * 0.5 days; values rounded only to keep the JSON small
        "series": {
            "days": (np.arange(raw.shape[0]) * STEP_TO_DAYS).tolist(),
            "gases": {g: np.round(raw[:, k], 4).tolist() for k, g in enumerate(b.gas_columns)},
        },
        "actual": None,
    }
    if sample_id is not None:
        true_cls = int(b.true_fdd(sample_id))
        analysis["actual"] = {"fdd": {"class": true_cls, "name": FDD_CLASS_NAMES[true_cls]},
                              "rul_days": float(b.true_rul_days(sample_id))}
    return analysis


def run_model(raw):
    """Runs the models. The lock means two visitors can never use the models at the same moment."""
    with MODEL_LOCK:
        return backend.analyse(raw)


def analyse_bytes(data: bytes, name: str):
    """CSV bytes -> analysis. Runs in a worker thread (see the endpoints)."""
    raw = backend.validate(parse_csv(data), name)
    return build_analysis(raw, run_model(raw), name)


def require_ready():
    if not state["ready"]:
        detail = ("The models failed to load: " + state["error"]) if state["error"] \
            else "The models are still loading. Please try again in a minute."
        raise HTTPException(status_code=503, detail=detail)


# =====================================================================
# The API
# =====================================================================
@asynccontextmanager
async def lifespan(app):
    # Runs once when the server starts: begin loading the models in the background
    threading.Thread(target=load_backend_in_background, daemon=True).start()
    yield


app = FastAPI(title="Transformer Health Check", docs_url=None, redoc_url=None, lifespan=lifespan)


@app.exception_handler(InputError)
async def input_error_handler(request, exc):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(RequestValidationError)
async def missing_file_handler(request, exc):
    return JSONResponse(status_code=422, content={"detail": "No file was received. Please choose a CSV file."})


@app.get("/api/health")
def health():
    return {"ready": state["ready"], "mode": "mock" if MOCK else "real", "error": state["error"]}


@app.get("/api/samples")
def list_samples():
    require_ready()
    return [{"id": i} for i in backend.sample_ids]


def check_sample_id(sample_id: str):
    """Only ids from our own list are accepted, so nobody can ask for other files on the server."""
    require_ready()
    if sample_id not in backend.sample_ids:
        raise HTTPException(status_code=404, detail="Unknown example transformer.")


@app.get("/api/samples/{sample_id}")
async def sample_analysis(sample_id: str):
    check_sample_id(sample_id)

    def work():
        raw = backend.sample_raw(sample_id)
        return build_analysis(raw, run_model(raw), sample_id, sample_id=sample_id)
    return await asyncio.to_thread(work)         # off the event loop, so the server stays responsive


@app.get("/api/samples/{sample_id}/csv")
def sample_csv(sample_id: str):
    check_sample_id(sample_id)
    return Response(content=backend.sample_csv(sample_id), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{sample_id}"'})


@app.post("/api/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    require_ready()
    too_big = HTTPException(status_code=413, detail="The file is larger than 2 MB. "
                            "A valid file is only a few tens of kilobytes.")
    # Refuse early if the browser announces a huge body, then check again on what we actually read
    if int(request.headers.get("content-length") or 0) > MAX_UPLOAD_BYTES + 10_000:
        raise too_big
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise too_big
    # The upload lives only in this variable: it is never written to disk.
    return await asyncio.to_thread(analyse_bytes, data, file.filename or "uploaded file")


@app.get("/api/model-info")
def model_info():
    require_ready()
    return {
        "mode": "mock" if MOCK else "real",
        "input": {"columns": backend.gas_columns, "rows": backend.n_rows, "hours_between_readings": 12},
        "fdd": {"classes": [{"class": c, "name": n} for c, n in FDD_CLASS_NAMES.items()],
                "low_confidence_threshold": backend.threshold, **PERFORMANCE["fdd"]},
        "rul": {"error_radius_days": backend.radius, "max_training_days": backend.cap_days,
                "near_cap_fraction": 0.9, **PERFORMANCE["rul"]},
    }


# The web page. Mounted LAST so the /api/... routes above are matched first.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    # PORT is set by Azure (8000). One worker only: the models are big and must be loaded once.
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
