"""
api_server.py — thin HTTP layer over the existing SIR backend. Serves both
the JSON API and the dashboard HTML from one process, so a single Render
web service is all that's needed for the UI + API together.

Run locally:
    pip install fastapi uvicorn --break-system-packages
    python api_server.py
    # or: uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

Run in production (Render sets $PORT for you - see render.yaml):
    uvicorn api_server:app --host 0.0.0.0 --port $PORT

AUTH: every route (including the dashboard page itself) requires HTTP
Basic Auth, via SIR_DASHBOARD_USER / SIR_DASHBOARD_PASSWORD env vars. This
is a deliberate, fail-CLOSED design - if those env vars aren't set, every
request gets a 503 rather than the service silently running open. This
dashboard has no other login of its own, and once deployed it's reachable
from the public internet, not just your laptop; without this, ~3M+ real
voters' names/addresses/ages/EPIC numbers would be an unauthenticated
public search tool. Set both env vars before deploying anywhere but your
own machine.

Endpoints:
    GET  /                 the dashboard (static HTML)
    GET  /api/health
    GET  /api/search?name=&epic=&house=&relation=&constituency=&mode=OR&min_match=2
    GET  /api/audit
    POST /api/sync?timeline=both&force=true   (see note in README - not
                                                 well-suited to running
                                                 synchronously in production)
"""

import os
import secrets
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from config import SIRConfig
from search_engine import SIRSearchEngine, SearchQuery
from anomaly_detector import SIRAnomalyDetector
from es_sync import sync_timeline

app = FastAPI(title="Dakshina Kannada SIR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

cfg = SIRConfig()
engine = SIRSearchEngine(cfg)

DASHBOARD_PATH = Path(__file__).parent / "sir_dashboard.html"

# ---------------------------------------------------------------------------
# Auth - fail CLOSED if credentials aren't configured (see module docstring)
# ---------------------------------------------------------------------------
security = HTTPBasic()
_DASH_USER = os.environ.get("SIR_DASHBOARD_USER", "")
_DASH_PASS = os.environ.get("SIR_DASHBOARD_PASSWORD", "")


def require_auth(credentials: HTTPBasicCredentials = Depends(security)):
    if not _DASH_USER or not _DASH_PASS:
        raise HTTPException(
            status_code=503,
            detail="SIR_DASHBOARD_USER / SIR_DASHBOARD_PASSWORD are not configured on the "
                   "server - refusing to serve requests rather than running unauthenticated.",
        )
    user_ok = secrets.compare_digest(credentials.username, _DASH_USER)
    pass_ok = secrets.compare_digest(credentials.password, _DASH_PASS)
    if not (user_ok and pass_ok):
        raise HTTPException(status_code=401, detail="Invalid credentials",
                             headers={"WWW-Authenticate": "Basic"})
    return True


def to_ui_record(rec: dict, timeline: str) -> dict:
    """Map internal ES field names to the flat shape the HTML expects.

    Deliberately does NOT surface religion/community, even though the ES
    documents carry those fields (from a separate classifier pipeline) -
    this API is being deployed to a public host, which is a different and
    much larger exposure than a local dev instance. See deployment notes."""
    return {
        "name": rec.get("voter_name") or "",
        "relation": rec.get("relation_name") or "",
        "house": rec.get("door_no") or "",
        "epic": rec.get("epic_no") or "",
        "status": (rec.get("status") or "ACTIVE").lower().replace("_", "-"),
        "constituency": rec.get("constituency") or "",
        "part_no": rec.get("part_no"),
        "age": rec.get("age"),
        "gender": rec.get("gender"),
        "score": rec.get("_score"),
        "timeline": timeline,
    }


@app.get("/")
def dashboard(_auth: bool = Depends(require_auth)):
    if not DASHBOARD_PATH.exists():
        raise HTTPException(status_code=500, detail=f"{DASHBOARD_PATH.name} not found next to api_server.py")
    return FileResponse(DASHBOARD_PATH)


@app.get("/api/health")
def health():
    # deliberately NOT behind auth - lets Render's health check (and you)
    # confirm the process is alive without needing credentials
    return {"status": "ok"}


@app.get("/api/search")
def search(
    name: Optional[str] = None,
    epic: Optional[str] = None,
    house: Optional[str] = None,
    relation: Optional[str] = None,
    constituency: Optional[str] = None,
    mode: str = Query("OR", pattern="^(OR|AND|MIN_N)$"),
    min_match: int = 2,
    _auth: bool = Depends(require_auth),
):
    if not any([name, epic, house, relation, constituency]):
        return {"results_2025": [], "results_2002": [], "count_2025": 0, "count_2002": 0,
                "name_variants_tried": [], "relation_variants_tried": []}

    q = SearchQuery(
        epic_no=epic or None,
        door_no=house or None,
        voter_name=name or None,
        relation_name=relation or None,
        constituency=constituency or None,
        combine_mode=mode,
        min_match=min_match,
    )
    try:
        result = engine.search(q)
    except Exception as e:
        return {"error": str(e), "results_2025": [], "results_2002": [], "count_2025": 0, "count_2002": 0,
                "name_variants_tried": [], "relation_variants_tried": []}

    results_2025 = [to_ui_record(r, "2025") for r in result["merged_results"] if r.get("_timeline") == "2025"]
    results_2002 = [to_ui_record(r, "2002") for r in result["merged_results"] if r.get("_timeline") == "2002"]
    return {
        "results_2025": results_2025,
        "results_2002": results_2002,
        "count_2025": result["count_2025"],
        "count_2002": result["count_2002"],
        "name_variants_tried": result.get("name_variants_tried", []),
        "relation_variants_tried": result.get("relation_variants_tried", []),
    }


@app.get("/api/audit")
def audit(_auth: bool = Depends(require_auth)):
    try:
        detector = SIRAnomalyDetector(cfg)
        anomalies = detector.run_full_audit()
    except Exception as e:
        return {"error": str(e), "summary": {}, "anomalies": []}

    return {
        "summary": detector.summary(),
        "anomalies": [
            {"category": a.category, "severity": a.severity, "epic_no": a.epic_no, "details": a.details}
            for a in anomalies
        ],
    }


@app.post("/api/sync")
def sync(timeline: str = "both", force: bool = False, _auth: bool = Depends(require_auth)):
    # NOTE: this runs the full Mongo->ES sync SYNCHRONOUSLY inside the HTTP
    # request. Locally that's fine; on Render, a 1.5M+ document sync can
    # run well past typical platform request-timeout limits and tie up a
    # worker the whole time. For production use, prefer running
    # `python es_sync.py --timeline both --force` from your own machine
    # (pointed at the deployed Mongo/ES) or as a separate Render Cron Job,
    # rather than hitting this endpoint - kept here for local/manual use.
    timelines = ["2025", "2002"] if timeline == "both" else [timeline]
    try:
        for t in timelines:
            sync_timeline(cfg, t, force_reindex=force)
    except Exception as e:
        return {"status": "error", "detail": str(e)}
    return {"status": "ok", "synced": timelines}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))   # Render assigns $PORT - must bind to it
    uvicorn.run(app, host="0.0.0.0", port=port)
