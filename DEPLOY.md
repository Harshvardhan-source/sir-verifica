# Deploying to Render.com

## Two different lifecycles — read this first

`python api_server.py` and `python main.py sync --timeline both --force` are
**not** the same kind of thing on Render, and don't get "run" the same way:

- **`api_server.py`** is a long-running process that Render manages *for
  you*: you don't manually type this command. Render builds your repo once,
  then keeps `uvicorn api_server:app` running continuously as the `sir-dk-api`
  Web Service, restarting it automatically if it ever crashes. This is what
  `render.yaml`'s `startCommand` sets up — see "2. Create the Render web
  service" below.
- **`main.py sync ...`** is a one-off batch job — it runs once, does its
  work (rebuilds the ES index from 1.5M+ Mongo docs), and exits. It should
  **not** be the web service's start command. See "4. Running the sync
  command" below for how to trigger it on demand.

## What gets deployed, and what doesn't

**Deployed as one Render web service** (dashboard + API together, one process):
- `api_server.py`, `sir_dashboard.html`, `search_engine.py`, `anomaly_detector.py`,
  `config.py`, `schema_mapping.py`, `name_variants.py`, `es_sync.py`, `requirements.txt`

**Stays local / run manually, not deployed as a service:**
- `mongo_voter_classifier.py`, `hmc_classifier_final.joblib` — not referenced by
  anything the API imports, and not something to run as scheduled cloud
  infrastructure. If you need to re-run it, do that from your own machine
  against Mongo directly, the same way you have been.
- `seed_sample_data.py`, `check_connection.py`, `main.py` — CLI/one-off
  tools, run locally when needed, pointed at the deployed Mongo/ES via env vars.
- `.env` / `_env` — never deploy this file anywhere. Render env vars replace it.

## 0. Before anything else

- **Rotate the Atlas password** if you haven't (see chat) — do this before
  putting the connection string into Render's env vars, so you're not
  copying a credential you already know is compromised.
- **Mongo Atlas Network Access**: Render's outbound IPs aren't fixed on
  standard plans, so Atlas needs to allow connections from anywhere:
  Atlas → Network Access → Add IP Address → `0.0.0.0/0` ("Allow access from
  anywhere"). This is standard for PaaS deployments without a static IP,
  but it does mean Atlas is relying on your DB **username/password** as the
  real access control from here on — another reason the password must be
  strong and freshly rotated. (If you want a real fix instead of `0.0.0.0/0`,
  Render's paid plans offer a static outbound IP add-on you can allow-list instead.)

## 1. Elasticsearch — pick one

Render has no managed Elasticsearch product, so ES needs to live somewhere else:

**Option A — managed ES/OpenSearch host (recommended).** Elastic Cloud,
Bonsai.io, or AWS OpenSearch Service all work — you get a URL + credentials,
set them as `SIR_ES_HOSTS` / `SIR_ES_USERNAME` / `SIR_ES_PASSWORD` in Render,
done. No JVM/disk/memory tuning on your end. This is the path I'd take
first — running ES yourself is real operational work for not much benefit
at this scale.

**Option B — run ES yourself as a Render Private Service (Docker).** More
control, more to manage: needs a paid instance type with enough RAM (ES
wants ~1-2GB heap minimum — the free/starter tier's 512MB won't run it), a
persistent disk (also a paid feature, or you lose the index every restart),
and **you must enable ES security** (`xpack.security.enabled: true` +
a real password) — the local dev instructions in the old README explicitly
turn security OFF, which is fine on your own laptop and NOT fine on any
shared host, private network or not. I haven't built this option out fully
since it's a bigger, costlier commitment — say if you want to go this route
and I'll put together the Dockerfile + render.yaml service block for it.

Either way: **once ES is reachable, run the sync** — see "4. Running the
sync command" below for how to do this on Render itself (Dashboard Shell or
a Cron Job), or run it from your own machine against the deployed
Mongo/ES if you'd rather:
```bash
export SIR_MONGO_URI="<your Atlas URI>"
export SIR_ES_HOSTS="<your ES URL>"
export SIR_ES_USERNAME="<...>"    # if your ES needs auth (Option B: it must)
export SIR_ES_PASSWORD="<...>"
python es_sync.py --timeline both --force
```

## 2. Create the Render web service

1. Push this repo to GitHub (you're already partway there).
2. Render dashboard → New → Blueprint → point it at your repo. Render will
   read `render.yaml` and create the `sir-dk-api` web service automatically.
   (No blueprint? New → Web Service → same repo → Render auto-detects
   Python; set Build Command `pip install -r requirements.txt` and Start
   Command `uvicorn api_server:app --host 0.0.0.0 --port $PORT` manually.)
3. Service → Environment tab → set every variable `render.yaml` declared
   (`SIR_MONGO_URI`, `SIR_ES_HOSTS`, `SIR_ES_USERNAME`, `SIR_ES_PASSWORD`,
   `SIR_DASHBOARD_USER`, `SIR_DASHBOARD_PASSWORD`) with real values.
   **Pick a real password for `SIR_DASHBOARD_PASSWORD`** — this is the only
   thing standing between the public internet and a search tool over ~3M
   real voters' names, addresses, ages, and EPIC numbers. The service
   refuses to serve requests at all if this isn't set (see `api_server.py`) -
   that's intentional, not a bug if you see 503s before setting it.
4. Deploy. Render builds and starts the service, gives you a URL like
   `https://sir-dk-api.onrender.com`.
5. Open that URL — your browser will prompt for the username/password you
   set in step 3 (standard HTTP Basic Auth dialog), then load the dashboard.

## 3. After deploying

- `GET /api/health` is intentionally the one route *without* auth, so
  Render's health checks (and you) can confirm the process is alive without
  credentials. It returns `{"status": "ok"}` and nothing else - no voter
  data reachable through it.
- If search comes back empty, it's almost always the ES side: confirm
  `SIR_ES_HOSTS` is reachable from Render (not `localhost`) and that you've
  actually run the sync (next section) against that ES instance.
- Render's free/starter web services spin down after inactivity and take
  ~30-60s to wake back up on the next request - expect a slow first load
  after idle periods unless you're on a paid always-on plan.

## 4. Running the sync command (`main.py sync --timeline both --force`)

Two ways to do this on Render, once your `sir-dk-api` web service exists
(both require it to be on a **paid** plan, e.g. `starter` — already set in
`render.yaml`; Render doesn't support Shell/one-off jobs on free instances):

### Option A — Dashboard Shell (simplest, no extra service, use this first)

1. Render Dashboard → `sir-dk-api` service → **Shell** tab. No SSH key
   setup needed for this - it's a browser-based terminal.
2. Once connected, run:
   ```bash
   python main.py sync --timeline both --force
   ```
3. This runs on a temporary, isolated instance of your service (same build,
   same environment variables as the live service) - it does **not** affect
   the live dashboard/API traffic, and Render deletes the instance
   automatically when you disconnect.
4. Watch the output directly in the shell - same progress logging you saw
   running this locally.

### Option B — Cron Job service with "Trigger Run" (if you want a button,
### or a recurring schedule, instead of opening Shell each time)

`render.yaml` (updated) includes an optional `sir-dk-es-sync` cron job
service, set up to run **only when you manually trigger it** (no automatic
schedule by default):

1. If you used the Blueprint to deploy, this service was already created
   alongside `sir-dk-api` - set its env vars the same way (Environment tab:
   `SIR_MONGO_URI`, `SIR_ES_HOSTS`, `SIR_ES_USERNAME`, `SIR_ES_PASSWORD`).
   If you deployed the web service manually (not via Blueprint), create
   this one the same way: New → Cron Job → same repo → Build Command
   `pip install -r requirements.txt` → Command `python main.py sync --timeline both --force`.
2. Render Dashboard → `sir-dk-es-sync` → **Trigger Run** button, whenever
   you want to resync.
3. Want it automatic too? Add a real `schedule:` to that service in
   `render.yaml` (5-field cron expression, evaluated in **UTC**), e.g.
   `schedule: "0 3 * * *"` for daily at 3am UTC, then push/redeploy.
4. Cost: cron job services bill by the second while running, with a $1/month
   minimum just for the service existing - skip this option and just use
   Shell (Option A) if you don't need repeatable/scheduled runs.
5. Runs are capped at 12 hours by Render regardless of plan - not a concern
   at your current record counts (a full sync finished in well under a
   minute in your last local run), just worth knowing if the dataset grows a lot.

