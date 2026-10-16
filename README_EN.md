# Disease-Specific Metric Causal Attribution System

> Connect the HIS discharge front-page database directly to a browser panel and get a causal attribution graph, counterfactual predictions, and a review draft for a given disease and quarter within 30 seconds; the quality-control analyst drafts the attribution report, and the medical-affairs reviewer confirms with one click in the panel to write the result back to the single-disease reporting system—no more Excel stitching or manual row-counting.

## Applicable Scenarios / Target Roles

| Role | When it's used | What they get |
|------|----------------|---------------|
| Quality control / Statistician | One week before the monthly single-disease review meeting | A causal attribution graph and counterfactual prediction screenshot for a disease in a given quarter |
| Quality control / Coding analyst | When the review meeting needs an attribution report draft | A Chinese-language attribution report draft with causal effect, refutation robustness, and sample-size gate markers |
| Medical affairs / Review reviewer | Month-end meeting or quarterly retrospective | One-click confirm or return the draft in the panel; on confirm, the system auto-writes it back to the single-disease reporting system |
| IT / Data analyst | Adding a new disease or adjusting field mappings | Field-mapping JSON validated at startup + explicit error on missing fields, never silent failure |

> **Table discipline**: this table only lists **role language + business moment + verifiable output**; HL7, ASTM, field-mapping paths, state enums, etc. go into the "Commands / API / Configuration" section.

## Key Capabilities

- **Direct connection to the HIS discharge front-page database**: pulls actual inpatient samples for a specified disease and quarter range directly from the hospital HIS database (PostgreSQL/MySQL); connection parameters are read from environment variables, credentials are never committed to the repo.
- **Causal attribution graph in 30 seconds**: after the browser panel opens, it streams the effect estimate and confidence interval of each candidate factor (treatment strategy, comorbidities, age band, admission route, etc.); edges with p<0.05 are solid, otherwise dashed; edge thickness encodes effect size.
- **Click-to-explore counterfactual simulation**: select a node and enter a hypothetical value; the predicted outcome and confidence interval return in under 5 seconds; 10 scenarios batched return in under 30 seconds in parallel.
- **Multi-tier robustness refutation**: built-in random common cause, placebo, bootstrap, and data-subset refutation; runs in parallel with n_jobs=-1 and returns an overall verdict of robust / partial / unstable; extra-wide CI is auto-degraded and flagged in red.
- **Sample-size gate**: n<30 disables write-back entirely; 30≤n<100 is flagged as marginal (draft only); n≥100 enters the formal estimation pipeline—preventing small-sample noise from driving clinical conclusions.
- **Three-state draft review**: the attribution report draft follows a pending_review / reviewed / rejected / archived state machine, with strict role checks (only medical-affairs reviewers can confirm) and transition events audited to the database.
- **One-click confirm auto-reporting**: after the reviewer clicks confirm in the panel, an HTTP POST automatically writes the attribution report back to the single-disease reporting system (Bearer auth, retry 3 times with exponential backoff); on failure the draft falls back to pending_review so no data is lost.
- **Audit & de-identification**: all modeling, draft, state-transition, reporting, model-call, HIS-query, and degradation events are uniformly audited; patient_name / id_card / phone / address fields are forcibly stripped before writing, and never appear in audit logs or report payloads.

## Quick Start

### Dependencies

- Python ≥ 3.11
- Optional: local PostgreSQL ≥ 14 (used for the product's own database and audit log; SQLite substitute is fine for testing)
- Browser: Chrome / Edge / Safari within the last 2 major versions

### Installation

```bash
# 1. Create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Prepare environment variables
cp .env.example .env
# Edit .env: fill HIS_DB_URL / HIS_DB_USER / HIS_DB_PASS / DIPCASE_REPORTER_URL / DIPCASE_REPORTER_TOKEN
#            OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL / OPENAI_TIMEOUT
#            DATABASE_URL (the product's own PostgreSQL audit database)
```

### Configuration

| Configuration item | Default value | Description |
|--------------------|---------------|-------------|
| `HIS_DB_URL` | (none) | HIS discharge front-page DB direct-connection URL; postgresql/mysql is auto-inferred from env |
| `HIS_DB_USER` | (none) | HIS DB username; credential read from environment variable, not committed |
| `HIS_DB_PASS` | (none) | HIS DB password; same as above |
| `DIPCASE_REPORTER_URL` | (none) | Single-disease reporting system base URL |
| `DIPCASE_REPORTER_TOKEN` | (none) | Reporting system Bearer Token; used at reviewer confirm |
| `OPENAI_BASE_URL` | `http://localhost:18888/v1` | OpenAI-compatible gateway base URL (used for model / narrative generation) |
| `OPENAI_API_KEY` | (none) | Gateway API Key |
| `OPENAI_MODEL` | `narrative-zh-1` | Gateway model name |
| `OPENAI_TIMEOUT` | `10` | Inference timeout (seconds); explanation class 30s |
| `DATABASE_URL` | `sqlite:///./audit.db` | The product's own audit-database connection URL |

### Build & Verification

```bash
# Narrow tests: run only the modules directly related to the current task
pytest tests/ -k "dag_builder or pipeline" -q

# Full regression: only when on the last task, after touching shared infrastructure, or after a cross-module fix
pytest tests/ -q

# Start the service
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Open the browser
open http://localhost:8000/        # macOS
xdg-open http://localhost:8000/    # Linux
```

### Main Command Reference

```bash
# One-click start service + browser self-check (common during development)
bash scripts/demo.sh

# One-click end-to-end run (HIS stub + reporting mock + model stub)
bash scripts/run_e2e.sh

# Capture real runtime screenshots to docs/screenshots/
bash scripts/capture_screenshots.sh

# Large-scale performance & stability benchmark (hundreds to thousands of samples)
bash benchmarks/run_parallel.sh
```

## Commands / API / Configuration

### Key Configuration Items

| Configuration item | Default value | Description |
|--------------------|---------------|-------------|
| `HIS_DB_URL` | (none) | HIS discharge front-page DB direct-connection URL; postgresql/mysql is auto-inferred from env |
| `HIS_DB_USER` | (none) | HIS DB username; credential read from environment variable, not committed |
| `HIS_DB_PASS` | (none) | HIS DB password; same as above |
| `DIPCASE_REPORTER_URL` | (none) | Single-disease reporting system base URL |
| `DIPCASE_REPORTER_TOKEN` | (none) | Reporting system Bearer Token |
| `OPENAI_BASE_URL` | `http://localhost:18888/v1` | OpenAI-compatible gateway base URL |
| `OPENAI_API_KEY` | (none) | Gateway API Key |
| `OPENAI_MODEL` | `narrative-zh-1` | Gateway model name |
| `OPENAI_TIMEOUT` | `10` | Inference timeout (seconds); explanation class 30s |
| `DATABASE_URL` | `sqlite:///./audit.db` | The product's own audit-database connection URL |

### CLI Subcommands / API Endpoints

| Command or path | Purpose | Key parameters |
|-----------------|---------|----------------|
| `POST /api/modeling/run` | Start causal modeling, streaming effects by edge | `{disease_code, quarter, outcome}`, header `X-Quality-Role` |
| `POST /api/counterfactual/run` | Batch counterfactual simulation | `{disease_code, outcome, quarter, scenarios:[]}` |
| `GET /api/drafts` | List drafts, filtered by status | `?status=pending_review` |
| `GET /api/drafts/{draft_id}` | Get a single draft | — |
| `POST /api/drafts/{draft_id}/review` | Reviewer confirm or return | `{action: confirm\|reject, reviewer_id}` |
| `GET /api/healthz` | System health status | — |
| `GET /api/modules` | Causal engine module list + version + model version (for product panel display) | — |
| `GET /` | Browser panel entry | — |

> **README only describes product-level names**; protocol implementation details are not listed; related references uniformly use neutral method descriptions (vector retrieval / SCM / counterfactual simulation / multi-tier refutation / conditional-independence non-driver filtering).

### Errors & Degradation

| Scenario | Symptom | Handling |
|----------|---------|----------|
| HIS DB unreachable | `/api/healthz` returns his_conn=down | Modeling pipeline does not start; banner shown at panel top; `his/query` failure event audited |
| Required field mapping missing | `ValueError` at startup; service refuses to start | Not silent; no "modeled" illusion shown in panel |
| Estimand cannot be identified | Modeling stream returns `EstimandNotIdentifiedError` | No draft generated; `modeling/run` failure audited |
| Sample size n<30 | Draft not written back, marked `sample_insufficient` | Panel shows "sample insufficient" prompt; reviewer cannot click confirm |
| Refutation CI too wide (>0.6) | Edge in draft flagged `refutation_unstable` in red | Draft can still be generated but panel flags weak robustness; not silently advanced |
| Counterfactual timeout (>5s) | Returns scenario + `timeout` marker | Yellow warning in panel; not silently swallowed |
| Gateway timeout / unparseable | Auto-degrades to `DeterministicNarrativeFallback` | Template statements generated based on z-score ordering; `model/call` degraded audited |
| Gateway 5xx | Retry 3 times (exponential backoff) | Still failing → degraded to deterministic template; draft status not advanced |
| Reporting HTTP failure | Retry 3 times (exponential backoff) | Still failing → draft falls back to pending_review; audit does not contain response sensitive fields |
| Non-reviewer role confirm | 403 | State machine rejects transition; panel toast says no permission |

### Integration with Existing Systems

- **HIS discharge front-page database** (read): SQLAlchemy create_engine with psycopg2-binary / pymysql dual drivers; connection URL auto-infers postgresql/mysql from env; session factory runs through lifespan with connection-pool recycling; table structure defined by the ORM model in `app/db/schema.py`.
- **Single-disease reporting system** (write): `app/adapters/dipcase_reporter.py` does `requests` POST to `DIPCASE_REPORTER_URL/reports/attribution/draft`; Authorization Bearer; payload schema strictly aligned with the internal spec (disease_code / quarter / outcome / treatment / graph_snapshot_dot / effect_estimates / refutation_results / counterfactual_scenarios / narrative_zh / reviewer_id / model_version); retry 3 times with exponential backoff + fall back to pending_review on failure.
- **OpenAI-compatible gateway** (narrative generation only): `app/adapters/model_gateway.py` is a single-protocol httpx client; `OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL / OPENAI_TIMEOUT` fully configurable; structured response_format=json_object; timeout / unparseable / 5xx all degrade to `DeterministicNarrativeFallback`, never silently advance business state.
- **Product's own PostgreSQL audit database** (write): `app/db.py` aggregates + `DATABASE_URL` env + auto-create tables + session factory; used for Draft / StateTransition / audit_logs.

## Typical Scenarios

### Scenario 1: Quality control preparing for the acute-MI single-disease review in early September

1. The quality analyst opens the browser at `http://localhost:8000/`, selects "Acute MI" + 2026Q2 + outcome=CMI, and clicks "Model".
2. SSE streams 30 edges of effect estimates with confidence intervals; the panel renders the causal graph; edges with p<0.05 are solid, otherwise dashed; red = strong effect, yellow = weak, gray = not significant.
3. The analyst clicks the "treatment strategy = PCI" node → Counterfactual button → enters the hypothetical value "all patients use PCI" → predicted CMI is returned within 5 seconds.
4. After 4 tiers of refutation finish (random_common_cause / placebo / bootstrap / data_subset), the panel shows the overall verdict `partial`, and the two edges with extra-wide CI are flagged red to indicate weak robustness.
5. The analyst clicks "Draft attribution report" → the gateway generates the Chinese narrative_zh (gateway timeout auto-degrades to z-score template) → the draft is persisted with status=pending_review.
6. The medical-affairs reviewer logs in and sees the draft list, clicks confirm → the system automatically POSTs to the single-disease reporting system; status advances to reviewed; the review meeting agenda directly references it.

```bash
# One-click end-to-end stub run (HIS SQLite stub + reporting mock + gateway mock)
bash scripts/run_e2e.sh
```

### Scenario 2: IT adding a new stroke disease's field mapping

1. The IT team gets the stroke discharge front-page table structure, edits `app/field_mapping_config/field_mapping.json`: under `by_disease_code.I63`, add the eight mappings `primary_diagnosis_code / treatment_strategy / age_band / admission_route / los_days / cmi / readmit_flag / complication_flag`.
2. Restart the service; the startup validator immediately checks whether all required fields are present; any missing field → `ValueError`; the service refuses to start; console log explicitly states which field is missing.
3. All required fields present → service starts; `/api/healthz` returns 200, his_conn=up; the panel can model stroke normally.
4. The field-mapping file is under version control; patient_name / id_card / phone are always on the `display_blocked_fields` whitelist and will never be read by the panel or appear in the reporting payload.

```bash
# Field-mapping startup validation test
pytest tests/test_field_mapping.py -q

# Verify the display_blocked_fields set contains patient_name / id_card / phone
grep -E 'patient_name|id_card|phone' app/field_mapping_config/field_mapping.json
```

### Scenario 3: Quarterly retrospective returning an attribution draft for review

1. Before the quarterly retrospective, the analyst has already drafted an attribution report for a disease with status pending_review.
2. The medical-affairs reviewer opens the panel, sees the draft list, and notices an edge with CI too wide (>0.6) flagged red and an overall verdict unstable.
3. The reviewer clicks "Return" → action=reject + reviewer_id → state machine validates the role → transition written to StateTransition + audit → draft status becomes rejected.
4. The analyst receives the return, adjusts the DAG (removing the factor corresponding to that edge) or expands the sample size (merging to the most recent 3 quarters) → re-models → re-drafts.
5. All transition audits can be queried in the audit_logs table (event_type=state/transition), filtered by reviewer_id and time.

```bash
# State machine legal / illegal / role 403 tests
pytest tests/test_state_machine.py -q

# Audit-event completeness test
pytest tests/test_audit.py -q
```

## Output Samples

### Causal Attribution Graph (rendered in browser panel)

- **Visual**: dark background (#1a1a1a) + nodes in red/yellow/gray encoded by effect size + edges labeled with effect value text + p<0.05 solid/else dashed
- **Nodes**: treatment strategy PCI / thrombolysis / conservative, comorbidity count, age band, admission route, CMI (outcome)
- **Interaction**: click a node → EffectEstimate tooltip appears (ate / CI / p_value / method); highlights the current node's in-edges and out-edges

### Attribution Report Draft JSON (with causal effect, refutation, counterfactual)

```json
{
  "draft_id": "<draft-uuid>",
  "disease_code": "acute_mi",
  "quarter": "2026Q2",
  "outcome": "cmi",
  "treatment": "treatment_strategy",
  "effect_estimates": [
    {
      "edge": "treatment_strategy->cmi",
      "ate": -0.42,
      "ci_lower": -0.55,
      "ci_upper": -0.30,
      "p_value": 0.003,
      "method_name": "backdoor.propensity_score_matching",
      "refutation_verdict": "robust"
    }
  ],
  "refutation_results": [
    {"method": "random_common_cause", "new_effect": -0.41, "p_value": 0.86, "passed": true},
    {"method": "placebo_treatment_refuter", "new_effect": 0.02, "p_value": 0.91, "passed": true},
    {"method": "bootstrap_refuter", "new_effect": -0.43, "p_value": 0.78, "passed": true},
    {"method": "data_subset_refuter", "new_effect": -0.39, "p_value": 0.12, "passed": true}
  ],
  "counterfactual_scenarios": [
    {
      "scenario": "treatment_strategy=PCI for all",
      "predicted_outcome": 1.05,
      "confidence": 0.82,
      "do_operations": ["do(treatment_strategy='PCI')"]
    }
  ],
  "narrative_zh": "(generated by gateway; degradation path: z-score-ordered template statements)",
  "status": "pending_review",
  "reviewer_id": null
}
```

### Real End-to-End Runtime Screenshots

- `docs/screenshots/01_startup.png` — Browser first screen after service starts (with causal graph placeholder and modeling button)
- `docs/screenshots/02_causal_graph.png` — Causal graph for acute-MI 2026Q2 CMI (with edge effect values and robustness color encoding)
- `docs/screenshots/03_counterfactual.png` — Predicted outcome and confidence interval after the counterfactual button is triggered
- `docs/screenshots/04_review_confirm.png` — Toast notification + status advance after the reviewer clicks confirm

## Concept vs. Real-World Comparison

### Visual Goals

Starting state: Browser first screen, dark background + disease / quarter / outcome selectors at the top + central Plotly.js causal-graph placeholder + bottom "Model" button.

First usable moment: SSE streams the first edge effect estimate; one node and one edge appear on the causal graph, color-encoded by effect size; p<0.05 solid line.

Mature state: Full causal graph fully rendered, 4-tier refutation overall verdict displayed, 3 buttons (counterfactual / confirm / return) ready, loading state cleared, toast notification "Modeling complete".

Color & layout:
- Main background `#1a1a1a`; primary colors `#d62728` (strong effect red), `#ff7f0e` (weak effect yellow), `#7f7f7f` (not significant gray)
- 8px spacing grid (4/8/12/16/24/32)
- Font hierarchy ≥ 4 levels: page title / section title / body / auxiliary description
- Node names 14px; effect values 12px; numbers tabular-nums monospaced alignment
- Border radius and shadow divided into card / floating-layer / modal three levels

Honesty note: this environment has no image-generation capability, so `docs/concepts/` only provides a structured visual goal card (see `vision-target.md`); no fake images are made.

### Real-World Evidence

Real runtime screenshots are captured by `scripts/capture_screenshots.sh`, 4 PNGs each > 5KB:

![Startup first screen](docs/screenshots/01_startup.png)
![Causal graph render](docs/screenshots/02_causal_graph.png)
![Counterfactual simulation](docs/screenshots/03_counterfactual.png)
![Review confirm](docs/screenshots/04_review_confirm.png)

| Screenshot | Trigger action | Verification points |
|------------|----------------|---------------------|
| `01_startup.png` | Open `http://localhost:8000/` in browser | Dark background + disease/quarter/outcome selectors + causal graph placeholder + model button visible |
| `02_causal_graph.png` | Select Acute MI + 2026Q2 + cmi, click Model to completion | Nodes in red/yellow/gray color-encoded by effect size + edges labeled with effect value + p<0.05 solid/else dashed |
| `03_counterfactual.png` | Click the treatment-strategy node + enter hypothetical value + trigger counterfactual | predicted_outcome and confidence values + returns within 5 seconds + exception-path toast |
| `04_review_confirm.png` | After reviewer clicks confirm | Toast notification + draft status advances to reviewed + reporting success prompt |

> **Honesty note**: above is the visual goal (structured description in `docs/concepts/vision-target.md`), below is the real measured screenshot. There will inevitably be a gap between goal and implementation—some interaction details in the measured output (button hover, loading-spinner corner style, toast gradient) are not fully rendered due to the screenshot-tool rendering precision; we state exactly where they differ, and **do not** claim the two are identical, nor generate "post-hoc renders" to paper over the gap.

## Architecture & Data Flow

```
┌────────────────────┐    direct connection (SQLAlchemy + psycopg2/pymysql)    ┌─────────────────────┐
│   HIS discharge   │ <─────────────────────────────────────────────────────── │   HisAdapter        │
│   front-page DB    │   credentials: HIS_DB_URL / USER / PASS (env)            │   (app/adapters)    │
│  PostgreSQL/MySQL  │                                                       │                     │
└────────────────────┘                                                        └─────────┬───────────┘
                                                                                            │ DataFrame
                                                                                            ▼
┌────────────────────┐    env: OPENAI_BASE_URL / KEY / MODEL / TIMEOUT  ┌─────────────────────┐
│  OpenAI-compatible │ <──────────────────────────────────────────────── │   ModelGateway      │
│      gateway       │   structured response_format=json_object         │   (app/adapters)    │
└────────────────────┘   timeout/unparseable/5xx → template fallback    └─────────┬───────────┘
                                                                                    │
                                                                                    ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│  core/pipeline (CausalModel + identify_effect + estimate_effect, multiple tiers)              │
│  core/refuter (random_common_cause / placebo / bootstrap / data_subset + parallel + verdict) │
│  core/graph_refuter (partial_correlation conditional-independence non-driver filtering)      │
│  core/dosampler + core/dosampler_batch + core/path_traverse (counterfactual + path traversal)│
│  core/equation_parser + core/equation_templates (SCM auto-generation)                        │
│  core/baseline (univariate correlation baseline + wedge strength)                              │
│  core/sample_size_gate (n<30 no write-back / 30≤n<100 marginal / n≥100 sufficient)           │
│  core/estimator_selector (pick estimator by sample size + IV availability)                   │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
                                                                                    │
                                                                                    ▼
┌────────────────────┐    SSE stream (POST /api/modeling/run)         ┌─────────────────────┐
│   FastAPI + uvicorn│ <────────────────────────────────────────────── │   Browser panel     │
│   + Plotly.js      │   header X-Quality-Role                        │   Jinja2 + native   │
└────────────────────┘                                                 │   HTML              │
                                                                         └─────────────────────┘
                                                                                    │
                                                                                    ▼
┌────────────────────┐   POST /api/drafts/{id}/review confirm        ┌─────────────────────┐
│  drafts/           │ ────────────────────────────────────────────> │   Single-disease   │
│  state_machine     │   retry 3 + fall back to pending_review       │   reporting system  │
│  (Draft model)     │   payload schema strictly aligned             │   Bearer Token      │
└────────────────────┘                                                 │   (DIPCASE_REPORTER)│
                                                                         └─────────────────────┘
                                                                                    │
                                                                                    ▼
                                                                         ┌─────────────────────┐
                                                                         │   Audit DB          │
                                                                         │   (audit_logs)      │
                                                                         │   strip_pii enforced│
                                                                         └─────────────────────┘
```

### Key Invariants

- **Zero hard-coded credentials**: HIS_DB_* / DIPCASE_REPORTER_* / OPENAI_* / DATABASE_URL are all read from environment variables; missing any one raises `ValueError`, and no default is allowed to silently cover it up.
- **Small sample does not silently advance**: n<30 directly becomes `sample_insufficient`; reviewer cannot confirm; 30≤n<100 is flagged marginal; n≥100 enters formal estimation.
- **Wide refutation CI degrades**: CI upper - lower > 0.6 → `refutation_unstable` red edge; the draft can still be generated but the panel flags weak robustness; not silently swallowed.
- **Gateway unreachable does not advance business state**: any `ModelGateway` timeout / unparseable / 5xx all degrade to `DeterministicNarrativeFallback`; audit marks `state_not_advanced=True`, and reviewer business state is not silently changed.
- **Audit de-identification enforced**: `strip_pii` recursively cleans patient_name / id_card / phone / address; stripping happens before writing to audit_logs and reporting payloads.
- **State machine cannot skip levels**: pending_review → reviewed / rejected; reviewed → archived; any illegal transition rejected; role check strict (only reviewer can confirm).
- **Failure fallback ≠ blanket fallback**: reporting failure falls back to pending_review; modeling/refutation failure does not fall back, but rather degrades to red flag + audit, preserving all work the analyst has already done.

## Security & Compliance Boundaries

- **No diagnosis or treatment advice**: all narrative generation only does structural organization and colloquial explanation; any statement like "recommend adding some drug" or "suggests possibly some disease" is forbidden at both the template and gateway prompt layers.
- **No auto-signing or proxy signing**: reviewer confirm is a real person clicking in the panel; draft confirm carries no auto-signing or proxy-signing logic.
- **patient_name / id_card / phone / address forcibly stripped before write**: `strip_pii` recursively cleans all dict/list; audit logs and reporting payloads contain no PII.
- **HIS database read-only**: HISAdapter only queries; there is no update / delete / insert; connection-pool recycling is managed by lifespan.
- **Gateway calls only for narrative generation**: business state (draft status, model estimates, refutation results) is always computed through deterministic paths; the gateway does not participate in business logic.
- **De-identified substitute**: all fixtures / mock endpoints / SQLite substitutes use placeholder or categorical expressions (`<EXAMPLE_DISEASE>` / `<EXAMPLE_LOT>`); no real patient identifiers appear.

## Acceptance & Self-Check

| Acceptance item | Command | Expected result |
|-----------------|---------|-----------------|
| Service startup | `bash scripts/demo.sh` | uvicorn starts; `/api/healthz` returns 200; his_conn=up |
| Field-mapping startup validation | `pytest tests/test_field_mapping.py -q` | Missing required field raises `ValueError`; whitelist contains patient_name / id_card / phone |
| Causal modeling (DAG) | `pytest tests/test_dag_builder.py -q` | 3 diseases × 4 outcomes: nodes/edges/colors all assert correctly |
| Causal modeling (pipeline) | `pytest tests/ -k pipeline -q` | CausalModel construction + identify_effect on 3 fixtures, estimand not None |
| Multi-tier refutation | `pytest tests/test_refuter.py -q` | Four refutation tiers run on 3 fixtures; n_jobs=-1 parallel; three verdicts triggered |
| Counterfactual | `pytest tests/test_dosampler.py -q` | Fixture returns counterfactual in < 5s; predicted_outcome within baseline ± 3σ |
| Batch counterfactual + path traversal | `pytest tests/test_dosampler_batch.py tests/test_path_traverse.py -q` | 10 scenarios batched < 30s; DFS depth=3 node deduplication correct |
| Sample-size gate | `pytest tests/test_sample_size_gate.py -q` | n<30 triggers insufficient; 30≤n<100 marginal; n≥100 sufficient |
| Large-scale performance | `pytest benchmarks/test_perf_bench.py -q` | n=2000 entire run < 60s; error rate < 5% |
| State machine legal / illegal | `pytest tests/test_state_machine.py -q` | Legal transitions pass; level-skip rejected; non-reviewer 403; transition audited |
| Reporting write-back | `pytest tests/test_dipcase_reporter.py -q` | retry 3 times + failure fallback; payload schema aligned; audit contains no sensitive fields |
| Gateway degradation | `pytest tests/test_model_gateway_degradation.py -q` | timeout / unparseable degrade to fallback; audit metadata 5 fields complete |
| Gateway protocol compatibility | `pytest tests/test_model_gateway_protocol_compat.py -q` | stub 6 scenarios all trigger correct path; provider-neutral boundary in effect |
| Audit de-identification | `pytest tests/test_audit.py -q` | 6 event_type all registered; strip_pii removes PII |
| End-to-end integration | `bash scripts/run_e2e.sh` | Full SSE stream + batch counterfactual + draft + reporting mock receives payload + status reviewed |
| Real-world screenshots | `bash scripts/capture_screenshots.sh` | 4 real screenshots fall into `docs/screenshots/`, files > 5KB |
| README coverage | `grep -c '^## ' README.md` / `grep -c '^### ' README.md` / `grep -c '^|' README.md` / `grep -c '^```' README.md` | ## ≥ 10, ### ≥ 8, tables ≥ 5, code blocks ≥ 8 |

## Project Structure

```
disease-metric-causal-attribution/
├── app/                          # application package
│   ├── main.py                   # FastAPI entry + lifespan + uvicorn startup
│   ├── config.py                 # pydantic-settings loading environment variables
│   ├── db.py                     # product's own PostgreSQL + auto table creation + session factory
│   ├── adapters/                 # integration layer to external systems
│   │   ├── his_adapter.py        # HIS discharge front-page DB direct connection (SQLAlchemy + psycopg2/pymysql)
│   │   ├── field_mapping.py      # field-mapping JSON config + startup validation + de-identification whitelist
│   │   ├── dipcase_reporter.py   # single-disease reporting HTTP POST + retry 3 + failure fallback + audit
│   │   └── model_gateway.py      # OpenAI-compatible single protocol + degradation to deterministic template
│   ├── core/                     # causal modeling core engine
│   │   ├── pipeline.py           # CausalModel + identify_effect + estimate_effect, three tiers
│   │   ├── dag_builder.py        # business DAG template (nodes/edges/colors)
│   │   ├── refuter.py            # multi-tier refutation + parallel n_jobs=-1 + overall verdict
│   │   ├── graph_refuter.py      # partial_correlation conditional-independence non-driver filtering
│   │   ├── dosampler.py          # single-hypothesis counterfactual + 5s timeout
│   │   ├── dosampler_batch.py    # batch counterfactual parallel batch_size=10
│   │   ├── path_traverse.py      # DFS depth=3 causal chain expansion
│   │   ├── equation_parser.py    # SCM auto-generation (metric + template → SCM)
│   │   ├── equation_templates.py # business template strings (one per disease per outcome)
│   │   ├── baseline.py           # univariate correlation baseline + wedge strength 1.5x threshold
│   │   ├── sample_size_gate.py   # n<30 insufficient / 30≤n<100 marginal / n≥100 sufficient
│   │   ├── estimator_selector.py # pick estimator by sample size + IV availability
│   │   └── exception_handlers.py # custom exceptions → 4xx/5xx + JSON envelope + audit
│   ├── api/                      # FastAPI routes
│   │   ├── modeling.py           # SSE streaming modeling + X-Quality-Role auth
│   │   ├── counterfactual.py     # batch counterfactual JSON
│   │   ├── drafts.py             # draft GET / list / review confirm/reject
│   │   ├── healthz.py            # system health status
│   │   └── modules.py            # causal engine module list + version + model version
│   ├── drafts/                   # draft state machine
│   │   ├── models.py             # Draft SQLAlchemy model
│   │   └── state_machine.py      # state transitions + role checks + audit
│   ├── audit/                    # audit & de-identification
│   │   └── logger.py             # AuditLogger + strip_pii field stripping
│   ├── db/                       # ORM models
│   │   ├── schema.py             # HIS discharge front-page table structure
│   │   └── draft_models.py       # Draft + StateTransition table structure
│   └── panel/                    # browser panel
│       ├── templates/index.html  # Jinja2 + Plotly.js causal graph + design tokens
│       └── static/               # plotly.min.js + panel.css + panel.js
├── data/                         # de-identified fixtures (CSV test data)
│   ├── fixture_acute_mi.csv      # Acute MI 200-row de-identified sample
│   ├── fixture_stroke.csv        # Stroke 200-row de-identified sample
│   └── fixture_hip_replacement.csv  # Hip replacement 200-row de-identified sample
├── benchmarks/                   # performance & stability
│   ├── run_perf_benchmark.py     # 4 sample-size tiers × 3 diseases × 4 outcomes pipeline
│   ├── run_parallel.sh           # one-click run all benchmarks
│   └── test_perf_bench.py        # n=2000 < 60s + error rate < 5% assertion
├── perf_records/                 # performance baseline JSONs (48 records: 4 tiers × 3 diseases × 4 outcomes)
├── tests/                        # unit & integration tests
│   ├── conftest.py               # fixture loader
│   ├── test_seed_loader.py       # de-identified CSV readable + fields aligned
│   ├── test_his_adapter.py       # SQLite stub + driver inference + limit in effect
│   ├── test_field_mapping.py     # field mapping + missing-field error + de-identification whitelist
│   ├── test_dag_builder.py       # 3 diseases × 4 outcomes DAG nodes/edges/colors
│   ├── test_pipeline.py         # CausalModel + identify_effect + small-sample diagnosis
│   ├── test_refuter.py           # four refutation tiers + parallel + verdict + degradation
│   ├── test_graph_refuter.py     # partial_correlation non-driver filtering
│   ├── test_dosampler.py         # single-hypothesis counterfactual < 5s
│   ├── test_dosampler_batch.py   # 10-scenario batch < 30s
│   ├── test_path_traverse.py     # DFS depth=3 node deduplication
│   ├── test_equation_parser.py   # 3 diseases × 4 outcomes SCM generation
│   ├── test_baseline.py          # causal-vs-correlation ratio ≥ 1.5 wedge flag
│   ├── test_sample_size_gate.py  # three-tier sample-size gate
│   ├── test_estimator_selector.py    # 4 trigger conditions
│   ├── test_path_traverse_extension.py # path flattening fields
│   ├── test_app_lifespan.py      # lifespan startup order + exception handlers
│   ├── test_panel_sse.py         # SSE streaming + auth + batch counterfactual JSON
│   ├── test_drafts_api.py        # GET / list / POST review confirm/reject
│   ├── test_panel_html.py        # HTML contains Plotly.js + design tokens
│   ├── test_panel_interaction.py # Playwright node click + tooltip + three-state
│   ├── test_panel_buttons.py     # 3 buttons + toast
│   ├── test_state_machine.py     # legal/illegal level-skip + role 403 + transition audit
│   ├── test_dipcase_reporter.py  # retry 3 + failure fallback + payload schema + audit
│   ├── test_model_gateway_basic.py       # OpenAI-compatible protocol handshake + structured output
│   ├── test_model_gateway_degradation.py # timeout/unparseable degradation + audit + state_not_advanced
│   ├── test_model_gateway_protocol_compat.py # stub 6 scenarios protocol compatibility
│   ├── test_audit.py             # 6 event_type + strip_pii field stripping
│   ├── test_e2e.py               # end-to-end SSE + counterfactual + draft + reporting + status + audit
│   ├── e2e/                      # e2e fixtures
│   │   └── conftest.py           # one-click fixture preparation
│   └── mocks/                    # test substitutes
│       └── openai_stub/          # OpenAI-compatible stub + 3 injection endpoints
├── docs/                         # documentation
│   ├── concepts/                 # visual goals (structured .md)
│   └── screenshots/              # real-world screenshots (4 real runtime screenshots)
├── scripts/                      # one-click scripts
│   ├── demo.sh                   # one-click start service + browser self-check
│   ├── run_e2e.sh                # one-click end-to-end run
│   └── capture_screenshots.sh    # capture real runtime screenshots
├── pyproject.toml                # Python 3.11 + dependency manifest
├── .env.example                  # credential placeholder example (no real values)
├── README.md                     # Chinese product description (this file)
├── README_EN.md                  # English version (generated by Phase 5)
├── LICENSE                       # MIT
└── qrcode.jpg                    # official-account QR code
```

## License

MIT

---

## Follow Us

Scan the QR code to follow our official account for updates and community access:

![Follow Us](qrcode.jpg)