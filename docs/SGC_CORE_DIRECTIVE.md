# SGC-Core — Platform Architecture Directive

**Version:** 1.0
**Status:** Active
**Repo:** `/dev/sgccore`
**Scope:** SGC-Core is the domain-agnostic analysis execution platform. This
document governs its architecture, feature set, and development principles.

---

## 1. What SGC-Core Is

SGC-Core is a **generic, governed, auditable analysis execution platform.** It
knows about users, organizations, projects, datasets, plugins, jobs, and
results. It knows nothing about any specific domain — not biostatistics, not
finance, not genomics.

```
SGC-Core              (the platform — closed source)
  ├── Auth, RBAC, org/project isolation
  ├── Plugin registry + schema-driven UI
  ├── Job queue + worker dispatch (CPU/GPU)
  ├── Dataset model + object storage abstraction
  ├── Audit log (immutable)
  └── Generic result viewer

Plugin packs plug into SGC-Core:
  SGC-Bio             (biotech/pharma — first vertical)
  SGC-Financial       (quant finance — future)
  SGC-Genomic         (bioinformatics — future)
```

SGC-Core is the engine. Plugin packs are the games. Customer extensions are
mods. All use the same plugin system.

**SGC-Core must be fully functional and testable with zero domain plugins.**
Development uses toy plugins (sum two numbers, compute column mean) to validate
the platform before any real plugin pack is connected.

**SGC-Core has zero dependencies on any other SGC repo or package.** It does
not depend on PyStatistics, PyStatsBio, or any plugin pack. Its only Python
dependencies are its own chosen stack (FastAPI/Django, Postgres driver,
Celery, etc.). The toy test plugins in `plugins/_dev/` use only stdlib and
possibly numpy. This means:

- SGC-Core can be built, tested, and deployed without any domain library
  installed
- The Core test suite runs fast — no scipy, no torch, no heavy dependencies
- A developer working on SGC-Core needs zero knowledge of biostatistics,
  finance, genomics, or any domain
- Plugin packs are installed into the worker environment at deployment time,
  not at development time

---

## 2. Plugin Architecture

### Principle

**Every analysis type is a plugin, including all built-in ones.**

There is no "first-class" vs "second-class" distinction. There are no special
code paths for built-in analyses. The Skyrim model: the game ships as a mod of
itself.

### What a Plugin Is

An analysis plugin consists of:

1. **A Python callable** that takes structured input (parameters dict +
   DatasetHandle) and returns a structured result (dict serializable to JSON)
2. **A parameter schema** (JSON Schema) describing its inputs — types, ranges,
   defaults, descriptions, UI hints. The frontend renders forms from this
   schema automatically.
3. **A result schema** (JSON Schema) describing the output structure, enabling
   generic result rendering without analysis-specific templates. **Results are
   validated against this schema before storage** — non-conforming results are
   rejected.
4. **Metadata**: unique ID, display name, category, description, version,
   execution hints, plugin pack source

### Plugin Registry Lifecycle

**The plugin catalog is static and resolved at worker startup.** Plugin packs
are imported when the worker process boots. Registration happens at import
time. Changing the available plugins requires a container restart.

Do not build dynamic plugin loading, hot-reloading, or runtime plugin
discovery. Plugins are code. Changing code without restart is a footgun.

### Plugin Callable Contract

```python
from sgccore.plugins import registry, DatasetHandle

@registry.register(
    id="bio.pk.nca",
    name="Non-Compartmental Analysis",
    category="PK",
    description="AUC, Cmax, half-life, clearance, Vz",
    version="1.0.0",
    execution={"backend": "cpu"},
    parameter_schema={...},   # JSON Schema
    result_schema={...},      # JSON Schema
)
def run_nca(params: dict, dataset: DatasetHandle) -> dict:
    """Plugin callable — wraps pystatsbio.pk.nca()."""
    time = dataset.column_as_array("time")
    conc = dataset.column_as_array("concentration")
    ...
    return result_dict
```

### DatasetHandle: The Plugin Data Contract

**Plugins must never import Django, access the ORM, or know about storage
backends.** The worker loads the dataset file from storage, parses it into a
pandas DataFrame, and wraps it in a `DatasetHandle` — a minimal,
platform-defined object that plugins receive.

```python
class DatasetHandle:
    """Platform-provided data wrapper. Plugins receive this, not ORM models."""

    @property
    def df(self) -> pd.DataFrame:
        """The full dataset as a pandas DataFrame."""

    def column_as_array(self, name: str) -> np.ndarray:
        """Extract a single column as a numpy float64 array."""

    @property
    def columns(self) -> list[str]:
        """Available column names."""

    def column_metadata(self, name: str) -> dict:
        """Per-column metadata (units, type hints, display labels, etc.).

        Returns an empty dict if no metadata is available for the column.
        Populated from dataset upload metadata or inferred at parse time.
        Plugins use this for labeling (e.g. axis units, summary table headers).
        """

    @property
    def metadata(self) -> dict:
        """Dataset-level metadata (name, version, row count, etc.)."""
```

**Key rules:**

- `DatasetHandle` is defined in `sgccore.plugins` — the only SGC-Core module
  that plugins are allowed to import
- It wraps a pandas DataFrame but does not expose Django models, file paths,
  or storage details
- The worker is responsible for loading the file, parsing CSV/Excel into a
  DataFrame, and constructing the DatasetHandle before calling the plugin
- Plugins interact with data exclusively through this interface

### Result Validation

**Plugin results are validated against the declared result_schema before
storage.** The worker runs JSON Schema validation on the returned dict. If the
result does not conform:

- The job is marked as failed
- The validation error is logged
- The non-conforming result is NOT stored

This prevents silent drift between plugin versions and ensures the frontend
can always render what it expects. Validation on output is just as important
as validation on input.

### Execution Metadata

Plugin metadata includes an `execution` dict instead of a simple boolean GPU
flag. This is cheap to define now and avoids a migration later.

```python
execution = {
    "backend": "cpu",        # "cpu" | "gpu" | "either"
    # Future fields (not implemented in v1, but the dict is extensible):
    # "resource_profile": "small" | "large",
    # "timeout_seconds": 300,
}
```

For v1:

- `"cpu"` → route to CPU worker
- `"gpu"` → route to GPU worker, fall back to CPU if unavailable
- `"either"` → worker decides based on availability (same as "gpu" with
  fallback in v1, but semantically distinct for future optimization)

The `execution` dict is extensible. Future fields (resource profiles, timeout
hints, memory requirements) can be added without changing the registration API.

### Architectural Rules

1. **Analysis types are data, not code paths.** No `if analysis_type ==
   "power_t_test"` switch statements anywhere. A registry maps analysis type
   IDs to handler functions. Adding a new analysis type means adding a
   registry entry, not modifying dispatcher code.

2. **Parameter schemas are declarative.** The frontend renders forms from
   schemas. There are zero hardcoded form components per analysis type. If
   the schema says `"float, min=0, max=1, default=0.05, label='Significance
   level'"`, the frontend renders an appropriate input.

3. **Result rendering is generic.** The result viewer renders any structured
   result that conforms to the result schema. No analysis-specific display
   templates. Key-value pairs render as tables. Summary strings render as
   text.

4. **The worker is analysis-agnostic.** It receives: plugin reference,
   parameters, dataset reference. It loads the dataset into a DatasetHandle,
   executes the plugin callable, validates the result against the schema,
   and stores it. It does not know or care what it is running.

5. **Plugins are sandboxed from the platform.** Plugins may only import
   `sgccore.plugins` (for `DatasetHandle` and `registry`). They must not
   import Django, access the database, or interact with storage directly.

### Extensibility Exposure (Staged)

| Phase | What's Exposed |
|-------|---------------|
| v1 | Nothing — internal plugin system only, no public API |
| Future | Plugin developer docs, upload/registration endpoint, sandboxing |

Because the internals use the plugin architecture from day one, opening it up
is documentation + one API endpoint, not a rewrite.

---

## 3. Deployment Architecture

### Principle

**SaaS-first, on-prem-capable from day one.**

A single codebase supports both. Containerized deployment with abstracted
storage and configuration.

### System Components

```
┌─────────────────────────────────────────────┐
│  Browser (Web Frontend)                     │
│  - Stateless SPA                            │
│  - Schema-driven forms (from plugin specs)  │
│  - Generic result renderer                  │
│  - Responsive (mobile read-only dashboard)  │
└─────────────┬───────────────────────────────┘
              │ HTTPS
┌─────────────▼───────────────────────────────┐
│  API Backend (SGC-Core)                     │
│  - Authentication & authorization           │
│  - Project/dataset/analysis CRUD            │
│  - Plugin registry (serves schemas to UI)   │
│  - Job submission & routing                 │
│  - Result retrieval                         │
│  - Audit logging                            │
└──────┬──────────────────┬───────────────────┘
       │                  │
┌──────▼──────┐    ┌──────▼──────────────────┐
│  Database   │    │  Worker Layer            │
│  (Postgres) │    │  - CPU workers           │
│             │    │  - GPU workers           │
└─────────────┘    │  - Analysis-agnostic:    │
                   │    receives plugin ref + │
┌─────────────┐    │    params, returns result│
│  Object     │    │  - Imports plugin packs  │
│  Storage    │    │    at startup            │
│  (S3 / local│    └─────────────────────────┘
│   filesystem│
│   abstract) │
└─────────────┘
```

### Tenant Model

**Multi-tenant with strong logical isolation.**

- Row-level security in Postgres (organization_id on every table)
- Schema-per-tenant if a customer demands stronger isolation
- Single-tenant container isolation only for on-prem or regulated customers
- Do not run N copies of everything at early stage

### SaaS Infrastructure (Initial)

- **AWS** (single region to start)
- EC2 or ECS for backend + workers
- RDS Postgres for database
- S3 for object storage
- GPU instances on-demand (spot instances where possible)

### On-Prem Deployment

- Docker Compose for simple deployments
- Kubernetes-ready for enterprise IT
- All external dependencies (S3, RDS) have local equivalents (filesystem,
  local Postgres)
- Configuration via environment variables, not code changes

---

## 4. Tech Stack

### Backend

- **Django 5+** with **Django REST Framework (DRF)** for the API layer
  - Django provides: auth, ORM, migrations, admin panel, RBAC, session
    management — all out of the box
  - DRF provides: serializers, viewsets, permissions, browsable API
  - The admin panel doubles as an internal debugging/ops tool for free
  - Performance is irrelevant — the bottleneck is the Celery worker running
    analysis plugins, not the API. There will never be 10,000 concurrent
    requests; there will be 5 biostatisticians.
- **Postgres** for relational data (orgs, users, projects, datasets, analyses,
  audit log)
- **Celery + Redis** for async job queue
  - Redis as both Celery broker and result backend
  - Celery workers import plugin packs directly — no subprocess calls, no REST
    wrappers around the math
  - Workers are analysis-agnostic: receive plugin reference + params, execute,
    return result
- **Django's ORM** for all database access — no raw SQL, no SQLAlchemy

### Frontend

- **React** with **TypeScript**
  - Plain React SPA — no Next.js (SSR is unnecessary)
  - Schema-driven form renderer: reads JSON Schema from API, generates UI
    components automatically
  - Generic result viewer: renders any structured result without
    analysis-specific templates
  - Communicates with backend via DRF REST API
  - State management: React context or Zustand (decide at implementation time,
    keep it simple)

### Infrastructure

- **Docker** for all services
- **docker-compose.yml** for local development and simple deployment
  - Services: `web` (Django), `worker` (Celery), `redis`, `postgres`,
    `frontend` (nginx serving React build)
- Container-ready from first commit
- `docker-compose up` must produce a fully working system

### Key Library Versions (Pin at Implementation Time)

- Python 3.12+
- Django 5.x
- djangorestframework 3.15+
- celery 5.x
- redis (via django-redis)
- psycopg 3.x (Postgres driver)
- React 18+
- TypeScript 5+

---

## 5. UI Platform

**Web application only.**

- No native apps (macOS, Windows, iOS)
- No Electron

The UI must:

- Be browser-based (zero-install for enterprise environments)
- Be responsive — read-only mobile dashboard acceptable (check job status
  from phone)
- Work in Chrome, Firefox, Safari (latest two versions)
- Render all forms from plugin parameter schemas (zero hardcoded form
  components per analysis type)
- Render all results generically from result schemas (zero analysis-specific
  display templates)

---

## 6. Core Feature Set

Sequenced by dependency order. All features are domain-agnostic.

### Phase 1a: Foundation

1. **Plugin registry**
   - Register plugins with: callable, parameter schema, result schema,
     metadata (name, category, description, GPU flag)
   - Serve plugin catalog to frontend (list of available analyses + schemas)
   - Built and tested first, with toy plugins in `plugins/_dev/`

2. **Authentication system**
   - Organization-based accounts
   - Role-based access control: admin, analyst, viewer
   - Email/password auth initially (add SSO later)

3. **Project model**
   - Projects belong to an organization
   - Projects contain datasets and analyses
   - Strict project isolation (users only see their org's projects)

4. **Dataset registration**
   - Upload CSV/Excel files
   - Store file in object storage, metadata in Postgres
   - Immutable once registered (new version = new upload)
   - Dataset versioning: simple monotonic version counter, not git

### Phase 1b: Execution Engine

5. **Job queue + worker dispatch**
   - Async execution via Celery worker
   - Job states: queued, running, completed, failed
   - Worker receives: plugin ID, parameters, dataset reference
   - Worker loads dataset from storage into DatasetHandle (pandas DataFrame
     wrapper — see Section 2)
   - Worker executes the registered plugin callable with params + DatasetHandle
   - GPU routing: read `execution.backend` from plugin metadata, dispatch
     accordingly
   - CPU fallback if GPU unavailable
   - Job status visible in UI (and on mobile)

6. **Result storage + viewer**
   - Worker validates plugin result against the declared result_schema
     (JSON Schema validation) before storage — non-conforming results are
     rejected and the job marked as failed
   - Store validated result artifact (JSON) in object storage
   - Generic result renderer: key-value tables, summary text, downloadable
     JSON
   - No analysis-specific display templates
   - No fancy charts in v1 — structured tables are fine

7. **Audit log**
   - Every analysis run recorded automatically
   - Fields: who, when, which dataset (version), which plugin, parameter
     snapshot, plugin pack version, result artifact ID
   - Immutable (append-only table, no deletes, no updates)
   - Viewable in UI

### Phase 1c: Polish

- Export results to CSV
- Re-run analysis with modified parameters (clone + edit)
- Basic dashboard: list of recent analyses, job status overview
- Organization settings (invite users, manage roles)

### What NOT to Build in SGC-Core

- PDF report generators
- Data visualization / charting platform
- AI recommender ("you should use analysis X")
- Enterprise workflow engine (approval chains, sign-offs)
- Multi-cloud abstraction layer
- Batch execution / pipeline orchestration
- Plugin marketplace or public plugin API (architecture supports it; expose
  when ready)
- Plugin developer documentation (write when opening up extensibility)
- Any domain-specific logic whatsoever

---

## 7. Compliance & Governance

These are platform features — domain-agnostic, built into SGC-Core.

- **Immutable audit trail** — append-only Postgres table, no DELETE/UPDATE
- **Dataset versioning** — monotonic version counter per dataset, old versions
  never deleted
- **Analysis parameter snapshot** — full parameter dict stored as JSONB with
  each analysis run
- **Plugin version tracking** — plugin pack version recorded with each run
- **User access controls** — RBAC with org isolation
- **Project isolation** — row-level security, no cross-org data leakage

**What we do NOT need yet:**

- 21 CFR Part 11 (electronic signatures) — add when regulated customers appear
- HIPAA — add when PII/PHI handling is required
- SOC 2 — important eventually, not for design partner phase

---

## 8. GPU Strategy

GPU compute is handled at the infrastructure layer, transparent to the user.
Plugin metadata declares backend requirements via the `execution` dict (see
Section 2: Execution Metadata).

SGC-Core must:

- Read `execution.backend` from plugin metadata at job submission time
- Route `"gpu"` and `"either"` jobs to GPU worker if available
- Fall back to CPU transparently if GPU worker is unavailable
- Route `"cpu"` jobs directly to CPU worker (never to GPU)
- Log which backend was actually used in the audit trail

GPU-capable plugins are defined by plugin packs, not by SGC-Core. SGC-Core
just routes based on the metadata.

---

## 9. Design Philosophy

The platform assumes:

- The user understands their domain methodology
- The user is responsible for analysis choice
- The UI removes infrastructure friction, not professional judgment

The UI must:

- Expose the full plugin catalog for installed plugin packs
- Not block "incorrect" analysis choices
- Provide parameter validation (type checking, range checking from schema)
  but not domain-specific advice
- Show clear, structured results — not black-box summaries

---

## 10. Development Principles

1. **Core is domain-agnostic.** SGC-Core never imports any domain library. It
   only knows about plugins, schemas, and results.
2. **Separate computation from presentation.** The API layer never does math.
   Workers execute plugin callables and return results.
3. **Structured result objects only.** No parsing text output. Plugins return
   JSON-serializable structured results.
4. **Plugin-first internals.** Every analysis type goes through the plugin
   registry. No special code paths for any analysis.
5. **Schema-driven UI.** The frontend renders forms from declarative parameter
   schemas. Adding an analysis type never requires frontend code changes.
6. **Container-ready from first commit.** `docker-compose up` must work.
7. **Single architecture for SaaS and on-prem.** Environment variables, not
   code branches.
8. **Tests from day one.** API tests, integration tests, plugin registration
   tests. Toy plugins are the test fixtures.

---

## 11. Repository Structure

```
sgccore/
  core/
    auth/             ← authentication, RBAC, org management
    projects/         ← project CRUD, isolation
    datasets/         ← dataset upload, versioning, storage abstraction
    plugins/          ← plugin registry, schema engine, dispatch
    jobs/             ← job queue, worker management, GPU routing
    audit/            ← immutable audit log
    results/          ← result storage, retrieval
    api/              ← REST API endpoints (ties everything together)
  plugins/
    _dev/             ← toy plugins for platform testing
  frontend/           ← web UI (SPA, TypeScript, schema-driven)
  docker/             ← Dockerfiles, docker-compose.yml
  tests/
    core/             ← platform tests
    plugins/          ← plugin registration tests
    integration/      ← end-to-end tests with toy plugins
```

---

## 12. Development Machines

- **Powerhouse** (Mac Studio M2 Max, 96GB) — primary development, MPS GPU
- **Forge** (AMD Ryzen 5 7600X, RTX 5070 Ti, Ubuntu 24.04) — CUDA GPU,
  staging/CI target, production-like environment

---

## 13. Relationship to Plugin Packs

SGC-Core defines the plugin contract. Plugin packs implement it.

- SGC-Core provides: registry API, worker execution, schema validation,
  result storage, audit logging
- Plugin packs provide: callables, parameter schemas, result schemas, metadata
- Plugin packs are installed into the worker environment (via pip or as a
  local package in a monorepo)
- SGC-Core never depends on any plugin pack. Plugin packs depend on SGC-Core's
  registry API.

**Dependency direction is strictly one-way:**

```
SGC-Bio  ──depends on──▶  SGC-Core
SGC-Bio  ──depends on──▶  PyStatsBio  ──depends on──▶  PyStatistics

SGC-Core ──depends on──▶  (nothing in our ecosystem)
```

SGC-Core is the root. It can be developed, tested, and deployed in complete
isolation from every other repo. Plugin packs are added at deployment time
by installing them into the worker environment and listing them in
configuration.

The first plugin pack is **SGC-Bio** (see `SGC_BIO_DIRECTIVE.md`).

---

*This document governs SGC-Core development. All platform architectural
decisions should reference it. Update it when strategy evolves.*
