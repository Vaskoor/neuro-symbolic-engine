# Architecture Decision Records (ADRs)

> A catalogue of the principal design choices in the Neuro-Symbolic Reasoning Engine, written in the ADR style (Michael Nygard, 2011). Each record carries the same structure: **Context · Decision · Status · Consequences · Alternatives Considered**.

---

## ADR-001 — Use RDFLib for the Knowledge Graph

**Date:** 2024-10-01 · **Status:** Accepted

### Context
The engine must persist triples, load them from Turtle, answer SPARQL 1.1 queries, and expose OWL/SHACL hooks. We want a Python-only, offline-installable, well-tested library — not a Java triple-store or a hosted graph service.

### Decision
Use **[RDFLib](https://rdflib.readthedocs.io/)** 7.x as the in-memory graph backend. The `KnowledgeGraph` class is a thin wrapper exposing:
- `add_triple(s, p, o)`,
- `load_turtle(path)`,
- `query_sparql(q) -> list[dict]`,
- `triples(s=None, p=None, o=None) -> Iterator[Triple]`,
- `__len__` returning the number of triples,
- `serialise(format="turtle")` for roundtrip.

### Consequences
- ✅ Pure Python, offline, MIT-licensed, in PyPI since 2002.
- ✅ Implements SPARQL 1.1 + RDFS/OWL-RL entailment out of the box.
- ✅ Plugs into `pySHACL` for SHACL validation (optional, not required at runtime).
- ⚠️ RDFLib's `Graph` is single-process — no concurrent writers. The API must serialise writes via a lock.
- ⚠️ SPARQL queries can be slow on very large graphs; we mitigate by keeping the demo ontology small and recommending a remote SPARQL endpoint for production scale.

### Alternatives Considered
- **Oxigraph** — faster, Rust-backed, but harder to install in offline CI.
- **Apache Jena/Fuseki** — Java, network service; violates the "single Python process" constraint.
- **Custom triple store** — would need to re-implement SPARQL parsing; absurd for a 1k-line project.

---

## ADR-002 — Use Datalog (with OWL 2 RL rule templates) for Rule-Based Inference

**Date:** 2024-10-02 · **Status:** Accepted

### Context
We need rule-based reasoning over the KG that is (a) explainable (each derived fact carries a derivation chain), (b) decidable (terminates on every input), and (c) simple enough to maintain in <300 lines. Full OWL 2 DL reasoning is undecidable in the worst case and would pull in a heavyweight reasoner (Pellet, HermiT).

### Decision
Implement a small **positive Datalog interpreter** with **semi-naive fixpoint forward-chaining**. Add a thin layer of OWL 2 RL rule templates (subclass transitivity, property domain/range, inverse-of, symmetric, transitive, disjointness) so that a pure-Datalog forward pass reproduces the common OWL entailments the demo ontology exercises. Rules live in `config/rules.yaml` in a `head :- body1, body2, …` syntax.

### Consequences
- ✅ Decidable — fixpoint forward-chaining always terminates.
- ✅ Explainable — every derived fact carries a `ProofStep(rule, premises, conclusion)`.
- ✅ Easy to test — the test-suite verifies each of the 10 rules derives its expected fact.
- ⚠️ No negation, no aggregation. Realistic domain ontologies with default reasoning would need stratified Datalog; out of scope for v0.1.
- ⚠️ OWL 2 RL ≠ OWL 2 DL. Some ontologies require DL-only features (cardinality on transitive properties, etc.); the engine will not derive those.

### Alternatives Considered
- **Pellet/HermiT (OWL-DL reasoner)** — Java; needs network or heavy install; not in scope for a Python offline engine.
- **swipl (SWI-Prolog)** — would require an external runtime; complicates the Docker image.
- **Apache Jena Rules** — same issue as ADR-001.
- **Custom Python OWL reasoner** — too much surface area for one person.

---

## ADR-003 — Use TransE for Knowledge-Graph Embeddings

**Date:** 2024-10-03 · **Status:** Accepted

### Context
We want a small, well-understood KGE method that:
- trains in seconds on the demo ontology (≈50 triples),
- exposes a `score(s, p, o)` in a comparable range across entities,
- has a clear, citable reference (Bordes et al., 2013),
- can be implemented in pure `numpy` so the test-suite runs without `torch`.

### Decision
Implement **TransE** (Bordes et al., *Translating Embeddings for Modeling Multi-relational Data*, NeurIPS 2013) in `neurosym.neural.embeddings.TransEEmbedder`. The model embeds each entity `h` and each relation `r` as a `d`-dim vector; the score of a triple `(h, r, t)` is `-||h + r - t||_p` (typically `p=2`). Training minimises a margin-based ranking loss against randomly corrupted negatives. Pure numpy + `numpy.random.default_rng(seed)`.

### Consequences
- ✅ Trivially implementable in ~80 lines of numpy.
- ✅ Deterministic for a fixed seed; the test-suite can pin the seed and assert reproducibility.
- ✅ Strong baseline for **1-to-1** relations; still used as the canonical KGE reference.
- ⚠️ TransE is **provably unable** to model symmetric (`spouse_of`), 1-to-N (`parent_of`), and N-to-1 (`born_in`) relations. The README and `DOCS/research_notes.md` document this explicitly.
- ⚠️ Margin and learning rate need tuning; defaults (margin=1.0, lr=0.01, epochs=200, dim=64) work for the demo but are not optimal.

### Alternatives Considered
- **RotatE** (Sun et al., 2019) — handles symmetric/antisymmetric/inversion/composition; would be a better default. Rejected for v0.1 because the implementation is ~2× more complex (complex-valued embeddings).
- **ComplEx** (Trouillon et al., 2016) — uses complex-valued embeddings; same argument as RotatE.
- **DistMult** (Yang et al., 2015) — symmetric only; worse than TransE on most benchmarks.
- **ConvE** (Dettmers et al., 2018) — CNN-based; overkill for our scale.

The engine is structured so that swapping TransE → RotatE is a drop-in replacement behind the `Embedder` interface; we expect to ship RotatE in v0.2.

---

## ADR-004 — Numpy GNN Fallback when Torch is Unavailable

**Date:** 2024-10-04 · **Status:** Accepted

### Context
We want a graph-neural link predictor (`GNNLinkPredictor`) that:
- uses `torch` if available (real GNN training),
- falls back to a **pure-numpy** implementation when torch is absent — so the engine, the API, and the test-suite all run on a clean Python worker without GPU/CUDA,
- exposes the **same API surface** in both modes (`train`, `predict_link`).

### Decision
Implement `GNNLinkPredictor` with two backends selected at construction time:
1. `torch` backend — when `import torch` succeeds, use a small 2-layer GCN with mean aggregation and an MLP scoring head.
2. `numpy` fallback — a **single-hop message-passing model** that mirrors the GCN API: aggregate neighbour features (mean), apply a 1-layer transform, and score with a dot-product head. Trained by simple SGD on margin loss against corrupted negatives.

The backend is selected via `GNNLinkPredictor(force_numpy=True)` or automatically by probing `import torch`. Both backends return ranked candidates from `predict_link(s, p)`.

### Consequences
- ✅ The test-suite is runnable in offline CI without torch.
- ✅ The numpy fallback is **deterministic** (seeded RNG), so the engine has the same outputs in CI and production.
- ⚠️ The numpy fallback is a **caricature** of a GNN; do not benchmark it against real GNNs. Its purpose is to keep the API contract stable.
- ⚠️ In production, callers should set `force_numpy=False` and ensure `torch` is installed.

### Alternatives Considered
- **Hard-fail when torch absent** — violates the "runs offline" constraint.
- **Skip the GNN component entirely when torch absent** — would force callers to write conditional code; bad API.
- **Use `dgl` or `torch-geometric`** — heavy install, conflicts with the "no torch required" constraint.
- **Use `spektral` (TensorFlow)** — same issue with TF.

---

## ADR-005 — LLM Bridge with Deterministic Templates

**Date:** 2024-10-05 · **Status:** Accepted

### Context
The engine needs a natural-language fallback for queries that the KG/symbolic/neural layers cannot answer. We want:
- a deterministic, offline default (no API key, no network),
- an optional SDK-backed mode (`z-ai-web-dev-sdk`) for richer NL answers,
- the LLM answer **always** marked as `source="llm_fallback"` and `confidence<0.5` so downstream consumers cannot accidentally treat it as a high-trust fact.

### Decision
Implement `LLMBridge` with two modes:
1. **Deterministic templates** (default) — hand-written prompt-and-parse templates that:
   - recognise common question shapes ("Is X a Y?", "What does X do?", "List all X"),
   - emit a structured `ReasoningResult` with `answer`, `source="llm_fallback"`, `confidence=0.4`,
   - never call out to a network or a clock.
2. **Optional SDK** — when `z-ai-web-dev-sdk` is installed **and** `NEUROSYM_LLM_ENABLED=1`, the bridge routes the prompt to the hosted model. The response is **still** wrapped with `source="llm_fallback"` and a templated proof step, so the answer is auditable and distinguishable from a high-trust fact.

### Consequences
- ✅ The engine runs offline in CI; the LLM component is a no-op until explicitly enabled.
- ✅ Source attribution makes the LLM **always** a "soft" answer — protects downstream decision systems.
- ⚠️ In offline mode, the "LLM" is **not actually a language model**; the answers are intentionally generic and low-confidence.
- ⚠️ When the SDK is enabled, the engine has a network dependency and may have non-deterministic responses (the SDK itself should set `temperature=0` to mitigate).

### Alternatives Considered
- **Always-on LLM with caching** — violates offline constraint.
- **No LLM fallback at all** — leaves a UX gap (users get "I don't know" for genuinely unknown queries).
- **Multiple LLM backends** — premature; v0.2 work.

---

## ADR-006 — Fixpoint Forward-Chaining for the Datalog Engine

**Date:** 2024-10-06 · **Status:** Accepted

### Context
A Datalog program defines a unique least-Herbrand-model model. The natural implementation is fixpoint iteration: repeatedly apply all rules to the current fact set until no new facts are derived.

### Decision
Implement **semi-naive fixpoint forward-chaining**:
1. Maintain `facts: set[Triple]` and `delta: set[Triple]` (newly-added facts this iteration).
2. In each iteration, join rule bodies against `delta` (semi-naive optimisation: at least one body atom must match a `delta` fact), generating new candidates.
3. Add candidates not already in `facts` to `delta` for the next iteration.
4. Stop when `delta` is empty (the fixpoint is reached).

### Consequences
- ✅ Always terminates — the Herbrand base is finite (closed by the union of constants appearing in facts and rules).
- ✅ Idempotent — calling `forward_chain()` twice in a row yields identical results.
- ✅ Explainable — every derived fact carries a derivation chain we can reconstruct as a proof.
- ⚠️ Semi-naive is correct but not optimal; the engine is fine for ontologies with ≤10⁴ triples. Above that, consider magic-sets or DRed for incremental updates.

### Alternatives Considered
- **Magic-sets rewriting** — much faster for selective queries; out of scope for v0.1.
- **Bottom-up vs. top-down (SLD resolution)** — bottom-up is simpler and yields all derivable facts in one pass, which suits our use case (we want the closure, not a single proof).
- **DRed (delete-rederive)** — needed for incremental updates; v0.2 work.

---

## ADR-007 — Manual SHACL Evaluator for a Small Shape Subset

**Date:** 2024-10-07 · **Status:** Accepted

### Context
The engine needs to expose a `/validate` endpoint that checks the KG against a small set of SHACL shapes. Full SHACL is a complex specification (≈100+ constraint components with `sh:property`, `sh:node`, `sh:not`, etc.). We do not want to reimplement all of it.

### Decision
Implement a **manual SHACL evaluator** that handles the **five most common constraint components**:
- `sh:class` (target must have `rdf:type` of the given class),
- `sh:minCount` / `sh:maxCount` (lower/upper bounds on a property's values),
- `sh:nodeKind` (IRI / BlankNode / Literal),
- `sh:datatype` (literal must have the given XSD datatype),
- `sh:in` (value must be in an enumeration).

For full SHACL, optionally delegate to `pySHACL` when installed (`pip install pyshacl`). The manual evaluator returns a `ValidationReport` with `conforms: bool` and a list of `Violation` records.

### Consequences
- ✅ Zero mandatory dependencies beyond RDFLib.
- ✅ The five supported constraints cover the majority of practical SHACL use-cases.
- ⚠️ Shapes using unsupported constraints (e.g. `sh:sparql`, `sh:qualifiedValueShape`) will be silently ignored unless `pySHACL` is installed.
- ⚠️ The manual evaluator does not support SHACL Advanced (targeting by SPARQL, shape-based targets).

### Alternatives Considered
- **Always require `pySHACL`** — adds a runtime dep; minor, but we want the core engine installable from `requirements.txt` only.
- **Skip SHACL entirely** — `/validate` would be a 501; bad UX.
- **Wrap `pySHACL` and require it for the manual subset too** — over-engineering for v0.1.

---

## ADR-008 — Belief Cascade: KG → Symbolic → Neural → LLM Fallback

**Date:** 2024-10-08 · **Status:** Accepted

### Context
The engine has four sources of belief with very different epistemic profiles. We need a single `query(text)` that returns a single `ReasoningResult`, but we must preserve source attribution so callers can route answers differently.

### Decision
`NeuroSymbolicEngine.query()` walks the four sources in **strict order of decreasing trust**:
1. **Knowledge Graph** — direct triple lookup. If found, return `source="knowledge_graph"`, `confidence=1.0`.
2. **Symbolic Reasoner** — Datalog + OWL inference + SHACL constraints. If a derivation exists, return `source="symbolic"`, `confidence=0.95`.
3. **Neural Layer** — TransE score + GNN prediction. If a confident candidate exists (score ≥ threshold), return `source="neural"`, `confidence=score`.
4. **LLM Fallback** — templated NL answer (or SDK-backed). Always returns `source="llm_fallback"`, `confidence≤0.5`.

Every `ReasoningResult` carries `source`, `proof`, `trace_id`, `latency_ms`, `confidence`.

### Consequences
- ✅ Callers can trivially filter on `source` for high-stakes decisions.
- ✅ Proof traces are always available (for KG/symbolic; templated for neural/LLM).
- ✅ Latency is bounded — the cascade short-circuits at the first source that yields an answer.
- ⚠️ The cascade order assumes KG > symbolic > neural > LLM in trust. This is a **policy choice**, not a theorem — some applications may prefer neural over symbolic (e.g. when the KG is sparse). The engine exposes `engine.query(text, policy=...)` for callers to override (v0.2 work).

### Alternatives Considered
- **Parallel query + vote** — would require a confidence calibration step across sources; more complex.
- **Always return all four answers** — clutters the response; bad UX for the simple case.
- **Hard-coded LLM-first** — would defeat the "explainable" goal; rejected.

---

## ADR-009 — `pydantic` for All Public Data Models

**Date:** 2024-10-09 · **Status:** Accepted

### Context
The API returns structured `ReasoningResult`, `ValidationReport`, `HealthResponse` etc. We want runtime type validation, JSON-schema generation for OpenAPI, and ergonomic construction.

### Decision
Use **pydantic v2** `BaseModel` for every public data model in `neurosym.core.types`. The FastAPI app binds request/response models directly to these types — automatic OpenAPI generation comes for free.

### Consequences
- ✅ OpenAPI schema is generated automatically; the `/docs` UI is rich.
- ✅ Runtime validation catches malformed requests before they hit the engine.
- ✅ `model_dump()` makes JSON serialisation trivial.
- ⚠️ Pydantic v2 has a non-trivial cold-start cost (~10 ms) on import. Acceptable for a long-running API.

### Alternatives Considered
- **`dataclasses` + manual JSON schema** — less boilerplate but loses runtime validation.
- **`msgspec`** — faster, but smaller ecosystem; not worth the friction for v0.1.
- **`attrs` + `cattrs`** — same trade-off as dataclasses.

---

## ADR-010 — FastAPI for the HTTP Layer

**Date:** 2024-10-10 · **Status:** Accepted

### Context
The engine needs an HTTP API for (a) the `/query` cascade, (b) `/assert`, (c) `/sparql`, (d) `/reason`, (e) `/explain`, (f) `/validate`, (g) `/health`. We want OpenAPI docs, async middleware, dependency injection, and an active community.

### Decision
Use **FastAPI** with **uvicorn** as the ASGI server. The app is constructed in `neurosym.api.main.create_app()` and exposed as `app` for `uvicorn neurosym.api.main:app`. Middleware: CORS, structured logging, trace-ID header. The engine is **lazily constructed** on first request so the module imports cleanly even if the core layer is not yet wired (CI smoke test).

### Consequences
- ✅ OpenAPI UI at `/docs`, ReDoc at `/redoc`.
- ✅ Pydantic models compose directly with route signatures.
- ✅ Async middleware (CORS + logging + trace-ID) is trivial.
- ⚠️ Single-worker uvicorn in development; production should run `uvicorn --workers N` or `gunicorn -k uvicorn.workers.UvicornWorker`.

### Alternatives Considered
- **Flask** — synchronous, less ergonomic with pydantic.
- **Starlette (raw)** — lower-level; we'd re-implement much of what FastAPI provides.
- **Litestar** — newer; ecosystem is smaller.
- **aiohttp** — fine, but a separate ecosystem from pydantic/OpenAPI tooling.
