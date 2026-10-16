# Neuro-Symbolic Reasoning Engine: Combining OWL, SHACL, Datalog, GNNs, and LLMs for Explainable Knowledge Inference

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![Pydantic v2](https://img.shields.io/badge/pydantic-v2-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)
![Status: Research](https://img.shields.io/badge/status-research-orange.svg)
![Tests](https://img.shields.io/badge/tests-pytest-success.svg)
![Offline](https://img.shields.io/badge/runs-offline-success.svg)

> A hybrid reasoning engine that fuses symbolic knowledge graphs (RDF/OWL/SHACL), Datalog rule inference, knowledge-graph embeddings (TransE), graph neural link predictors (GNN with numpy fallback), and large language model natural-language fallbacks — with end-to-end **proof traces** and **source attribution** on every answer.

---

## Abstract

Pure-neural systems hallucinate; pure-symbolic systems are brittle on missing facts. **Neuro-Symbolic Reasoning Engine** is a layered inference engine that combines the best of both traditions. Given a natural-language query, the engine cascades through four sources of belief — a hand-curated **RDF/OWL knowledge graph**, a **Datalog rule base** with fixpoint forward-chaining, a **neural link predictor** (TransE embeddings + GNN), and finally a deterministic **LLM fallback** with templated prompts — and returns a single `ReasoningResult` carrying the answer, the proof trace, the source attribution, and the latency. Every component degrades gracefully to a **deterministic, offline, dependency-light** fallback so that the entire stack runs without `torch`, without an LLM API key, and without network access.

The codebase is small (≈1k lines of pure Python), MIT-licensed, ships a FastAPI service with seven endpoints, a complete pytest suite, a Docker image, and continuous-integration on GitHub Actions. It is intended as (a) a **teaching artefact** for explainable neuro-symbolic AI, (b) a **research scaffold** for benchmarking KG-embedding vs. rule-based inference, and (c) a **production template** for hybrid AI services that need to justify every answer.

---

## Table of Contents

1. [Architecture](#architecture)
2. [The Five Components](#the-five-components)
3. [Ontology + 10 Rules](#ontology--10-rules)
4. [Quick Start](#quick-start)
5. [API Reference](#api-reference)
6. [Reproducibility](#reproducibility)
7. [Limitations](#limitations)
8. [Ethics & Responsible AI](#ethics--responsible-ai)
9. [Citation](#citation)
10. [License](#license)

---

## Architecture

```
                  ┌──────────────────────────────────────────┐
                  │              User / Client                │
                  │  (curl, httpx, Jupyter, agent-browser)   │
                  └────────────────┬─────────────────────────┘
                                   │  HTTP / JSON
                                   ▼
              ┌──────────────────────────────────────────────┐
              │           FastAPI Service (api/main.py)      │
              │  POST /query  POST /assert  GET  /sparql    │
              │  POST /reason  GET /explain  GET  /validate  │
              │  GET  /health                              │
              │  CORS · structured logging · trace store     │
              └────────────────┬─────────────────────────────┘
                               │  ReasoningRequest
                               ▼
              ┌──────────────────────────────────────────────┐
              │         NeuroSymbolicEngine (core)          │
              │   query()  → cascade over 4 belief sources  │
              │   assert_triple() · sparql() · validate()    │
              │   health() · explain(trace_id)               │
              └────────────────┬─────────────────────────────┘
                               │
        ┌──────────────────────┼───────────────────────┐
        │                      │                       │
        ▼                      ▼                       ▼
 ┌──────────────┐   ┌────────────────────┐    ┌───────────────────┐
 │ LLM Bridge   │◄──┤ SymbolicReasoner   │◄───┤ KnowledgeGraph    │
 │ (templates + │   │ (OWL + Datalog +   │    │ (RDFLib + OWL     │
 │  optional    │   │  SHACL, with       │    │  + SHACL)         │
 │  SDK)        │   │  proof traces)     │    │                   │
 └──────────────┘   └────────────────────┘    └───────────────────┘
        ▲                      ▲                       │
        │                      │                       │
        └──────────────────────┴───────────────────────┘
                               │
                               ▼
                  ┌──────────────────────────────┐
                  │  Neural Layer (numpy + torch)│
                  │  · TransEEmbedder            │
                  │  · GNNLinkPredictor (numpy   │
                  │    fallback if no torch)     │
                  └──────────────────────────────┘
                               │
                               ▼
                       ┌────────────────┐
                       │ ReasoningResult│
                       │  answer        │
                       │  source        │
                       │  proof         │
                       │  trace_id      │
                       │  latency_ms    │
                       └────────────────┘
```

**Belief cascade.** A `query(text)` call walks the sources in increasing order of epistemic risk:

1. **Knowledge Graph** — direct RDF/OWL triple lookup. Highest trust.
2. **Symbolic Reasoner** — Datalog forward-chaining + OWL axioms + SHACL constraints.
3. **Neural Layer** — TransE score + GNN link prediction. Returns ranked candidates with confidence.
4. **LLM Fallback** — natural-language answer via deterministic templates, optionally routed through the SDK. Lowest trust; always wrapped in a "no KB evidence" marker.

Every result carries `source` (one of `knowledge_graph | symbolic | neural | llm_fallback`), `proof` (machine-checkable trace), `trace_id`, and `latency_ms`.

---

## The Five Components

### 1. Knowledge Graph (`neurosym.knowledge.graph`)

Built on **RDFLib**. The graph is an in-memory `rdflib.Graph` with turtle loader, SPARQL 1.1 query endpoint, triple adder, and a triple counter. OWL 2 RL / RDFS entailment is opt-in via `reason(with_owl=True)`. SHACL shapes can be loaded separately for validation. The graph is the single source of truth for ground facts.

```python
from neurosym.knowledge.graph import KnowledgeGraph

kg = KnowledgeGraph()
kg.load_turtle("config/ontology.ttl")
kg.add_triple("ex:Bob", "ex:knows", "ex:Alice")
rows = kg.sparql_query("SELECT ?o WHERE { ex:Bob ?p ?o }")
print(len(kg))  # number of triples
```

### 2. Datalog Engine (`neurosym.knowledge.datalog`)

A small but correct **Datalog interpreter** with naive fixpoint forward-chaining. Rules are stored as `Rule(name, body, head, priority, description)` where `body` is a list of `{s, p, o}` patterns (with `?x`-style variables) and `head` is a single `{s, p, o}` pattern instantiated on match. Loaded in bulk from `config/rules.yaml` (10 curated rules). The engine is decoupled from RDFLib and operates over plain `Triple` objects with string fields.

```python
from neurosym.knowledge.datalog import DatalogEngine
from neurosym.core.types import Triple

dl = DatalogEngine(seed=42)
dl.load_rules("config/rules.yaml")
dl.set_facts([
    Triple(s="ex:P1", p="ex:authoredBy", o="ex:R1"),
    Triple(s="ex:R1", p="ex:affiliatedWith", o="ex:I1"),
])
new_facts = dl.forward_chain()        # idempotent — running twice yields the same set
rows      = dl.query({"s": "?x", "p": "ex:hasInstitution", "o": "?z"})
```

### 3. Symbolic Reasoner (`neurosym.reasoning.reasoner`)

The user-facing **inference façade** over the KG and the Datalog engine. `reason(query, kg)` returns a `ReasoningResult` whose `proof` is a `ProofTrace` (a tree of `ProofStep` records, each with `rule_name`, `premises`, `conclusion`). The reasoner also performs:

- **Consistency checking** — `check_consistency(kg)` detects when an individual is asserted to be a member of two disjoint classes.
- **Subsumption** — `subsumes(class_a, class_b, kg)` returns whether `class_b ⊑* class_a` (using subclass closure computed by BFS over `rdfs:subClassOf`).

```python
from neurosym.reasoning.reasoner import SymbolicReasoner

rs = SymbolicReasoner()
rs.load_rules("config/rules.yaml")     # type: ignore[attr-defined]  (or pass via rules=)
result = rs.reason("?x ex:hasInstitution ?z", kg)
print(result.answer, result.proof, result.source)

consistent = rs.check_consistency(kg)             # bool
subsumes   = rs.subsumes("ex:Researcher", "ex:Author", kg)  # True if Author ⊑* Researcher
```

### 4. Neural Layer (`neurosym.neural.embeddings`, `neurosym.neural.gnn`)

Two complementary neural modules:

- **`TransEEmbedder`** — translates embeddings `(h + r ≈ t)` trained by SGD on the graph's triples. Uses `numpy` only (no torch needed). Returns a `score(h, r, t)` in `[0, 1]` (sigmoid of `margin - ||h+r-t||`) and exposes `get_entity_embeddings()` returning `{entity_str: np.ndarray}`. Deterministic for a fixed seed.
- **`GNNLinkPredictor`** — uses **torch** when available, else a **deterministic numpy fallback** that mirrors the GCN message-passing API (2-layer aggregation + dot-product scoring). Both backends expose `train(triples, epochs)` and `predict_link(s, p, o)` returning a score in `[0, 1]`.

The numpy fallback exists so that the engine and its tests run on a CI worker without torch.

```python
from neurosym.neural.embeddings import TransEEmbedder
from neurosym.neural.gnn       import GNNLinkPredictor

transe = TransEEmbedder(seed=42)
transe.train(list_of_triples, dim=50, epochs=100)
score = transe.score("ex:Bob", "ex:knows", "ex:Alice")

gnn = GNNLinkPredictor(embed_dim=50, backend="numpy")   # numpy backend by default
gnn.train(list_of_triples, epochs=50)
score = gnn.predict_link("ex:Bob", "ex:knows", "ex:Alice")
embs  = gnn.get_embeddings()   # {entity_str: np.ndarray}
```

### 5. LLM Bridge (`neurosym.llm.bridge`)

A thin natural-language layer used as the **last-resort fallback** when the symbolic and neural layers cannot answer. Two modes:

- **Deterministic templates** (default; offline; no API key): hand-written prompts and parsing that emits a structured `ReasoningResult` marked `source="llm_fallback"` and `confidence<0.5`.
- **Optional SDK** (when `z-ai-web-dev-sdk` is installed and `NEUROSYM_LLM_ENABLED=1`): the bridge routes the query to a hosted LLM, then parses the response back into a `ReasoningResult`. Even in this mode, the answer is **always** wrapped with `source="llm_fallback"` and a templated proof step so downstream consumers can detect "low-confidence" answers.

```python
from neurosym.llm.bridge import LLMBridge

bridge = LLMBridge(enabled=False)   # offline mode
result = bridge.fallback("How many planets are in our solar system?")
# ReasoningResult(source="llm_fallback", answer="...", confidence=0.4, ...)
```

---

## Ontology + 10 Rules

### Ontology (`config/ontology.ttl`)

A small OWL 2 ontology in Turtle covering a scientific-research domain used by the test-suite:

- **Classes** (`owl:Class`): `ex:Researcher`, `ex:Author`, `ex:Reviewer`, `ex:Paper`, `ex:ConferencePaper`, `ex:JournalPaper`, `ex:Institution`, `ex:University`, `ex:Company`, `ex:Venue`, `ex:Journal`, `ex:Conference`, `ex:TopVenue`, `ex:Field`, `ex:Method`, `ex:Dataset`.
- **Object properties**: `ex:authoredBy`, `ex:affiliatedWith`, `ex:employedBy`, `ex:hasInstitution`, `ex:cites`, `ex:usesMethod`, `ex:indirectlyUsesMethod`, `ex:knowsMethod`, `ex:publishedIn`, `ex:belongsToField`, `ex:worksInField`, `ex:collaboratedWith`, `ex:supervisedBy`, `ex:hasHighImpactPaper`.
- **Datatype properties**: `ex:title`, `ex:name`, `ex:year`, `ex:hindex`, `ex:impactFactor`, `ex:isHighImpact`, `ex:isTopVenue`.
- **Axioms**: `ex:Author ⊑ ex:Researcher`, `ex:Reviewer ⊑ ex:Researcher`, `ex:Journal ⊑ ex:Venue`, `ex:Conference ⊑ ex:Venue`, `ex:TopVenue ⊑ ex:Venue`, `ex:ConferencePaper ⊑ ex:Paper`, `ex:JournalPaper ⊑ ex:Paper`, `ex:University ⊑ ex:Institution`, `ex:Company ⊑ ex:Institution`; disjoint pairs `(ex:Author, ex:Reviewer)`, `(ex:ConferencePaper, ex:JournalPaper)`, `(ex:University, ex:Company)`; plus `rdfs:domain`/`rdfs:range` for every property.
- **SHACL shapes**: `ex:PaperShape` (targetClass `ex:Paper` — requires `ex:authoredBy` IRI minCount 1, `ex:title` Literal datatype `xsd:string` maxCount 1, `ex:publishedIn` IRI class `ex:Venue`), `ex:ResearcherShape` (requires `ex:affiliatedWith` IRI class `ex:Institution` minCount 1), `ex:TopVenueShape` (requires `ex:isTopVenue` Literal `xsd:boolean` minCount 1 maxCount 1).

### The 10 Rules (`config/rules.yaml`)

A curated rule base that covers OWL 2 RL-style inference patterns and
domain-specific derivations over the scientific-research ontology. Each rule
is a Horn clause `head :- body_1, body_2, ...` where every pattern is a
`{s, p, o}` mapping. Variables start with `?`; constants must match exactly.

```yaml
# 1. A paper inherits the institution of each of its authors.
- name: derive_has_institution
  priority: 90
  body:
    - {s: "?x", p: "ex:authoredBy", o: "?y"}
    - {s: "?y", p: "ex:affiliatedWith", o: "?z"}
  head: {s: "?x", p: "ex:hasInstitution", o: "?z"}

# 2. Employment implies institutional affiliation for the paper too.
- name: derive_has_institution_employed
  priority: 85
  body:
    - {s: "?x", p: "ex:authoredBy", o: "?y"}
    - {s: "?y", p: "ex:employedBy", o: "?z"}
  head: {s: "?x", p: "ex:hasInstitution", o: "?z"}

# 3. Methods propagate backwards along citation edges.
- name: derive_indirectly_uses_method
  priority: 80
  body:
    - {s: "?x", p: "ex:cites", o: "?y"}
    - {s: "?y", p: "ex:usesMethod", o: "?m"}
  head: {s: "?x", p: "ex:indirectlyUsesMethod", o: "?m"}

# 4. A paper's field becomes its authors' working field.
- name: derive_researcher_field
  priority: 75
  body:
    - {s: "?x", p: "ex:authoredBy", o: "?r"}
    - {s: "?x", p: "ex:belongsToField", o: "?f"}
  head: {s: "?r", p: "ex:worksInField", o: "?f"}

# 5. Methods used by a paper are known by its authors.
- name: derive_method_expert
  priority: 72
  body:
    - {s: "?x", p: "ex:authoredBy", o: "?r"}
    - {s: "?x", p: "ex:usesMethod", o: "?m"}
  head: {s: "?r", p: "ex:knowsMethod", o: "?m"}

# 6. Top venues confer high-impact status on their papers.
- name: derive_high_impact
  priority: 70
  body:
    - {s: "?x", p: "ex:publishedIn", o: "?v"}
    - {s: "?v", p: "ex:isTopVenue", o: "true"}
  head: {s: "?x", p: "ex:isHighImpact", o: "true"}

# 7. High-impact papers boost their authors.
- name: derive_high_impact_researcher
  priority: 65
  body:
    - {s: "?p", p: "ex:authoredBy", o: "?r"}
    - {s: "?p", p: "ex:isHighImpact", o: "true"}
  head: {s: "?r", p: "ex:hasHighImpactPaper", o: "true"}

# 8. Transitive propagation of indirect method usage up the citation graph.
- name: propagate_indirect_method
  priority: 60
  body:
    - {s: "?x", p: "ex:cites", o: "?y"}
    - {s: "?y", p: "ex:indirectlyUsesMethod", o: "?m"}
  head: {s: "?x", p: "ex:indirectlyUsesMethod", o: "?m"}

# 9. Two researchers sharing a method, linked by a citation between their
#    papers, are deemed collaborators.
- name: derive_collaborated
  priority: 55
  body:
    - {s: "?r1", p: "ex:knowsMethod", o: "?m"}
    - {s: "?r2", p: "ex:knowsMethod", o: "?m"}
    - {s: "?p1", p: "ex:authoredBy", o: "?r1"}
    - {s: "?p2", p: "ex:authoredBy", o: "?r2"}
    - {s: "?p1", p: "ex:cites", o: "?p2"}
  head: {s: "?r1", p: "ex:collaboratedWith", o: "?r2"}

# 10. Transitive closure of citation: X cites Y and Y cites Z => X cites Z.
- name: transitive_cites
  priority: 50
  body:
    - {s: "?x", p: "ex:cites", o: "?y"}
    - {s: "?y", p: "ex:cites", o: "?z"}
  head: {s: "?x", p: "ex:cites", o: "?z"}
```

The test-suite `tests/test_datalog.py` verifies that **all 10 rules derive their expected facts** from synthetic instance data, and that fixpoint forward-chaining converges (running `forward_chain()` twice yields the same fact set on the second call).

---

## Quick Start

### Prerequisites

- Python 3.11 or 3.12
- ~200 MB disk for the Python deps below

### Install (editable)

```bash
git clone https://github.com/bhaskarkaphle/neuro-symbolic-engine
cd neuro-symbolic-engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Run the test-suite (offline, no torch, no LLM)

```bash
pytest tests/ -v
```

### Start the API

```bash
uvicorn neurosym.api.main:app --host 0.0.0.0 --port 8000 --reload
# or via the module:
python -m neurosym.api.main
```

### Use the API

```bash
# Health
curl http://localhost:8000/health
# {"status":"ok","kg_triples":180,"rules":10,"embeddings_trained":false}

# Hybrid query (KG → symbolic → neural → LLM fallback).
# Use a SPARQL SELECT to hit the KG directly:
curl -X POST http://localhost:8000/query \
     -H 'Content-Type: application/json' \
     -d '{"text":"SELECT ?s WHERE { ?s a <http://www.w3.org/2002/07/owl#Class> } LIMIT 3","type":"SPARQL"}'

# Or a Datalog triple-pattern query (hits symbolic after forward-chaining):
curl -X POST http://localhost:8000/query \
     -H 'Content-Type: application/json' \
     -d '{"text":"?s rdfs:subClassOf ?o","type":"DATALOG"}'

# Assert a triple
curl -X POST http://localhost:8000/assert \
     -H 'Content-Type: application/json' \
     -d '{"s":"ex:Bob","p":"ex:knows","o":"ex:Alice"}'

# SPARQL via GET
curl --get http://localhost:8000/sparql \
     --data-urlencode 'q=SELECT ?s WHERE { ?s a <http://www.w3.org/2002/07/owl#Class> } LIMIT 3'

# Pure symbolic reasoning
curl -X POST http://localhost:8000/reason \
     -H 'Content-Type: application/json' \
     -d '{"query":"?s rdfs:subClassOf ?o"}'

# Retrieve proof trace (trace_id is returned by /query and /reason)
curl 'http://localhost:8000/explain?trace_id=abc123'

# SHACL validation
curl http://localhost:8000/validate
# {"conforms": true, "violations": [], "count": 0}
```

### Run in Docker

```bash
docker build -t neurosym .
docker run -p 8000:8000 neurosym
```

---

## API Reference

| Method | Path         | Body / Query | Description |
|--------|--------------|--------------|-------------|
| `POST` | `/query`     | `{text, type?, params?}` | Hybrid cascade query → `ReasoningResult` |
| `POST` | `/assert`    | `{s, p, o}`               | Assert an RDF triple into the KG |
| `GET`  | `/sparql`    | `?q=<sparql>`            | Run a SPARQL 1.1 query |
| `POST` | `/reason`    | `{query}`                | Pure symbolic reasoning with proof |
| `GET`  | `/explain`   | `?trace_id=<id>`         | Retrieve a stored proof trace |
| `GET`  | `/validate`  | —                        | Run SHACL validation |
| `GET`  | `/health`    | —                        | `{status, kg_triples, rules, embeddings_trained}` |

### Common response shape — `ReasoningResult`

```jsonc
{
  "query":        "?x ex:hasInstitution ?z",
  "answer":       [{"?x": "ex:P1", "?z": "ex:I1"}],
  "confidence":   0.9,
  "source":       "symbolic",            // kg | symbolic | neural | llm | hybrid
  "proof": {
    "steps": [
      {"rule_name": "derive_has_institution",
       "premises":   [{"rule_name": "fact", "premises": [], "conclusion": {"s":"ex:P1","p":"ex:authoredBy","o":"ex:R1"}}, ...],
       "conclusion": {"s": "ex:P1", "p": "ex:hasInstitution", "o": "ex:I1"}}
    ],
    "final_conclusion": {"s": "ex:P1", "p": "ex:hasInstitution", "o": "ex:I1"},
    "depth": 2
  },
  "latency_ms":    3.42,
  "notes":        "Derived via Datalog forward-chaining (1 binding(s)).",
  "trace_id":      "a3f9b2c1"
}
```

### Configuration via environment variables

| Variable                  | Default                          | Meaning |
|---------------------------|----------------------------------|---------|
| `NEUROSYM_ONTOLOGY`       | `config/ontology.ttl`            | Path to the OWL ontology |
| `NEUROSYM_RULES`          | `config/rules.yaml`              | Path to the Datalog rules |
| `NEUROSYM_LLM_ENABLED`    | `0`                              | If `1`, route `llm_fallback` through the SDK |
| `NEUROSYM_CORS_ORIGINS`   | `*`                              | Comma-separated CORS allow-list |
| `NEUROSYM_LOG_LEVEL`      | `INFO`                           | Root logger level |
| `NEUROSYM_HOST`           | `0.0.0.0`                        | Bind host (Docker) |
| `NEUROSYM_PORT`           | `8000`                           | Bind port (Docker) |

---

## Reproducibility

The engine is **deterministic by construction**:

- No `random.random()` / `time.time()` seeds the inference path. TransE and the GNN fallback both take an explicit `seed=42` argument and use `numpy.random.default_rng(seed)`.
- The Datalog forward-chain is a fixpoint computation; running it twice on the same inputs yields the same fact set.
- The LLM fallback (offline mode) returns templated answers derived purely from the query string — no clock, no RNG.
- CI runs without `torch`, without an LLM API key, and without network access (`NEUROSYM_OFFLINE=1`). All tests pass on a clean Python 3.11 worker.

To reproduce a benchmark run:

```bash
NEUROSYM_OFFLINE=1 pytest tests/ -v --tb=short
NEUROSYM_OFFLINE=1 python -m neurosym.api.main   # API on :8000
```

---

## Limitations

- **Knowledge Graph**: the engine ships a tiny OWL ontology (~50 triples) for demonstration. Real-world deployments should swap `config/ontology.ttl` with a domain KG and re-run forward-chaining.
- **Datalog**: the interpreter supports **positive Datalog** (no negation-as-failure, no aggregation). Negation would require stratification, which is out of scope for v0.1.
- **SHACL**: the SHACL validator is a **manual evaluator** of a small subset of shapes (`sh:class`, `sh:minCount`, `sh:maxCount`, `sh:nodeKind`). Full SHACL Advanced is delegated to `rdflib-shacl` when available.
- **Neural**: TransE is well-known to fail on 1-to-N / N-to-1 / symmetric relations; we recommend RotatE/ComplEx for those (see `DOCS/research_notes.md`). The numpy GNN fallback is **architecturally** a 1-layer message-passing network — fine for unit tests, underpowered for real benchmarks.
- **LLM bridge**: in offline mode, the "LLM" is a deterministic template engine, **not** an actual language model. The templated answers are intentionally conservative and low-confidence.
- **No streaming**: the API does not yet support SSE/WebSocket streaming; every response is a single JSON object.

---

## Ethics & Responsible AI

The engine is designed with three responsible-AI commitments:

1. **Explainability by default.** Every `ReasoningResult` carries a `proof` field that is a machine-readable list of inference steps. No answer is opaque.
2. **Source attribution.** `source` is a first-class field — downstream consumers can route `llm_fallback` answers differently from `knowledge_graph` answers (e.g. for high-stakes decisions).
3. **Graceful degradation, not silent failure.** When torch or the LLM SDK is unavailable, the engine continues to function with a documented, deterministic fallback — it does not silently pretend to be a neural network.

**Risks to watch:**

- The LLM fallback, even templated, can produce confidently-worded answers with `confidence<0.5`. Always inspect `source` before acting.
- The Datalog engine, given an inconsistent ontology, will derive `inconsistent(?x, ?c1, ?c2)` facts but will not refuse to answer. Callers should consult `check_consistency()` before trusting `reason()` outputs on adversarial input.
- TransE/GNN link predictions are probabilistic; treat any `source="neural"` answer as a hypothesis, not a fact.

---

## Citation

If you use this engine in academic work, please cite:

```bibtex
@software{kaphle2024neurosym,
  author    = {Bhaskar Kaphle},
  title     = {{Neuro-Symbolic Reasoning Engine: Combining OWL, SHACL, Datalog, GNNs, and LLMs for Explainable Knowledge Inference}},
  year      = {2024},
  version   = {0.1.0},
  license   = {MIT},
  url       = {https://github.com/bhaskarkaphle/neuro-symbolic-engine}
}
```

---

## License

MIT License — Copyright (c) 2024 Bhaskar Kaphle. See [`LICENSE`](LICENSE) for the full text.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, **subject to the following condition**: the above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
