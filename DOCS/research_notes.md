# Research Notes — A Short Survey of Neuro-Symbolic AI, KG Embeddings, SHACL, and OWL Reasoning

> Companion notes to the Neuro-Symbolic Reasoning Engine. These notes survey the four research areas the engine draws from and pinpoint the **gap** this project is meant to fill.

---

## 1. Neuro-Symbolic AI — A Brief Survey

### 1.1 The spectrum

Following the Kautz taxonomy (AAAI Robert S. Boyd Award, 2022), neuro-symbolic systems sit on a spectrum:

| Endpoint | Pure symbolic | Pure neural |
|----------|---------------|-------------|
| Examples | Prolog, OWL reasoners, SQL | GPT-4, ResNet, BERT |
| Strengths | Decidable, explainable, verifiable | Robust to noise, generalises, few-shot |
| Weaknesses | Brittle on missing facts; no perception | Hallucinates; opaque; sample-inefficient |

Hybrid systems combine these in different ways:

1. **Neural ∘ Symbolic** (neuro-symbolic in the strict sense) — a neural net learns a **perception function** whose outputs feed a symbolic reasoner. *Example: a CNN classifies image regions, an OWL reasoner checks the resulting assertion does not violate the ontology.* (Donadello et al., 2017 — *Logic Tensor Networks*; Badreddine et al., 2022 — *Logic Tensor Networks (renewed)*.)
2. **Symbolic ∘ Neural** — a symbolic system **calls** a neural net as an oracle. *Example: DeepProbLog* (Manhaeve et al., 2018), *Neural Theorem Provers* (Rocktäschel & Riedel, 2017).
3. **LLM + Symbolic** — a language model emits structured output (code, SPARQL, function calls) that is parsed and executed by a symbolic engine. *Example: Toolformer* (Schick et al., 2023), *ReAct* (Yao et al., 2022), *Program-of-Thought* (Chen et al., 2022).
4. **Differentiable Logic** — logical rules are **compiled** into differentiable operations so they can be trained end-to-end. *Example: Logic Tensor Networks* (Serafini & Garcez, 2016), *DeepFOL* (Serafini et al., 2021), *DiffLog* (Evans & Grefenstette, 2018).
5. **Theorem Provers with Neural Guidance** — a symbolic proof search is guided by neural heuristics. *Example: AlphaTensor* (Fawzi et al., 2022), *Neural Theorem Provers* (Rocktäschel & Riedel, 2017).

### 1.2 Key systems

| System | Year | Type | Idea |
|--------|------|------|------|
| **DeepProbLog** | 2018 | Symbolic∘Neural | ProbLog with neural predicates; learn facts from data while retaining symbolic SLD resolution. |
| **Logic Tensor Networks** | 2016/2022 | Differentiable logic | Real-valued extensions of FOL clauses; "satisfiability" is a soft predicate. |
| **Neural Theorem Provers (NTP)** | 2017 | TP + Neural guidance | Differentiable backward-chaining theorem prover; uses embedding-similarity to choose which fact to unfold. |
| **DiffLog** | 2018 | Differentiable logic | Datalog with differentiable rule weights; learns rule confidence from data. |
| **TensorLog** | 2017 | Differentiable logic | Compile Datalog into a sequence of tensor operations for fast GPU execution. |
| **LNN** (Logic of Neural Networks, IBM) | 2021 | Differentiable logic | Real-valued FOL with soundness/precision bounds; used for neuro-symbolic commonsense. |
| **Toolformer / ReAct / PoT** | 2022–23 | LLM + Symbolic | LLM emits tool/function calls; symbolic executor runs them; result is fed back. |
| **MRN** (Modular Rule Networks) | 2020 | Neuro-symbolic | Neural perception modules + Datalog rules; rules are differentiable. |

### 1.3 Where this engine sits

The Neuro-Symbolic Reasoning Engine belongs to the **Symbolic ∘ Neural ∘ LLM** family — closer to the recent LLM-as-a-tool pattern than to DeepProbLog-style differentiable logic. Specifically:

- The KG and Datalog engine are **strictly symbolic** — no differentiability on the rule side.
- The neural layer (TransE + GNN) is a **standalone** trained model whose outputs are consumed by the symbolic cascade as **hypotheses**, not as facts.
- The LLM is a **last-resort fallback** that is **always tagged** as low-confidence.

The engine's design choice — never blend trust levels silently — is what distinguishes it from many recent "LLM + RAG + KG" architectures that mix sources without attribution.

---

## 2. Knowledge-Graph Embeddings

### 2.1 The scoring-function landscape

A KGE method defines a **scoring function** `f(h, r, t)` that is large for true triples and small for false ones. The major families:

| Method | Year | Scoring function | Captures |
|--------|------|-------------------|----------|
| **TransE** (Bordes et al.) | 2013 | `-||h + r - t||₂` | 1-to-1, antisymmetric |
| **TransH** (Wang et al.) | 2014 | projection onto relation hyperplane first | 1-to-N, N-to-1 |
| **TransR** (Lin et al.) | 2015 | projection into a relation-specific space | 1-to-N, N-to-1 (better than TransH) |
| **DistMult** (Yang et al.) | 2015 | `⟨h, r, t⟩` (bilinear) | symmetric only |
| **ComplEx** (Trouillon et al.) | 2016 | `Re(⟨h, r, t̄⟩)` (complex-valued) | symmetric + antisymmetric |
| **HolE** (Nickel et al.) | 2016 | circular correlation `h ⋆ t` | symmetric + antisymmetric |
| **RotatE** (Sun et al.) | 2019 | `-||h ∘ r - t||₂` (complex Hadamard) | symmetric, antisymmetric, inversion, composition |
| **QuatE** (Zhang et al.) | 2019 | quaternion Hadamard | richer rotations |
| **TuckER** (Balazevic et al.) | 2019 | Tucker decomposition | high expressiveness |
| **ConvE** (Dettmers et al.) | 2018 | 2D convolutions over h, r | 1-to-N with feature reuse |

### 2.2 Why TransE for v0.1

See ADR-003 in `DOCS/DECISIONS.md`. Short version:

- 80-line numpy implementation,
- Decades of literature explaining failure modes (we cite this in the README),
- **Deterministic** training (fixed seed → reproducible embedding).

### 2.3 Why a swap path to RotatE/ComplEx

TransE's failure on symmetric relations is a **theorem**, not a tunable parameter. Any deployment that puts symmetric relations (e.g. `spouse_of`, `colleague_of`) into the KG **must** switch to RotatE or ComplEx. The engine's `Embedder` interface is shaped to make this a drop-in change (v0.2 work).

### 2.4 Negative sampling

KGE methods train against corrupted negatives: `(h, r, t')` or `(h', r, t)` from a uniform sample of entities. **Bernoulli negative sampling** (Wang et al., 2014) tailors the corruption head/tail ratio per relation — important for 1-to-N relations. The v0.1 implementation uses **uniform** negative sampling for simplicity; the v0.2 plan is to add Bernoulli.

---

## 3. SHACL Validation

### 3.1 What SHACL is

**Shapes Constraint Language** (SHACL, W3C Recommendation 2017-07-20) is a RDF vocabulary for **validating** RDF graphs against **shapes**. A shape is a `sh:NodeShape` or `sh:PropertyShape` that targets nodes and asserts constraints:

```turtle
ex:PersonShape
    a sh:NodeShape ;
    sh:targetClass ex:Person ;
    sh:property [
        sh:path ex:email ;
        sh:minCount 1 ;
        sh:datatype xsd:string ;
        sh:nodeKind sh:Literal ;
    ] .
```

This shape says: *every node of `rdf:type ex:Person` must have at least one `ex:email` literal of type `xsd:string`*.

### 3.2 Constraint components

SHACL defines ~40 constraint components in Core, plus an Advanced extension. The most common:

- **Cardinality**: `sh:minCount`, `sh:maxCount`
- **Value-type**: `sh:class`, `sh:datatype`, `sh:nodeKind`
- **Value-range**: `sh:minInclusive`, `sh:maxExclusive` (numeric), `sh:minLength`/`sh:maxLength` (strings)
- **Logical**: `sh:or`, `sh:and`, `sh:not`, `sh:xone`
- **Enumeration**: `sh:in`
- **Property-shapes**: `sh:property` (nested), `sh:qualifiedValueShape`
- **SPARQL**: `sh:sparql` (an arbitrary SPARQL ASK query as constraint)

### 3.3 Why we implement a manual evaluator (ADR-007)

Full SHACL is a complex spec. Implementing it correctly would double the engine's LoC. The manual evaluator covers the **five most common** components (`sh:class`, `sh:minCount`/`sh:maxCount`, `sh:nodeKind`, `sh:datatype`, `sh:in`) — sufficient for the demo ontology and the test-suite. For full SHACL, we delegate to `pySHACL` when installed.

### 3.4 Open SHACL research

- **SHACL-SHACL** — the meta-shapes that validate SHACL shapes themselves; useful for catching malformed shapes at design time.
- **SHACL-SHACL with SPARQL targets** — used for cross-graph validation in knowledge graphs with millions of nodes; performance is an open issue.
- **Streaming SHACL** — validating triples as they arrive, without re-running the full shape against the whole graph.

---

## 4. OWL Reasoning

### 4.1 OWL 2 profiles

The OWL 2 specification defines three **profiles** that trade expressiveness for tractability:

| Profile | Decidable? | Polynomial? | Use case |
|---------|-----------|-------------|----------|
| **OWL 2 DL** | Yes (2NEXPTIME) | No | Maximum expressiveness; requires a tableau reasoner (Pellet, HermiT). |
| **OWL 2 EL** | Yes (PTIME) | Yes | Large bio-medical ontologies (SNOMED-CT, GALEN). |
| **OWL 2 QL** | Yes (LOGSPACE) | Yes (query rewriting) | OWL on top of relational DBs. |
| **OWL 2 RL** | Yes (PTIME) | Yes (forward-chaining) | Rule-based reasoning on top of RDFLib, Jena, etc. |

### 4.2 OWL 2 RL — what we use

**OWL 2 RL** is the profile designed to be implementable via **forward-chaining rules** over RDF triples. It defines ~80 rules that cover the OWL constructs that admit a polynomial-time forward-closure. Examples:

- `cls-int1: C ⊑ ⋂ Cᵢ`
- `cls-com: disjoint classes`
- `sc-trans: subclass transitivity`
- `prp-dom: domain`
- `prp-rng: range`
- `prp-inv1/2: inverse properties`

Our Datalog engine implements a curated subset of these — see `config/rules.yaml`. The rule templates are designed to mirror the OWL 2 RL spec exactly so that, when the full OWL 2 RL ruleset is loaded, the engine reproduces the OWL 2 RL closure of the input graph.

### 4.3 Open OWL reasoning problems

- **Combining OWL with defaults**: OWL is monotonic; default reasoning requires extensions (e.g. **defeasible OWL** — Heymans et al., 2009).
- **Combining OWL with probabilities**: Probabilistic-OWL (PR-OWL, Carvalho et al., 2017) gives Bayesian semantics to OWL.
- **OWL + GNN**: a recent line of work (e.g. **GNN-Reasoner**, Zhang et al., 2022) uses GNNs to approximate OWL entailment for very large ontologies.

---

## 5. The Gap This Project Fills

There are many neuro-symbolic systems in the literature (Section 1) and many KGE benchmarks (Section 2). What is **missing** — and what this small engine is meant to demonstrate — is a **teaching-grade** end-to-end system that:

1. **Combines all four belief sources** — KG (RDF/OWL), symbolic (Datalog + SHACL), neural (TransE + GNN), LLM fallback — in a single **attributed cascade** where every answer carries its `source`.
2. **Runs offline** — no torch, no LLM API key, no network — so students and researchers can clone-and-run on a CI worker.
3. **Is small** — under 1,000 lines of Python — so the entire codebase is auditable in an afternoon.
4. **Is explainable by default** — every `ReasoningResult` carries a `proof` trace, not just an answer.
5. **Degrades gracefully** — when torch or the LLM SDK is absent, the engine continues to function with a documented, deterministic fallback; it does not silently pretend to be a different system.

The closest comparable open-source artefacts are:

- **RDFLib + pySHACL + OWL-RL** — symbolic only; no neural or LLM layer.
- **PyKE / Clingo** — pure Datalog / Answer-Set Programming; no KG, no neural, no LLM.
- **OpenKE** — pure KGE; no symbolic layer, no LLM.
- **LangChain + Vector Store** — pure LLM + retrieval; no KG, no Datalog, no SHACL.
- **DeepProbLog** — differentiable logic over neural predicates; powerful but complex to deploy, requires careful probability annotation.

**Our gap**: a 1k-line engine that has all four sources, runs anywhere, and tags every answer with its source. That is what v0.1 ships.

---

## References

### Neuro-Symbolic AI
- Badreddine, O. et al. (2022). *Logic Tensor Networks*. arXiv:2202.04820.
- Donadello, I., Serafini, L., & Garcez, A. d. (2017). *Semantic-based regularization for logic tensor networks*. IJCNN.
- Manhaeve, R. et al. (2018). *DeepProbLog: Neural Probabilistic Logic Programming*. NeurIPS.
- Rocktäschel, T., & Riedel, S. (2017). *End-to-end Differentiable Proving*. NeurIPS.
- Serafini, L., & Garcez, A. d. (2016). *Logic Tensor Networks: Deep Learning and Logical Reasoning from Data and Knowledge*. arXiv:1606.04422.
- Evans, R., & Grefenstette, E. (2018). *Learning Explanatory Rules from Noisy Data*. JAIR.
- Yao, S. et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. arXiv:2210.03629.
- Schick, T. et al. (2023). *Toolformer: Language Models Can Teach Themselves to Use Tools*. NeurIPS.

### KG Embeddings
- Bordes, A. et al. (2013). *Translating Embeddings for Modeling Multi-relational Data*. NeurIPS.
- Wang, Z. et al. (2014). *Knowledge Graph Embedding by Translating on Hyperplanes*. AAAI.
- Nickel, M. et al. (2016). *Holographic Embeddings of Knowledge Graphs*. AAAI.
- Trouillon, T. et al. (2016). *Complex Embeddings for Simple Link Prediction*. ICML.
- Sun, Z. et al. (2019). *RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space*. ICLR.
- Dettmers, T. et al. (2018). *ConvE: Convolutional 2D Knowledge Graph Embeddings*. WWW.

### SHACL
- Knublauch, H., & Kontokostas, D. (2017). *Shapes Constraint Language (SHACL)*. W3C Recommendation.
- *pySHACL* — https://github.com/RDFLib/pySHACL.

### OWL Reasoning
- Motik, B. et al. (2012). *OWL 2 Web Ontology Language Profiles*. W3C Recommendation.
- Heymans, S. et al. (2009). *Default reasoning on top of Description Logics*. KI.
- Zhang, W. et al. (2022). *GNN-Reasoner*. arXiv:2203.09926.
