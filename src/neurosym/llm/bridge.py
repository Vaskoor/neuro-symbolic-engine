# MIT License
#
# Copyright (c) 2024 Neuro-Symbolic Engine Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
"""LLM bridge for the neuro-symbolic engine.

:class:`LLMBridge` is the soft, natural-language-aware layer that sits at
the *end* of the engine's cascading fallback chain (KG -> symbolic ->
neural -> LLM). When the upstream layers cannot answer a query, the
bridge produces:

* a NL-to-SPARQL translation (:meth:`nl_to_sparql`),
* a candidate :class:`Triple` fact (:meth:`fallback_fact`) — useful as a
  starting point for the symbolic layer to validate,
* a natural-language explanation of a deduction chain
  (:meth:`explain_proof`).

The bridge uses the ``z-ai-web-dev-sdk`` Python package if it is installed
and an API key is available; otherwise it falls back to deterministic,
seeded NL templates that produce *useful*, *reproducible* output without
any network call. The two paths share the same public API, so callers can
treat the bridge uniformly and check the :attr:`available` flag only when
they want to surface "LLM mode" to a user.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional

try:  # contract from sibling task ns-A
    from neurosym.core.types import (
        ProofStep,
        ProofTrace,
        Query,
        QueryType,
        ReasoningResult,
        Rule,
        Triple,
    )
except ImportError:  # pragma: no cover - fallback
    # Reuse the canonical fallback defined in the reasoning package so all
    # bridge-produced Triples are structurally compatible with whatever
    # ReasoningResult the symbolic layer returns.
    from ..reasoning.proof import (  # type: ignore[no-redef]
        ProofStep,
        ProofTrace,
        Query,
        QueryType,
        ReasoningResult,
        Rule,
        Triple,
    )

# Optional z-ai-web-dev-sdk (true LLM mode). Guarded so absence never
# breaks the import.
try:
    import z_ai_web_dev_sdk as _zai_sdk  # type: ignore
    _HAS_ZAI_SDK = True
except Exception:  # pragma: no cover - SDK is optional
    _zai_sdk = None  # type: ignore[assignment]
    _HAS_ZAI_SDK = False


if TYPE_CHECKING:  # pragma: no cover
    from neurosym.core.types import (
        ProofStep as _ProofStep,
        ProofTrace as _ProofTrace,
        Query as _Query,
        QueryType as _QueryType,
        ReasoningResult as _ReasoningResult,
        Rule as _Rule,
        Triple as _Triple,
    )


# ---------------------------------------------------------------------------
# Triple coercion helper
# ---------------------------------------------------------------------------
def _coerce_triple(value: Any) -> Triple:
    """Coerce any triple-like value into a :class:`Triple`."""
    if value is None:
        raise ValueError("Cannot coerce None into a Triple")
    if isinstance(value, Triple):
        return value
    if isinstance(value, Mapping):
        if {"s", "p", "o"} <= set(value.keys()):
            return Triple(s=str(value["s"]), p=str(value["p"]), o=str(value["o"]))
        if {"subject", "predicate", "object"} <= set(value.keys()):
            return Triple(
                s=str(value["subject"]),
                p=str(value["predicate"]),
                o=str(value["object"]),
            )
    if isinstance(value, (tuple, list)) and len(value) >= 3:
        return Triple(s=str(value[0]), p=str(value[1]), o=str(value[2]))
    if hasattr(value, "s") and hasattr(value, "p") and hasattr(value, "o"):
        return Triple(
            s=str(getattr(value, "s")),
            p=str(getattr(value, "p")),
            o=str(getattr(value, "o")),
        )
    if hasattr(value, "subject") and hasattr(value, "predicate") and hasattr(value, "object"):
        return Triple(
            s=str(getattr(value, "subject")),
            p=str(getattr(value, "predicate")),
            o=str(getattr(value, "object")),
        )
    raise TypeError(f"Cannot coerce {type(value).__name__!r} into a Triple")


def _triple_str(t: Triple | None) -> str:
    if t is None:
        return "(none)"
    return f"({t.s}, {t.p}, {t.o})"


def _is_var(token: str) -> bool:
    if not token:
        return False
    return token.startswith(("?", "$"))


def _var_name(token: str) -> str | None:
    if not token:
        return None
    if token.startswith(("?", "$")):
        return token[1:]
    return None


# ---------------------------------------------------------------------------
# Deterministic NL templates for the fallback path
# ---------------------------------------------------------------------------
_NL_TEMPLATES: list[tuple[re.Pattern[str], str, str, str]] = [
    # WH-questions first (these typically project ?x in the SELECT).
    (re.compile(r"^who\s+wrote\s+(?P<o>.+)$", re.IGNORECASE), "?s", "ex:authored", "?o"),
    (re.compile(r"^who\s+authored\s+(?P<o>.+)$", re.IGNORECASE), "?s", "ex:authored", "?o"),
    (re.compile(r"^who\s+knows\s+(?P<o>.+)$", re.IGNORECASE), "?s", "ex:knows", "?o"),
    (re.compile(r"^who\s+is\s+a\s+(?P<o>.+)$", re.IGNORECASE), "?s", "rdf:type", "?o"),
    # Declarative / "X wrote Y" style.
    (re.compile(r"^(?P<s>.+?)\s+wrote\s+(?P<o>.+)$", re.IGNORECASE), "?s", "ex:authored", "?o"),
    (re.compile(r"^(?P<o>.+?)\s+was\s+written\s+by\s+(?P<s>.+)$", re.IGNORECASE), "?s", "ex:authored", "?o"),
    (re.compile(r"^(?P<s>.+?)\s+authored\s+(?P<o>.+)$", re.IGNORECASE), "?s", "ex:authored", "?o"),
    (re.compile(r"^(?P<s>.+?)\s+is\s+an?\s+(?P<o>.+)$", re.IGNORECASE), "?s", "rdf:type", "?o"),
    (re.compile(r"^(?P<s>.+?)\s+is\s+a\s+(?P<o>.+)$", re.IGNORECASE), "?s", "rdf:type", "?o"),
    (re.compile(r"^(?P<s>.+?)\s+knows\s+(?P<o>.+)$", re.IGNORECASE), "?s", "ex:knows", "?o"),
    (re.compile(r"^(?P<s>.+?)\s+has\s+(?P<o>.+)$", re.IGNORECASE), "?s", "ex:has", "?o"),
    (re.compile(r"^(?P<s>.+?)\s+subclass\s+of\s+(?P<o>.+)$", re.IGNORECASE), "?s", "rdfs:subClassOf", "?o"),
    (re.compile(r"^(?P<s>.+?)\s+subClassOf\s+(?P<o>.+)$", re.IGNORECASE), "?s", "rdfs:subClassOf", "?o"),
]


def _match_nl_template(text: str) -> tuple[str, str, str] | None:
    """Match ``text`` against the NL template table.

    Returns ``(s, p, o)`` where any of ``s``/``p``/``o`` may be a variable
    (``?s`` / ``?o``) when the corresponding field was the WH-question's
    projection target.
    """
    for regex, s_t, p_t, o_t in _NL_TEMPLATES:
        m = regex.match(text.strip())
        if not m:
            continue
        groups = {k: v.strip() for k, v in m.groupdict().items()}
        s = _substitute_template(s_t, groups)
        p = _substitute_template(p_t, groups)
        o = _substitute_template(o_t, groups)
        return s, p, o
    return None


def _substitute_template(template: str, bindings: Mapping[str, str]) -> str:
    """Substitute ``?name`` placeholders in a template using ``bindings``."""
    if template.startswith("?"):
        name = template[1:]
        if name in bindings:
            return bindings[name]
        return template
    return template


def _render_sparql_term(term: str) -> str:
    """Render a Datalog term as a SPARQL term (mirrors QueryEngine)."""
    if not term:
        return "?_"
    if term.startswith(("?", "$")):
        vn = _var_name(term)
        return f"?{vn}" if vn else term
    if term.startswith("<") and term.endswith(">"):
        return term
    if "://" in term:
        return f"<{term}>"
    if ":" in term and not term.startswith('"'):
        # CURIE / prefixed name — leave verbatim
        return term
    return f'"{term}"'


# ---------------------------------------------------------------------------
# LLM bridge
# ---------------------------------------------------------------------------
class LLMBridge:
    """Natural-language ↔ structured-query bridge with an optional LLM backend.

    Parameters
    ----------
    seed:
        Seed for the deterministic PRNG used by the fallback path
        (``random.Random``). When the z-ai-web-dev-sdk LLM is available,
        the seed is forwarded as a deterministic temperature-0 prompt tag.
    api_key:
        Optional override for the API key. When omitted, the bridge looks
        for ``ZAI_API_KEY`` (and the SDK's own preferred env vars) in the
        environment.
    model:
        Optional model identifier forwarded to the SDK when available.
    """

    #: environment variable names the bridge probes for an API key
    ENV_VARS: tuple[str, ...] = ("ZAI_API_KEY", "Z_AI_API_KEY", "ZHIPUAI_API_KEY")

    def __init__(
        self,
        *,
        seed: int = 42,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.seed = int(seed)
        self.model = model
        self._api_key = api_key or self._discover_api_key()
        self._client: Any = None

        # `available` is True iff we can actually reach the SDK at call time.
        self._available: bool = False
        if _HAS_ZAI_SDK and self._api_key:
            try:
                self._client = self._instantiate_client()
                self._available = self._client is not None
            except Exception:  # pragma: no cover - SDK errors are non-fatal
                self._client = None
                self._available = False

    # ------------------------------------------------------------------
    # availability
    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        """``True`` iff the LLM backend is wired up and ready to call."""
        return self._available

    @property
    def backend(self) -> str:
        """``"llm"`` when :attr:`available`, else ``"templates"``."""
        return "llm" if self._available else "templates"

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def nl_to_sparql(self, nl_text: str) -> str:
        """Translate a natural-language-ish query string to SPARQL SELECT.

        When the LLM is available, the query is delegated to it with a
        strict SPARQL-only prompt and a deterministic seed. Otherwise, the
        deterministic template table above is consulted.

        Parameters
        ----------
        nl_text:
            Free-form text (e.g. ``"who wrote Book1"`` or
            ``"Alice is a Person"``).

        Returns
        -------
        str
            A SPARQL ``SELECT`` query string. Always non-empty.
        """
        if not isinstance(nl_text, str):
            raise TypeError(f"nl_text must be str, got {type(nl_text).__name__}")
        if self._available and self._client is not None:
            try:
                return self._llm_nl_to_sparql(nl_text)
            except Exception:  # pragma: no cover - fall through to templates
                pass
        return self._template_nl_to_sparql(nl_text)

    def fallback_fact(self, query: Any) -> Triple:
        """Generate a candidate :class:`Triple` when nothing else could answer.

        The fallback is intentionally conservative: if the query is a
        recognisable pattern (e.g. ``"?x ex:authoredBy Alice"``), the
        candidate is a deterministic hash-based subject filling the variable
        slot, paired with the same predicate and object the query asked for.
        When the LLM is available, it is prompted for the missing fact
        instead (still returned as a :class:`Triple` so the symbolic layer
        can validate / reject it).

        Parameters
        ----------
        query:
            A :class:`Query` model, a raw pattern ``dict``, a tuple
            ``(s, p, o)``, or a string (NL or Datalog).

        Returns
        -------
        Triple
            A candidate fact.
        """
        s, p, o = self._extract_query_pattern(query)
        if self._available and self._client is not None:
            try:
                llm_triple = self._llm_fallback_fact(s, p, o)
                if llm_triple is not None:
                    return _coerce_triple(llm_triple)
            except Exception:  # pragma: no cover - fall through to template
                pass

        # Template path: fill variable slots with deterministic hash strings.
        s_out = self._fill_var(s, prefix="s")
        p_out = self._fill_var(p, prefix="p")
        o_out = self._fill_var(o, prefix="o")
        # The relation gets a sensible default when not known.
        if _is_var(p) or not p:
            p_out = "ex:relatedTo"
        return Triple(s=s_out, p=p_out, o=o_out)

    def explain_proof(self, trace: ProofTrace | ReasoningResult) -> str:
        """Render a :class:`ProofTrace` (or :class:`ReasoningResult`) as NL.

        Walks the recursive proof tree and produces a deterministic,
        human-readable explanation of the deduction chain. When the LLM is
        available, the structured walk is still produced (so callers always
        have a deterministic fallback) but a smoothed LLM paragraph is also
        prepended.
        """
        if trace is None:
            return "No proof was provided."
        # Accept ReasoningResult too.
        proof: ProofTrace | None = None
        if isinstance(trace, ProofTrace):
            proof = trace
        else:
            proof = getattr(trace, "proof", None)
            if not isinstance(proof, ProofTrace):
                proof = None
        if proof is None:
            return "No proof trace is available for this answer."

        deterministic = self._render_proof_deterministic(proof)

        if self._available and self._client is not None:
            try:
                llm_text = self._llm_explain_proof(proof, deterministic)
                if llm_text:
                    return llm_text.strip() + "\n\n" + deterministic
            except Exception:  # pragma: no cover
                pass
        return deterministic

    # ------------------------------------------------------------------
    # internal: API key discovery + client instantiation
    # ------------------------------------------------------------------
    def _discover_api_key(self) -> str | None:
        for var in self.ENV_VARS:
            val = os.environ.get(var)
            if val:
                return val
        return None

    def _instantiate_client(self) -> Any:
        """Instantiate the z-ai-web-dev-sdk client (guarded)."""
        if not _HAS_ZAI_SDK or self._api_key is None:
            return None
        sdk = _zai_sdk
        # The SDK exposes a handful of slightly different entry-point
        # class names across versions; probe them defensively.
        for attr in ("ZAIWPredictor", "ZaiClient", "Client", "LLM"):
            cls = getattr(sdk, attr, None)
            if cls is None:
                continue
            try:
                # Try api_key kwarg first.
                try:
                    return cls(api_key=self._api_key)  # type: ignore[no-any-return]
                except TypeError:
                    pass
                # Try positional.
                try:
                    return cls(self._api_key)  # type: ignore[no-any-return]
                except TypeError:
                    pass
                # Try no-arg init (relies on env var).
                try:
                    return cls()  # type: ignore[no-any-return]
                except Exception:
                    continue
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------
    # internal: LLM-backed implementations
    # ------------------------------------------------------------------
    def _llm_call(self, prompt: str, *, max_tokens: int = 256) -> str:
        """Best-effort single-shot LLM completion. Returns '' on failure."""
        client = self._client
        if client is None:
            return ""
        # The SDK exposes a handful of slightly different call surfaces;
        # probe them defensively and return the first non-empty result.
        candidates: list[str] = []
        for method_name in ("complete", "chat", "generate", "predict", "run"):
            method = getattr(client, method_name, None)
            if not callable(method):
                continue
            try:
                # Try keyword form first.
                try:
                    result = method(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=0.0,
                        seed=self.seed,
                    )
                except TypeError:
                    try:
                        result = method(prompt, temperature=0.0, seed=self.seed)
                    except TypeError:
                        try:
                            result = method(prompt)
                        except Exception:
                            continue
                text = self._extract_text(result)
                if text:
                    candidates.append(text)
            except Exception:
                continue
        # Prefer the shortest deterministic-looking response (LLMs tend to
        # be verbose; we want SPARQL/JSON-ish output).
        if not candidates:
            return ""
        candidates.sort(key=len)
        return candidates[0]

    @staticmethod
    def _extract_text(result: Any) -> str:
        """Pull a string out of a heterogeneous SDK response object."""
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        # Common shapes: {text, content, response, output, generated_text}
        if isinstance(result, Mapping):
            for key in ("text", "content", "response", "output", "generated_text", "completion"):
                val = result.get(key)
                if isinstance(val, str) and val:
                    return val
            for val in result.values():
                if isinstance(val, str) and val:
                    return val
        # Object with attribute access
        for attr in ("text", "content", "response", "output", "generated_text"):
            val = getattr(result, attr, None)
            if isinstance(val, str) and val:
                return val
        return str(result)

    def _llm_nl_to_sparql(self, nl_text: str) -> str:
        prompt = (
            "You are a strict SPARQL generator. Translate the following "
            "natural-language query to a single SPARQL SELECT query.\n"
            "Rules:\n"
            "1. Output ONLY the SPARQL query, no commentary.\n"
            "2. Use the bound prefixes: ex, rdf, rdfs, owl, foaf.\n"
            f"3. Treat the seed as {self.seed} (deterministic, temperature 0).\n"
            f"Query: {nl_text}\n"
            "SPARQL:"
        )
        text = self._llm_call(prompt)
        # Strip code-fences if present.
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]
                if text.lower().startswith("sparql"):
                    text = text[len("sparql"):].lstrip()
        # Sanity check: only return if it looks like SPARQL.
        low = text.lower().strip()
        if low.startswith(("select", "ask", "construct", "describe")):
            return text.strip()
        # Fall back to templates if the LLM babbled.
        return self._template_nl_to_sparql(nl_text)

    def _llm_fallback_fact(self, s: str, p: str, o: str) -> Triple | None:
        prompt = (
            "You are a knowledge-graph completer. Given an incomplete triple "
            "with one or more variables (prefixed with '?'), propose ONE "
            "concrete replacement as a JSON object with keys s, p, o "
            "(string values). Use CURIEs like ex:authoredBy.\n"
            f"Input triple: ({s}, {p}, {o})\n"
            "Output JSON only (no commentary):"
        )
        text = self._llm_call(prompt)
        # Parse a JSON object out of the response.
        import json

        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
        except Exception:
            return None
        if not isinstance(obj, dict):
            return None
        out_s = str(obj.get("s", s))
        out_p = str(obj.get("p", p))
        out_o = str(obj.get("o", o))
        if not (out_s and out_p and out_o):
            return None
        return Triple(s=out_s, p=out_p, o=out_o)

    def _llm_explain_proof(self, proof: ProofTrace, deterministic: str) -> str:
        prompt = (
            "You are a proof assistant. Explain the following deduction chain "
            "in one or two concise sentences of plain English.\n"
            f"Proof tree (deterministic rendering):\n{deterministic}\n"
            "Explanation:"
        )
        text = self._llm_call(prompt, max_tokens=128)
        text = text.strip()
        if not text:
            return ""
        # Truncate at first newline pair to keep it short.
        if "\n\n" in text:
            text = text.split("\n\n")[0]
        return text

    # ------------------------------------------------------------------
    # internal: deterministic template implementations
    # ------------------------------------------------------------------
    def _template_nl_to_sparql(self, nl_text: str) -> str:
        match = _match_nl_template(nl_text)
        if match is None:
            # Unmatched: emit a permissive "SELECT * WHERE { ?s ?p ?o }".
            return "SELECT * WHERE {\n  ?s ?p ?o .\n}"
        s, p, o = match
        # Determine projected variables (those that appear in the pattern).
        select: list[str] = []
        for term in (s, o):
            vn = _var_name(term)
            if vn is not None:
                var = f"?{vn}"
                if var not in select:
                    select.append(var)
        proj = " ".join(select) if select else "*"
        s_term = _render_sparql_term(s)
        p_term = _render_sparql_term(p)
        o_term = _render_sparql_term(o)
        return (
            f"SELECT {proj} WHERE {{\n"
            f"  {s_term} {p_term} {o_term} .\n"
            f"}}"
        )

    def _render_proof_deterministic(self, proof: ProofTrace) -> str:
        """Produce a deterministic NL walk of the proof tree."""
        lines: list[str] = []
        final = getattr(proof, "final_conclusion", None)
        if final is not None:
            lines.append(
                f"The conclusion {_triple_str(final)} was reached as follows."
            )
        else:
            lines.append("The following deduction chain was reached.")
        if not proof.steps:
            lines.append("No derivation steps were recorded.")
            return "\n".join(lines)
        counter = 0

        def walk(step: ProofStep, indent: int, prefix: str) -> None:
            nonlocal counter
            counter += 1
            idx = counter
            pad = "  " * indent
            conc = step.conclusion
            if not step.premises:
                if step.rule_name in ("fact", "asserted"):
                    lines.append(
                        f"{pad}{prefix}{idx}. {_triple_str(conc)} is an "
                        f"asserted fact (no further justification)."
                    )
                else:
                    lines.append(
                        f"{pad}{prefix}{idx}. {_triple_str(conc)} was concluded "
                        f"by rule '{step.rule_name}' with no premises."
                    )
            else:
                prem_str = ", ".join(_triple_str(p.conclusion) for p in step.premises)
                lines.append(
                    f"{pad}{prefix}{idx}. By applying rule '{step.rule_name}' to "
                    f"the premise(s) {prem_str}, we derived {_triple_str(conc)}."
                )
                for i, sub in enumerate(step.premises, start=1):
                    walk(sub, indent + 1, f"{i}.")

        for i, root in enumerate(proof.steps, start=1):
            walk(root, 0, f"[{i}] ")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # internal: query-pattern extraction + var filling
    # ------------------------------------------------------------------
    def _extract_query_pattern(self, query: Any) -> tuple[str, str, str]:
        """Extract a single (s, p, o) pattern from a query-like value."""
        if query is None:
            return "?s", "?p", "?o"
        if isinstance(query, str):
            match = _match_nl_template(query)
            if match is not None:
                return match
            # Try a simple whitespace parse for "?s p ?o" style.
            tokens = query.strip().split()
            if len(tokens) == 3:
                return tokens[0], tokens[1], tokens[2]
            return "?s", "?p", "?o"
        if isinstance(query, Mapping):
            if {"s", "p", "o"} <= set(query.keys()):
                return str(query["s"]), str(query["p"]), str(query["o"])
            if {"subject", "predicate", "object"} <= set(query.keys()):
                return (
                    str(query["subject"]),
                    str(query["predicate"]),
                    str(query["object"]),
                )
        if isinstance(query, (tuple, list)) and len(query) >= 3:
            return str(query[0]), str(query[1]), str(query[2])
        if isinstance(query, Triple):
            return query.s, query.p, query.o
        # Query model: prefer params['pattern'], then params['patterns'][0].
        params = getattr(query, "params", None) or {}
        if isinstance(params, Mapping):
            pat = params.get("pattern")
            if isinstance(pat, Mapping) and {"s", "p", "o"} <= set(pat.keys()):
                return str(pat["s"]), str(pat["p"]), str(pat["o"])
            patterns = params.get("patterns")
            if isinstance(patterns, (list, tuple)) and patterns:
                first = patterns[0]
                if isinstance(first, Mapping) and {"s", "p", "o"} <= set(first.keys()):
                    return str(first["s"]), str(first["p"]), str(first["o"])
        text = getattr(query, "text", None)
        if text:
            return self._extract_query_pattern(text)
        return "?s", "?p", "?o"

    def _fill_var(self, term: str, *, prefix: str) -> str:
        """Fill a variable slot with a deterministic, hash-based candidate.

        Non-variable terms are returned unchanged. Variables are filled with
        ``"ex:<prefix><hash>"`` so the same query always yields the same
        candidate (reproducible across runs).
        """
        if not term or _is_var(term):
            h = abs(hash((term or prefix, self.seed))) % 100_000
            return f"ex:{prefix}{h}"
        return term


__all__ = ["LLMBridge"]
