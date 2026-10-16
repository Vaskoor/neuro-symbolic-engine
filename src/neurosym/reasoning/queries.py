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
"""Query parsing and translation for the symbolic reasoner.

The :class:`QueryEngine` is a stateless, deterministic translator that
sits between human-friendly query surfaces (Datalog triple patterns,
NL-ish phrasings) and the formal query languages the underlying engines
speak (SPARQL, Datalog ``{s,p,o}`` patterns).

Two public methods form the contract surface required by the ``ns-B``
specification:

* :meth:`QueryEngine.parse_datalog` — turn a string of triple patterns
  (e.g. ``"?x authoredBy ?y"``) into a list of ``{s, p, o}`` dicts.
* :meth:`QueryEngine.to_sparql`    — turn a list of such patterns into
  a SPARQL ``SELECT`` query.

A handful of additional helpers (:meth:`build_query`, :meth:`parse_sparql`,
:func:`is_variable`, :func:`variable_name`, :func:`normalize_term`) are
exposed for the symbolic reasoner's convenience.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Iterable

try:  # contract from sibling task ns-A
    from neurosym.core.types import Query, QueryType, Triple
except ImportError:  # pragma: no cover - fallback when ns-A not yet shipped
    # Reuse the canonical fallback defined in :mod:`neurosym.reasoning.proof`
    # so every reasoning submodule shares structurally-compatible model
    # classes (a Query built here is the same class as one built in proof.py
    # / reasoner.py).
    from .proof import Query, QueryType, Triple


if TYPE_CHECKING:  # pragma: no cover
    from neurosym.core.types import Query as _Query
    from neurosym.core.types import QueryType as _QueryType
    from neurosym.core.types import Triple as _Triple


# ---------------------------------------------------------------------------
# Token-level helpers
# ---------------------------------------------------------------------------
_VAR_RE = re.compile(r"^[?$](\w+)$")
_QUOTED_RE = re.compile(r'^"(.*)"$')
_IRI_RE = re.compile(r"^<(.*)>$")

_PATTERN_SEPARATOR_RE = re.compile(
    r"\s*(?:\.\s+|\s*,\s*|\s*;\s*|\s+AND\s+|\s+and\s+)\s*"
)


def is_variable(token: str) -> bool:
    """Return True when ``token`` is a Datalog/SPARQL variable.

    Variables start with ``?`` or ``$`` (e.g. ``?x``, ``$y``).
    """
    if not token:
        return False
    return bool(_VAR_RE.match(token))


def variable_name(token: str) -> str | None:
    """Return the bare variable name (without ``?`` / ``$``) or None."""
    if not token:
        return None
    m = _VAR_RE.match(token)
    return m.group(1) if m else None


def normalize_term(token: str) -> str:
    """Strip enclosing quotes/IRI brackets from a term.

    * ``<http://example.org/foo>`` -> ``http://example.org/foo``
    * ``"Alice"`` -> ``Alice``
    * ``?x`` -> ``?x`` (unchanged — variables keep their marker)
    * bare tokens are returned unchanged.
    """
    if token is None:
        return ""
    token = token.strip()
    if not token:
        return ""
    m = _IRI_RE.match(token)
    if m:
        return m.group(1)
    m = _QUOTED_RE.match(token)
    if m:
        return m.group(1)
    return token


def _split_patterns(query_text: str) -> list[str]:
    """Split a multi-pattern query string into individual triple patterns."""
    text = query_text.strip().rstrip(".")
    if not text:
        return []
    parts = [p.strip() for p in _PATTERN_SEPARATOR_RE.split(text) if p.strip()]
    return parts


def _tokenize_triple(pattern_text: str) -> list[str]:
    """Tokenise a single triple pattern into its s/p/o tokens.

    Handles bracketed IRIs and double-quoted literals which may contain
    whitespace.
    """
    tokens: list[str] = []
    i = 0
    n = len(pattern_text)
    while i < n:
        c = pattern_text[i]
        if c.isspace():
            i += 1
            continue
        if c == "<":
            j = pattern_text.find(">", i + 1)
            if j == -1:
                tokens.append(pattern_text[i:].strip())
                break
            tokens.append(pattern_text[i : j + 1])
            i = j + 1
        elif c == '"':
            j = i + 1
            while j < n:
                if pattern_text[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if pattern_text[j] == '"':
                    break
                j += 1
            tokens.append(pattern_text[i : j + 1])
            i = j + 1
        else:
            j = i
            while j < n and not pattern_text[j].isspace():
                j += 1
            tokens.append(pattern_text[i:j])
            i = j
    return tokens


def _parse_one_pattern(pattern_text: str) -> dict[str, str]:
    """Parse a single triple pattern into ``{s, p, o}``.

    Accepted shapes (after whitespace tokenisation):

    * ``?x authoredBy ?y``                 -> subject/predicate/object
    * ``?x <http://.../authoredBy> ?y``     -> IRI predicate
    * ``?x authoredBy "Alice"``             -> quoted literal object
    * ``(?x, authoredBy, ?y)``              -> parenthesised, comma-separated
    * ``?x ?p ?y``                          -> predicate is also a variable
    """
    text = pattern_text.strip().strip("()")
    if "," in text:
        parts = [p.strip().strip('"').strip("<>") for p in text.split(",") if p.strip()]
        if len(parts) >= 3:
            s, p, o = parts[0], parts[1], parts[2]
            return {"s": normalize_term(s), "p": normalize_term(p), "o": normalize_term(o)}
        if len(parts) == 2:
            return {"s": normalize_term(parts[0]), "p": "?p", "o": normalize_term(parts[1])}
        raise ValueError(f"Malformed pattern: {pattern_text!r}")

    tokens = _tokenize_triple(text)
    if len(tokens) == 0:
        raise ValueError(f"Empty pattern: {pattern_text!r}")
    if len(tokens) == 1:
        return {"s": normalize_term(tokens[0]), "p": "?p", "o": "?o"}
    if len(tokens) == 2:
        return {"s": normalize_term(tokens[0]), "p": "?p", "o": normalize_term(tokens[1])}
    # take the first three tokens; join any remainder onto the object
    s, p = tokens[0], tokens[1]
    o_tokens = tokens[2:]
    if o_tokens[0].startswith('"') and not o_tokens[0].endswith('"'):
        o = " ".join(o_tokens)
    elif o_tokens[0].startswith("<") and not o_tokens[0].endswith(">"):
        o = " ".join(o_tokens)
    else:
        o = o_tokens[0] if len(o_tokens) == 1 else " ".join(o_tokens)
    return {"s": normalize_term(s), "p": normalize_term(p), "o": normalize_term(o)}


# ---------------------------------------------------------------------------
# QueryEngine
# ---------------------------------------------------------------------------
class QueryEngine:
    """Translates natural-language-ish queries to Datalog patterns / SPARQL.

    The engine is fully deterministic and side-effect free; it never reaches
    out to a network, LLM, or RNG.  It accepts three loosely-coupled query
    surfaces:

    * **Datalog triple patterns** — whitespace- or comma-separated triples
      of the form ``?x authoredBy ?y`` (with optional ``<IRI>`` predicates
      and ``"literal"`` objects).
    * **SPARQL SELECT** strings — round-tripped into Datalog patterns and
      back out (useful for query inspection).
    * **NL-ish** phrasings — short phrases like "X wrote Y" or "Y written
      by X" are mapped onto a triple pattern via a small template table.

    The two methods in the public contract are :meth:`parse_datalog` and
    :meth:`to_sparql`; the remaining helpers are conveniences used by the
    symbolic reasoner.
    """

    # A small, deterministic template table for NL-ish phrasings.
    # Each entry maps a regex (with named groups) to a (s, p, o) template.
    _NL_TEMPLATES: list[tuple[re.Pattern[str], str, str, str]] = [
        (re.compile(r"^(?P<s>.+?)\s+wrote\s+(?P<o>.+)$"), "?s", "ex:authored", "?o"),
        (re.compile(r"^(?P<o>.+?)\s+was\s+written\s+by\s+(?P<s>.+)$"), "?s", "ex:authored", "?o"),
        (re.compile(r"^(?P<s>.+?)\s+authored\s+(?P<o>.+)$"), "?s", "ex:authored", "?o"),
        (re.compile(r"^(?P<s>.+?)\s+is\s+an?\s+(?P<o>.+)$"), "?s", "rdf:type", "?o"),
        (re.compile(r"^(?P<s>.+?)\s+is\s+a\s+(?P<o>.+)$"), "?s", "rdf:type", "?o"),
        (re.compile(r"^(?P<s>.+?)\s+is\s+(?P<o>.+)$"), "?s", "ex:is", "?o"),
        (re.compile(r"^(?P<s>.+?)\s+has\s+(?P<o>.+)$"), "?s", "ex:has", "?o"),
        (re.compile(r"^(?P<s>.+?)\s+knows\s+(?P<o>.+)$"), "?s", "ex:knows", "?o"),
        (re.compile(r"^(?P<s>.+?)\s+subclass\s+of\s+(?P<o>.+)$"), "?s", "rdfs:subClassOf", "?o"),
        (re.compile(r"^(?P<s>.+?)\s+subClassOf\s+(?P<o>.+)$"), "?s", "rdfs:subClassOf", "?o"),
    ]

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def parse_datalog(self, query_text: str) -> list[dict[str, str]]:
        """Parse a Datalog-style triple-pattern query into a list of patterns.

        Each pattern is a ``dict`` with the keys ``s``, ``p``, ``o``
        (subject/predicate/object). Variables keep their ``?`` marker so
        callers can distinguish them from constants.

        Examples
        --------
        >>> qe = QueryEngine()
        >>> qe.parse_datalog("?x authoredBy ?y")
        [{'s': '?x', 'p': 'authoredBy', 'o': '?y'}]
        >>> qe.parse_datalog('?x rdf:type Person . ?x knows ?y')
        [{'s': '?x', 'p': 'rdf:type', 'o': 'Person'}, {'s': '?x', 'p': 'knows', 'o': '?y'}]
        """
        if not isinstance(query_text, str):
            raise TypeError(f"query_text must be str, got {type(query_text).__name__}")
        cleaned = query_text.strip()
        if not cleaned:
            return []
        # Detect SPARQL and delegate
        if cleaned.lower().startswith(("select", "ask", "construct", "describe")):
            return self.parse_sparql(cleaned)
        # Detect NL-ish (single phrase, no variable markers / IRIs / commas)
        if (
            "?" not in cleaned
            and "$" not in cleaned
            and "<" not in cleaned
            and "," not in cleaned
        ):
            nl = self._match_nl_template(cleaned)
            if nl is not None:
                return [nl]
        # Otherwise treat as a Datalog triple-pattern sequence
        patterns: list[dict[str, str]] = []
        for piece in _split_patterns(cleaned):
            patterns.append(_parse_one_pattern(piece))
        return patterns

    def to_sparql(self, patterns: list[dict[str, str]]) -> str:
        """Build a SPARQL ``SELECT`` query from a list of Datalog patterns.

        All distinct variables appearing in the patterns are projected in
        the ``SELECT`` clause, in the order in which they first appear.
        Constants are emitted as SPARQL literals (IRIs for bare terms,
        ``"..."`` for strings that look like literals, ``<...>`` for
        explicit IRIs).
        """
        if not patterns:
            return "SELECT * WHERE { }"
        select_vars: list[str] = []
        seen_vars: set[str] = set()
        where_parts: list[str] = []
        for pat in patterns:
            for key in ("s", "p", "o"):
                term = pat.get(key, "")
                if not term:
                    continue
                vn = variable_name(term)
                if vn is not None:
                    var = f"?{vn}"
                    if var not in seen_vars:
                        seen_vars.add(var)
                        select_vars.append(var)
            s = self._render_sparql_term(pat.get("s", ""))
            p = self._render_sparql_term(pat.get("p", ""))
            o = self._render_sparql_term(pat.get("o", ""))
            where_parts.append(f"  {s} {p} {o} .")
        select_clause = " ".join(select_vars) if select_vars else "*"
        where_clause = "\n".join(where_parts)
        return (
            f"SELECT {select_clause} WHERE {{\n"
            f"{where_clause}\n"
            f"}}"
        )

    # ------------------------------------------------------------------
    # convenience / sibling-module helpers
    # ------------------------------------------------------------------
    def build_query(
        self,
        query_text: str,
        *,
        query_type: QueryType | None = None,
    ) -> Query:
        """Build a :class:`Query` model from raw text.

        ``query_type`` defaults to :attr:`QueryType.DATALOG` for queries
        that contain variable markers, otherwise
        :attr:`QueryType.SPARQL` (the contract enum has no NATURAL_LANGUAGE
        member, so NL-ish queries are also labelled DATALOG — the parsed
        patterns are what the reasoner consumes).
        """
        patterns = self.parse_datalog(query_text)
        if query_type is None:
            query_type = QueryType.DATALOG
        params: dict[str, Any] = {"patterns": patterns}
        if len(patterns) == 1:
            params["pattern"] = patterns[0]
        # Surface projected variable names for the symbolic reasoner.
        select: list[str] = []
        seen: set[str] = set()
        for pat in patterns:
            for key in ("s", "p", "o"):
                term = pat.get(key, "")
                vn = variable_name(term)
                if vn is not None and vn not in seen:
                    seen.add(vn)
                    select.append(vn)
        if select:
            params["select"] = select
        return Query(  # type: ignore[call-arg]
            text=query_text,
            type=query_type,
            params=params,
        )

    def parse_sparql(self, sparql_text: str) -> list[dict[str, str]]:
        """Extract Datalog patterns from a SPARQL SELECT/ASK/CONSTRUCT query.

        Only the ``WHERE`` clause triple patterns are extracted; FILTER,
        OPTIONAL, UNION, property paths and other SPARQL constructs are
        ignored (this is a deterministic surface translator, not a full
        SPARQL parser).
        """
        text = sparql_text.strip()
        low = text.lower()
        where_idx = low.find("where")
        if where_idx == -1:
            where_idx = low.find("{")
            if where_idx == -1:
                return []
        brace_open = text.find("{", where_idx)
        if brace_open == -1:
            return []
        depth = 0
        brace_close = -1
        for i in range(brace_open, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    brace_close = i
                    break
        if brace_close == -1:
            return []
        body = text[brace_open + 1 : brace_close]
        patterns: list[dict[str, str]] = []
        for raw in re.split(r"\s*\.\s*", body):
            raw = raw.strip()
            if not raw:
                continue
            try:
                patterns.append(_parse_one_pattern(raw))
            except ValueError:
                continue
        return patterns

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _match_nl_template(self, text: str) -> dict[str, str] | None:
        """Match ``text`` against the NL template table; return a pattern."""
        for regex, s_template, p_template, o_template in self._NL_TEMPLATES:
            m = regex.match(text)
            if not m:
                continue
            bindings = {k: normalize_term(v.strip()) for k, v in m.groupdict().items()}
            s = self._substitute_template(s_template, bindings)
            p = self._substitute_template(p_template, bindings)
            o = self._substitute_template(o_template, bindings)
            return {"s": s, "p": p, "o": o}
        return None

    @staticmethod
    def _substitute_template(template: str, bindings: dict[str, str]) -> str:
        """Substitute ``?name`` placeholders in a template using ``bindings``.

        If the template is a single ``?name`` placeholder, the bound value is
        substituted verbatim. If the template starts with ``?`` but has no
        matching binding, it stays a variable (useful for the predicate
        being a constant like ``ex:authored``).
        """
        if template.startswith("?"):
            name = template[1:]
            if name in bindings:
                return bindings[name]
            return template
        return template

    @staticmethod
    def _render_sparql_term(term: str) -> str:
        """Render a Datalog term as a SPARQL term.

        * ``?x`` / ``$x`` -> ``?x``
        * ``<http://...>`` -> unchanged IRI
        * ``http://...`` -> wrapped as ``<...>``
        * a CURIE-like token (``rdf:type``) -> kept verbatim (SPARQL allows
          prefixed names when a PREFIX is declared; callers should prepend
          PREFIX declarations as needed)
        * everything else -> quoted literal ``"..."``
        """
        if not term:
            return "?_"
        if term.startswith(("?", "$")):
            return f"?{variable_name(term)}"
        if term.startswith("<") and term.endswith(">"):
            return term
        if "://" in term:
            return f"<{term}>"
        if ":" in term and not term.startswith('"'):
            # CURIE / prefixed name — leave verbatim
            return term
        return f'"{term}"'


__all__ = [
    "QueryEngine",
    "is_variable",
    "variable_name",
    "normalize_term",
]
