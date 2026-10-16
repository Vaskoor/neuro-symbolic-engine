# SPDX-License-Identifier: MIT
#
# neurosym.knowledge.graph
# -------------------------
# A thin, well-typed wrapper around ``rdflib.Graph``.
#
# Design notes
# ~~~~~~~~~~~~
# The wrapper exposes triple-level operations on **string** terms so that the
# rest of the engine (notably the pure-Python Datalog reasoner) never has to
# import rdflib. The translation strategy is:
#
# * Subject and predicate strings are always URIs. CURIEs such as
#   ``ex:authoredBy`` are expanded against the graph's bound namespaces; bare
#   absolute IRIs are used verbatim.
# * Object strings are URIs if they expand to a bound CURIE / absolute IRI;
#   otherwise they are stored as ``rdflib.Literal`` (lexical value = string).
#
# When *reading* triples back out (``get_triples``, ``sparql_query``), URI
# terms are compacted back to CURIEs using the bound namespaces, so the
# string representation round-trips with the rule patterns stored in YAML.

"""RDFLib-backed knowledge graph with CURIE string round-tripping."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.term import Node

from ..core.types import Triple

# Bound by default so CURIEs like ``sh:NodeShape`` resolve even before any
# Turtle file is loaded.
_EX_NS = "http://example.org/sci#"
_SH_NS = "http://www.w3.org/ns/shacl#"


class KnowledgeGraph:
    """RDFLib graph wrapper with a stable string-facing API."""

    def __init__(self, base_namespace: str = _EX_NS) -> None:
        self.base_namespace: str = base_namespace
        self._g: Graph = Graph()
        # Bind common prefixes; rdflib already binds rdf/rdfs/owl/xsd.
        self._g.bind("ex", URIRef(base_namespace))
        self._g.bind("sh", URIRef(_SH_NS))
        self._g.bind("foaf", URIRef("http://xmlns.com/foaf/0.1/"))

    # ------------------------------------------------------------------
    # Access to the underlying rdflib graph (advanced use).
    # ------------------------------------------------------------------
    @property
    def graph(self) -> Graph:
        return self._g

    @property
    def namespaces(self) -> list[tuple[str, str]]:
        return [(p, str(n)) for p, n in self._g.namespaces()]

    # ------------------------------------------------------------------
    # CURIE expansion / compaction helpers
    # ------------------------------------------------------------------
    def _lookup_namespace(self, prefix: str) -> Optional[str]:
        if prefix == "":
            return self.base_namespace
        for pfx, ns in self._g.namespaces():
            if pfx == prefix:
                return str(ns)
        return None

    def _expand(self, value: str) -> Optional[str]:
        """Expand a CURIE (``ex:Foo``) to an absolute IRI.

        Returns ``None`` if ``value`` is neither an absolute IRI nor a CURIE
        resolvable against bound namespaces — in which case the caller should
        treat it as a literal.
        """
        if value.startswith(("http://", "https://", "urn:", "file:")):
            return value
        if ":" in value:
            prefix, _, local = value.partition(":")
            # Guard against xsd-style or unknown prefixes.
            ns = self._lookup_namespace(prefix)
            if ns is not None:
                return ns + local
        return None

    def _to_subject(self, value: str) -> Node:
        expanded = self._expand(value)
        return URIRef(expanded if expanded is not None else value)

    def _to_predicate(self, value: str) -> Node:
        # Predicates are always URIs.
        return self._to_subject(value)

    def _to_object(self, value: str) -> Node:
        expanded = self._expand(value)
        if expanded is not None:
            return URIRef(expanded)
        return Literal(value)

    def _qname(self, iri: str) -> str:
        """Compact an absolute IRI to a CURIE using bound namespaces."""
        for pfx, ns in self._g.namespaces():
            ns_str = str(ns)
            if ns_str and iri.startswith(ns_str):
                local = iri[len(ns_str):]
                return f"{pfx}:{local}" if pfx else f":{local}"
        return iri

    def _term_to_str(self, term: Node) -> str:
        if isinstance(term, URIRef):
            return self._qname(str(term))
        if isinstance(term, Literal):
            return str(term)
        if isinstance(term, BNode):
            return f"_:{str(term)}"
        return str(term)

    # ------------------------------------------------------------------
    # Triple mutation
    # ------------------------------------------------------------------
    def add_triple(self, s: str, p: str, o: str) -> None:
        """Add a triple from three string terms.

        Subjects and predicates must be URIs (absolute or CURIE). Objects are
        URIs when they expand, literals otherwise.
        """
        self._g.add((self._to_subject(s), self._to_predicate(p), self._to_object(o)))

    def assert_triple(self, triple: Triple) -> None:
        self.add_triple(triple.s, triple.p, triple.o)

    def remove_triple(self, s: Optional[str], p: Optional[str], o: Optional[str]) -> int:
        """Remove all matching triples. Returns the number removed."""
        sn = self._to_subject(s) if s else None
        pn = self._to_predicate(p) if p else None
        on = self._to_object(o) if o else None
        triples = list(self._g.triples((sn, pn, on)))
        for t in triples:
            self._g.remove(t)
        return len(triples)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def load_turtle(self, path: str | Path) -> int:
        """Parse a Turtle file into the graph. Returns # new triples added."""
        before = len(self._g)
        self._g.parse(str(path), format="turtle")
        return len(self._g) - before

    def save_turtle(self, path: str | Path) -> None:
        """Serialize the graph to a Turtle file (creating parent dirs)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._g.serialize(destination=str(p), format="turtle")

    def to_turtle(self) -> str:
        return self._g.serialize(format="turtle")

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------
    def sparql_query(self, query: str) -> list[dict[str, str]]:
        """Run a SPARQL SELECT; return rows as ``{var: str}`` dicts.

        rdflib terms are compacted/lexified with the same strategy as
        :meth:`get_triples` so callers see CURIEs and lexical literals.
        """
        qres = self._g.query(query)
        rows: list[dict[str, str]] = []
        varlist = [str(v) for v in (qres.vars or [])]
        for row in qres:
            d: dict[str, str] = {}
            raw = None
            if hasattr(row, "asdict"):
                try:
                    raw = row.asdict()
                except Exception:
                    raw = None
            if raw:
                for var, term in raw.items():
                    d[str(var)] = self._term_to_str(term) if term is not None else ""
            else:
                for vname in varlist:
                    term = getattr(row, vname, None)
                    d[vname] = self._term_to_str(term) if term is not None else ""
            rows.append(d)
        return rows

    def get_triples(
        self,
        s: Optional[str] = None,
        p: Optional[str] = None,
        o: Optional[str] = None,
    ) -> list[dict[str, str]]:
        """Match triples by (optional) s/p/o string patterns.

        ``None`` (or empty string) means "any". Returns a deterministic list
        sorted by ``(s, p, o)`` so downstream behaviour is reproducible.
        """
        sn = self._to_subject(s) if s else None
        pn = self._to_predicate(p) if p else None
        on = self._to_object(o) if o else None
        out: list[dict[str, str]] = []
        for (s_, p_, o_) in self._g.triples((sn, pn, on)):
            out.append(
                {
                    "s": self._term_to_str(s_),
                    "p": self._term_to_str(p_),
                    "o": self._term_to_str(o_),
                }
            )
        out.sort(key=lambda d: (d["s"], d["p"], d["o"]))
        return out

    def objects_for(self, s: str, p: str) -> list[str]:
        """Convenience accessor: all object strings for ``s p ?``."""
        sn = self._to_subject(s)
        pn = self._to_predicate(p)
        return sorted(
            self._term_to_str(o) for o in self._g.objects(sn, pn)
        )

    def has_triple(
        self,
        s: Optional[str] = None,
        p: Optional[str] = None,
        o: Optional[str] = None,
    ) -> bool:
        sn = self._to_subject(s) if s else None
        pn = self._to_predicate(p) if p else None
        on = self._to_object(o) if o else None
        for _ in self._g.triples((sn, pn, on)):
            return True
        return False

    def count(self) -> int:
        """Number of triples in the graph."""
        return len(self._g)

    # ------------------------------------------------------------------
    # Graphviz export
    # ------------------------------------------------------------------
    def to_dot(self) -> str:
        """Render the graph as Graphviz DOT text.

        Nodes are subjects/objects; edges are labelled with the predicate
        (compacted to a CURIE). Literal values are rendered as ellipses, URI
        nodes as boxes, and blank nodes as their ``_:id`` form.
        """
        lines: list[str] = [
            "digraph kg {",
            "  rankdir=LR;",
            '  node [fontname="Helvetica"];',
            '  edge [fontname="Helvetica", fontsize=10];',
        ]
        seen_nodes: dict[str, str] = {}  # label -> shape
        edges: list[str] = []
        for (s_, p_, o_) in self._g:
            sl = self._term_to_str(s_)
            pl = self._term_to_str(p_)
            ol = self._term_to_str(o_)
            for label, term in ((sl, s_), (ol, o_)):
                if label not in seen_nodes:
                    shape = "ellipse" if isinstance(term, Literal) else "box"
                    seen_nodes[label] = shape
                    nid = self._dot_id(label)
                    lines.append(
                        f'  "{nid}" [label="{self._escape(label)}", shape={shape}];'
                    )
            edges.append(
                f'  "{self._dot_id(sl)}" -> "{self._dot_id(ol)}" '
                f'[label="{self._escape(pl)}"];'
            )
        lines.extend(edges)
        lines.append("}")
        return "\n".join(lines)

    @staticmethod
    def _dot_id(label: str) -> str:
        return label.replace(":", "_").replace("/", "_").replace("#", "_")

    @staticmethod
    def _escape(label: str) -> str:
        return label.replace("\\", "\\\\").replace('"', '\\"')

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._g)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"KnowledgeGraph(triples={self.count()}, namespaces={len(self.namespaces)})"


__all__ = ["KnowledgeGraph"]
