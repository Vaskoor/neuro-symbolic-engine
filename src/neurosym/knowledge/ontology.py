# SPDX-License-Identifier: MIT
#
# neurosym.knowledge.ontology
# -----------------------------
# A small programmatic OWL DL ontology builder on top of rdflib.
#
# The builder produces a Turtle file describing classes, properties, a
# subclass hierarchy, existential/universal restrictions and disjointness
# axioms. It is intentionally a thin convenience layer over rdflib terms so
# that ontologies can be generated from code (e.g. tests, scaffolding) rather
# than hand-written Turtle.

"""Programmatic OWL DL ontology builder producing Turtle output."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import rdflib
from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

from .graph import KnowledgeGraph

# Datatype aliases accepted by ``add_datatype_property(range=...)``.
_XSD_ALIASES = {
    "string": XSD.string,
    "int": XSD.integer,
    "integer": XSD.integer,
    "decimal": XSD.decimal,
    "boolean": XSD.boolean,
    "double": XSD.double,
    "float": XSD.float,
    "date": XSD.date,
    "dateTime": XSD.dateTime,
}


class OntologyBuilder:
    """Build an OWL ontology incrementally and serialize it to Turtle."""

    def __init__(
        self,
        base_namespace: str = "http://example.org/sci#",
        prefix: str = "ex",
        ontology_iri: Optional[str] = None,
    ) -> None:
        self.ns = rdflib.Namespace(base_namespace)
        self.prefix = prefix
        self._g: Graph = Graph()
        self._g.bind(prefix, URIRef(base_namespace))
        # Header
        ont = URIRef(ontology_iri or base_namespace)
        self._g.add((ont, RDF.type, OWL.Ontology))
        self._g.add((ont, RDFS.label, Literal(f"{prefix} ontology")))
        self._ontology_iri = ont

    # ------------------------------------------------------------------
    # Term helpers
    # ------------------------------------------------------------------
    def _uri(self, name: str) -> URIRef:
        """Resolve a name to a URIRef.

        Absolute IRIs (``http://...``) are used verbatim; bare names are
        appended to the ontology's namespace; CURIEs (``ex:Foo``) are
        expanded against the bound namespaces.
        """
        if name.startswith(("http://", "https://", "urn:")):
            return URIRef(name)
        if ":" in name:
            prefix, _, local = name.partition(":")
            for pfx, ns in self._g.namespaces():
                if pfx == prefix:
                    return URIRef(str(ns) + local)
        return URIRef(self.ns[name])

    def _maybe(self, value: Optional[str]) -> Optional[URIRef]:
        return self._uri(value) if value else None

    # ------------------------------------------------------------------
    # Classes
    # ------------------------------------------------------------------
    def add_class(self, name: str, label: Optional[str] = None) -> URIRef:
        cls = self._uri(name)
        self._g.add((cls, RDF.type, OWL.Class))
        if label:
            self._g.add((cls, RDFS.label, Literal(label)))
        return cls

    def add_subclass(self, sub: str, sup: str) -> None:
        """Declare ``sub rdfs:subClassOf sup`` (creates the classes if absent)."""
        self._g.add((self._uri(sub), RDFS.subClassOf, self._uri(sup)))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    def add_object_property(
        self,
        name: str,
        domain: Optional[str] = None,
        range: Optional[str] = None,
        label: Optional[str] = None,
        sub_property_of: Optional[str] = None,
    ) -> URIRef:
        prop = self._uri(name)
        self._g.add((prop, RDF.type, OWL.ObjectProperty))
        if domain:
            self._g.add((prop, RDFS.domain, self._uri(domain)))
        if range:
            self._g.add((prop, RDFS.range, self._uri(range)))
        if label:
            self._g.add((prop, RDFS.label, Literal(label)))
        if sub_property_of:
            self._g.add((prop, RDFS.subPropertyOf, self._uri(sub_property_of)))
        return prop

    def add_datatype_property(
        self,
        name: str,
        domain: Optional[str] = None,
        range: Optional[str] = None,
        label: Optional[str] = None,
    ) -> URIRef:
        prop = self._uri(name)
        self._g.add((prop, RDF.type, OWL.DatatypeProperty))
        if domain:
            self._g.add((prop, RDFS.domain, self._uri(domain)))
        if range:
            self._g.add((prop, RDFS.range, self._resolve_datatype(range)))
        if label:
            self._g.add((prop, RDFS.label, Literal(label)))
        return prop

    def _resolve_datatype(self, range_name: str) -> URIRef:
        if range_name.startswith(("http://", "https://")):
            return URIRef(range_name)
        if ":" in range_name:
            prefix, _, local = range_name.partition(":")
            for pfx, ns in self._g.namespaces():
                if pfx == prefix:
                    return URIRef(str(ns) + local)
        return _XSD_ALIASES.get(range_name, XSD.string)

    # ------------------------------------------------------------------
    # Restrictions
    # ------------------------------------------------------------------
    def _add_restriction(
        self,
        cls: str,
        prop: str,
        filler: str,
        kind: URIRef,
    ) -> BNode:
        """Add ``cls rdfs:subClassOf [ a owl:Restriction; ... ]``."""
        r = BNode()
        self._g.add((r, RDF.type, OWL.Restriction))
        self._g.add((r, OWL.onProperty, self._uri(prop)))
        self._g.add((r, kind, self._uri(filler)))
        self._g.add((self._uri(cls), RDFS.subClassOf, r))
        return r

    def add_existential_restriction(self, cls: str, prop: str, filler: str) -> BNode:
        """``cls rdfs:subClassOf owl:someValuesFrom filler`` on ``prop``."""
        return self._add_restriction(cls, prop, filler, OWL.someValuesFrom)

    def add_universal_restriction(self, cls: str, prop: str, filler: str) -> BNode:
        """``cls rdfs:subClassOf owl:allValuesFrom filler`` on ``prop``."""
        return self._add_restriction(cls, prop, filler, OWL.allValuesFrom)

    def add_cardinality_restriction(
        self,
        cls: str,
        prop: str,
        cardinality: int,
        kind: URIRef = OWL.minCardinality,
    ) -> BNode:
        """Add a min/max/exact cardinality restriction on ``prop``."""
        r = BNode()
        self._g.add((r, RDF.type, OWL.Restriction))
        self._g.add((r, OWL.onProperty, self._uri(prop)))
        self._g.add((r, kind, Literal(cardinality)))
        self._g.add((self._uri(cls), RDFS.subClassOf, r))
        return r

    # ------------------------------------------------------------------
    # Disjointness & equivalence
    # ------------------------------------------------------------------
    def add_disjoint(self, a: str, b: str) -> None:
        self._g.add((self._uri(a), OWL.disjointWith, self._uri(b)))

    def add_equivalent_class(self, a: str, b: str) -> None:
        self._g.add((self._uri(a), OWL.equivalentClass, self._uri(b)))

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def to_turtle(self) -> str:
        return self._g.serialize(format="turtle")

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._g.serialize(destination=str(p), format="turtle")

    def to_knowledge_graph(self) -> KnowledgeGraph:
        """Materialize the ontology into a fresh :class:`KnowledgeGraph`."""
        kg = KnowledgeGraph(base_namespace=str(self.ns))
        for triple in self._g:
            kg.graph.add(triple)
        return kg

    @property
    def graph(self) -> Graph:
        return self._g

    def triple_count(self) -> int:
        return len(self._g)


__all__ = ["OntologyBuilder"]
