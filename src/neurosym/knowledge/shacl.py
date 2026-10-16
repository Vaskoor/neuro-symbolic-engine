# SPDX-License-Identifier: MIT
#
# neurosym.knowledge.shacl
# --------------------------
# A minimal, dependency-free SHACL validator.
#
# It implements the subset of SHACL needed by this project:
#
#   * ``sh:NodeShape`` with targets ``sh:targetClass``, ``sh:targetNode``,
#     ``sh:targetSubjectsOf`` and ``sh:targetObjectsOf``.
#   * Property shapes declared via ``sh:property`` (or directly on the node
#     shape) with ``sh:path``.
#   * Constraints: ``sh:minCount``, ``sh:maxCount``, ``sh:datatype``,
#     ``sh:class`` and ``sh:nodeKind``.
#
# No dependency on ``pySHACL`` is required; the evaluator walks the shapes
# graph with plain rdflib SPARQL-free triple access and returns
# ``list[Violation]``.

"""Manual SHACL validator returning ``Violation`` objects."""

from __future__ import annotations

from typing import Optional

import rdflib
from rdflib import BNode, Literal, URIRef
from rdflib.namespace import RDF
from rdflib.term import Node

from ..core.types import Violation
from .graph import KnowledgeGraph

SH = rdflib.Namespace("http://www.w3.org/ns/shacl#")

# NodeKind URIs (compared as strings).
_NK_IRI = str(SH.IRI)
_NK_LITERAL = str(SH.Literal)
_NK_BLANK = str(SH.BlankNode)
_NK_BLANK_OR_IRI = str(SH.BlankNodeOrIRI)
_NK_BLANK_OR_LITERAL = str(SH.BlankNodeOrLiteral)
_NK_IRI_OR_LITERAL = str(SH.IRIOrLiteral)


class SHACLValidator:
    """Validate a :class:`KnowledgeGraph` against a SHACL shapes graph.

    Parameters
    ----------
    kg:
        The data graph to validate.
    shapes:
        A separate shapes graph. If ``None``, shapes are looked up in the
        data graph itself (a common SHACL convention).
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        shapes: Optional[KnowledgeGraph] = None,
    ) -> None:
        self.kg = kg
        self.shapes_kg = shapes if shapes is not None else kg
        self._g = kg.graph
        self._sg = self.shapes_kg.graph
        self._sg.bind("sh", URIRef(str(SH)))

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def validate(self) -> list[Violation]:
        violations: list[Violation] = []
        for shape in self._sg.subjects(RDF.type, SH.NodeShape):
            violations.extend(self._validate_shape(shape))
        # Deterministic ordering for stable test output.
        violations.sort(
            key=lambda v: (
                v.focus_node,
                v.path or "",
                v.constraint,
                v.value or "",
                v.message,
            )
        )
        return violations

    # ------------------------------------------------------------------
    # Targeting
    # ------------------------------------------------------------------
    def _targets(self, shape: Node) -> set[Node]:
        targets: set[Node] = set()
        # sh:targetClass C -> all subjects of rdf:type C.
        for cls in self._sg.objects(shape, SH.targetClass):
            for inst in self._g.subjects(RDF.type, cls):
                targets.add(inst)
            # Also honour subclass-ish direct typing only; no OWL inference.
        # sh:targetNode N.
        for node in self._sg.objects(shape, SH.targetNode):
            targets.add(node)
        # sh:targetSubjectsOf P.
        for prop in self._sg.objects(shape, SH.targetSubjectsOf):
            for s in self._g.subjects(prop):
                targets.add(s)
        # sh:targetObjectsOf P.
        for prop in self._sg.objects(shape, SH.targetObjectsOf):
            for _, _, o in self._g.triples((None, prop, None)):
                targets.add(o)
        return targets

    # ------------------------------------------------------------------
    # Shape evaluation
    # ------------------------------------------------------------------
    def _validate_shape(self, shape: Node) -> list[Violation]:
        out: list[Violation] = []
        shape_str = self.kg._term_to_str(shape)
        for focus in sorted(self._targets(shape), key=str):
            focus_str = self.kg._term_to_str(focus)
            # Node-level constraints (applied to the focus node directly).
            out.extend(self._check_node_class(shape_str, shape, focus, focus_str))
            out.extend(self._check_node_kind(shape_str, shape, focus, focus_str))
            # Property shapes.
            for pshape in self._sg.objects(shape, SH.property):
                out.extend(
                    self._validate_property(shape_str, pshape, focus, focus_str)
                )
            # Inline constraints directly on the node shape (no sh:path).
            out.extend(
                self._check_min_max(shape_str, shape, focus, focus_str, node_level=True)
            )
        return out

    # ------------------------------------------------------------------
    # Constraint checks
    # ------------------------------------------------------------------
    def _check_node_class(
        self,
        shape_str: str,
        shape: Node,
        focus: Node,
        focus_str: str,
    ) -> list[Violation]:
        out: list[Violation] = []
        for cls in self._sg.objects(shape, SH["class"]):
            if (focus, RDF.type, cls) not in self._g:
                out.append(
                    Violation(
                        focus_node=focus_str,
                        constraint="sh:class",
                        value=focus_str,
                        message=(
                            f"Node {focus_str} is not of type "
                            f"{self.kg._term_to_str(cls)}"
                        ),
                        shape=shape_str,
                        source_shape=shape_str,
                    )
                )
        return out

    def _check_node_kind(
        self,
        shape_str: str,
        shape: Node,
        focus: Node,
        focus_str: str,
    ) -> list[Violation]:
        out: list[Violation] = []
        nk = self._sg.value(shape, SH.nodeKind)
        if nk is not None and not self._node_kind_ok(focus, nk):
            out.append(
                Violation(
                    focus_node=focus_str,
                    constraint="sh:nodeKind",
                    value=focus_str,
                    message=(
                        f"Node {focus_str} does not satisfy nodeKind "
                        f"{self.kg._term_to_str(nk)}"
                    ),
                    shape=shape_str,
                    source_shape=shape_str,
                )
            )
        return out

    def _validate_property(
        self,
        shape_str: str,
        pshape: Node,
        focus: Node,
        focus_str: str,
    ) -> list[Violation]:
        out: list[Violation] = []
        path = self._sg.value(pshape, SH.path)
        if path is None:
            return out
        path_str = self.kg._term_to_str(path)
        values = list(self._g.objects(focus, path))
        out.extend(
            self._check_min_max(
                shape_str, pshape, focus, focus_str, node_level=False,
                path=path, path_str=path_str, values=values,
            )
        )
        out.extend(self._check_datatype(shape_str, pshape, focus, focus_str, path, path_str, values))
        out.extend(self._check_class(shape_str, pshape, focus, focus_str, path, path_str, values))
        out.extend(self._check_node_kind_property(shape_str, pshape, focus, focus_str, path, path_str, values))
        return out

    def _check_min_max(
        self,
        shape_str: str,
        pshape: Node,
        focus: Node,
        focus_str: str,
        node_level: bool,
        path: Optional[Node] = None,
        path_str: Optional[str] = None,
        values: Optional[list[Node]] = None,
    ) -> list[Violation]:
        out: list[Violation] = []
        if node_level:
            # Node-level min/max count is ill-defined; skip.
            return out
        assert path is not None and values is not None
        mc = self._sg.value(pshape, SH.minCount)
        if mc is not None and len(values) < self._as_int(mc):
            out.append(
                Violation(
                    focus_node=focus_str,
                    path=path_str,
                    constraint="sh:minCount",
                    message=(
                        f"Expected at least {self._as_int(mc)} value(s) for "
                        f"{path_str} on {focus_str}, found {len(values)}"
                    ),
                    shape=shape_str,
                    source_shape=self.kg._term_to_str(pshape),
                )
            )
        maxc = self._sg.value(pshape, SH.maxCount)
        if maxc is not None and len(values) > self._as_int(maxc):
            idx = self._as_int(maxc)
            offending = values[idx] if idx < len(values) else values[-1]
            out.append(
                Violation(
                    focus_node=focus_str,
                    path=path_str,
                    constraint="sh:maxCount",
                    value=self.kg._term_to_str(offending),
                    message=(
                        f"Expected at most {self._as_int(maxc)} value(s) for "
                        f"{path_str} on {focus_str}, found {len(values)}"
                    ),
                    shape=shape_str,
                    source_shape=self.kg._term_to_str(pshape),
                )
            )
        return out

    @staticmethod
    def _as_int(term: Node) -> int:
        """Best-effort integer coercion (handles rdflib Literal or plain str)."""
        try:
            return int(term)
        except (TypeError, ValueError):
            return int(str(term))

    def _check_datatype(
        self,
        shape_str: str,
        pshape: Node,
        focus: Node,
        focus_str: str,
        path: Node,
        path_str: str,
        values: list[Node],
    ) -> list[Violation]:
        out: list[Violation] = []
        dt = self._sg.value(pshape, SH.datatype)
        if dt is None:
            return out
        for v in values:
            if not isinstance(v, Literal):
                out.append(
                    self._make_violation(
                        shape_str, pshape, focus_str, path_str, v,
                        "sh:datatype",
                        f"Value {self.kg._term_to_str(v)} is not a literal of "
                        f"datatype {self.kg._term_to_str(dt)}",
                    )
                )
                continue
            if v.datatype is None or str(v.datatype) != str(dt):
                out.append(
                    self._make_violation(
                        shape_str, pshape, focus_str, path_str, v,
                        "sh:datatype",
                        f"Value {self.kg._term_to_str(v)} has datatype "
                        f"{v.datatype or 'none'}, expected "
                        f"{self.kg._term_to_str(dt)}",
                    )
                )
        return out

    def _check_class(
        self,
        shape_str: str,
        pshape: Node,
        focus: Node,
        focus_str: str,
        path: Node,
        path_str: str,
        values: list[Node],
    ) -> list[Violation]:
        out: list[Violation] = []
        cls = self._sg.value(pshape, SH["class"])
        if cls is None:
            return out
        for v in values:
            if isinstance(v, Literal):
                out.append(
                    self._make_violation(
                        shape_str, pshape, focus_str, path_str, v, "sh:class",
                        f"Value {self.kg._term_to_str(v)} is a literal, expected "
                        f"an instance of {self.kg._term_to_str(cls)}",
                    )
                )
                continue
            if (v, RDF.type, cls) not in self._g:
                out.append(
                    self._make_violation(
                        shape_str, pshape, focus_str, path_str, v, "sh:class",
                        f"Value {self.kg._term_to_str(v)} is not an instance of "
                        f"{self.kg._term_to_str(cls)}",
                    )
                )
        return out

    def _check_node_kind_property(
        self,
        shape_str: str,
        pshape: Node,
        focus: Node,
        focus_str: str,
        path: Node,
        path_str: str,
        values: list[Node],
    ) -> list[Violation]:
        out: list[Violation] = []
        nk = self._sg.value(pshape, SH.nodeKind)
        if nk is None:
            return out
        for v in values:
            if not self._node_kind_ok(v, nk):
                out.append(
                    self._make_violation(
                        shape_str, pshape, focus_str, path_str, v, "sh:nodeKind",
                        f"Value {self.kg._term_to_str(v)} does not satisfy nodeKind "
                        f"{self.kg._term_to_str(nk)}",
                    )
                )
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _make_violation(
        self,
        shape_str: str,
        pshape: Node,
        focus_str: str,
        path_str: str,
        value: Node,
        constraint: str,
        message: str,
    ) -> Violation:
        return Violation(
            focus_node=focus_str,
            path=path_str,
            value=self.kg._term_to_str(value),
            constraint=constraint,
            message=message,
            shape=shape_str,
            source_shape=self.kg._term_to_str(pshape),
        )

    @staticmethod
    def _node_kind_ok(value: Node, nk: Node) -> bool:
        s = str(nk)
        if s == _NK_IRI:
            return isinstance(value, URIRef)
        if s == _NK_LITERAL:
            return isinstance(value, Literal)
        if s == _NK_BLANK:
            return isinstance(value, BNode)
        if s == _NK_BLANK_OR_IRI:
            return isinstance(value, (URIRef, BNode))
        if s == _NK_BLANK_OR_LITERAL:
            return isinstance(value, (Literal, BNode))
        if s == _NK_IRI_OR_LITERAL:
            return isinstance(value, (URIRef, Literal))
        # Unknown nodeKind -> permissive.
        return True


__all__ = ["SHACLValidator", "SH"]
