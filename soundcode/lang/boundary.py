"""BoundaryDetector protocol — language-agnostic offset finder for
checkpointable positions in generated source.

Phase 0: Rust uses depth-0 `;` or `}` outside strings/comments/parens/
brackets. C++/Java will share the same rule. Python (Phase 1+) will use
newline-at-indent-0. The protocol method returns a flat list of integer
offsets (NOT richer Boundary objects) so per-language detectors are free
to use whatever internal representation makes sense without forcing the
existing `soundcode/eval/boundary.Boundary` dataclass on every backend.

Alternative considered: have the protocol return the existing
`Boundary(offset, kind)` dataclass directly. Chose offsets-only because
the kind field is Rust-specific (`;` vs `}`) and downstream callers
(`Code._update_boundary_checkpoints`) only use the offset; punting the
richer type to Phase 1 keeps the abstraction minimal.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class BoundaryDetector(Protocol):
    """Detect statement/block boundaries in incremental source.

    Returns a list of offsets where the *next* checkpoint should be placed
    — i.e., the boundary character itself is at `offset - 1`. This matches
    the existing `Code._update_boundary_checkpoints` convention
    (`offset = b.offset + 1`).
    """

    def find_boundaries(self, source: str) -> list[int]: ...
