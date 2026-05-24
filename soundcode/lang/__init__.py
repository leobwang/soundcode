"""Per-language abstractions for SoundCode.

The three Protocols (`Checker`, `BoundaryDetector`, `Workspace`) are the
contract a language pack must implement. Concrete impls live in the
per-language modules: `rust.py`, `java.py`, `cpp.py`, `python.py`. Each
module supplies a sync checker (subprocess-based or compile()-based) and
an async LSP-based checker, plus a boundary detector and workspace.
"""

from soundcode.lang.boundary import BoundaryDetector
from soundcode.lang.checker import Checker
from soundcode.lang.cpp import (
    ClangdLspChecker,
    CppBoundaryDetector,
    CppWorkspace,
    GccChecker,
)
from soundcode.lang.java import (
    JavaBoundaryDetector,
    JavaWorkspace,
    JavacChecker,
    JdtLspChecker,
)
from soundcode.lang.python import (
    PyrightLspChecker,
    PythonBoundaryDetector,
    PythonCompileChecker,
    PythonWorkspace,
)
from soundcode.lang.rust import (
    RustAnalyzerLspChecker,
    RustBoundaryDetector,
    RustCargoChecker,
    RustWorkspace,
)
from soundcode.lang.workspace import Workspace

__all__ = [
    # Protocols
    "BoundaryDetector",
    "Checker",
    "Workspace",
    # Rust
    "RustCargoChecker",
    "RustAnalyzerLspChecker",
    "RustBoundaryDetector",
    "RustWorkspace",
    # Java
    "JavacChecker",
    "JdtLspChecker",
    "JavaBoundaryDetector",
    "JavaWorkspace",
    # C++
    "GccChecker",
    "ClangdLspChecker",
    "CppBoundaryDetector",
    "CppWorkspace",
    # Python
    "PythonCompileChecker",
    "PyrightLspChecker",
    "PythonBoundaryDetector",
    "PythonWorkspace",
]
