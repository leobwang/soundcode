"""Headless rust-analyzer wrapper for code generation verification.

Provides a simple async interface to rust-analyzer's LSP capabilities,
optimized for programmatic use during code generation: incremental file
updates, pull diagnostics, completions, and hover.
"""

from __future__ import annotations

import asyncio
import enum
from collections.abc import Callable, Coroutine, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import AsyncGenerator, Any, TypeVar

import lsprotocol.types as lsp_type
from lsp_client import RustAnalyzerClient, Position, Range
from lsp_client.jsonrpc.exception import JsonRpcResponseError

T = TypeVar("T")

SERVER_CANCELLED = -32802
CONTENT_MODIFIED = -32801
_RETRYABLE_CODES = {SERVER_CANCELLED, CONTENT_MODIFIED}


async def _retry_on_cancel(
    fn: Callable[[], Coroutine[Any, Any, T]],
    *,
    max_retries: int = 5,
    delay: float = 0.5,
) -> T:
    """Retry an LSP request if the server cancels it or reports content modified."""
    for attempt in range(max_retries):
        try:
            return await fn()
        except JsonRpcResponseError as e:
            if e.code not in _RETRYABLE_CODES or attempt == max_retries - 1:
                raise
            await asyncio.sleep(delay)
    raise RuntimeError("max_retries must be >= 1")


class DiagnosticSeverity(enum.IntEnum):
    ERROR = 1
    WARNING = 2
    INFORMATION = 3
    HINT = 4


@dataclass(frozen=True)
class Diagnostic:
    """Simplified diagnostic from rust-analyzer."""

    message: str
    severity: DiagnosticSeverity
    range: Range
    code: str | None = None
    source: str | None = None

    @classmethod
    def from_lsp(cls, d: lsp_type.Diagnostic) -> Diagnostic:
        code = None
        if d.code is not None:
            code = str(d.code)
        return cls(
            message=d.message,
            severity=DiagnosticSeverity(int(d.severity)) if d.severity else DiagnosticSeverity.ERROR,
            range=d.range,
            code=code,
            source=d.source,
        )

    @property
    def is_error(self) -> bool:
        return self.severity == DiagnosticSeverity.ERROR

    @property
    def start_line(self) -> int:
        return self.range.start.line

    @property
    def end_line(self) -> int:
        return self.range.end.line

    def overlaps_lines(self, start: int, end: int) -> bool:
        """Check if this diagnostic overlaps a line range (inclusive)."""
        return self.start_line <= end and self.end_line >= start


_COMPLETION_KIND_NAMES: dict[int, str] = {
    1: "text", 2: "method", 3: "function", 4: "constructor",
    5: "field", 6: "variable", 7: "class", 8: "interface",
    9: "module", 10: "property", 11: "unit", 12: "value",
    13: "enum", 14: "keyword", 15: "snippet", 16: "color",
    17: "file", 18: "reference", 19: "folder", 20: "enum_member",
    21: "constant", 22: "struct", 23: "event", 24: "operator",
    25: "type_parameter",
}


@dataclass(frozen=True)
class CompletionItem:
    """Simplified completion item."""

    label: str
    kind: str | None = None
    detail: str | None = None
    insert_text: str | None = None

    @classmethod
    def from_lsp(cls, item: lsp_type.CompletionItem) -> CompletionItem:
        kind_name = None
        if item.kind is not None:
            kind_val = int(item.kind)
            kind_name = _COMPLETION_KIND_NAMES.get(kind_val, str(kind_val))
        return cls(
            label=item.label,
            kind=kind_name,
            detail=item.detail,
            insert_text=item.insert_text or item.label,
        )


# Diagnostic codes that are noisy on partial/incomplete code and should
# not trigger rollback.
SUPPRESSED_CODES: set[str] = {
    "unused-variables",
    "incorrect-ident-case",
}


class _TrackedClient(RustAnalyzerClient):
    """RustAnalyzerClient subclass that captures push diagnostics."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._push_diagnostics: dict[str, list[lsp_type.Diagnostic]] = {}

    async def _receive_publish_diagnostics(
        self, params: lsp_type.PublishDiagnosticsParams
    ) -> None:
        self._push_diagnostics[params.uri] = list(params.diagnostics)


class RustAnalyzer:
    """Headless rust-analyzer wrapper for code generation.

    Manages file state independently from lsp-client's DocumentStateManager
    and calls internal LSP methods directly to avoid conflicts with the
    library's automatic file open/close lifecycle.

    Usage::

        async with RustAnalyzer("/path/to/cargo/project") as ra:
            await ra.open_file("src/main.rs", "fn main() {\\n\\n}\\n")
            await ra.update_file("src/main.rs", "fn main() {\\n    let x: i32 = 42;\\n}\\n")
            errors = await ra.get_errors("src/main.rs")
    """

    def __init__(
        self,
        project_path: str | Path,
        *,
        check_on_save: bool = False,
        experimental_diagnostics: bool = False,
        request_timeout: float = 30.0,
    ) -> None:
        self._project_path = Path(project_path).resolve()
        self._check_on_save = check_on_save
        self._experimental_diagnostics = experimental_diagnostics
        self._request_timeout = request_timeout
        self._client: _TrackedClient | None = None
        self._files: dict[str, str] = {}  # path -> current content
        self._versions: dict[str, int] = {}  # path -> version counter

    @asynccontextmanager
    async def _create_client(self) -> AsyncGenerator[_TrackedClient]:
        client = _TrackedClient(
            workspace=str(self._project_path),
            server="local",
            request_timeout=self._request_timeout,
            initialization_options={
                "cargo": {
                    "buildScripts": {"enable": True},
                    "allTargets": True,
                },
                "procMacro": {"enable": True},
                "checkOnSave": {"enable": self._check_on_save},
                "diagnostics": {
                    "enable": True,
                    "experimental": {"enable": self._experimental_diagnostics},
                },
            },
        )
        async with client as c:
            yield c

    async def __aenter__(self) -> RustAnalyzer:
        self._ctx = self._create_client()
        self._client = await self._ctx.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        for path in list(self._files):
            try:
                await self._close_file(path)
            except Exception:
                pass
        try:
            await self._ctx.__aexit__(exc_type, exc_val, exc_tb)
        except (ExceptionGroup, BaseExceptionGroup):
            # lsp-client's server teardown can race with pending
            # background tasks (cargo check, workspace loading). This is
            # not actionable — suppress it during shutdown.
            pass
        self._client = None

    @property
    def client(self) -> _TrackedClient:
        if self._client is None:
            raise RuntimeError("RustAnalyzer is not initialized. Use 'async with RustAnalyzer(...) as ra:'")
        return self._client

    def _resolve_path(self, file_path: str) -> str:
        """Resolve a relative path against the project root."""
        p = Path(file_path)
        if not p.is_absolute():
            p = self._project_path / p
        return str(p)

    def _uri(self, resolved_path: str) -> str:
        return self.client.as_uri(resolved_path)

    def _text_doc_id(self, resolved_path: str) -> lsp_type.TextDocumentIdentifier:
        return lsp_type.TextDocumentIdentifier(uri=self._uri(resolved_path))

    # ── File management ──────────────────────────────────────────────

    async def open_file(self, file_path: str, content: str) -> None:
        """Open a file in rust-analyzer's VFS with the given content.

        The file does not need to exist on disk.
        """
        resolved = self._resolve_path(file_path)
        self._files[resolved] = content
        self._versions[resolved] = 0
        await self.client.notify_text_document_opened(resolved, content)

    async def _close_file(self, resolved_path: str) -> None:
        await self.client.notify_text_document_closed(resolved_path)
        self._files.pop(resolved_path, None)
        self._versions.pop(resolved_path, None)

    async def update_file(self, file_path: str, content: str) -> None:
        """Replace the entire file content. Opens the file if not already open."""
        resolved = self._resolve_path(file_path)
        if resolved not in self._files:
            await self.open_file(file_path, content)
            return
        self._versions[resolved] += 1
        self._files[resolved] = content
        change = lsp_type.TextDocumentContentChangeWholeDocument(text=content)
        await self.client.notify_text_document_changed(
            resolved,
            content_changes=[change],
            version=self._versions[resolved],
        )

    async def append_to_file(self, file_path: str, text: str) -> None:
        """Append text to the end of a tracked file."""
        resolved = self._resolve_path(file_path)
        if resolved not in self._files:
            raise RuntimeError(f"File not open: {file_path}")
        old = self._files[resolved]
        lines = old.split("\n")
        last_line = len(lines) - 1
        last_char = len(lines[-1])
        self._versions[resolved] += 1
        self._files[resolved] = old + text
        change = lsp_type.TextDocumentContentChangePartial(
            range=Range(
                start=Position(line=last_line, character=last_char),
                end=Position(line=last_line, character=last_char),
            ),
            text=text,
        )
        await self.client.notify_text_document_changed(
            resolved,
            content_changes=[change],
            version=self._versions[resolved],
        )

    def get_content(self, file_path: str) -> str:
        """Return the current in-memory content of a tracked file."""
        resolved = self._resolve_path(file_path)
        if resolved not in self._files:
            raise RuntimeError(f"File not open: {file_path}")
        return self._files[resolved]

    # ── Diagnostics ──────────────────────────────────────────────────

    async def get_diagnostics(
        self,
        file_path: str,
        *,
        min_severity: DiagnosticSeverity = DiagnosticSeverity.HINT,
    ) -> list[Diagnostic]:
        """Pull native diagnostics from rust-analyzer.

        Returns diagnostics filtered by severity. Does not include cargo
        check / flycheck diagnostics (those arrive via push and are
        available separately via get_push_diagnostics).
        """
        resolved = self._resolve_path(file_path)
        report = await _retry_on_cancel(lambda: self.client._request_diagnostic(
            lsp_type.DocumentDiagnosticParams(
                text_document=self._text_doc_id(resolved),
            )
        ))
        items: Sequence[lsp_type.Diagnostic] = ()
        if isinstance(report, lsp_type.RelatedFullDocumentDiagnosticReport):
            items = report.items
        return [
            Diagnostic.from_lsp(d)
            for d in items
            if d.severity is not None and int(d.severity) <= min_severity.value
        ]

    async def get_errors(
        self,
        file_path: str,
        *,
        suppress_noisy: bool = True,
        line_range: tuple[int, int] | None = None,
    ) -> list[Diagnostic]:
        """Get error-severity native diagnostics, with optional filtering.

        Args:
            file_path: Path to the file.
            suppress_noisy: If True, filter out diagnostics with codes in
                SUPPRESSED_CODES (e.g. unused-variables).
            line_range: If set, only return diagnostics overlapping this
                (start_line, end_line) range (inclusive, 0-indexed).
        """
        diags = await self.get_diagnostics(
            file_path, min_severity=DiagnosticSeverity.ERROR
        )
        if suppress_noisy:
            diags = [d for d in diags if d.code not in SUPPRESSED_CODES]
        if line_range is not None:
            start, end = line_range
            diags = [d for d in diags if d.overlaps_lines(start, end)]
        return diags

    def get_push_diagnostics(self, file_path: str) -> list[Diagnostic]:
        """Get the latest push diagnostics (from cargo check / flycheck).

        These arrive asynchronously after checkOnSave or manual flycheck.
        Returns an empty list if none have been received.
        """
        resolved = self._resolve_path(file_path)
        uri = self._uri(resolved)
        raw = self.client._push_diagnostics.get(uri, [])
        return [Diagnostic.from_lsp(d) for d in raw]

    async def run_flycheck(self, file_path: str) -> None:
        """Manually trigger cargo check for a file.

        Results will arrive asynchronously and be available via
        get_push_diagnostics. Note: this is slow (2-30s).
        """
        resolved = self._resolve_path(file_path)
        await self.client.notify(
            lsp_type.DidSaveTextDocumentNotification(
                params=lsp_type.DidSaveTextDocumentParams(
                    text_document=self._text_doc_id(resolved),
                )
            )
        )

    # ── Completions ──────────────────────────────────────────────────

    async def get_completions(
        self,
        file_path: str,
        line: int,
        character: int,
        *,
        trigger_character: str | None = None,
    ) -> list[CompletionItem]:
        """Get completion items at a position.

        Args:
            file_path: Path to the file.
            line: 0-indexed line number.
            character: 0-indexed character offset.
            trigger_character: Optional trigger (e.g. ".", ":").
        """
        resolved = self._resolve_path(file_path)
        trigger_kind = lsp_type.CompletionTriggerKind.Invoked
        if trigger_character is not None:
            trigger_kind = lsp_type.CompletionTriggerKind.TriggerCharacter
        context = lsp_type.CompletionContext(
            trigger_kind=trigger_kind,
            trigger_character=trigger_character,
        )
        result = await _retry_on_cancel(lambda: self.client._request_completion(
            lsp_type.CompletionParams(
                text_document=self._text_doc_id(resolved),
                position=Position(line=line, character=character),
                context=context,
            )
        ))
        if result is None:
            return []
        if isinstance(result, lsp_type.CompletionList):
            items = result.items
        elif isinstance(result, list):
            items = result
        else:
            items = []
        return [CompletionItem.from_lsp(item) for item in items]

    # ── Hover ────────────────────────────────────────────────────────

    async def get_hover(
        self, file_path: str, line: int, character: int
    ) -> str | None:
        """Get hover information (type, docs) at a position.

        Returns the hover content as a string, or None if nothing is
        available at the position.
        """
        resolved = self._resolve_path(file_path)
        result = await _retry_on_cancel(lambda: self.client._request_hover(
            lsp_type.HoverParams(
                text_document=self._text_doc_id(resolved),
                position=Position(line=line, character=character),
            )
        ))
        if result is None:
            return None
        if isinstance(result.contents, lsp_type.MarkupContent):
            return result.contents.value
        if isinstance(result.contents, str):
            return result.contents
        return str(result.contents)

    # ── Navigation ───────────────────────────────────────────────────

    async def get_definition(
        self, file_path: str, line: int, character: int
    ) -> list[tuple[str, Range]] | None:
        """Get definition location(s) for the symbol at a position.

        Returns a list of (file_uri, range) tuples, or None.
        """
        resolved = self._resolve_path(file_path)
        result = await _retry_on_cancel(lambda: self.client._request_definition(
            lsp_type.DefinitionParams(
                text_document=self._text_doc_id(resolved),
                position=Position(line=line, character=character),
            )
        ))
        if result is None:
            return None
        locations: list[tuple[str, Range]] = []
        if isinstance(result, lsp_type.Location):
            locations.append((result.uri, result.range))
        elif isinstance(result, list):
            for item in result:
                if isinstance(item, lsp_type.Location):
                    locations.append((item.uri, item.range))
                elif isinstance(item, lsp_type.LocationLink):
                    locations.append((item.target_uri, item.target_range))
        return locations or None

    async def get_references(
        self,
        file_path: str,
        line: int,
        character: int,
        *,
        include_declaration: bool = True,
    ) -> list[tuple[str, Range]] | None:
        """Find all references to the symbol at a position.

        Returns a list of (file_uri, range) tuples, or None.
        """
        resolved = self._resolve_path(file_path)
        result = await _retry_on_cancel(lambda: self.client._request_references(
            lsp_type.ReferenceParams(
                text_document=self._text_doc_id(resolved),
                position=Position(line=line, character=character),
                context=lsp_type.ReferenceContext(
                    include_declaration=include_declaration
                ),
            )
        ))
        if result is None:
            return None
        return [(loc.uri, loc.range) for loc in result]
