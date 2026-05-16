"""Send a Rust source string to rust-analyzer; print the raw LSP response."""

import asyncio
import json
import pathlib

from multilspy import LanguageServer
from multilspy.multilspy_config import MultilspyConfig
from multilspy.multilspy_logger import MultilspyLogger

WORKSPACE = pathlib.Path(__file__).resolve().parent.parent / "rust-samples" / "sample0"
TARGET = WORKSPACE / "src" / "main.rs"

class DiagnosticType(Enum):
    NON_BLOCKING = 0
    BLOCKING = 1

class Diagnostic:
    # a single dianostic message
    def __init__(self, ...):
        self.type: DiagnosticType
        self.message: str
        self.range: dict

class StateType(Enum):
    INACTIVE = 0
    NON_BLOCKING = 1
    BLOCKING = 2

class State:
    # state of a .rs file
    def __init__(self, path: str | pathlib.Path):
        self.path: pathlib.Path
        self.content: str
        self.diagnostics: list[Diagnostic]


class Server:
    # One instance per project
    def __init__(self):
        # TODO: 
    
    async def receive_token(self, token: str):


async def diagnose(source: str, wait_s: float = 8.0) -> list[dict]:
    lsp = LanguageServer.create(
        MultilspyConfig.from_dict({"code_language": "rust"}),
        MultilspyLogger(),
        str(WORKSPACE),
    )
    uri = TARGET.as_uri()
    captured: list[dict] = []
    original = TARGET.read_text()
    TARGET.write_text(source)
    try:
        async with lsp.start_server():
            lsp.server.on_notification(
                "textDocument/publishDiagnostics",
                lambda p: captured.append(p) if p["uri"] == uri else None,
            )
            with lsp.open_file("src/main.rs"):
                await asyncio.sleep(wait_s)
    finally:
        TARGET.write_text(original)
    return captured


if __name__ == "__main__":
    code = 'fn main() { let x: i32 = "oops"; }'
    print(json.dumps(asyncio.run(diagnose(code)), indent=2))
