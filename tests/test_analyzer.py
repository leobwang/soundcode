"""Smoke test for RustAnalyzer wrapper.

Run with: uv run python tests/test_analyzer.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from soundcode import RustAnalyzer, DiagnosticSeverity


PROJECT = Path(__file__).parent.parent / "test_workspace"


async def test_diagnostics_clean():
    """A valid program should produce no errors."""
    async with RustAnalyzer(PROJECT) as ra:
        await ra.open_file("src/main.rs", "fn main() {\n    let x: i32 = 42;\n    println!(\"{}\", x);\n}\n")
        # Give rust-analyzer a moment to process
        await asyncio.sleep(1)
        errors = await ra.get_errors("src/main.rs")
        print(f"[clean] errors: {len(errors)}")
        for e in errors:
            print(f"  {e.code}: {e.message}")
        assert len(errors) == 0, f"Expected no errors, got {errors}"


async def test_diagnostics_type_error():
    """A type mismatch should produce an error diagnostic."""
    async with RustAnalyzer(PROJECT) as ra:
        code = 'fn main() {\n    let x: String = 42;\n}\n'
        await ra.open_file("src/main.rs", code)
        await asyncio.sleep(1)
        errors = await ra.get_errors("src/main.rs")
        print(f"[type_error] errors: {len(errors)}")
        for e in errors:
            print(f"  L{e.start_line} {e.code}: {e.message}")
        assert len(errors) > 0, "Expected at least one error for type mismatch"


async def test_diagnostics_unresolved_name():
    """An unresolved name should produce an error diagnostic."""
    async with RustAnalyzer(PROJECT) as ra:
        code = "fn main() {\n    let x = nonexistent_function();\n}\n"
        await ra.open_file("src/main.rs", code)
        await asyncio.sleep(1)
        errors = await ra.get_errors("src/main.rs")
        print(f"[unresolved] errors: {len(errors)}")
        for e in errors:
            print(f"  L{e.start_line} {e.code}: {e.message}")
        assert len(errors) > 0, "Expected at least one error for unresolved name"


async def test_incremental_update():
    """Updating file content should produce fresh diagnostics."""
    async with RustAnalyzer(PROJECT) as ra:
        # Start clean
        await ra.open_file("src/main.rs", "fn main() {\n    let x: i32 = 42;\n}\n")
        await asyncio.sleep(1)
        errors1 = await ra.get_errors("src/main.rs")
        print(f"[incremental] before: {len(errors1)} errors")

        # Introduce error
        await ra.update_file("src/main.rs", 'fn main() {\n    let x: String = 42;\n}\n')
        await asyncio.sleep(1)
        errors2 = await ra.get_errors("src/main.rs")
        print(f"[incremental] after:  {len(errors2)} errors")
        for e in errors2:
            print(f"  L{e.start_line} {e.code}: {e.message}")

        assert len(errors1) == 0, f"Expected no initial errors, got {errors1}"
        assert len(errors2) > 0, "Expected errors after introducing type mismatch"


async def test_append():
    """Appending to a file should work via incremental edit."""
    async with RustAnalyzer(PROJECT) as ra:
        await ra.open_file("src/main.rs", "fn main() {\n")
        await ra.append_to_file("src/main.rs", "    let x: i32 = 42;\n")
        await ra.append_to_file("src/main.rs", "}\n")
        content = ra.get_content("src/main.rs")
        print(f"[append] content:\n{content}")
        assert content == "fn main() {\n    let x: i32 = 42;\n}\n"

        await asyncio.sleep(1)
        errors = await ra.get_errors("src/main.rs")
        print(f"[append] errors: {len(errors)}")
        assert len(errors) == 0, f"Expected no errors, got {errors}"


async def test_completions():
    """Completions after '.' on a Vec should include push, len, etc."""
    async with RustAnalyzer(PROJECT) as ra:
        code = "fn main() {\n    let v: Vec<i32> = vec![1, 2, 3];\n    v.\n}\n"
        await ra.open_file("src/main.rs", code)
        await asyncio.sleep(1)
        completions = await ra.get_completions("src/main.rs", line=2, character=6, trigger_character=".")
        labels = [c.label for c in completions]
        print(f"[completions] got {len(completions)} items: {labels[:10]}...")
        assert any("push" in l for l in labels) or any("len" in l for l in labels), f"Expected Vec methods in completions, got {labels[:20]}"


async def test_hover():
    """Hover on a variable should return type info."""
    async with RustAnalyzer(PROJECT) as ra:
        code = "fn main() {\n    let x: Vec<i32> = vec![1, 2, 3];\n}\n"
        await ra.open_file("src/main.rs", code)
        await asyncio.sleep(1)
        hover = await ra.get_hover("src/main.rs", line=1, character=8)
        print(f"[hover] result: {hover[:100] if hover else None}")
        assert hover is not None, "Expected hover info for variable"


async def test_line_range_filter():
    """Line range filter should restrict diagnostics to specified lines."""
    async with RustAnalyzer(PROJECT) as ra:
        code = (
            "fn main() {\n"
            "    let x: String = 42;\n"       # line 1: error
            "    let y: i32 = \"hello\";\n"   # line 2: error
            "}\n"
        )
        await ra.open_file("src/main.rs", code)
        await asyncio.sleep(1)

        all_errors = await ra.get_errors("src/main.rs")
        line1_errors = await ra.get_errors("src/main.rs", line_range=(1, 1))
        print(f"[line_range] total errors: {len(all_errors)}, line 1 only: {len(line1_errors)}")
        for e in line1_errors:
            print(f"  L{e.start_line} {e.code}: {e.message}")


async def main():
    tests = [
        ("clean program", test_diagnostics_clean),
        ("type error", test_diagnostics_type_error),
        ("unresolved name", test_diagnostics_unresolved_name),
        ("incremental update", test_incremental_update),
        ("append", test_append),
        ("completions", test_completions),
        ("hover", test_hover),
        ("line range filter", test_line_range_filter),
    ]

    passed = 0
    failed = 0
    for name, test in tests:
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print(f"{'='*60}")
        try:
            await test()
            print(f"PASSED")
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'='*60}")
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
