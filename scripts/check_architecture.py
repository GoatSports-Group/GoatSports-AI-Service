"""Fail CI when an inner Clean Architecture layer imports an outer layer."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULES = {
    "domain": ("app.application", "app.infrastructure", "app.presentation"),
    "application": ("app.infrastructure", "app.presentation"),
    "infrastructure": ("app.presentation",),
}


def imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def main() -> None:
    violations: list[str] = []
    for layer, forbidden_prefixes in RULES.items():
        for path in (ROOT / "app" / layer).rglob("*.py"):
            for module in imports(path):
                if module.startswith(forbidden_prefixes):
                    violations.append(f"{path.relative_to(ROOT)} imports {module}")
    if violations:
        raise SystemExit("Architecture violations:\n" + "\n".join(violations))
    print("ARCHITECTURE_CHECK_OK")


if __name__ == "__main__":
    main()
