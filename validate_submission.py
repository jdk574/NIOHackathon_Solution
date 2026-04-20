from __future__ import annotations

"""Small structural validator for generated NAS submission files."""

import argparse
from pathlib import Path


def validate_file(path: Path) -> list[str]:
    """Check only the file structure required by the competition format."""
    issues: list[str] = []
    lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]
    if not lines:
        return ["File is empty."]

    if lines[0].strip() != "BEGIN BULK":
        issues.append("Missing BEGIN BULK header.")
    if lines[-1].strip() != "ENDDATA":
        issues.append("Missing ENDDATA footer.")

    seen_grid = False
    seen_element = False

    for line_number, line in enumerate(lines[1:-1], start=2):
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        card = parts[0]

        if card == "GRID":
            seen_grid = True
            if len(parts) < 5:
                issues.append(f"Line {line_number}: GRID row has too few fields.")
        elif card in {"CTRIA3", "CQUAD4"}:
            seen_element = True
            expected = 6 if card == "CTRIA3" else 7
            if len(parts) < expected:
                issues.append(f"Line {line_number}: {card} row has too few fields.")
        else:
            issues.append(f"Line {line_number}: Unexpected card '{card}'.")

    if not seen_grid:
        issues.append("No GRID rows found.")
    if not seen_element:
        issues.append("No element rows found.")

    return issues


def main() -> None:
    """Validate one NAS file or every NAS file inside a directory."""
    parser = argparse.ArgumentParser(description="Validate generated NAS files structurally.")
    parser.add_argument("path", type=Path, help="A single NAS file or a directory of NAS files.")
    args = parser.parse_args()

    if args.path.is_dir():
        files = sorted(args.path.glob("*.nas"))
    else:
        files = [args.path]

    total_issues = 0
    for file_path in files:
        issues = validate_file(file_path)
        if issues:
            total_issues += len(issues)
            print(f"{file_path}:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"{file_path}: OK")

    if total_issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
