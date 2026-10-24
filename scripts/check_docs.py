"""Check (and optionally fix) the shape of the tracked Markdown docs.

The docs follow one shape so diffs stay about meaning:

- one blank line between blocks, never two
- no blank line inside a bullet or numbered list
- one blank line after every heading, and before it
- paragraphs on a single line (no hard wrap)
- LF line endings, no trailing whitespace, exactly one final newline

ADRs and `archive/` are records, not documents we edit, so they are skipped.

Usage:
    python -m scripts.check_docs           # report problems, exit 1
    python -m scripts.check_docs --fix     # rewrite the files
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Paths that keep their own shape: decision records and the frozen archive.
SKIP_PARTS = ("archive/",)
SKIP_NAMES = ("CHANGELOG.md",)

FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^#{1,6}\s")
LIST_ITEM = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+")
TABLE_ROW = re.compile(r"^\s*\|")
QUOTE = re.compile(r"^\s*>\s?")
HTML_BLOCK = re.compile(r"^\s*<[a-zA-Z/!]")
THEMATIC_BREAK = re.compile(r"^\s*([-*_])\1{2,}\s*$")
SENTENCE_END = (".", "!", "?", ":", ";", "。", "！", "？")


def tracked_docs() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "*.md"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    ).stdout
    docs = []
    for raw in out.split(b"\0"):
        if not raw:
            continue
        rel = raw.decode()
        if any(f"/{part}" in f"/{rel}" or rel.startswith(part) for part in SKIP_PARTS):
            continue
        if Path(rel).name in SKIP_NAMES or re.search(r"ADR-\d", rel):
            continue
        docs.append(REPO_ROOT / rel)
    return docs


def block_kind(line: str) -> str:
    if not line.strip():
        return "blank"
    if FENCE.match(line):
        return "fence"
    if HEADING.match(line):
        return "heading"
    if TABLE_ROW.match(line):
        return "table"
    if THEMATIC_BREAK.match(line):
        return "break"
    if LIST_ITEM.match(line):
        return "item"
    if QUOTE.match(line):
        return "quote" if QUOTE.sub("", line).strip() else "blank"
    if HTML_BLOCK.match(line):
        return "html"
    return "text"


def to_blocks(lines: list[str]) -> list[tuple[str, list[str]]]:
    """Group the document into blocks, joining wrapped continuation lines."""
    blocks: list[tuple[str, list[str]]] = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            i += 1
            continue

        if FENCE.match(line):
            fence = line.strip()[:3]
            body = [line]
            i += 1
            while i < len(lines):
                raw = lines[i].rstrip()
                body.append(raw)
                i += 1
                if raw.strip().startswith(fence):
                    break
            blocks.append(("fence", body))
            continue

        kind = block_kind(line)
        if kind == "heading":
            blocks.append(("heading", [line]))
            i += 1
            continue
        if kind in ("table", "html", "break"):
            body = [line]
            i += 1
            while i < len(lines) and block_kind(lines[i].rstrip()) == kind:
                body.append(lines[i].rstrip())
                i += 1
            blocks.append((kind, body))
            continue

        # A blockquote holds one or more paragraphs, split by a lone `>` line.
        if kind == "quote":
            body: list[str] = []
            para: list[str] = []

            first = QUOTE.sub("", line).strip()
            if LIST_ITEM.match(first):
                body.append(f"> {first}")
            else:
                para.append(first)
            i += 1
            while i < len(lines):
                raw = lines[i].rstrip()
                if not raw.strip():
                    look = i + 1
                    while look < len(lines) and not lines[look].strip():
                        look += 1
                    if look < len(lines) and QUOTE.match(lines[look]):
                        if para:
                            body.append(f"> {' '.join(para)}".rstrip())
                            para = []
                        if body and body[-1] != ">":
                            body.append(">")
                        i = look
                        continue
                    break
                if not QUOTE.match(raw):
                    break
                content = QUOTE.sub("", raw).strip()
                if not content:
                    if para:
                        body.append(f"> {' '.join(para)}".rstrip())
                        para = []
                    if body and body[-1] != ">":
                        body.append(">")
                elif LIST_ITEM.match(content):
                    if para:
                        body.append(f"> {' '.join(para)}".rstrip())
                        para = []
                    body.append(f"> {content}")
                else:
                    para.append(content)
                i += 1
            if para:
                body.append(f"> {' '.join(para)}".rstrip())
            while body and body[-1] == ">":
                body.pop()
            blocks.append(("quote", body))
            continue

        # List item or paragraph: gather continuations, then join.
        marker = ""
        if kind == "item":
            match = LIST_ITEM.match(line)
            assert match is not None
            marker = match.group(0)
            parts = [line[len(marker) :].strip()]
        else:
            parts = [line.strip()]

        i += 1
        while i < len(lines):
            nxt = lines[i].rstrip()
            nkind = block_kind(nxt)
            if nkind in (
                "blank",
                "fence",
                "heading",
                "table",
                "break",
                "html",
                "item",
                "quote",
            ):
                break
            parts.append(nxt.strip())
            i += 1

        joined = " ".join(part for part in parts if part)
        blocks.append((kind, [f"{marker}{joined}".rstrip()]))

    return blocks


def normalise(lines: list[str]) -> list[str]:
    """Render the blocks with exactly one blank line between them."""
    rendered: list[str] = []
    previous = ""
    for kind, body in to_blocks(lines):
        if rendered:
            tight = kind == "item" and previous == "item"
            if not tight:
                rendered.append("")
        rendered.extend(body)
        previous = kind
    while rendered and not rendered[-1].strip():
        rendered.pop()
    return rendered


def body_lines(text: str) -> list[str]:
    """Split into lines, dropping only the trailing newline (never real content)."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def check_text(text: str) -> list[str]:
    """Return human-readable findings for one document."""
    problems: list[str] = []
    lines = body_lines(text)
    if not text.endswith("\n"):
        problems.append("missing final newline")
    elif text.endswith("\n\n"):
        problems.append("blank line at end of file")

    in_fence = False
    fence_line = 0
    for i, line in enumerate(lines, 1):
        if FENCE.match(line):
            in_fence = not in_fence
            fence_line = i
            continue
        if in_fence:
            continue
        if line != line.rstrip():
            problems.append(f"{i}: trailing whitespace")
        if "\t" in line:
            problems.append(f"{i}: tab character")
        if not line.strip() and i > 1:
            prev = lines[i - 2]
            nxt = lines[i] if i < len(lines) else ""
            if not prev.strip():
                problems.append(f"{i}: two blank lines in a row")
            elif LIST_ITEM.match(prev) and LIST_ITEM.match(nxt):
                problems.append(f"{i}: blank line inside a list")
        if HEADING.match(line) and i < len(lines) and lines[i].strip():
            problems.append(f"{i}: no blank line after heading")
    if in_fence:
        problems.append(f"{fence_line}: unbalanced code fence")

    renamed = normalise(lines)
    if renamed != lines:
        for i, (got, want) in enumerate(zip(lines, renamed, strict=False), 1):
            joined = block_kind(got) in ("text", "quote", "item") and bool(want)
            if got != want and joined and want.startswith(got.strip()):
                problems.append(f"{i}: hard-wrapped paragraph")
                break
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true", help="rewrite the files")
    parser.add_argument("paths", nargs="*", help="limit the run to these files")
    args = parser.parse_args()

    docs = tracked_docs()
    if args.paths:
        wanted = {Path(p).resolve() for p in args.paths}
        docs = [d for d in docs if d.resolve() in wanted]

    failures = 0
    for path in docs:
        rel = path.relative_to(REPO_ROOT).as_posix()
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            print(f"{rel}: not UTF-8")
            failures += 1
            continue
        crlf = b"\r\n" in raw
        if crlf:
            text = text.replace("\r\n", "\n")
        problems = check_text(text)
        if crlf:
            problems.insert(0, "CRLF line endings")
        if not problems:
            continue
        if args.fix:
            fixed = "\n".join(normalise(body_lines(text))) + "\n"
            if fixed != raw.decode("utf-8"):
                path.write_text(fixed, encoding="utf-8", newline="\n")
                print(f"fixed {rel}")
        else:
            failures += 1
            print(rel)
            for problem in problems[:6]:
                print(f"  {problem}")

    if not args.fix:
        if failures:
            print(f"\n{failures} file(s) need --fix")
            return 1
        print(f"{len(docs)} doc file(s) OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
