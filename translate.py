#!/usr/bin/env python3
"""Ren'Py translation tool: fills empty translation entries using Ollama (Qwen3)."""

import argparse
import json
import re
import shutil
from datetime import datetime
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = {
    "ollama_host": "http://localhost:11434",
    "model": "qwen3:14b",
    "batch_size": 5,
    "context_lines": 5,   # preceding dialogue lines as translation context
    "num_ctx": 8192,       # Ollama context window tokens
}


def load_config() -> dict:
    """Load config from config.json next to this script.

    Returns DEFAULT_CONFIG values for any missing keys.
    Silently falls back to defaults if the file doesn't exist or is invalid.
    """
    config = dict(DEFAULT_CONFIG)
    config_path = SCRIPT_DIR / "config.json"
    if config_path.is_file():
        try:
            with open(config_path, encoding="utf-8") as f:
                user = json.load(f)
            for key in DEFAULT_CONFIG:
                if key in user:
                    config[key] = user[key]
        except (json.JSONDecodeError, OSError):
            pass
    return config


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Dialogue format:
#     # speaker "english text"
#     speaker ""
# Speaker can be absent (narrator), e.g.:
#     # "english text"
#     ""
DIALOGUE_RE = re.compile(
    r'^(?P<comment_indent> +)# (?P<speaker>\S+ )?"(?P<original>.*)".*$\n'
    r'^(?P<line_indent> +)(?:\S+ )?"(?P<translation>)".*$',
    re.MULTILINE,
)

# Force-mode: match entries that already have translations
DIALOGUE_FILLED_RE = re.compile(
    r'^(?P<comment_indent> +)# (?P<speaker>\S+ )?"(?P<original>.*)".*$\n'
    r'^(?P<line_indent> +)(?:\S+ )?"(?P<translation>.+)".*$',
    re.MULTILINE,
)

# String format:
#     old "english text"
#     new ""
STRING_RE = re.compile(
    r'^(?P<old_indent> +)old "(?P<original>.*)"$\n'
    r'^(?P<new_indent> +)new "(?P<translation>)"$',
    re.MULTILINE,
)

# Force-mode: match string entries that already have translations
STRING_FILLED_RE = re.compile(
    r'^(?P<old_indent> +)old "(?P<original>.*)"$\n'
    r'^(?P<new_indent> +)new "(?P<translation>.+)"$',
    re.MULTILINE,
)


def find_entries(content: str, force: bool = False) -> list[dict]:
    """Find untranslated (and optionally already-translated) entries in a .rpy file.

    When force=True, also includes entries that already have translations.

    Returns a list of dicts with keys:
      - original: the English text
      - old_translation: existing translation ("" if untranslated)
      - match: the re.Match object (for replacement)
      - kind: 'dialogue' or 'string'
    """
    entries = []
    seen_positions: set[int] = set()

    for m in DIALOGUE_RE.finditer(content):
        original = m.group("original")
        if not original:
            continue
        seen_positions.add(m.start())
        entries.append({"original": original, "old_translation": "",
                        "match": m, "kind": "dialogue"})

    for m in STRING_RE.finditer(content):
        original = m.group("original")
        if not original:
            continue
        seen_positions.add(m.start())
        entries.append({"original": original, "old_translation": "",
                        "match": m, "kind": "string"})

    if force:
        for m in DIALOGUE_FILLED_RE.finditer(content):
            if m.start() in seen_positions:
                continue
            original = m.group("original")
            if not original:
                continue
            entries.append({"original": original,
                            "old_translation": m.group("translation"),
                            "match": m, "kind": "dialogue"})

        for m in STRING_FILLED_RE.finditer(content):
            if m.start() in seen_positions:
                continue
            original = m.group("original")
            if not original:
                continue
            entries.append({"original": original,
                            "old_translation": m.group("translation"),
                            "match": m, "kind": "string"})

    # Sort by position in file
    entries.sort(key=lambda e: e["match"].start())
    return entries


# ---------------------------------------------------------------------------
# Ren'Py tag validation
# ---------------------------------------------------------------------------

TAG_RE = re.compile(r'\{[^}]+\}|\[[^\]]+\]|\\n')


def _is_whitespace_content(text: str) -> bool:
    """Check if text contains only whitespace after removing Ren'Py tags."""
    return not TAG_RE.sub('', text).strip()


def extract_tags(text: str) -> list[str]:
    """Extract all Ren'Py tags ([...] and {...}) from text."""
    return TAG_RE.findall(text)


_POSSESSIVE_RE = re.compile(r"(\[[^\]]+\])'s\b")


def preprocess_for_translation(text: str) -> str:
    """Pre-process text before sending to LLM."""
    # [tag]'s → [tag]的 (English possessive → Chinese possessive)
    text = _POSSESSIVE_RE.sub(r"\1的", text)
    return text


def mask_tags(text: str) -> tuple[str, list[str]]:
    """Replace Ren'Py tags with numbered placeholders."""
    tags: list[str] = []
    def replacer(m: re.Match) -> str:
        tags.append(m.group(0))
        return f"<#{len(tags)}>"
    masked = TAG_RE.sub(replacer, text)
    return masked, tags


def unmask_tags(text: str, tags: list[str]) -> str:
    """Restore numbered placeholders back to original tags."""
    for i, tag in enumerate(tags, 1):
        text = text.replace(f"<#{i}>", tag, 1)
    # Remove any hallucinated <#N> placeholders from LLM output
    text = re.sub(r'<#\d+>', '', text)
    return text


_LEADING_TAGS_RE = re.compile(r'^((?:\{[^}]+\}|\[[^\]]+\])+)')
_TRAILING_TAGS_RE = re.compile(r'((?:\{[^}]+\}|\[[^\]]+\])+)([.?!。？！…～~]*)$')


def strip_boundary_tags(text: str) -> tuple[str, str, str]:
    """Strip leading/trailing {…} tags, wrapping dashes, and trailing punctuation.
    Returns (prefix, core, suffix).
    """
    prefix = ""
    m = _LEADING_TAGS_RE.match(text)
    if m:
        prefix = m.group(1)
        text = text[m.end():]

    suffix = ""
    m = _TRAILING_TAGS_RE.search(text)
    if m:
        suffix = m.group(1) + m.group(2)   # tags + punct
        text = text[:m.start()]

    # Handle wrapping dashes: -text-
    if (len(text) > 2 and text[0] == '-' and text[-1] == '-'
            and text[1] not in '- ' and text[-2] not in '- '):
        prefix += '-'
        suffix = '-' + suffix
        text = text[1:-1]

    return prefix, text, suffix


def validate_tags(original: str, translated: str) -> tuple[bool, str]:
    """Check that all tags in original are preserved in translated and no extras added."""
    orig_tags = extract_tags(original)
    trans_tags = extract_tags(translated)

    # Check missing tags (in original but not in translated)
    remaining = list(trans_tags)
    missing = []
    for tag in orig_tags:
        if tag in remaining:
            remaining.remove(tag)
        else:
            missing.append(tag)

    # Check extra tags (in translated but not in original)
    remaining2 = list(orig_tags)
    extra = []
    for tag in trans_tags:
        if tag in remaining2:
            remaining2.remove(tag)
        else:
            extra.append(tag)

    problems = []
    if missing:
        problems.append(f"Missing tags: {missing}")
    if extra:
        problems.append(f"Extra tags: {extra}")

    if problems:
        return False, "; ".join(problems)
    return True, ""


def escape_for_rpy(text: str) -> str:
    """Escape text for safe insertion into a Ren'Py double-quoted string."""
    # Convert actual newlines to \n escape sequences
    text = text.replace('\n', '\\n')
    # Escape unescaped double quotes
    text = re.sub(r'(?<!\\)"', '\\"', text)
    return text


def strip_outer_quotes(text: str) -> str:
    """Remove at most one layer of surrounding quotes from LLM output."""
    text = text.strip()
    if len(text) >= 2:
        if (text[0] == '"' and text[-1] == '"') or (text[0] == "'" and text[-1] == "'"):
            text = text[1:-1]
    return text


def clean_translation(text: str) -> str:
    """Clean up raw LLM output into a usable translation."""
    text = strip_outer_quotes(text)
    # LLM often adds line breaks; Ren'Py entries are single lines
    text = text.replace('\n', ' ').strip()
    # Normalize full-width angle brackets that LLMs sometimes produce
    text = text.replace('＜', '<').replace('＞', '>')
    return text


# ---------------------------------------------------------------------------
# Ollama API
# ---------------------------------------------------------------------------


class OllamaError(Exception):
    """Error communicating with Ollama API."""
    pass


class TranslationStopped(Exception):
    """Raised when translation is interrupted but partial work may have been saved."""
    def __init__(self, translated_count: int):
        self.translated_count = translated_count


SYSTEM_PROMPT = """\
You are a professional English to Traditional Chinese (繁體中文) translator specializing in dialogue from a visual novel game.
Use a conversational, natural tone appropriate for game dialogue.

Rules:
1. Translate the English text into natural, fluent Traditional Chinese.
2. Keep any <#N> placeholders (e.g. <#1>, <#2>) exactly where they are. Do NOT translate, remove, or reorder them.
3. Do NOT add line breaks.
4. Output ONLY the translated text, nothing else. No quotes, no explanations.
5. You may receive preceding dialogue lines for context — use them to maintain consistency but do NOT translate them."""

BATCH_SYSTEM_PROMPT = """\
You are a professional English to Traditional Chinese (繁體中文) translator specializing in dialogue from a visual novel game.
Use a conversational, natural tone appropriate for game dialogue.

Rules:
1. Translate each English line into natural, fluent Traditional Chinese.
2. Keep any <#N> placeholders (e.g. <#1>, <#2>) exactly where they are. Do NOT translate, remove, or reorder them.
3. Do NOT add line breaks.
4. You will receive lines numbered like 1|text. Output translations in the same numbered format.
5. Output ONLY the numbered translations. No extra commentary.
6. You may receive preceding dialogue lines for context — use them to maintain consistency but do NOT translate them."""


def call_ollama(prompt: str, model: str, system: str = SYSTEM_PROMPT,
                 base_url: str = "http://localhost:11434",
                 num_ctx: int = 8192) -> str:
    """Call Ollama generate API and return the response text."""
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 4096,
            "num_ctx": num_ctx,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("response", "").strip()
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        raise OllamaError(f"Error connecting to Ollama: {e}") from e
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise OllamaError(f"Invalid response from Ollama: {e}") from e


def translate_batch(entries: list[dict], model: str,
                    base_url: str = "http://localhost:11434",
                    context: list[tuple[str, str]] | None = None,
                    num_ctx: int = 8192) -> tuple[list[str], list[tuple[str, str]]]:
    """Translate a batch of entries using numbered format.

    Tags are masked before sending to the LLM and restored afterwards.
    context: list of (masked_input, cleaned_llm_output) pairs from preceding dialogue.
    Returns (translations, ctx_pairs):
      - translations: assembled translated strings in the same order as entries
      - ctx_pairs: (masked_input, cleaned_llm_output) for each entry, for use as future context
    """
    # Strip boundary tags, then mask remaining inner tags
    masked_texts: list[str] = []
    tag_maps: list[list[str]] = []
    boundary_maps: list[tuple[str, str]] = []
    ws_maps: list[tuple[str, str]] = []  # (leading_ws, trailing_ws)
    for entry in entries:
        prefix, core, suffix = strip_boundary_tags(entry["original"])
        core = preprocess_for_translation(core)
        masked, tags = mask_tags(core)
        # Preserve leading/trailing whitespace from core
        lstripped = masked.lstrip()
        leading_ws = masked[:len(masked) - len(lstripped)]
        rstripped = masked.rstrip()
        trailing_ws = masked[len(rstripped):]
        masked_texts.append(masked.strip())
        tag_maps.append(tags)
        boundary_maps.append((prefix, suffix))
        ws_maps.append((leading_ws, trailing_ws))

    # Build context block from preceding translations
    context_block = ""
    if context:
        ctx_lines = []
        for orig, trans in context:
            ctx_lines.append(f"- \"{orig}\" → \"{trans}\"")
        context_block = "Preceding dialogue (for reference, do NOT translate):\n" + "\n".join(ctx_lines) + "\n\n"

    if len(entries) == 1:
        # Single entry: use simple prompt
        prompt = context_block + f"/no_think\nTranslate to Traditional Chinese:\n{masked_texts[0]}"
        result = call_ollama(prompt, model, base_url=base_url, num_ctx=num_ctx)
        cleaned = clean_translation(result)
        prefix, suffix = boundary_maps[0]
        leading_ws, trailing_ws = ws_maps[0]
        translated = prefix + leading_ws + unmask_tags(cleaned, tag_maps[0]) + trailing_ws + suffix
        return [translated], [(masked_texts[0], cleaned)]

    # Build numbered prompt
    lines = []
    for i, masked in enumerate(masked_texts, 1):
        lines.append(f"{i}|{masked}")
    prompt = context_block + "/no_think\nTranslate each line to Traditional Chinese:\n" + "\n".join(lines)

    result = call_ollama(prompt, model, system=BATCH_SYSTEM_PROMPT, base_url=base_url, num_ctx=num_ctx)

    # Parse numbered results
    translations = {}
    for line in result.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Match patterns like "1|translation" or "1. translation" or "1: translation"
        m = re.match(r'^(\d+)\s*[|.:)\-]\s*(.+)$', line)
        if m:
            idx = int(m.group(1))
            text = clean_translation(m.group(2))
            translations[idx] = text

    # Build result list, falling back to single translation if parsing fails
    results = []
    ctx_pairs = []
    for i, entry in enumerate(entries, 1):
        prefix, suffix = boundary_maps[i - 1]
        leading_ws, trailing_ws = ws_maps[i - 1]
        if i in translations:
            translated = prefix + leading_ws + unmask_tags(translations[i], tag_maps[i - 1]) + trailing_ws + suffix
            results.append(translated)
            ctx_pairs.append((masked_texts[i - 1], translations[i]))
        else:
            # Fallback: translate individually (masked)
            prompt = context_block + f"Translate to Traditional Chinese:\n{masked_texts[i - 1]}"
            text = call_ollama(prompt, model, base_url=base_url, num_ctx=num_ctx)
            cleaned = clean_translation(text)
            translated = prefix + leading_ws + unmask_tags(cleaned, tag_maps[i - 1]) + trailing_ws + suffix
            results.append(translated)
            ctx_pairs.append((masked_texts[i - 1], cleaned))
    return results, ctx_pairs


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------

def _apply_and_write(
    filepath: Path,
    content: str,
    replacements: list[tuple[dict, str]],
    has_bom: bool,
    has_crlf: bool = False,
) -> None:
    """Apply collected replacements to content and write to file."""
    replacements.sort(key=lambda r: r[0]["match"].start(), reverse=True)
    for entry, translated in replacements:
        m = entry["match"]
        old_text = m.group(0)
        translated_escaped = escape_for_rpy(translated)

        old_translation = entry.get("old_translation", "")
        old_quoted = f'"{old_translation}"'
        new_quoted = f'"{translated_escaped}"'

        lines = old_text.split("\n")
        lines[1] = lines[1].replace(old_quoted, new_quoted, 1)
        new_text = "\n".join(lines)

        content = content[: m.start()] + new_text + content[m.end() :]

    # Restore original line endings
    if has_crlf:
        content = content.replace("\n", "\r\n")

    encoding = "utf-8-sig" if has_bom else "utf-8"
    tmp_path = filepath.with_suffix(filepath.suffix + ".tmp")
    tmp_path.write_bytes(content.encode(encoding))
    tmp_path.replace(filepath)


def process_file(
    filepath: Path,
    model: str,
    dry_run: bool = False,
    batch_size: int = 5,
    base_url: str = "http://localhost:11434",
    force: bool = False,
    context_lines: int = 5,
    num_ctx: int = 8192,
) -> int:
    """Process a single .rpy translation file.

    When force=True, re-translate entries that already have translations.
    Returns the number of entries translated.
    """
    raw = filepath.read_bytes()
    has_bom = raw.startswith(b'\xef\xbb\xbf')
    has_crlf = b'\r\n' in raw
    content = raw.decode("utf-8-sig")
    # Normalize line endings for consistent regex matching
    if has_crlf:
        content = content.replace('\r\n', '\n')
    entries = find_entries(content, force=force)

    # Handle whitespace-only entries (including tag + whitespace): copy original directly
    whitespace_entries = [e for e in entries if _is_whitespace_content(e["original"])]
    entries = [e for e in entries if not _is_whitespace_content(e["original"])]
    whitespace_replacements = [(e, e["original"]) for e in whitespace_entries]

    if not entries and not whitespace_entries:
        msg = "No entries found" if force else "No untranslated entries found"
        print(f"  {msg} in {filepath.name}")
        return 0

    total = len(entries)
    label = "entries (force)" if force else "untranslated entries"
    if total:
        print(f"  Found {total} {label} in {filepath.name}")

    # Create dated backup before modifying
    if not dry_run:
        backup_dir = filepath.parent / "bak"
        backup_dir.mkdir(exist_ok=True)
        date_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = filepath.stem  # e.g. "sleep"
        suffix = filepath.suffix  # e.g. ".rpy"
        backup_path = backup_dir / f"{stem}_{date_tag}{suffix}.bak"
        if not backup_path.exists():
            shutil.copy2(filepath, backup_path)

    replacements = list(whitespace_replacements)
    translated_count = len(whitespace_replacements)
    if whitespace_replacements:
        print(f"  Copied {len(whitespace_replacements)} whitespace-only entries as-is")

    recent_translations: list[tuple[str, str]] = []  # (masked_input, cleaned_llm_output) for context
    i = 0
    try:
        while i < total:
            batch = entries[i : i + batch_size]
            batch_end = i + len(batch)

            print(f"  Translating {i + 1}-{batch_end}/{total}...", end="", flush=True)
            start_time = time.time()

            ctx = recent_translations[-context_lines:] if context_lines > 0 else None
            translations, ctx_pairs = translate_batch(batch, model, base_url=base_url, context=ctx, num_ctx=num_ctx)
            elapsed = time.time() - start_time

            for j, (entry, translated) in enumerate(zip(batch, translations)):
                idx = i + j + 1
                original = entry["original"]
                ctx_pair = ctx_pairs[j]

                # Fallback to original on empty translation
                if not translated.strip():
                    print(f"\n  WARNING [{idx}/{total}]: Empty translation, using original.")
                    translated = original
                    ctx_pair = None

                # Validate tags (safety net after unmask)
                valid, msg = validate_tags(original, translated)
                if not valid:
                    print(f"\n  WARNING [{idx}/{total}]: Tag mismatch! {msg}")
                    print(f"    Original:   {original}")
                    print(f"    Translated: {translated}")
                    # Retry once individually with masked text
                    prefix, core, suffix = strip_boundary_tags(original)
                    core = preprocess_for_translation(core)
                    masked, tags = mask_tags(core)
                    core_leading = masked[:len(masked) - len(masked.lstrip())]
                    core_trailing = masked[len(masked.rstrip()):]
                    retry_prompt = f"Translate to Traditional Chinese:\n{masked.strip()}"
                    retry_raw = call_ollama(retry_prompt, model, base_url=base_url, num_ctx=num_ctx)
                    retry_cleaned = clean_translation(retry_raw)
                    translated = prefix + core_leading + unmask_tags(retry_cleaned, tags) + core_trailing + suffix
                    valid2, msg2 = validate_tags(original, translated)
                    if not valid2:
                        print(f"  RETRY FAILED: Still missing tags: {msg2}, using original.")
                        translated = original
                        ctx_pair = None
                    else:
                        print(f"  RETRY OK: {translated}")
                        ctx_pair = (masked.strip(), retry_cleaned)

                replacements.append((entry, translated))
                translated_count += 1
                if ctx_pair is not None:
                    recent_translations.append(ctx_pair)

                if dry_run:
                    print(f"\n    [{idx}/{total}] {original}")
                    print(f"             → {translated}")

            if not dry_run:
                print(f" done ({elapsed:.1f}s)")

            i = batch_end

    except KeyboardInterrupt:
        print("\n  Interrupted.", file=sys.stderr)
        if replacements and not dry_run:
            print(f"  Saving {len(replacements)} completed translations...")
            _apply_and_write(filepath, content, replacements, has_bom, has_crlf)
            print(f"  Partial result saved to {filepath.name}")
        raise TranslationStopped(translated_count)
    except OllamaError as e:
        print(f"\n  {e}", file=sys.stderr)
        print("  Make sure Ollama is running: ollama serve", file=sys.stderr)
        if replacements and not dry_run:
            print(f"  Saving {len(replacements)} completed translations before exiting...")
            _apply_and_write(filepath, content, replacements, has_bom, has_crlf)
            print(f"  Partial result saved to {filepath.name}")
        raise TranslationStopped(translated_count) from e

    if dry_run:
        print(f"\n  Dry run complete. {translated_count} entries would be translated.")
        return translated_count

    if not replacements:
        return 0

    _apply_and_write(filepath, content, replacements, has_bom, has_crlf)
    print(f"  Wrote {translated_count} translations to {filepath.name}")

    return translated_count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    config = load_config()

    parser = argparse.ArgumentParser(
        description="Translate Ren'Py .rpy files from English to Traditional Chinese using Ollama."
    )
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="One or more paths to .rpy files or directories containing .rpy files.",
    )
    parser.add_argument(
        "--host",
        default=config["ollama_host"],
        help=f"Ollama API base URL (default from config: {config['ollama_host']}).",
    )
    parser.add_argument(
        "--model",
        default=config["model"],
        help=f"Ollama model to use (default from config: {config['model']}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview translations without writing to files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config["batch_size"],
        choices=range(1, 21),
        metavar="N",
        help=f"Number of entries to translate per API call (1-20, default from config: {config['batch_size']}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-translate entries that already have translations.",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recursively search directories for .rpy files.",
    )
    args = parser.parse_args()

    # Strip trailing slash from host URL
    base_url = args.host.rstrip("/")

    files: list[Path] = []
    for path in args.paths:
        if path.is_file():
            if path.suffix != ".rpy":
                print(f"Error: {path} is not a .rpy file.", file=sys.stderr)
                sys.exit(1)
            files.append(path)
        elif path.is_dir():
            glob_func = path.rglob if args.recursive else path.glob
            found = sorted(glob_func("*.rpy"), key=lambda f: f.stat().st_size)
            if not found:
                print(f"Warning: No .rpy files found in {path}", file=sys.stderr)
            files.extend(found)
        else:
            print(f"Error: {path} does not exist.", file=sys.stderr)
            sys.exit(1)

    if not files:
        print("Error: No .rpy files to process.", file=sys.stderr)
        sys.exit(1)

    total_translated = 0
    start_time = time.time()

    try:
        for filepath in files:
            print(f"\nProcessing {filepath}...")
            file_start = time.time()
            count = process_file(
                filepath, args.model,
                dry_run=args.dry_run, batch_size=args.batch_size,
                base_url=base_url, force=args.force,
                context_lines=config["context_lines"],
                num_ctx=config["num_ctx"],
            )
            file_elapsed = time.time() - file_start
            if count > 0:
                print(f"  {filepath.name}: {count} entries in {file_elapsed:.1f}s")
            total_translated += count
    except TranslationStopped as e:
        total_translated += e.translated_count
    except KeyboardInterrupt:
        print("\n  Interrupted.", file=sys.stderr)

    elapsed = time.time() - start_time
    print(f"\nDone. Translated {total_translated} entries in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
