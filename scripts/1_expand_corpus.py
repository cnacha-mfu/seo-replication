"""
Corpus Expander for JCSSE 2026 Paper.

Searches GitHub for repositories containing CLAUDE.md, AGENTS.md, or other
agent manifest files and collects their content, expanding the corpus from
23 to ~50–80 repos.

Manifest types collected (in priority order):
  1. CLAUDE.md           — Claude Code instructions
  2. AGENTS.md           — OpenAI Codex / multi-agent instructions
  3. .github/copilot-instructions.md  — GitHub Copilot instructions
  4. .cursorrules        — Cursor AI instructions
  5. .clinerules         — Cline instructions
  6. .windsurfrules      — Windsurf instructions

Output format matches phase1_repos.json so content_classifier.py runs unchanged.

Paper: "What Do Human Agent Manifests Miss? An Empirical Content Analysis of CLAUDE.md Files"
Venue: JCSSE 2026

Usage:
    # Dry run (estimate candidates without downloading)
    python experiments/jcsse2026/expand_corpus.py --token $GITHUB_TOKEN --dry-run

    # Full expansion: collect up to 80 repos total (existing 23 + new)
    python experiments/jcsse2026/expand_corpus.py --token $GITHUB_TOKEN --target 80

    # Append to existing corpus
    python experiments/jcsse2026/expand_corpus.py --token $GITHUB_TOKEN --target 80 \\
        --existing experiments/paper1_ontology_manifests/results/real/phase1_repos.json \\
        --output experiments/jcsse2026/results/expanded_corpus.json
"""

import argparse
import base64
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Manifest file definitions ─────────────────────────────────────────────────

# (search_filename, manifest_type, path_in_repo)
MANIFEST_TARGETS = [
    ("CLAUDE.md",                          "claude_code",  "CLAUDE.md"),
    ("AGENTS.md",                          "codex",        "AGENTS.md"),
    ("copilot-instructions.md",            "copilot",      ".github/copilot-instructions.md"),
    (".cursorrules",                        "cursor",       ".cursorrules"),
    (".clinerules",                         "cline",        ".clinerules"),
    (".windsurfrules",                      "windsurf",     ".windsurfrules"),
]

# Search queries to run — diversified by language to avoid result caps
SEARCH_QUERIES = [
    # CLAUDE.md — primary target
    ('filename:CLAUDE.md size:>300', 'CLAUDE.md'),
    ('filename:CLAUDE.md language:Python size:>300', 'CLAUDE.md'),
    ('filename:CLAUDE.md language:TypeScript size:>300', 'CLAUDE.md'),
    ('filename:CLAUDE.md language:JavaScript size:>300', 'CLAUDE.md'),
    ('filename:CLAUDE.md language:Go size:>300', 'CLAUDE.md'),
    ('filename:CLAUDE.md language:Java size:>300', 'CLAUDE.md'),
    ('filename:CLAUDE.md language:Rust size:>300', 'CLAUDE.md'),
    # AGENTS.md — secondary target
    ('filename:AGENTS.md size:>300', 'AGENTS.md'),
    ('filename:AGENTS.md language:Python size:>300', 'AGENTS.md'),
    ('filename:AGENTS.md language:TypeScript size:>300', 'AGENTS.md'),
    # Other manifests
    ('filename:copilot-instructions.md path:.github size:>300', 'copilot-instructions.md'),
    ('filename:.cursorrules size:>300', '.cursorrules'),
    ('filename:.clinerules size:>300', '.clinerules'),
    ('filename:.windsurfrules size:>300', '.windsurfrules'),
]

# Inclusion thresholds
MIN_STARS = 30
MIN_HISTORY_MONTHS = 6
MIN_MANIFEST_CHARS = 200
MAX_MANIFEST_CHARS = 200_000  # skip auto-generated mega-files


# ── GitHub client ─────────────────────────────────────────────────────────────

class GitHubClient:
    """GitHub REST API client with rate limiting."""

    BASE = "https://api.github.com"

    def __init__(self, token: str):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        self.calls = 0
        self._last_search = 0.0

    def _wait_if_needed(self, is_search: bool = False) -> None:
        """Enforce rate limits: search API ≤ 10 req/min, core ≤ 5000/hr."""
        if is_search:
            elapsed = time.time() - self._last_search
            if elapsed < 6.5:  # ~9 req/min to stay safe
                time.sleep(6.5 - elapsed)
            self._last_search = time.time()

    def get(self, endpoint: str, params: dict = None, is_search: bool = False):
        self._wait_if_needed(is_search)
        url = endpoint if endpoint.startswith("http") else f"{self.BASE}/{endpoint}"
        self.calls += 1
        for attempt in range(3):
            try:
                r = self.session.get(url, params=params, timeout=30)
                remaining = int(r.headers.get("X-RateLimit-Remaining", 999))
                if remaining < 10:
                    reset = int(r.headers.get("X-RateLimit-Reset", 0))
                    wait = max(reset - time.time(), 0) + 5
                    logger.warning("Rate limit low (%d). Sleeping %.0fs", remaining, wait)
                    time.sleep(wait)
                if r.status_code == 200:
                    return r.json()
                elif r.status_code in (403, 429):
                    wait = 60 * (attempt + 1)
                    logger.warning("Status %d — sleeping %ds", r.status_code, wait)
                    time.sleep(wait)
                elif r.status_code == 422:
                    # Validation error — search query issue
                    logger.debug("Search 422: %s", r.json().get("message", ""))
                    return None
                else:
                    logger.debug("Status %d for %s", r.status_code, url)
                    return None
            except requests.RequestException as exc:
                logger.warning("Request error (attempt %d): %s", attempt + 1, exc)
                time.sleep(5)
        return None

    def search_code(self, query: str, per_page: int = 100, max_pages: int = 3) -> list:
        """Search code API. Returns list of result items."""
        items = []
        for page in range(1, max_pages + 1):
            data = self.get(
                "search/code",
                {"q": query, "per_page": per_page, "page": page},
                is_search=True,
            )
            if not data or "items" not in data:
                break
            items.extend(data["items"])
            total = data.get("total_count", 0)
            logger.info(
                "  search_code page %d: %d items (total ~%d)",
                page, len(data["items"]), total,
            )
            if len(data["items"]) < per_page:
                break
        return items

    def get_repo(self, full_name: str) -> dict | None:
        return self.get(f"repos/{full_name}")

    def get_file_content(self, full_name: str, path: str) -> str | None:
        """Fetch decoded text content of a file. Returns None if not found."""
        data = self.get(f"repos/{full_name}/contents/{path}")
        if not data or not isinstance(data, dict):
            return None
        encoding = data.get("encoding", "")
        content = data.get("content", "")
        if encoding == "base64":
            try:
                return base64.b64decode(content).decode("utf-8", errors="replace")
            except Exception:
                return None
        return content or None


# ── Filtering ─────────────────────────────────────────────────────────────────

def is_eligible(repo_data: dict, existing_names: set[str]) -> tuple[bool, str]:
    """Return (eligible, reason) for a candidate repo."""
    name = repo_data.get("full_name", "")

    if name in existing_names:
        return False, "already in corpus"
    if repo_data.get("fork", False):
        return False, "fork"
    if repo_data.get("archived", False):
        return False, "archived"
    if repo_data.get("private", False):
        return False, "private"
    if (repo_data.get("stargazers_count", 0) or 0) < MIN_STARS:
        return False, f"stars < {MIN_STARS}"

    # Check age
    created_str = repo_data.get("created_at", "")
    if created_str:
        try:
            created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            age_months = (
                (datetime.now(timezone.utc) - created).days / 30.44
            )
            if age_months < MIN_HISTORY_MONTHS:
                return False, f"age < {MIN_HISTORY_MONTHS} months"
        except ValueError:
            pass

    return True, "ok"


# ── Collection ────────────────────────────────────────────────────────────────

def collect_manifest(
    client: GitHubClient,
    full_name: str,
    manifest_path: str,
    manifest_type: str,
    repo_data: dict,
) -> dict | None:
    """Fetch manifest content and build a corpus entry."""
    content = client.get_file_content(full_name, manifest_path)
    if not content:
        return None

    content = content.strip()
    n_chars = len(content)
    if n_chars < MIN_MANIFEST_CHARS:
        logger.debug("  %s: manifest too short (%d chars)", full_name, n_chars)
        return None
    if n_chars > MAX_MANIFEST_CHARS:
        logger.debug("  %s: manifest too large (%d chars) — skipping", full_name, n_chars)
        return None

    return {
        "full_name": full_name,
        "language": (repo_data.get("language") or "unknown").lower(),
        "stars": repo_data.get("stargazers_count", 0),
        "created_at": repo_data.get("created_at", ""),
        "default_branch": repo_data.get("default_branch", "main"),
        "description": repo_data.get("description", "") or "",
        "manifest_type": manifest_type,       # new field vs phase1_repos.json
        "manifest_path": manifest_path,        # new field
        "claude_md_content": content,          # keep same key for classifier compatibility
        "claude_md_length": n_chars,
    }


# ── Main expander ─────────────────────────────────────────────────────────────

class CorpusExpander:
    """Expand the CLAUDE.md corpus by searching GitHub."""

    def __init__(self, token: str, existing_path: str | None = None):
        self.client = GitHubClient(token)
        self.existing: list[dict] = []
        self.existing_names: set[str] = set()

        if existing_path and Path(existing_path).exists():
            with open(existing_path) as f:
                self.existing = json.load(f)
            self.existing_names = {r["full_name"] for r in self.existing}
            logger.info(
                "Loaded %d existing repos from %s", len(self.existing), existing_path
            )

    def run(
        self,
        target: int = 80,
        output_path: str = "experiments/jcsse2026/results/expanded_corpus.json",
        dry_run: bool = False,
    ) -> list[dict]:
        """Collect repos until target total (existing + new) is reached."""
        new_entries: list[dict] = []
        seen_repos: set[str] = set(self.existing_names)

        need = target - len(self.existing)
        logger.info(
            "Target: %d total repos. Have %d. Need %d more.",
            target, len(self.existing), need,
        )
        if need <= 0:
            logger.info("Already at target — nothing to do.")
            return self.existing

        for query, filename in SEARCH_QUERIES:
            if len(new_entries) >= need:
                break

            logger.info("Query: %s", query)
            results = self.client.search_code(query, per_page=100, max_pages=2)
            logger.info("  %d raw results", len(results))

            for item in results:
                if len(new_entries) >= need:
                    break

                repo_info = item.get("repository", {})
                full_name = repo_info.get("full_name", "")
                if not full_name or full_name in seen_repos:
                    continue

                seen_repos.add(full_name)

                # Fetch full repo metadata
                repo_data = self.client.get_repo(full_name)
                if not repo_data:
                    continue

                eligible, reason = is_eligible(repo_data, self.existing_names)
                if not eligible:
                    logger.debug("  SKIP %s: %s", full_name, reason)
                    continue

                if dry_run:
                    logger.info(
                        "  CANDIDATE %s (%d★, %s)",
                        full_name,
                        repo_data.get("stargazers_count", 0),
                        repo_data.get("language", "?"),
                    )
                    new_entries.append({"full_name": full_name, "dry_run": True})
                    continue

                # Find which manifest file to collect (prefer in priority order)
                entry = None
                for _, mtype, mpath in MANIFEST_TARGETS:
                    # Check if this file is the one we found, or try each in order
                    entry = collect_manifest(
                        self.client, full_name, mpath, mtype, repo_data
                    )
                    if entry:
                        break

                if not entry:
                    logger.debug("  SKIP %s: no manifest content", full_name)
                    continue

                logger.info(
                    "  ADD %s (%d★, %s, %s, %d chars)",
                    full_name,
                    entry["stars"],
                    entry["language"],
                    entry["manifest_type"],
                    entry["claude_md_length"],
                )
                new_entries.append(entry)

            logger.info(
                "After query: %d new entries collected (need %d)",
                len(new_entries), need,
            )

        if dry_run:
            logger.info(
                "\nDRY RUN: found %d candidates (would add to reach target %d)",
                len(new_entries), target,
            )
            return new_entries

        # Combine existing + new
        combined = self.existing + new_entries
        logger.info(
            "\nCorpus: %d existing + %d new = %d total",
            len(self.existing), len(new_entries), len(combined),
        )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        logger.info("Saved → %s", out)

        # Print summary table
        self._print_summary(new_entries)

        return combined

    def _print_summary(self, new_entries: list[dict]) -> None:
        if not new_entries:
            return
        print(f"\n{'Repository':<50} {'Type':<20} {'Stars':>6} {'Lang':<12} {'Chars':>7}")
        print("-" * 100)
        for e in sorted(new_entries, key=lambda x: -x.get("stars", 0)):
            print(
                f"{e['full_name']:<50} {e.get('manifest_type','?'):<20} "
                f"{e.get('stars',0):>6} {e.get('language','?'):<12} "
                f"{e.get('claude_md_length',0):>7}"
            )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand CLAUDE.md corpus for JCSSE 2026"
    )
    parser.add_argument(
        "--token",
        default=None,
        help="GitHub personal access token (or set GITHUB_TOKEN env var)",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=80,
        help="Target total corpus size (existing + new). Default: 80",
    )
    parser.add_argument(
        "--existing",
        default="experiments/paper1_ontology_manifests/results/real/phase1_repos.json",
        help="Path to existing phase1_repos.json",
    )
    parser.add_argument(
        "--output",
        default="experiments/jcsse2026/results/expanded_corpus.json",
        help="Output path for expanded corpus",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List candidates without downloading content",
    )
    args = parser.parse_args()

    import os
    token = args.token or os.environ.get("GITHUB_TOKEN")
    if not token:
        parser.error("Provide --token or set GITHUB_TOKEN environment variable")

    expander = CorpusExpander(token=token, existing_path=args.existing)
    expander.run(
        target=args.target,
        output_path=args.output,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
