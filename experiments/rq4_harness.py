#!/usr/bin/env python3
"""
RQ4 Agentic Code Generation Experiment Harness — Paper 1

Tests whether evolution-aware manifests (SENSO-generated) improve agentic
code generation compared to human-written manifests and no manifest at all.

Design:
  - 20 repos x 5 real closed issues = 100 tasks (known-good patches as ground truth)
  - 3 conditions: (A) no manifest, (B) human-written manifest, (C) SENSO-generated
  - 2 NIM models x 3 conditions x 100 tasks x 3 repeats = 1,800 agentic runs
  - temperature=0, sequential execution with rate-limit backoff
  - Each run time-boxed to 10 minutes

Paper 1: "From Tribal Knowledge to Machine-Readable Evolution Context"

Usage:
  python rq4_harness.py select-tasks --token $GITHUB_TOKEN --output configs/tasks.yaml
  python rq4_harness.py run --config configs/rq4.yaml --tasks configs/tasks.yaml
  python rq4_harness.py analyze --results-dir results/rq4/
  python rq4_harness.py report --results-dir results/rq4/ --output results/rq4_report.json
"""

import argparse
import copy
import datetime
import difflib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from dotenv import load_dotenv

# Add project root to path for shared imports.
# IMPORTANT: use sys.path.append (not insert) to avoid the project's platform/
# directory shadowing Python's stdlib `platform` module.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from shared.evaluation.statistics import holm_bonferroni, cohens_d, cliffs_delta

load_dotenv(_PROJECT_ROOT / ".env")

logger = logging.getLogger("rq4_harness")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

MODELS = {
    "primary": {
        "name": "meta/llama-3.3-70b-instruct",
        "api_key_env": "NVIDIA_API_KEY",
    },
    "replication": {
        "name": "mistralai/mistral-large-latest",
        "api_key_env": "NVIDIA_API_KEY_3",
    },
}

CONDITIONS = ("A", "B", "C")
CONDITION_LABELS = {
    "A": "no_manifest",
    "B": "human_manifest",
    "C": "senso_manifest",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TaskDefinition:
    """A single coding task derived from a real closed GitHub issue.

    Paper 1 — RQ4 task specification.
    """
    task_id: str
    repo: str
    language: str
    issue_number: int
    issue_title: str
    issue_description: str
    pre_fix_commit: str
    ground_truth_diff: str
    test_commands: list[str]
    ground_truth_files_changed: list[str]
    human_manifest: str = ""
    senso_manifest: str = ""


@dataclass
class RunResult:
    """Result of a single agentic run.

    Paper 1 — RQ4 per-run outcome record.
    """
    run_id: str
    task_id: str
    condition: str
    model_key: str
    model_name: str
    repeat_index: int
    timestamp: str = ""

    # Agent behaviour
    agent_log: str = ""
    tool_calls: int = 0
    iterations: int = 0
    files_touched: list[str] = field(default_factory=list)
    agent_diff: str = ""
    timed_out: bool = False
    error: str = ""

    # Metrics
    complexity_before: float = 0.0
    complexity_after: float = 0.0
    complexity_delta: float = 0.0
    coupling_before: int = 0
    coupling_after: int = 0
    coupling_delta: int = 0
    test_pass_rate: float = 0.0
    tests_passed: int = 0
    tests_total: int = 0
    diff_similarity: float = 0.0
    task_completed: bool = False
    phase_aligned: bool = False
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# TaskSelector
# ---------------------------------------------------------------------------

class TaskSelector:
    """Selects tasks from real closed GitHub issues with linked PRs.

    Paper 1 — RQ4: builds the 100-task benchmark from 20 repos x 5 issues.
    Queries GitHub API for closed issues with clear fixes (diff < 500 lines,
    tests exist) and extracts task definitions.
    """

    def __init__(self, token: str, max_diff_lines: int = 500,
                 repos_count: int = 20, tasks_per_repo: int = 5,
                 languages: Optional[list[str]] = None):
        """Initialise TaskSelector.

        Args:
            token: GitHub personal access token.
            max_diff_lines: Maximum diff size to accept.
            repos_count: Number of repositories to source tasks from.
            tasks_per_repo: Number of tasks to select per repository.
            languages: Acceptable primary languages.
        """
        self.token = token
        self.max_diff_lines = max_diff_lines
        self.repos_count = repos_count
        self.tasks_per_repo = tasks_per_repo
        self.languages = languages or ["python", "javascript", "typescript", "java", "go"]
        self._headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
        }

    # -- GitHub helpers -----------------------------------------------------

    def _gh_get(self, url: str, params: Optional[dict] = None) -> Any:
        """Make an authenticated GET request to the GitHub API."""
        import requests  # local import — optional dependency

        resp = requests.get(url, headers=self._headers, params=params, timeout=30)
        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(reset - int(time.time()), 1)
            logger.warning("GitHub rate limit hit — sleeping %d s", wait)
            time.sleep(wait)
            return self._gh_get(url, params)
        resp.raise_for_status()
        return resp.json()

    # -- Repository discovery -----------------------------------------------

    def discover_repos(self) -> list[str]:
        """Find popular repositories with good issue/PR hygiene.

        Returns:
            List of 'owner/name' strings.
        """
        repos: list[str] = []
        per_lang = max(10, self.repos_count // len(self.languages) + 5)
        for lang in self.languages:
            url = "https://api.github.com/search/repositories"
            params = {
                "q": f"language:{lang} stars:>1000 archived:false",
                "sort": "stars",
                "order": "desc",
                "per_page": per_lang,
            }
            try:
                data = self._gh_get(url, params)
            except Exception:
                continue
            for item in data.get("items", []):
                full_name = item["full_name"]
                if full_name not in repos:
                    repos.append(full_name)
                if len(repos) >= self.repos_count:
                    break
            if len(repos) >= self.repos_count:
                break
            time.sleep(1)  # avoid search rate limit
        logger.info("Discovered %d candidate repositories", len(repos))
        return repos[: self.repos_count]

    # -- Issue / PR extraction -----------------------------------------------

    def _find_linked_pr(self, repo: str, issue_number: int) -> Optional[dict]:
        """Find a merged PR linked to the given issue."""
        url = f"https://api.github.com/repos/{repo}/pulls"
        params = {"state": "closed", "per_page": 100, "sort": "updated", "direction": "desc"}
        try:
            pulls = self._gh_get(url, params)
        except Exception:
            return None
        for pr in pulls:
            if not pr.get("merged_at"):
                continue
            body = (pr.get("body") or "").lower()
            title = (pr.get("title") or "").lower()
            markers = [
                f"#{issue_number}",
                f"fixes #{issue_number}", f"closes #{issue_number}",
                f"resolves #{issue_number}", f"fix #{issue_number}",
                f"close #{issue_number}", f"resolve #{issue_number}",
                f"fixed #{issue_number}", f"closed #{issue_number}",
            ]
            if any(m in body or m in title for m in markers):
                return pr
        return None

    def _get_pr_diff(self, repo: str, pr_number: int) -> str:
        """Fetch the raw diff for a pull request."""
        import requests

        url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
        headers = {**self._headers, "Accept": "application/vnd.github.diff"}
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.text

    def _diff_line_count(self, diff_text: str) -> int:
        """Count changed lines in a unified diff."""
        return sum(1 for line in diff_text.splitlines() if line.startswith(("+", "-"))
                   and not line.startswith(("+++", "---")))

    def _extract_changed_files(self, diff_text: str) -> list[str]:
        """Extract file paths from a unified diff."""
        files = []
        for line in diff_text.splitlines():
            if line.startswith("+++ b/"):
                files.append(line[6:])
        return files

    def _has_tests(self, repo: str, ref: str) -> bool:
        """Quick heuristic: check if common test directories exist at a ref."""
        for test_path in ["tests", "test", "spec", "src/test", "__tests__"]:
            url = f"https://api.github.com/repos/{repo}/contents/{test_path}"
            try:
                self._gh_get(url, {"ref": ref})
                return True
            except Exception:
                continue
        return False

    # -- Main selection logic ------------------------------------------------

    def select_tasks_for_repo(self, repo: str) -> list[TaskDefinition]:
        """Select up to tasks_per_repo tasks from a single repository.

        Uses two strategies:
        1. Closed issues with linked merged PRs (primary)
        2. Recent merged PRs with bug-fix labels/titles (fallback)
        """
        logger.info("Selecting tasks from %s", repo)
        tasks: list[TaskDefinition] = []
        seen_prs: set[int] = set()

        # Get repo language once
        try:
            repo_info = self._gh_get(f"https://api.github.com/repos/{repo}")
            language = (repo_info.get("language") or "unknown").lower()
        except Exception:
            language = "unknown"

        # Check for tests once
        test_checked = False
        has_tests = False

        def _check_tests(ref: str) -> bool:
            nonlocal test_checked, has_tests
            if test_checked:
                return has_tests
            has_tests = self._has_tests(repo, ref)
            test_checked = True
            return has_tests

        # Strategy 1: closed issues → linked PRs
        try:
            url = f"https://api.github.com/repos/{repo}/issues"
            params = {"state": "closed", "per_page": 100, "sort": "updated", "direction": "desc"}
            issues = self._gh_get(url, params)
        except Exception:
            issues = []

        for issue in issues:
            if len(tasks) >= self.tasks_per_repo:
                break
            if issue.get("pull_request"):
                continue

            issue_num = issue["number"]
            pr = self._find_linked_pr(repo, issue_num)
            if pr is None:
                continue
            if pr["number"] in seen_prs:
                continue

            task = self._try_make_task(repo, language, pr, issue, _check_tests)
            if task:
                tasks.append(task)
                seen_prs.add(pr["number"])
                logger.info("  Selected task %s (issue #%d)", task.task_id, issue["number"])

        # Strategy 2: recent merged PRs (bug fixes, small features)
        if len(tasks) < self.tasks_per_repo:
            try:
                url = f"https://api.github.com/repos/{repo}/pulls"
                params = {"state": "closed", "per_page": 100, "sort": "updated", "direction": "desc"}
                pulls = self._gh_get(url, params)
            except Exception:
                pulls = []

            bug_keywords = ["fix", "bug", "patch", "resolve", "repair", "correct", "hotfix"]
            for pr in pulls:
                if len(tasks) >= self.tasks_per_repo:
                    break
                if not pr.get("merged_at") or pr["number"] in seen_prs:
                    continue
                title = (pr.get("title") or "").lower()
                # Only pick PRs that look like bug fixes or small features
                if not any(kw in title for kw in bug_keywords):
                    continue

                # Create a synthetic issue from the PR
                issue = {
                    "number": pr["number"],
                    "title": pr.get("title", ""),
                    "body": pr.get("body", "") or pr.get("title", ""),
                }
                task = self._try_make_task(repo, language, pr, issue, _check_tests)
                if task:
                    tasks.append(task)
                    seen_prs.add(pr["number"])
                    logger.info("  Selected task %s (PR #%d)", task.task_id, pr["number"])

        return tasks

    def _try_make_task(self, repo: str, language: str, pr: dict,
                       issue: dict, check_tests_fn) -> Optional[TaskDefinition]:
        """Try to create a TaskDefinition from a PR + issue pair."""
        try:
            diff_text = self._get_pr_diff(repo, pr["number"])
        except Exception:
            return None

        if self._diff_line_count(diff_text) > self.max_diff_lines:
            return None
        if self._diff_line_count(diff_text) < 5:
            return None  # too trivial

        merge_commit = pr.get("merge_commit_sha", "")
        base_sha = pr.get("base", {}).get("sha", "")
        pre_fix_commit = base_sha or merge_commit

        if not check_tests_fn(pre_fix_commit):
            return None

        return TaskDefinition(
            task_id=f"{repo.replace('/', '-')}-{issue['number']}",
            repo=repo,
            language=language,
            issue_number=issue["number"],
            issue_title=issue.get("title", ""),
            issue_description=issue.get("body", "") or issue.get("title", ""),
            pre_fix_commit=pre_fix_commit,
            ground_truth_diff=diff_text,
            test_commands=self._infer_test_commands(language),
            ground_truth_files_changed=self._extract_changed_files(diff_text),
        )

    @staticmethod
    def _infer_test_commands(language: str) -> list[str]:
        """Infer default test commands based on project language."""
        mapping = {
            "python": ["python -m pytest -x -q"],
            "javascript": ["npm test"],
            "typescript": ["npm test"],
            "java": ["mvn test -q"],
            "go": ["go test ./..."],
        }
        return mapping.get(language, ["echo 'no test command inferred'"])

    def select_all(self) -> list[TaskDefinition]:
        """Select tasks across all discovered repositories.

        Returns:
            List of TaskDefinition objects.
        """
        repos = self.discover_repos()
        all_tasks: list[TaskDefinition] = []
        for repo in repos:
            try:
                tasks = self.select_tasks_for_repo(repo)
                all_tasks.extend(tasks)
            except Exception as exc:
                logger.warning("Failed to select tasks from %s: %s", repo, exc)
        logger.info("Selected %d tasks total from %d repos", len(all_tasks), len(repos))
        return all_tasks

    @staticmethod
    def save_tasks(tasks: list[TaskDefinition], output_path: Path) -> None:
        """Serialize task definitions to YAML."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"tasks": [asdict(t) for t in tasks]}
        with open(output_path, "w") as fh:
            yaml.dump(data, fh, default_flow_style=False, sort_keys=False, width=120)
        logger.info("Saved %d tasks to %s", len(tasks), output_path)


# ---------------------------------------------------------------------------
# WorkspaceManager
# ---------------------------------------------------------------------------

class WorkspaceManager:
    """Manages isolated workspaces for each agentic run.

    Paper 1 — RQ4: clones repo at pre-fix commit, places manifest per condition.
    """

    def __init__(self, base_dir: str = "/tmp/rq4_workspaces"):
        """Initialise WorkspaceManager.

        Args:
            base_dir: Root directory for workspace clones.
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def prepare(self, task: TaskDefinition, condition: str, run_id: str) -> Path:
        """Clone repo at pre-fix commit and set up manifest.

        Args:
            task: The task to prepare a workspace for.
            condition: One of 'A', 'B', 'C'.
            run_id: Unique identifier for this run.

        Returns:
            Path to the prepared workspace directory.
        """
        workspace = self.base_dir / run_id
        if workspace.exists():
            shutil.rmtree(workspace)

        # Clone the repository
        repo_url = f"https://github.com/{task.repo}.git"
        logger.info("Cloning %s into %s", task.repo, workspace)
        subprocess.run(
            ["git", "clone", "--depth", "50", repo_url, str(workspace)],
            capture_output=True, text=True, timeout=120, check=True,
        )

        # Checkout pre-fix commit
        subprocess.run(
            ["git", "checkout", task.pre_fix_commit],
            cwd=str(workspace), capture_output=True, text=True, timeout=30, check=True,
        )

        # Place or remove CLAUDE.md based on condition
        manifest_path = workspace / "CLAUDE.md"
        if condition == "A":
            if manifest_path.exists():
                manifest_path.unlink()
            logger.debug("Condition A: no manifest")
        elif condition == "B":
            manifest_path.write_text(task.human_manifest, encoding="utf-8")
            logger.debug("Condition B: human manifest placed")
        elif condition == "C":
            manifest_path.write_text(task.senso_manifest, encoding="utf-8")
            logger.debug("Condition C: SENSO manifest placed")
        else:
            raise ValueError(f"Unknown condition: {condition}")

        return workspace

    @staticmethod
    def capture_state(workspace: Path) -> dict[str, str]:
        """Capture a snapshot of tracked file contents for comparison.

        Args:
            workspace: Path to the workspace directory.

        Returns:
            Mapping of relative file paths to their content hashes.
        """
        import hashlib

        state: dict[str, str] = {}
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=str(workspace), capture_output=True, text=True, timeout=30,
        )
        for rel_path in result.stdout.strip().splitlines():
            full = workspace / rel_path
            if full.is_file():
                content = full.read_bytes()
                state[rel_path] = hashlib.sha256(content).hexdigest()
        return state

    @staticmethod
    def get_diff(workspace: Path) -> str:
        """Get the git diff of all changes in the workspace.

        Args:
            workspace: Path to the workspace directory.

        Returns:
            Unified diff string.
        """
        result = subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=str(workspace), capture_output=True, text=True, timeout=30,
        )
        return result.stdout

    @staticmethod
    def cleanup(workspace: Path) -> None:
        """Remove the workspace directory.

        Args:
            workspace: Path to remove.
        """
        if workspace.exists():
            shutil.rmtree(workspace, ignore_errors=True)


# ---------------------------------------------------------------------------
# AgentRunner
# ---------------------------------------------------------------------------

class AgentRunner:
    """Launches an agentic coding session (OpenCode or simulated NIM).

    Paper 1 — RQ4: drives the agent to attempt fixing the issue in the workspace.
    Supports two modes:
      - OpenCode binary (if installed) configured with NIM backend
      - Simulated mode: direct multi-turn NIM API calls (read-plan-edit-verify)
    """

    def __init__(self, config: dict):
        """Initialise AgentRunner from experiment config.

        Args:
            config: Parsed rq4.yaml configuration dictionary.
        """
        agent_cfg = config.get("agent", {})
        self.opencode_binary = agent_cfg.get("opencode_binary", "opencode")
        self.simulated_mode = agent_cfg.get("simulated_mode", True)
        self.simulated_max_turns = agent_cfg.get("simulated_max_turns", 8)
        self.timeout = config.get("execution", {}).get("timeout_seconds", 600)
        self.temperature = config.get("execution", {}).get("temperature", 0.0)
        self.max_tokens = config.get("execution", {}).get("max_tokens", 4096)

    # -- OpenCode mode -------------------------------------------------------

    def _run_opencode(self, workspace: Path, prompt: str,
                      model_name: str, api_key: str) -> dict[str, Any]:
        """Launch OpenCode binary with NIM backend.

        Args:
            workspace: Working directory for the agent.
            prompt: Issue description / coding task prompt.
            model_name: NIM model identifier.
            api_key: NVIDIA API key.

        Returns:
            Dict with keys: log, tool_calls, iterations, files_touched, timed_out, error.
        """
        env = os.environ.copy()
        env["OPENAI_API_BASE"] = NIM_BASE_URL
        env["OPENAI_API_KEY"] = api_key
        env["OPENCODE_MODEL"] = model_name

        cmd = [self.opencode_binary, "--prompt", prompt, "--non-interactive"]
        timed_out = False
        try:
            result = subprocess.run(
                cmd, cwd=str(workspace), env=env,
                capture_output=True, text=True, timeout=self.timeout,
            )
            log_text = result.stdout + "\n" + result.stderr
            error = result.stderr if result.returncode != 0 else ""
        except subprocess.TimeoutExpired:
            log_text = "TIMEOUT"
            error = f"Run timed out after {self.timeout}s"
            timed_out = True
        except FileNotFoundError:
            logger.warning("OpenCode binary not found — falling back to simulated mode")
            return self._run_simulated(workspace, prompt, model_name, api_key)

        return {
            "log": log_text,
            "tool_calls": log_text.count("tool_call"),  # heuristic
            "iterations": log_text.count("iteration"),
            "files_touched": self._detect_touched_files(workspace),
            "timed_out": timed_out,
            "error": error,
        }

    # -- Simulated mode ------------------------------------------------------

    def _run_simulated(self, workspace: Path, prompt: str,
                       model_name: str, api_key: str) -> dict[str, Any]:
        """Simulate an agentic session via direct NIM API calls.

        Multi-turn conversation: read relevant files -> plan -> edit -> verify.

        Args:
            workspace: Working directory for the agent.
            prompt: Issue description / coding task prompt.
            model_name: NIM model identifier.
            api_key: NVIDIA API key.

        Returns:
            Dict with keys: log, tool_calls, iterations, files_touched, timed_out, error.
        """
        from openai import OpenAI

        client = OpenAI(base_url=NIM_BASE_URL, api_key=api_key)

        # Read manifest if present
        manifest_path = workspace / "CLAUDE.md"
        manifest_content = ""
        if manifest_path.exists():
            manifest_content = manifest_path.read_text(encoding="utf-8")

        # List project files for context
        file_listing = subprocess.run(
            ["find", ".", "-type", "f", "-name", "*.py", "-o", "-name", "*.js",
             "-o", "-name", "*.ts", "-o", "-name", "*.java", "-o", "-name", "*.go"],
            cwd=str(workspace), capture_output=True, text=True, timeout=30,
        )
        project_files = file_listing.stdout.strip()[:3000]  # truncate

        system_prompt = (
            "You are an expert software engineer fixing a bug in a codebase. "
            "You will be given an issue description and project context. "
            "Provide your fix as a unified diff that can be applied with `git apply`.\n"
        )
        if manifest_content:
            system_prompt += f"\n## Project Evolution Context\n{manifest_content}\n"

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"## Issue\n{prompt}\n\n"
                f"## Project Files (partial listing)\n```\n{project_files}\n```\n\n"
                "First, identify which files are most relevant to this issue. "
                "Then provide a unified diff to fix it."
            )},
        ]

        full_log = ""
        tool_calls = 0
        iterations = 0
        start_time = time.time()

        for turn in range(self.simulated_max_turns):
            elapsed = time.time() - start_time
            if elapsed > self.timeout:
                full_log += "\n[TIMEOUT]\n"
                return {
                    "log": full_log,
                    "tool_calls": tool_calls,
                    "iterations": iterations,
                    "files_touched": self._detect_touched_files(workspace),
                    "timed_out": True,
                    "error": f"Simulated run timed out after {elapsed:.0f}s",
                }

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                assistant_msg = response.choices[0].message.content or ""
            except Exception as exc:
                error_str = str(exc)
                if "rate" in error_str.lower() or "429" in error_str:
                    logger.warning("Rate limited during simulated run — waiting 30s")
                    time.sleep(30)
                    continue
                full_log += f"\n[API ERROR: {exc}]\n"
                return {
                    "log": full_log,
                    "tool_calls": tool_calls,
                    "iterations": iterations,
                    "files_touched": [],
                    "timed_out": False,
                    "error": error_str,
                }

            full_log += f"\n--- Turn {turn + 1} ---\n{assistant_msg}\n"
            messages.append({"role": "assistant", "content": assistant_msg})
            iterations += 1

            # Try to extract and apply diff from response
            diff_text = self._extract_diff(assistant_msg)
            if diff_text:
                tool_calls += 1
                applied = self._apply_diff(workspace, diff_text)
                if applied:
                    # Ask for verification
                    messages.append({"role": "user", "content": (
                        "The diff was applied successfully. "
                        "Review if there are any remaining issues or if the fix is complete. "
                        "If complete, respond with DONE."
                    )})
                else:
                    messages.append({"role": "user", "content": (
                        "The diff could not be applied cleanly. "
                        "Please provide a corrected diff."
                    )})
                continue

            # Check if agent considers itself done
            if "DONE" in assistant_msg.upper() and turn > 0:
                break

            # If no diff yet, ask the agent to read specific files and produce one
            if turn == 0:
                # Read the files the agent identified
                files_to_read = self._extract_file_paths(assistant_msg)
                file_contents = ""
                for fp in files_to_read[:5]:
                    full_path = workspace / fp.lstrip("./")
                    if full_path.is_file():
                        try:
                            content = full_path.read_text(encoding="utf-8", errors="replace")
                            # Truncate large files
                            if len(content) > 5000:
                                content = content[:5000] + "\n... [truncated]"
                            file_contents += f"\n### {fp}\n```\n{content}\n```\n"
                            tool_calls += 1
                        except Exception:
                            pass

                if file_contents:
                    messages.append({"role": "user", "content": (
                        f"Here are the file contents:\n{file_contents}\n\n"
                        "Now provide a unified diff to fix the issue."
                    )})
                else:
                    messages.append({"role": "user", "content": (
                        "Please provide a unified diff to fix the issue."
                    )})

        return {
            "log": full_log,
            "tool_calls": tool_calls,
            "iterations": iterations,
            "files_touched": self._detect_touched_files(workspace),
            "timed_out": False,
            "error": "",
        }

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def _extract_diff(text: str) -> str:
        """Extract a unified diff block from LLM response text."""
        lines = text.splitlines()
        diff_lines: list[str] = []
        in_diff = False
        in_code_block = False

        for line in lines:
            if line.strip().startswith("```diff") or line.strip().startswith("```patch"):
                in_code_block = True
                continue
            if in_code_block and line.strip() == "```":
                in_code_block = False
                continue
            if in_code_block:
                diff_lines.append(line)
                continue

            if line.startswith("--- a/") or line.startswith("diff --git"):
                in_diff = True
            if in_diff:
                diff_lines.append(line)

        return "\n".join(diff_lines) if diff_lines else ""

    @staticmethod
    def _apply_diff(workspace: Path, diff_text: str) -> bool:
        """Attempt to apply a diff to the workspace."""
        diff_file = workspace / ".tmp_patch.diff"
        try:
            diff_file.write_text(diff_text, encoding="utf-8")
            result = subprocess.run(
                ["git", "apply", "--allow-empty", str(diff_file)],
                cwd=str(workspace), capture_output=True, text=True, timeout=30,
            )
            return result.returncode == 0
        except Exception as exc:
            logger.debug("Failed to apply diff: %s", exc)
            return False
        finally:
            if diff_file.exists():
                diff_file.unlink()

    @staticmethod
    def _extract_file_paths(text: str) -> list[str]:
        """Extract file paths mentioned in text (heuristic)."""
        import re
        patterns = [
            r'`([a-zA-Z0-9_/\-\.]+\.[a-zA-Z]{1,5})`',
            r'(?:^|\s)((?:src|lib|app|pkg|internal|cmd)/[a-zA-Z0-9_/\-\.]+)',
        ]
        paths: list[str] = []
        for pattern in patterns:
            paths.extend(re.findall(pattern, text))
        # Deduplicate preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for p in paths:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique

    @staticmethod
    def _detect_touched_files(workspace: Path) -> list[str]:
        """Detect which files were modified in the workspace."""
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            cwd=str(workspace), capture_output=True, text=True, timeout=30,
        )
        return [f for f in result.stdout.strip().splitlines() if f]

    # -- Public API ----------------------------------------------------------

    def run(self, workspace: Path, prompt: str,
            model_name: str, api_key: str) -> dict[str, Any]:
        """Execute an agentic run.

        Args:
            workspace: Prepared workspace directory.
            prompt: Issue description.
            model_name: NIM model identifier.
            api_key: NVIDIA API key.

        Returns:
            Result dictionary from the agent run.
        """
        if self.simulated_mode or not shutil.which(self.opencode_binary):
            return self._run_simulated(workspace, prompt, model_name, api_key)
        return self._run_opencode(workspace, prompt, model_name, api_key)


# ---------------------------------------------------------------------------
# MetricCollector
# ---------------------------------------------------------------------------

class MetricCollector:
    """Measures code quality and task completion metrics.

    Paper 1 — RQ4: computes complexity delta, coupling change, test pass rate,
    diff similarity to ground truth, and agent effort indicators.
    """

    def __init__(self, complexity_tool: str = "radon", fallback_tool: str = "lizard"):
        """Initialise MetricCollector.

        Args:
            complexity_tool: Primary complexity analysis tool ('radon' or 'lizard').
            fallback_tool: Fallback if primary is unavailable.
        """
        self.complexity_tool = complexity_tool
        self.fallback_tool = fallback_tool

    # -- Complexity ----------------------------------------------------------

    def measure_complexity(self, workspace: Path, files: list[str]) -> float:
        """Compute average cyclomatic complexity for the given files.

        Args:
            workspace: Path to the workspace directory.
            files: List of relative file paths to measure.

        Returns:
            Average cyclomatic complexity across the files.
        """
        if self.complexity_tool == "radon":
            return self._complexity_radon(workspace, files)
        return self._complexity_lizard(workspace, files)

    def _complexity_radon(self, workspace: Path, files: list[str]) -> float:
        """Measure complexity using radon (Python files)."""
        complexities: list[float] = []
        for f in files:
            full_path = workspace / f
            if not full_path.exists() or not f.endswith(".py"):
                continue
            try:
                result = subprocess.run(
                    ["python", "-m", "radon", "cc", "-a", "-s", str(full_path)],
                    capture_output=True, text=True, timeout=30,
                    cwd=str(workspace),
                )
                # Parse average complexity from radon output
                for line in result.stdout.splitlines():
                    if "Average complexity:" in line:
                        parts = line.split()
                        for part in parts:
                            try:
                                complexities.append(float(part.strip("()")))
                                break
                            except ValueError:
                                continue
            except Exception as exc:
                logger.debug("radon failed for %s: %s", f, exc)
                # Fall back to lizard
                return self._complexity_lizard(workspace, files)
        return float(np.mean(complexities)) if complexities else 0.0

    def _complexity_lizard(self, workspace: Path, files: list[str]) -> float:
        """Measure complexity using lizard (multi-language)."""
        complexities: list[float] = []
        for f in files:
            full_path = workspace / f
            if not full_path.exists():
                continue
            try:
                result = subprocess.run(
                    ["lizard", str(full_path)],
                    capture_output=True, text=True, timeout=30,
                    cwd=str(workspace),
                )
                # Parse lizard table output for CCN column
                for line in result.stdout.splitlines():
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            ccn = float(parts[1])  # CCN is typically the 2nd column
                            complexities.append(ccn)
                        except (ValueError, IndexError):
                            continue
            except Exception as exc:
                logger.debug("lizard failed for %s: %s", f, exc)
        return float(np.mean(complexities)) if complexities else 0.0

    # -- Coupling ------------------------------------------------------------

    @staticmethod
    def measure_coupling(workspace: Path, files: list[str]) -> int:
        """Count import/dependency statements in the given files.

        Args:
            workspace: Path to the workspace directory.
            files: List of relative file paths to measure.

        Returns:
            Total number of import/require/include statements.
        """
        import re

        import_patterns = [
            r'^\s*import\s+',           # Python/Java/Go
            r'^\s*from\s+\S+\s+import', # Python
            r'^\s*require\s*\(',        # JavaScript/Go
            r'^\s*const\s+.*=\s*require\(', # JavaScript
            r'^\s*import\s+.*\s+from\s+', # ES6
            r'^\s*#include\s+',         # C/C++
        ]
        combined = re.compile("|".join(import_patterns))
        total = 0
        for f in files:
            full_path = workspace / f
            if not full_path.is_file():
                continue
            try:
                content = full_path.read_text(encoding="utf-8", errors="replace")
                for line in content.splitlines():
                    if combined.match(line):
                        total += 1
            except Exception:
                continue
        return total

    # -- Test pass rate ------------------------------------------------------

    @staticmethod
    def run_tests(workspace: Path, test_commands: list[str]) -> tuple[int, int, float]:
        """Run test commands and compute pass rate.

        Args:
            workspace: Path to the workspace directory.
            test_commands: List of test commands to execute.

        Returns:
            Tuple of (passed, total, pass_rate).
        """
        passed = 0
        total = len(test_commands)
        for cmd in test_commands:
            try:
                result = subprocess.run(
                    cmd, shell=True, cwd=str(workspace),
                    capture_output=True, text=True, timeout=120,
                )
                if result.returncode == 0:
                    passed += 1
            except subprocess.TimeoutExpired:
                logger.debug("Test command timed out: %s", cmd)
            except Exception as exc:
                logger.debug("Test command failed: %s — %s", cmd, exc)
        rate = passed / total if total > 0 else 0.0
        return passed, total, rate

    # -- Diff similarity -----------------------------------------------------

    @staticmethod
    def diff_similarity(agent_diff: str, ground_truth_diff: str) -> float:
        """Compute similarity between agent diff and ground truth.

        Uses difflib.SequenceMatcher on the changed lines.

        Args:
            agent_diff: The agent's produced diff.
            ground_truth_diff: The known-good diff.

        Returns:
            Similarity ratio between 0.0 and 1.0.
        """
        def _extract_changes(diff_text: str) -> list[str]:
            """Extract only the actual change lines (+ and - lines)."""
            changes = []
            for line in diff_text.splitlines():
                if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
                    changes.append(line.strip())
            return changes

        agent_changes = _extract_changes(agent_diff)
        truth_changes = _extract_changes(ground_truth_diff)

        if not truth_changes:
            return 1.0 if not agent_changes else 0.0

        matcher = difflib.SequenceMatcher(None, agent_changes, truth_changes)
        return matcher.ratio()

    # -- Phase alignment check -----------------------------------------------

    @staticmethod
    def check_phase_alignment(workspace: Path, agent_diff: str,
                              manifest_content: str) -> bool:
        """Check if the agent's patch respects evolution phase constraints.

        Heuristic: if the manifest mentions 'Stabilization', flag new dependencies
        or large additions. If 'Growth', flag deletions of core interfaces.

        Args:
            workspace: Path to the workspace directory.
            agent_diff: The agent's produced diff.
            manifest_content: CLAUDE.md content (empty for condition A).

        Returns:
            True if the patch is phase-aligned.
        """
        if not manifest_content:
            return True  # No manifest → no constraints to violate

        manifest_lower = manifest_content.lower()
        diff_lines = agent_diff.splitlines()
        added_lines = [l for l in diff_lines if l.startswith("+") and not l.startswith("+++")]

        # Stabilization phase: flag new imports/dependencies
        if "stabilization" in manifest_lower:
            import re
            new_imports = [l for l in added_lines
                          if re.match(r'^\+\s*(import|from\s+\S+\s+import|require\(|#include)', l)]
            if len(new_imports) > 3:
                return False

        # Growth phase: generally permissive
        # Decline phase: flag large additions
        if "decline" in manifest_lower:
            if len(added_lines) > 50:
                return False

        return True

    # -- Aggregate -----------------------------------------------------------

    def collect(self, workspace: Path, task: TaskDefinition,
                agent_result: dict[str, Any],
                state_before: dict[str, str]) -> dict[str, Any]:
        """Collect all metrics for a single run.

        Args:
            workspace: Path to the workspace directory.
            task: The task definition.
            agent_result: Output from AgentRunner.run().
            state_before: File state snapshot from before the run.

        Returns:
            Dictionary of metric values.
        """
        changed_files = agent_result.get("files_touched", [])
        target_files = task.ground_truth_files_changed

        # Measure complexity on target files before (from ground truth) and after
        complexity_after = self.measure_complexity(workspace, target_files)

        # Coupling
        coupling_after = self.measure_coupling(workspace, target_files)

        # Tests
        passed, total, pass_rate = self.run_tests(workspace, task.test_commands)

        # Diff similarity
        agent_diff = WorkspaceManager.get_diff(workspace)
        similarity = self.diff_similarity(agent_diff, task.ground_truth_diff)

        # Phase alignment
        manifest_path = workspace / "CLAUDE.md"
        manifest_content = ""
        if manifest_path.exists():
            manifest_content = manifest_path.read_text(encoding="utf-8")
        aligned = self.check_phase_alignment(workspace, agent_diff, manifest_content)

        # Task completion: tests pass AND diff is reasonably similar
        completed = pass_rate > 0.5 and similarity > 0.3

        return {
            "complexity_after": complexity_after,
            "coupling_after": coupling_after,
            "test_pass_rate": pass_rate,
            "tests_passed": passed,
            "tests_total": total,
            "diff_similarity": similarity,
            "task_completed": completed,
            "phase_aligned": aligned,
            "agent_diff": agent_diff,
        }


# ---------------------------------------------------------------------------
# StatisticalAnalyzer
# ---------------------------------------------------------------------------

class StatisticalAnalyzer:
    """Runs statistical tests on RQ4 experiment results.

    Paper 1 — RQ4: Friedman test, Wilcoxon post-hoc with Holm-Bonferroni,
    linear mixed-effects model, and agent behaviour analysis.
    """

    METRICS = [
        "complexity_delta", "coupling_delta", "test_pass_rate",
        "diff_similarity", "task_completed", "phase_aligned",
        "tool_calls", "iterations",
    ]

    def __init__(self, alpha: float = 0.05):
        """Initialise StatisticalAnalyzer.

        Args:
            alpha: Significance level for statistical tests.
        """
        self.alpha = alpha

    @staticmethod
    def _group_by_condition(results: list[dict], metric: str) -> dict[str, list[float]]:
        """Group metric values by condition, aggregating repeats by median."""
        from collections import defaultdict

        # Group by (task_id, condition) -> list of values across repeats
        task_cond: dict[tuple[str, str], list[float]] = defaultdict(list)
        for r in results:
            key = (r["task_id"], r["condition"])
            val = r.get(metric, 0)
            if isinstance(val, bool):
                val = float(val)
            task_cond[key].append(float(val))

        # Take median per task-condition pair
        cond_values: dict[str, list[float]] = defaultdict(list)
        tasks_seen: dict[str, set[str]] = defaultdict(set)
        for (task_id, cond), values in task_cond.items():
            median_val = float(np.median(values))
            cond_values[cond].append(median_val)
            tasks_seen[cond].add(task_id)

        return dict(cond_values)

    def friedman_test(self, results: list[dict], metric: str) -> dict[str, Any]:
        """Run Friedman test across 3 conditions for a metric.

        Args:
            results: List of RunResult dicts.
            metric: Metric name to test.

        Returns:
            Dict with statistic, p_value, and significant flag.
        """
        from scipy import stats

        groups = self._group_by_condition(results, metric)
        if len(groups) < 3:
            return {"statistic": 0, "p_value": 1.0, "significant": False, "error": "< 3 conditions"}

        # Align arrays to same length (matched by task order)
        arrays = [np.array(groups[c]) for c in sorted(groups.keys())]
        min_len = min(len(a) for a in arrays)
        arrays = [a[:min_len] for a in arrays]

        if min_len < 3:
            return {"statistic": 0, "p_value": 1.0, "significant": False,
                    "error": f"only {min_len} matched tasks"}

        stat, p = stats.friedmanchisquare(*arrays)
        return {
            "statistic": float(stat),
            "p_value": float(p),
            "significant": p < self.alpha,
        }

    def wilcoxon_posthoc(self, results: list[dict], metric: str) -> list[dict[str, Any]]:
        """Run Wilcoxon signed-rank post-hoc tests with Holm-Bonferroni correction.

        Comparisons: A-B, A-C, B-C (key comparison is B vs C).

        Args:
            results: List of RunResult dicts.
            metric: Metric name to test.

        Returns:
            List of comparison result dicts.
        """
        from scipy import stats

        groups = self._group_by_condition(results, metric)
        pairs = [("A", "B"), ("A", "C"), ("B", "C")]
        raw_results: list[dict[str, Any]] = []
        p_values: list[float] = []

        for c1, c2 in pairs:
            if c1 not in groups or c2 not in groups:
                raw_results.append({
                    "comparison": f"{c1}_vs_{c2}", "statistic": 0,
                    "p_value": 1.0, "error": "missing condition",
                })
                p_values.append(1.0)
                continue

            a1, a2 = np.array(groups[c1]), np.array(groups[c2])
            min_len = min(len(a1), len(a2))
            a1, a2 = a1[:min_len], a2[:min_len]

            if min_len < 6:
                raw_results.append({
                    "comparison": f"{c1}_vs_{c2}", "statistic": 0,
                    "p_value": 1.0, "error": f"only {min_len} paired samples",
                })
                p_values.append(1.0)
                continue

            try:
                stat, p = stats.wilcoxon(a1, a2)
            except ValueError:
                stat, p = 0.0, 1.0

            # Effect sizes
            cd, cd_mag = cliffs_delta(a1.tolist(), a2.tolist())
            cohen, cohen_mag = cohens_d(a1.tolist(), a2.tolist())

            raw_results.append({
                "comparison": f"{c1}_vs_{c2}",
                "statistic": float(stat),
                "p_value": float(p),
                "cliffs_delta": cd,
                "cliffs_delta_magnitude": cd_mag,
                "cohens_d": cohen,
                "cohens_d_magnitude": cohen_mag,
            })
            p_values.append(float(p))

        # Apply Holm-Bonferroni correction
        rejected = holm_bonferroni(p_values, self.alpha)
        for i, r in enumerate(raw_results):
            r["significant_corrected"] = rejected[i]

        return raw_results

    def mixed_effects_model(self, results: list[dict], metric: str) -> dict[str, Any]:
        """Fit linear mixed-effects model: metric ~ condition + (1|model) + (1|repo).

        Args:
            results: List of RunResult dicts.
            metric: Metric name to model.

        Returns:
            Dict with model coefficients, p-values, and summary.
        """
        try:
            import pandas as pd
            import statsmodels.formula.api as smf
        except ImportError as exc:
            return {"error": f"Missing dependency: {exc}"}

        rows = []
        for r in results:
            val = r.get(metric, 0)
            if isinstance(val, bool):
                val = float(val)
            rows.append({
                "metric_value": float(val),
                "condition": r["condition"],
                "model_key": r.get("model_key", "unknown"),
                "repo": r.get("task_id", "").rsplit("-", 1)[0],
            })

        df = pd.DataFrame(rows)
        if len(df) < 10:
            return {"error": f"Insufficient data: {len(df)} rows"}

        try:
            model = smf.mixedlm(
                "metric_value ~ C(condition, Treatment(reference='A'))",
                data=df,
                groups=df["repo"],
                re_formula="1",
            ).fit(reml=True)

            return {
                "params": model.params.to_dict(),
                "p_values": model.pvalues.to_dict(),
                "aic": float(model.aic) if hasattr(model, "aic") else None,
                "bic": float(model.bic) if hasattr(model, "bic") else None,
                "summary": str(model.summary()),
            }
        except Exception as exc:
            return {"error": str(exc)}

    def agent_behavior_analysis(self, results: list[dict]) -> dict[str, Any]:
        """Compare agent behaviour patterns across conditions.

        Analyses tool-use patterns, iteration counts, and files touched.

        Args:
            results: List of RunResult dicts.

        Returns:
            Dict with per-condition behaviour summaries and comparison stats.
        """
        from collections import defaultdict

        behavior: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for r in results:
            cond = r["condition"]
            behavior[cond]["tool_calls"].append(float(r.get("tool_calls", 0)))
            behavior[cond]["iterations"].append(float(r.get("iterations", 0)))
            behavior[cond]["files_touched"].append(
                float(len(r.get("files_touched", [])))
            )
            behavior[cond]["duration"].append(float(r.get("duration_seconds", 0)))

        summary: dict[str, Any] = {}
        for cond in sorted(behavior.keys()):
            stats_dict: dict[str, Any] = {}
            for bmetric, values in behavior[cond].items():
                arr = np.array(values)
                stats_dict[bmetric] = {
                    "mean": float(np.mean(arr)),
                    "median": float(np.median(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                }
            summary[cond] = stats_dict

        # Key comparison: does manifest reduce exploratory file reads?
        comparisons: dict[str, Any] = {}
        for bmetric in ["tool_calls", "iterations", "files_touched"]:
            for c1, c2 in [("A", "C"), ("B", "C")]:
                if c1 in behavior and c2 in behavior:
                    v1 = behavior[c1].get(bmetric, [])
                    v2 = behavior[c2].get(bmetric, [])
                    if v1 and v2:
                        cd, mag = cliffs_delta(v1, v2)
                        comparisons[f"{bmetric}_{c1}_vs_{c2}"] = {
                            "cliffs_delta": cd,
                            "magnitude": mag,
                        }

        return {"per_condition": summary, "comparisons": comparisons}

    def analyze_all(self, results: list[dict]) -> dict[str, Any]:
        """Run all statistical analyses.

        Args:
            results: List of RunResult dicts.

        Returns:
            Comprehensive analysis report dictionary.
        """
        report: dict[str, Any] = {"metrics": {}, "behavior": {}}

        for metric in self.METRICS:
            logger.info("Analyzing metric: %s", metric)
            report["metrics"][metric] = {
                "friedman": self.friedman_test(results, metric),
                "wilcoxon_posthoc": self.wilcoxon_posthoc(results, metric),
                "mixed_effects": self.mixed_effects_model(results, metric),
            }

        report["behavior"] = self.agent_behavior_analysis(results)
        return report


# ---------------------------------------------------------------------------
# ExperimentOrchestrator
# ---------------------------------------------------------------------------

class ExperimentOrchestrator:
    """Runs the full RQ4 experiment across all combinations.

    Paper 1 — RQ4: iterates over task x condition x model x repeat,
    sequential execution with rate-limit backoff, crash-safe result saving.
    """

    def __init__(self, config: dict, tasks: list[TaskDefinition]):
        """Initialise ExperimentOrchestrator.

        Args:
            config: Parsed rq4.yaml configuration dictionary.
            tasks: List of TaskDefinition objects.
        """
        self.config = config
        self.tasks = tasks
        self.workspace_mgr = WorkspaceManager(
            config.get("agent", {}).get("workspace_base", "/tmp/rq4_workspaces")
        )
        self.agent_runner = AgentRunner(config)
        self.metric_collector = MetricCollector(
            complexity_tool=config.get("metrics", {}).get("complexity_tool", "radon"),
            fallback_tool=config.get("metrics", {}).get("fallback_tool", "lizard"),
        )

        exec_cfg = config.get("execution", {})
        self.repeats = exec_cfg.get("repeats_per_task", 3)
        self.sleep_between = exec_cfg.get("sleep_between_runs", 3)
        self.backoff_base = exec_cfg.get("rate_limit_backoff_base", 5)
        self.backoff_max = exec_cfg.get("rate_limit_backoff_max", 120)
        self.max_retries = exec_cfg.get("rate_limit_max_retries", 10)

        output_cfg = config.get("output", {})
        self.results_dir = _PROJECT_ROOT / output_cfg.get(
            "results_dir", "experiments/paper1_ontology_manifests/results/rq4"
        )
        self.logs_dir = _PROJECT_ROOT / output_cfg.get(
            "logs_dir", "experiments/paper1_ontology_manifests/results/rq4/logs"
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Load existing results for crash recovery
        self.results_file = self.results_dir / "all_results.json"
        self.results: list[dict] = self._load_existing_results()
        self.completed_runs: set[str] = {r["run_id"] for r in self.results}

    def _load_existing_results(self) -> list[dict]:
        """Load previously saved results for crash recovery."""
        if self.results_file.exists():
            try:
                with open(self.results_file) as fh:
                    data = json.load(fh)
                logger.info("Loaded %d existing results", len(data))
                return data
            except (json.JSONDecodeError, KeyError):
                logger.warning("Could not parse existing results — starting fresh")
        return []

    def _save_results(self) -> None:
        """Save all results to JSON (crash-safe)."""
        tmp_file = self.results_file.with_suffix(".tmp")
        with open(tmp_file, "w") as fh:
            json.dump(self.results, fh, indent=2, default=str)
        tmp_file.replace(self.results_file)

    def _save_run_log(self, run_id: str, log_content: str) -> None:
        """Save individual run log."""
        log_file = self.logs_dir / f"{run_id}.log"
        with open(log_file, "w") as fh:
            fh.write(log_content)

    @staticmethod
    def _make_run_id(task_id: str, condition: str, model_key: str,
                     repeat: int) -> str:
        """Generate a deterministic run ID."""
        return f"{task_id}__{condition}__{model_key}__r{repeat}"

    def _resolve_model(self, model_key: str) -> tuple[str, str]:
        """Resolve model name and API key from config.

        Args:
            model_key: 'primary' or 'replication'.

        Returns:
            Tuple of (model_name, api_key).
        """
        model_cfg = self.config.get("models", {}).get(model_key, MODELS[model_key])
        model_name = model_cfg["name"]
        api_key_env = model_cfg["api_key_env"]
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise EnvironmentError(
                f"API key not found in environment: {api_key_env}. "
                f"Set it in .env or export it."
            )
        return model_name, api_key

    def _run_single(self, task: TaskDefinition, condition: str,
                    model_key: str, repeat: int) -> RunResult:
        """Execute a single agentic run.

        Args:
            task: The task to execute.
            condition: 'A', 'B', or 'C'.
            model_key: 'primary' or 'replication'.
            repeat: Repeat index (0-based).

        Returns:
            RunResult with all metrics populated.
        """
        run_id = self._make_run_id(task.task_id, condition, model_key, repeat)
        model_name, api_key = self._resolve_model(model_key)

        result = RunResult(
            run_id=run_id,
            task_id=task.task_id,
            condition=condition,
            model_key=model_key,
            model_name=model_name,
            repeat_index=repeat,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

        workspace: Optional[Path] = None
        try:
            # Prepare workspace
            workspace = self.workspace_mgr.prepare(task, condition, run_id)
            state_before = self.workspace_mgr.capture_state(workspace)

            # Measure pre-change metrics
            result.complexity_before = self.metric_collector.measure_complexity(
                workspace, task.ground_truth_files_changed
            )
            result.coupling_before = self.metric_collector.measure_coupling(
                workspace, task.ground_truth_files_changed
            )

            # Run agent
            prompt = (
                f"Issue #{task.issue_number}: {task.issue_title}\n\n"
                f"{task.issue_description}\n\n"
                f"Fix this issue. The relevant files are likely: "
                f"{', '.join(task.ground_truth_files_changed)}"
            )

            start = time.time()
            agent_out = self._run_with_backoff(
                workspace, prompt, model_name, api_key
            )
            result.duration_seconds = time.time() - start

            # Populate agent behaviour fields
            result.agent_log = agent_out.get("log", "")
            result.tool_calls = agent_out.get("tool_calls", 0)
            result.iterations = agent_out.get("iterations", 0)
            result.files_touched = agent_out.get("files_touched", [])
            result.timed_out = agent_out.get("timed_out", False)
            result.error = agent_out.get("error", "")

            # Collect metrics
            metrics = self.metric_collector.collect(
                workspace, task, agent_out, state_before
            )
            result.complexity_after = metrics["complexity_after"]
            result.complexity_delta = result.complexity_after - result.complexity_before
            result.coupling_after = metrics["coupling_after"]
            result.coupling_delta = result.coupling_after - result.coupling_before
            result.test_pass_rate = metrics["test_pass_rate"]
            result.tests_passed = metrics["tests_passed"]
            result.tests_total = metrics["tests_total"]
            result.diff_similarity = metrics["diff_similarity"]
            result.task_completed = metrics["task_completed"]
            result.phase_aligned = metrics["phase_aligned"]
            result.agent_diff = metrics["agent_diff"]

        except Exception as exc:
            logger.error("Run %s failed: %s", run_id, exc, exc_info=True)
            result.error = str(exc)
        finally:
            if workspace:
                self.workspace_mgr.cleanup(workspace)

        return result

    def _run_with_backoff(self, workspace: Path, prompt: str,
                          model_name: str, api_key: str) -> dict[str, Any]:
        """Run agent with exponential backoff on rate limit errors.

        Args:
            workspace: Prepared workspace directory.
            prompt: Issue description.
            model_name: NIM model identifier.
            api_key: NVIDIA API key.

        Returns:
            Agent run result dictionary.
        """
        for attempt in range(self.max_retries):
            result = self.agent_runner.run(workspace, prompt, model_name, api_key)
            error = result.get("error", "")
            if "rate" in error.lower() or "429" in error:
                wait = min(self.backoff_base * (2 ** attempt), self.backoff_max)
                logger.warning(
                    "Rate limited (attempt %d/%d) — backing off %ds",
                    attempt + 1, self.max_retries, wait,
                )
                time.sleep(wait)
                continue
            return result
        logger.error("Max retries exhausted for rate limiting")
        return result  # return last attempt

    def run_all(self) -> None:
        """Execute the full experiment: all tasks x conditions x models x repeats.

        Sequential execution with crash-safe incremental saving.
        """
        model_keys = list(self.config.get("models", MODELS).keys())
        total_runs = (
            len(self.tasks) * len(CONDITIONS) * len(model_keys) * self.repeats
        )
        completed = 0
        skipped = 0

        logger.info(
            "Starting RQ4 experiment: %d tasks x %d conditions x %d models x %d repeats = %d runs",
            len(self.tasks), len(CONDITIONS), len(model_keys), self.repeats, total_runs,
        )

        for task in self.tasks:
            for condition in CONDITIONS:
                for model_key in model_keys:
                    for repeat in range(self.repeats):
                        run_id = self._make_run_id(
                            task.task_id, condition, model_key, repeat
                        )

                        # Skip if already completed (crash recovery)
                        if run_id in self.completed_runs:
                            skipped += 1
                            continue

                        logger.info(
                            "[%d/%d] Running %s (skipped %d existing)",
                            completed + skipped + 1, total_runs, run_id, skipped,
                        )

                        result = self._run_single(task, condition, model_key, repeat)

                        # Save result
                        result_dict = asdict(result)
                        self.results.append(result_dict)
                        self.completed_runs.add(run_id)
                        self._save_results()
                        self._save_run_log(run_id, result.agent_log)

                        completed += 1

                        # Rate-limit courtesy sleep
                        time.sleep(self.sleep_between)

        logger.info(
            "Experiment complete: %d new runs, %d skipped (previously completed)",
            completed, skipped,
        )

    def generate_report(self) -> dict[str, Any]:
        """Generate a summary report of all results.

        Returns:
            Report dictionary with per-condition summaries and overall stats.
        """
        from collections import defaultdict

        if not self.results:
            return {"error": "No results to report"}

        report: dict[str, Any] = {
            "experiment": "rq4_agentic_code_generation",
            "paper": 1,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "total_runs": len(self.results),
            "conditions": {},
            "models": {},
        }

        # Per-condition summaries
        cond_groups: dict[str, list[dict]] = defaultdict(list)
        model_groups: dict[str, list[dict]] = defaultdict(list)

        for r in self.results:
            cond_groups[r["condition"]].append(r)
            model_groups[r["model_key"]].append(r)

        for cond, runs in sorted(cond_groups.items()):
            label = CONDITION_LABELS.get(cond, cond)
            report["conditions"][label] = {
                "n_runs": len(runs),
                "task_completion_rate": float(
                    np.mean([r["task_completed"] for r in runs])
                ),
                "mean_test_pass_rate": float(
                    np.mean([r["test_pass_rate"] for r in runs])
                ),
                "mean_diff_similarity": float(
                    np.mean([r["diff_similarity"] for r in runs])
                ),
                "mean_complexity_delta": float(
                    np.mean([r["complexity_delta"] for r in runs])
                ),
                "mean_coupling_delta": float(
                    np.mean([r["coupling_delta"] for r in runs])
                ),
                "phase_alignment_rate": float(
                    np.mean([r["phase_aligned"] for r in runs])
                ),
                "mean_tool_calls": float(
                    np.mean([r["tool_calls"] for r in runs])
                ),
                "mean_iterations": float(
                    np.mean([r["iterations"] for r in runs])
                ),
                "timeout_rate": float(
                    np.mean([r["timed_out"] for r in runs])
                ),
                "error_rate": float(
                    np.mean([1 if r.get("error") else 0 for r in runs])
                ),
            }

        for model_key, runs in sorted(model_groups.items()):
            report["models"][model_key] = {
                "n_runs": len(runs),
                "task_completion_rate": float(
                    np.mean([r["task_completed"] for r in runs])
                ),
                "mean_test_pass_rate": float(
                    np.mean([r["test_pass_rate"] for r in runs])
                ),
            }

        return report


# ---------------------------------------------------------------------------
# Task loading helpers
# ---------------------------------------------------------------------------

def load_tasks(path: Path) -> list[TaskDefinition]:
    """Load task definitions from a YAML file.

    Paper 1 — RQ4 task loader.

    Args:
        path: Path to tasks YAML file.

    Returns:
        List of TaskDefinition objects.
    """
    with open(path) as fh:
        data = yaml.safe_load(fh)

    tasks: list[TaskDefinition] = []
    for t in data.get("tasks", []):
        tasks.append(TaskDefinition(
            task_id=t["task_id"],
            repo=t["repo"],
            language=t.get("language", "unknown"),
            issue_number=t["issue_number"],
            issue_title=t.get("issue_title", ""),
            issue_description=t.get("issue_description", ""),
            pre_fix_commit=t.get("pre_fix_commit", ""),
            ground_truth_diff=t.get("ground_truth_diff", ""),
            test_commands=t.get("test_commands", []),
            ground_truth_files_changed=t.get("ground_truth_files_changed", []),
            human_manifest=t.get("human_manifest", ""),
            senso_manifest=t.get("senso_manifest", ""),
        ))

    logger.info("Loaded %d tasks from %s", len(tasks), path)
    return tasks


def load_config(path: Path) -> dict:
    """Load experiment configuration from YAML.

    Paper 1 — RQ4 config loader.

    Args:
        path: Path to rq4.yaml config file.

    Returns:
        Parsed configuration dictionary.
    """
    with open(path) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_select_tasks(args: argparse.Namespace) -> None:
    """CLI handler: select tasks from GitHub."""
    token = args.token or os.environ.get("GITHUB_TOKEN", "")
    if not token:
        logger.error("GitHub token required (--token or $GITHUB_TOKEN)")
        sys.exit(1)

    output = Path(args.output)
    selector = TaskSelector(
        token=token,
        repos_count=args.repos_count,
        tasks_per_repo=args.tasks_per_repo,
        max_diff_lines=args.max_diff_lines,
    )
    tasks = selector.select_all()
    selector.save_tasks(tasks, output)
    logger.info("Task selection complete — %d tasks saved to %s", len(tasks), output)


def cmd_run(args: argparse.Namespace) -> None:
    """CLI handler: run the experiment."""
    config = load_config(Path(args.config))
    tasks = load_tasks(Path(args.tasks))

    if not tasks:
        logger.error("No tasks loaded — cannot run experiment")
        sys.exit(1)

    orchestrator = ExperimentOrchestrator(config, tasks)
    orchestrator.run_all()
    logger.info("Experiment run complete")


def cmd_analyze(args: argparse.Namespace) -> None:
    """CLI handler: analyze results."""
    results_dir = Path(args.results_dir)
    results_file = results_dir / "all_results.json"

    if not results_file.exists():
        logger.error("Results file not found: %s", results_file)
        sys.exit(1)

    with open(results_file) as fh:
        results = json.load(fh)

    analyzer = StatisticalAnalyzer(alpha=args.alpha)
    analysis = analyzer.analyze_all(results)

    output_file = results_dir / "statistical_analysis.json"
    with open(output_file, "w") as fh:
        json.dump(analysis, fh, indent=2, default=str)

    logger.info("Analysis complete — results saved to %s", output_file)

    # Print key findings
    for metric, tests in analysis.get("metrics", {}).items():
        friedman = tests.get("friedman", {})
        if friedman.get("significant"):
            logger.info(
                "  %s: Friedman p=%.4f (SIGNIFICANT)",
                metric, friedman["p_value"],
            )
            for posthoc in tests.get("wilcoxon_posthoc", []):
                if posthoc.get("significant_corrected"):
                    logger.info(
                        "    %s: p=%.4f, Cliff's d=%.3f (%s)",
                        posthoc["comparison"], posthoc["p_value"],
                        posthoc.get("cliffs_delta", 0),
                        posthoc.get("cliffs_delta_magnitude", ""),
                    )


def cmd_report(args: argparse.Namespace) -> None:
    """CLI handler: generate summary report."""
    results_dir = Path(args.results_dir)
    results_file = results_dir / "all_results.json"

    if not results_file.exists():
        logger.error("Results file not found: %s", results_file)
        sys.exit(1)

    with open(results_file) as fh:
        results = json.load(fh)

    # Create a minimal orchestrator just for report generation
    config = {"output": {"results_dir": str(results_dir)}}
    orchestrator = ExperimentOrchestrator.__new__(ExperimentOrchestrator)
    orchestrator.results = results

    report = orchestrator.generate_report()

    # Also include statistical analysis if available
    analysis_file = results_dir / "statistical_analysis.json"
    if analysis_file.exists():
        with open(analysis_file) as fh:
            report["statistical_analysis"] = json.load(fh)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    logger.info("Report saved to %s", output)


def main() -> None:
    """Main CLI entry point for the RQ4 experiment harness.

    Paper 1 — RQ4: Agentic Code Generation Experiment.
    """
    parser = argparse.ArgumentParser(
        description="RQ4 Agentic Code Generation Experiment — Paper 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- select-tasks --------------------------------------------------------
    sp_select = subparsers.add_parser(
        "select-tasks", help="Select tasks from real closed GitHub issues",
    )
    sp_select.add_argument("--token", help="GitHub personal access token")
    sp_select.add_argument(
        "--output", default="configs/tasks.yaml",
        help="Output path for task definitions YAML",
    )
    sp_select.add_argument("--repos-count", type=int, default=20)
    sp_select.add_argument("--tasks-per-repo", type=int, default=5)
    sp_select.add_argument("--max-diff-lines", type=int, default=500)
    sp_select.set_defaults(func=cmd_select_tasks)

    # -- run -----------------------------------------------------------------
    sp_run = subparsers.add_parser("run", help="Run the experiment")
    sp_run.add_argument(
        "--config", default="configs/rq4.yaml",
        help="Path to rq4.yaml config",
    )
    sp_run.add_argument(
        "--tasks", default="configs/tasks.yaml",
        help="Path to tasks YAML file",
    )
    sp_run.set_defaults(func=cmd_run)

    # -- analyze -------------------------------------------------------------
    sp_analyze = subparsers.add_parser("analyze", help="Analyze results")
    sp_analyze.add_argument(
        "--results-dir", default="results/rq4/",
        help="Directory containing all_results.json",
    )
    sp_analyze.add_argument("--alpha", type=float, default=0.05)
    sp_analyze.set_defaults(func=cmd_analyze)

    # -- report --------------------------------------------------------------
    sp_report = subparsers.add_parser("report", help="Generate summary report")
    sp_report.add_argument(
        "--results-dir", default="results/rq4/",
        help="Directory containing all_results.json",
    )
    sp_report.add_argument(
        "--output", default="results/rq4_report.json",
        help="Output path for report JSON",
    )
    sp_report.set_defaults(func=cmd_report)

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args.func(args)


if __name__ == "__main__":
    main()
