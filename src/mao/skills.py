"""Skill discovery and runtime helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mao.storage import ExperienceTree

SKILL_FILE_NAME = "SKILL.md"
GUIDANCE_FILE_NAMES = ("AGENTS.md", "CLAUDE.md")


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}, text
    raw_meta = text[4:end]
    body = text[end + 5 :].lstrip()
    metadata: dict[str, str] = {}
    for line in raw_meta.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = _strip_quotes(value.strip())
    return metadata, body


@dataclass(slots=True)
class SkillMetadata:
    name: str
    description: str
    path: str
    root: str

    @property
    def directory(self) -> str:
        return str(Path(self.path).parent)


@dataclass(slots=True)
class WorkspaceGuidance:
    path: str
    content: str


def _existing_dirs(paths: list[Path]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for path in paths:
        normalized = str(path.resolve()) if path.exists() else str(path)
        if not path.is_dir() or normalized in seen:
            continue
        result.append(normalized)
        seen.add(normalized)
    return result


def discover_skill_roots(
    *,
    cwd: str | None = None,
    extra_paths: list[str] | None = None,
) -> list[str]:
    roots: list[Path] = []
    current = Path(cwd or os.getcwd()).resolve()

    env_paths = [
        Path(p).expanduser()
        for p in os.environ.get("MAO_SKILL_PATHS", "").split(os.pathsep)
        if p.strip()
    ]
    roots.extend(env_paths)
    if extra_paths:
        roots.extend(Path(p).expanduser() for p in extra_paths if p)

    for directory in [current, *current.parents]:
        roots.extend(
            [
                directory / ".codex" / "skills",
                directory / ".claude" / "skills",
                directory / ".agents" / "skills",
            ]
        )

    home = Path.home()
    roots.extend(
        [
            home / ".codex" / "skills",
            home / ".claude" / "skills",
            home / ".agents" / "skills",
        ]
    )
    return _existing_dirs(roots)


def discover_guidance_files(*, cwd: str | None = None) -> list[str]:
    current = Path(cwd or os.getcwd()).resolve()
    candidates: list[Path] = []
    for directory in [current, *current.parents]:
        for file_name in GUIDANCE_FILE_NAMES:
            candidates.append(directory / file_name)
    home = Path.home()
    candidates.extend(home / file_name for file_name in GUIDANCE_FILE_NAMES)
    seen: set[str] = set()
    result: list[str] = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        normalized = str(candidate.resolve())
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


class SkillRegistry:
    def __init__(
        self,
        *,
        cwd: str | None = None,
        skill_paths: list[str] | None = None,
    ) -> None:
        self.cwd = cwd or os.getcwd()
        self.skill_roots = discover_skill_roots(cwd=self.cwd, extra_paths=skill_paths)
        self.guidance_files = discover_guidance_files(cwd=self.cwd)
        self._skills: dict[str, SkillMetadata] = {}
        self._load_skills()

    def _load_skills(self) -> None:
        skills: dict[str, SkillMetadata] = {}
        for root in self.skill_roots:
            for path in Path(root).glob(f"*/{SKILL_FILE_NAME}"):
                try:
                    text = path.read_text(encoding="utf-8")
                except OSError:
                    continue
                metadata, _ = _parse_frontmatter(text)
                name = metadata.get("name") or path.parent.name
                description = metadata.get("description") or ""
                key = name.strip()
                if not key or key in skills:
                    continue
                skills[key] = SkillMetadata(
                    name=key,
                    description=description,
                    path=str(path.resolve()),
                    root=root,
                )
        self._skills = skills

    def list_skills(self) -> list[SkillMetadata]:
        return sorted(self._skills.values(), key=lambda item: item.name.lower())

    def get_skill(self, name: str) -> SkillMetadata | None:
        return self._skills.get(name)

    def read_skill(self, name: str) -> str | None:
        skill = self.get_skill(name)
        if not skill:
            return None
        return Path(skill.path).read_text(encoding="utf-8")

    def read_guidance(self) -> list[WorkspaceGuidance]:
        guidance: list[WorkspaceGuidance] = []
        for path in self.guidance_files:
            try:
                guidance.append(
                    WorkspaceGuidance(
                        path=path, content=Path(path).read_text(encoding="utf-8").strip()
                    )
                )
            except OSError:
                continue
        return guidance

    def render_selected_skills(self, selected: list[str] | None) -> str:
        if not selected:
            return ""
        blocks: list[str] = []
        for skill_name in selected:
            raw = self.read_skill(skill_name)
            if not raw:
                continue
            _, body = _parse_frontmatter(raw)
            blocks.append(f"## Skill: {skill_name}\n{body.strip()}")
        return "\n\n".join(blocks).strip()

    def render_guidance(self) -> str:
        entries = self.read_guidance()
        if not entries:
            return ""
        parts = []
        for entry in entries:
            parts.append(f"## Guidance from {Path(entry.path).name}\n{entry.content}")
        return "\n\n".join(parts).strip()

    async def recommend_skills(
        self,
        query: str,
        *,
        experience_tree: ExperienceTree | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        query_terms = {term for term in query.lower().split() if term}
        scored: list[tuple[float, SkillMetadata, list[str]]] = []
        experience_hits = []
        if experience_tree:
            experience_hits = await experience_tree.search_async(query, k=limit * 3)

        for skill in self.list_skills():
            reasons: list[str] = []
            haystack = f"{skill.name} {skill.description}".lower()
            score = 0.0
            overlap = sorted(term for term in query_terms if term in haystack)
            if overlap:
                score += float(len(overlap))
                reasons.append(f"metadata matched: {', '.join(overlap[:4])}")

            skill_tag = f"skill:{skill.name}"
            matching_hits = [
                hit
                for hit in experience_hits
                if skill_tag in (hit.get("tags") or [])
            ]
            if matching_hits:
                score += 2.5
                reasons.append("matched prior skill insights")

            if score > 0:
                scored.append((score, skill, reasons))

        scored.sort(key=lambda item: (-item[0], item[1].name.lower()))
        return [
            {
                "name": skill.name,
                "description": skill.description,
                "path": skill.path,
                "score": score,
                "reasons": reasons,
            }
            for score, skill, reasons in scored[:limit]
        ]

    async def save_skill_insight(
        self,
        *,
        skill_name: str,
        insight: str,
        experience_tree: ExperienceTree | None,
    ) -> bool:
        if not experience_tree or not self.get_skill(skill_name):
            return False
        await experience_tree.learn_from_experience_async(
            f"Skill insight for {skill_name}: {insight}",
            tags=["skill", "skill_insight", f"skill:{skill_name}"],
        )
        return True
