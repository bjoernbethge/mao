from pathlib import Path
import uuid

import pytest

from mao.storage import ExperienceTree
from mao.skills import SkillRegistry

try:
    from pytest_asyncio import fixture as asyncio_fixture
except ImportError:
    asyncio_fixture = pytest.fixture  # type: ignore


def _write_skill(root: Path, name: str, description: str, body: str) -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )


@pytest.fixture(scope="function")
def local_tmp_dir():
    path = Path.cwd() / ".test_tmp" / f"skills_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_skill_registry_discovers_project_and_explicit_paths(local_tmp_dir, monkeypatch):
    project_root = local_tmp_dir / "repo"
    project_root.mkdir()
    claude_skills = project_root / ".claude" / "skills"
    extra_skills = local_tmp_dir / "extra_skills"
    _write_skill(claude_skills, "alpha", "Alpha workflow", "Use alpha carefully.")
    _write_skill(extra_skills, "beta", "Beta workflow", "Use beta carefully.")
    (project_root / "AGENTS.md").write_text("Project agent guidance", encoding="utf-8")
    (project_root / "CLAUDE.md").write_text("Project claude guidance", encoding="utf-8")

    monkeypatch.setenv("MAO_SKILL_PATHS", "")
    registry = SkillRegistry(cwd=str(project_root), skill_paths=[str(extra_skills)])

    skills = {skill.name: skill for skill in registry.list_skills()}
    assert "alpha" in skills
    assert "beta" in skills
    assert str(claude_skills.resolve()) in registry.skill_roots
    assert str(extra_skills.resolve()) in registry.skill_roots

    guidance = registry.render_guidance()
    assert "Project agent guidance" in guidance
    assert "Project claude guidance" in guidance


@asyncio_fixture(scope="function")
async def experience_tree():
    tree = await ExperienceTree.create(
        db_path=":memory:",
        collection_name="skill_experience_collection",
        recreate_on_dim_mismatch=True,
    )
    await tree.clear_all_points_async()
    yield tree
    await tree.clear_all_points_async()


@pytest.mark.asyncio
async def test_skill_registry_saves_and_recommends_skill_insights(
    local_tmp_dir, monkeypatch, experience_tree
):
    skill_root = local_tmp_dir / "skills"
    _write_skill(skill_root, "reviewer", "Code review workflow", "Review code.")
    monkeypatch.setenv("MAO_SKILL_PATHS", "")
    registry = SkillRegistry(skill_paths=[str(skill_root)])

    saved = await registry.save_skill_insight(
        skill_name="reviewer",
        insight="Use this skill for review findings and regression checks.",
        experience_tree=experience_tree,
    )
    assert saved is True

    matches = await registry.recommend_skills(
        "review regressions in code", experience_tree=experience_tree
    )
    assert matches
    assert matches[0]["name"] == "reviewer"
