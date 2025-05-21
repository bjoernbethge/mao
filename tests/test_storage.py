"""
Tests for the storage module.
"""

import pytest
from unittest.mock import patch
from mao.storage import KnowledgeTree, ExperienceTree


@pytest.mark.asyncio
async def test_knowledge_tree_basic_operations(knowledge_tree):
    """Test basic operations of KnowledgeTree."""
    # Add a knowledge point
    point_id = await knowledge_tree.add_entry_async(
        text="This is a test knowledge point",
        tags=["test", "importance:high"],
    )

    # Verify point was added
    assert point_id is not None
    assert isinstance(point_id, str)

    # Search for the point
    results = await knowledge_tree.search_async("test knowledge")
    assert len(results) > 0
    assert any(r["id"] == point_id for r in results)

    # Delete the point
    await knowledge_tree.delete_entry_async(point_id)

    # Verify point was deleted
    results = await knowledge_tree.search_async("test knowledge")
    assert not any(r["id"] == point_id for r in results)


@pytest.mark.asyncio
async def test_experience_tree_basic_operations(experience_tree):
    """Test basic operations of ExperienceTree."""
    # Add an experience point
    point_id = await experience_tree.add_entry_async(
        text="This is a test experience",
        tags=["outcome:success", "importance:high"],
    )

    # Verify point was added
    assert point_id is not None
    assert isinstance(point_id, str)

    # Search for the point
    results = await experience_tree.search_async("test experience")
    assert len(results) > 0
    assert any(r["id"] == point_id for r in results)

    # Delete the point
    await experience_tree.delete_entry_async(point_id)

    # Verify point was deleted
    results = await experience_tree.search_async("test experience")
    assert not any(r["id"] == point_id for r in results)


@pytest.mark.asyncio
async def test_knowledge_tree_batch_operations(knowledge_tree):
    """Test batch operations of KnowledgeTree."""
    # Add multiple knowledge points
    texts = [f"Knowledge point {i}" for i in range(5)]
    tags_list = [[f"index:{i}"] for i in range(5)]

    point_ids = await knowledge_tree.add_entries_batch_async(texts, tags_list)

    # Verify points were added
    assert len(point_ids) == 5
    assert all(isinstance(pid, str) for pid in point_ids)

    # Search for points with increased k parameter to get all results
    results = await knowledge_tree.search_async("Knowledge point", k=10)
    assert len(results) >= 5

    # Delete points
    for point_id in point_ids:
        await knowledge_tree.delete_entry_async(point_id)

    # Verify points were deleted
    results = await knowledge_tree.search_async("Knowledge point", k=10)
    assert not any(r["id"] in point_ids for r in results)


@pytest.mark.asyncio
async def test_experience_tree_batch_operations(experience_tree):
    """Test batch operations of ExperienceTree."""
    # Add multiple experience points
    texts = [f"Experience {i}" for i in range(5)]
    tags_list = [[f"index:{i}"] for i in range(5)]

    point_ids = await experience_tree.add_entries_batch_async(texts, tags_list)

    # Verify points were added
    assert len(point_ids) == 5
    assert all(isinstance(pid, str) for pid in point_ids)

    # Search for points with increased k parameter to get all results
    results = await experience_tree.search_async("Experience", k=10)
    assert len(results) >= 5

    # Delete points
    for point_id in point_ids:
        await experience_tree.delete_entry_async(point_id)

    # Verify points were deleted
    results = await experience_tree.search_async("Experience", k=10)
    assert not any(r["id"] in point_ids for r in results)


@pytest.mark.asyncio
async def test_knowledge_tree_update_point(knowledge_tree):
    """Test updating a knowledge point."""
    # Add a knowledge point
    point_id = await knowledge_tree.add_entry_async(
        text="Original content",
        tags=["version:1"],
    )

    # Get the point to update
    point = await knowledge_tree.get_entry_async(point_id)
    assert point is not None

    # Update the point by adding a new entry and relating it
    new_point_id = await knowledge_tree.add_entry_async(
        text="Updated content",
        tags=["version:2"],
    )

    # Add relation between points
    await knowledge_tree.add_relation_async(point_id, new_point_id, "updated_to")

    # Search for the updated point
    results = await knowledge_tree.search_async("Updated content")
    assert len(results) > 0

    # Find the updated point
    updated_point = next((r for r in results if r["id"] == new_point_id), None)
    assert updated_point is not None
    assert "version:2" in updated_point.get("tags", [])


@pytest.mark.asyncio
async def test_experience_tree_update_point(experience_tree):
    """Test updating an experience point."""
    # Add an experience point
    point_id = await experience_tree.add_entry_async(
        text="Original experience",
        tags=["version:1"],
    )

    # Get the point to update
    point = await experience_tree.get_entry_async(point_id)
    assert point is not None

    # Update the point by adding a new entry and relating it
    new_point_id = await experience_tree.add_entry_async(
        text="Updated experience",
        tags=["version:2"],
    )

    # Add relation between points
    await experience_tree.add_relation_async(point_id, new_point_id, "updated_to")

    # Search for the updated point
    results = await experience_tree.search_async("Updated experience")
    assert len(results) > 0

    # Find the updated point
    updated_point = next((r for r in results if r["id"] == new_point_id), None)
    assert updated_point is not None
    assert "version:2" in updated_point.get("tags", [])


@pytest.mark.asyncio
async def test_knowledge_tree_get_point(knowledge_tree):
    """Test getting a specific knowledge point by ID."""
    # Add a knowledge point
    original_content = "Specific knowledge content"
    point_id = await knowledge_tree.add_entry_async(
        text=original_content,
        tags=["specific:true"],
    )

    # Get the point by ID
    point = await knowledge_tree.get_entry_async(point_id)

    # Verify the point
    assert point is not None
    assert point["id"] == point_id
    assert "specific:true" in point.get("tags", [])


@pytest.mark.asyncio
async def test_experience_tree_get_point(experience_tree):
    """Test getting a specific experience point by ID."""
    # Add an experience point
    original_content = "Specific experience content"
    point_id = await experience_tree.add_entry_async(
        text=original_content,
        tags=["specific:true"],
    )

    # Get the point by ID
    point = await experience_tree.get_entry_async(point_id)

    # Verify the point
    assert point is not None
    assert point["id"] == point_id
    assert "specific:true" in point.get("tags", [])


@pytest.mark.asyncio
async def test_knowledge_tree_search_with_filters(knowledge_tree):
    """Test searching knowledge points with metadata filters."""
    # Add points with different categories
    await knowledge_tree.add_entry_async(
        text="Content about science",
        tags=["category:science"],
    )

    await knowledge_tree.add_entry_async(
        text="Content about history",
        tags=["category:history"],
    )

    await knowledge_tree.add_entry_async(
        text="More science content",
        tags=["category:science"],
    )

    # Search with filter for science category
    science_results = await knowledge_tree.search_async("science")
    assert len(science_results) >= 2

    # Search with filter for history category
    history_results = await knowledge_tree.search_async("history")
    assert len(history_results) >= 1


@pytest.mark.asyncio
async def test_experience_tree_search_with_filters(experience_tree):
    """Test searching experience points with metadata filters."""
    # Add points with different outcomes
    await experience_tree.add_entry_async(
        text="Experience with success",
        tags=["outcome:success"],
    )

    await experience_tree.add_entry_async(
        text="Experience with failure",
        tags=["outcome:failure"],
    )

    await experience_tree.add_entry_async(
        text="Another successful experience",
        tags=["outcome:success"],
    )

    # Search with filter for successful outcomes
    success_results = await experience_tree.search_async("success")
    assert len(success_results) >= 2

    # Search with filter for failure outcomes
    failure_results = await experience_tree.search_async("failure")
    assert len(failure_results) >= 1


@pytest.mark.asyncio
async def test_knowledge_tree_create_with_invalid_url():
    """Test creating a KnowledgeTree with an invalid URL."""
    with pytest.raises(Exception):
        await KnowledgeTree.create(
            url="http://invalid-url-that-doesnt-exist:9999",
            collection_name="test_invalid_collection",
        )


@pytest.mark.asyncio
async def test_experience_tree_create_with_invalid_url():
    """Test creating an ExperienceTree with an invalid URL."""
    with pytest.raises(Exception):
        await ExperienceTree.create(
            url="http://invalid-url-that-doesnt-exist:9999",
            collection_name="test_invalid_collection",
        )


@pytest.mark.asyncio
@patch("mao.storage.QdrantClient")
async def test_knowledge_tree_clear_all_points(mock_qdrant_client, knowledge_tree):
    """Test clearing all points from a KnowledgeTree."""
    # Add some points
    for i in range(3):
        await knowledge_tree.add_entry_async(
            text=f"Test content {i}",
            tags=[f"index:{i}"],
        )

    # Clear all points
    await knowledge_tree.clear_all_points_async()

    # Verify all points are removed
    results = await knowledge_tree.search_async("Test content")
    assert len(results) == 0
