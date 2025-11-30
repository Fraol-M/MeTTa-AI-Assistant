"""
Integration tests for chunk annotation endpoints (chunk_annotation.py router).
Tests annotation endpoints with real database.
"""
import pytest
from httpx import AsyncClient
from bson import ObjectId
from datetime import datetime, timezone
from app.model.chunk import AnnotationStatus


@pytest.mark.asyncio
@pytest.mark.integration
class TestAnnotateSingleChunk:
    """Test POST /annotation/{chunk_id} endpoint."""
    
    async def test_annotate_chunk_success(
        self, async_client: AsyncClient, admin_headers, clean_db, mongo_db
    ):
        """Test successfully annotating a single chunk."""
        # Create a code chunk
        chunk_data = {
            "chunkId": "test-chunk-1",
            "chunk": "(= (factorial $n) (if (== $n 0) 1 (* $n (factorial (- $n 1)))))",
            "source": "code",
            "isEmbedded": False,
            "status": AnnotationStatus.RAW.value,
            "project": "test_project",
            "repo": "test_repo"
        }
        
        await mongo_db.chunks.insert_one(chunk_data)
        
        # Annotate the chunk
        response = await async_client.post(
            "/annotation/test-chunk-1",
            headers=admin_headers
        )
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        
        # Verify response structure
        assert "chunkId" in data
        assert "chunk" in data
        assert "status" in data
        assert data["chunkId"] == "test-chunk-1"
        
        # Verify annotation was added (description is an alias for annotation)
        annotation = data.get("description") or data.get("annotation")
        assert annotation is not None
        assert len(annotation) > 0
        
        # Verify status is ANNOTATED
        assert data["status"] == AnnotationStatus.ANNOTATED.value
        
        # Verify in database
        updated_chunk = await mongo_db.chunks.find_one({"chunkId": "test-chunk-1"})
        assert updated_chunk is not None
        assert updated_chunk.get("annotation") is not None
        assert updated_chunk["status"] == AnnotationStatus.ANNOTATED.value
    
    async def test_annotate_chunk_not_found(
        self, async_client: AsyncClient, admin_headers, clean_db
    ):
        """Test annotating a non-existent chunk."""
        fake_chunk_id = "nonexistent-chunk-id"
        
        response = await async_client.post(
            f"/annotation/{fake_chunk_id}",
            headers=admin_headers
        )
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    async def test_annotate_chunk_not_code_source(
        self, async_client: AsyncClient, admin_headers, clean_db, mongo_db
    ):
        """Test annotating a chunk that is not from code source."""
        # Create a documentation chunk (not code)
        chunk_data = {
            "chunkId": "doc-chunk-1",
            "chunk": "This is documentation content",
            "source": "documentation",
            "isEmbedded": False,
            "status": AnnotationStatus.RAW.value
        }
        
        await mongo_db.chunks.insert_one(chunk_data)
        
        # Try to annotate (should fail - only code chunks can be annotated)
        response = await async_client.post(
            "/annotation/doc-chunk-1",
            headers=admin_headers
        )
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    async def test_annotate_chunk_empty_content(
        self, async_client: AsyncClient, admin_headers, clean_db, mongo_db
    ):
        """Test annotating a chunk with empty content."""
        chunk_data = {
            "chunkId": "empty-chunk-1",
            "chunk": "",  # Empty chunk
            "source": "code",
            "isEmbedded": False,
            "status": AnnotationStatus.RAW.value
        }
        
        await mongo_db.chunks.insert_one(chunk_data)
        
        # Try to annotate (should fail validation)
        response = await async_client.post(
            "/annotation/empty-chunk-1",
            headers=admin_headers
        )
        
        # Should return 404 (chunk not found after validation fails) or 500
        assert response.status_code in [404, 500]
    
    async def test_annotate_chunk_unauthorized(
        self, async_client: AsyncClient, auth_headers, clean_db, mongo_db
    ):
        """Test that non-admin users cannot annotate chunks."""
        chunk_data = {
            "chunkId": "test-chunk-2",
            "chunk": "test code",
            "source": "code",
            "isEmbedded": False,
            "status": AnnotationStatus.RAW.value
        }
        
        await mongo_db.chunks.insert_one(chunk_data)
        
        response = await async_client.post(
            "/annotation/test-chunk-2",
            headers=auth_headers  # Regular user, not admin
        )
        
        assert response.status_code == 403
        assert "admin" in response.json()["detail"].lower()
    
    async def test_annotate_chunk_no_auth(self, async_client: AsyncClient, clean_db, mongo_db):
        """Test annotating without authentication."""
        chunk_data = {
            "chunkId": "test-chunk-3",
            "chunk": "test code",
            "source": "code",
            "isEmbedded": False,
            "status": AnnotationStatus.RAW.value
        }
        
        await mongo_db.chunks.insert_one(chunk_data)
        
        response = await async_client.post("/annotation/test-chunk-3")
        assert response.status_code == 401


@pytest.mark.asyncio
@pytest.mark.integration
class TestBatchAnnotationUnannotated:
    """Test POST /annotation/batch/unannotated endpoint."""
    
    async def test_batch_annotation_success(
        self, async_client: AsyncClient, admin_headers, clean_db, mongo_db
    ):
        """Test triggering batch annotation for unannotated chunks."""
        # Create multiple unannotated chunks
        chunks = [
            {
                "chunkId": f"batch-chunk-{i}",
                "chunk": f"(= (test-function-{i} $x) (* $x 2))",
                "source": "code",
                "isEmbedded": False,
                "status": AnnotationStatus.RAW.value,
                "project": "test_project",
                "repo": "test_repo"
            }
            for i in range(3)
        ]
        
        await mongo_db.chunks.insert_many(chunks)
        
        # Trigger batch annotation
        response = await async_client.post(
            "/annotation/batch/unannotated",
            headers=admin_headers
        )
        
        assert response.status_code == 202, f"Expected 202, got {response.status_code}: {response.text}"
        data = response.json()
        
        # Verify response structure
        assert "message" in data
        assert "status" in data
        assert "action" in data
        assert data["action"] == "batch_annotate_unannotated"
        assert "202" in data["status"]
        assert "limit" in data
        assert data["limit"] is None  # No limit specified
    
    async def test_batch_annotation_with_limit(
        self, async_client: AsyncClient, admin_headers, clean_db, mongo_db
    ):
        """Test batch annotation with a limit."""
        # Create chunks
        chunks = [
            {
                "chunkId": f"limit-chunk-{i}",
                "chunk": f"(= (func-{i} $x) $x)",
                "source": "code",
                "isEmbedded": False,
                "status": AnnotationStatus.RAW.value
            }
            for i in range(5)
        ]
        
        await mongo_db.chunks.insert_many(chunks)
        
        # Trigger batch annotation with limit
        response = await async_client.post(
            "/annotation/batch/unannotated?limit=2",
            headers=admin_headers
        )
        
        assert response.status_code == 202
        data = response.json()
        assert data["limit"] == 2
    
    async def test_batch_annotation_no_chunks(
        self, async_client: AsyncClient, admin_headers, clean_db
    ):
        """Test batch annotation when no unannotated chunks exist."""
        response = await async_client.post(
            "/annotation/batch/unannotated",
            headers=admin_headers
        )
        
        # Should still return 202 (background task initiated)
        assert response.status_code == 202
        data = response.json()
        assert "message" in data
    
    async def test_batch_annotation_unauthorized(
        self, async_client: AsyncClient, auth_headers, clean_db
    ):
        """Test that non-admin users cannot trigger batch annotation."""
        response = await async_client.post(
            "/annotation/batch/unannotated",
            headers=auth_headers  # Regular user
        )
        
        assert response.status_code == 403
        assert "admin" in response.json()["detail"].lower()
    
    async def test_batch_annotation_no_auth(self, async_client: AsyncClient, clean_db):
        """Test batch annotation without authentication."""
        response = await async_client.post("/annotation/batch/unannotated")
        assert response.status_code == 401


@pytest.mark.asyncio
@pytest.mark.integration
class TestBatchRetryFailed:
    """Test POST /annotation/batch/retry_failed endpoint."""
    
    async def test_retry_failed_annotations_success(
        self, async_client: AsyncClient, admin_headers, clean_db, mongo_db
    ):
        """Test triggering retry for failed annotations."""
        # Create failed chunks
        failed_chunks = [
            {
                "chunkId": f"failed-chunk-{i}",
                "chunk": f"(= (failed-{i} $x) $x)",
                "source": "code",
                "isEmbedded": False,
                "status": AnnotationStatus.FAILED_GEN.value,
                "retry_count": 1
            }
            for i in range(2)
        ]
        
        await mongo_db.chunks.insert_many(failed_chunks)
        
        # Trigger retry
        response = await async_client.post(
            "/annotation/batch/retry_failed",
            headers=admin_headers
        )
        
        assert response.status_code == 202, f"Expected 202, got {response.status_code}: {response.text}"
        data = response.json()
        
        # Verify response structure
        assert "message" in data
        assert "status" in data
        assert "action" in data
        assert data["action"] == "batch_retry_failed"
        assert "include_quota" in data
        assert data["include_quota"] is False
    
    async def test_retry_failed_with_quota(
        self, async_client: AsyncClient, admin_headers, clean_db, mongo_db
    ):
        """Test retry failed annotations including quota-exceeded chunks."""
        # Create failed chunks including quota failures
        failed_chunks = [
            {
                "chunkId": "quota-failed-1",
                "chunk": "(= (test $x) $x)",
                "source": "code",
                "isEmbedded": False,
                "status": AnnotationStatus.FAILED_QUOTA.value,
                "retry_count": 0
            },
            {
                "chunkId": "gen-failed-1",
                "chunk": "(= (test2 $x) $x)",
                "source": "code",
                "isEmbedded": False,
                "status": AnnotationStatus.FAILED_GEN.value,
                "retry_count": 1
            }
        ]
        
        await mongo_db.chunks.insert_many(failed_chunks)
        
        # Trigger retry with include_quota=True
        response = await async_client.post(
            "/annotation/batch/retry_failed?include_quota=true",
            headers=admin_headers
        )
        
        assert response.status_code == 202
        data = response.json()
        assert data["include_quota"] is True
    
    async def test_retry_failed_no_chunks(
        self, async_client: AsyncClient, admin_headers, clean_db
    ):
        """Test retry when no failed chunks exist."""
        response = await async_client.post(
            "/annotation/batch/retry_failed",
            headers=admin_headers
        )
        
        # Should still return 202 (background task initiated)
        assert response.status_code == 202
        data = response.json()
        assert "message" in data
    
    async def test_retry_failed_unauthorized(
        self, async_client: AsyncClient, auth_headers, clean_db
    ):
        """Test that non-admin users cannot trigger retry."""
        response = await async_client.post(
            "/annotation/batch/retry_failed",
            headers=auth_headers  # Regular user
        )
        
        assert response.status_code == 403
        assert "admin" in response.json()["detail"].lower()
    
    async def test_retry_failed_no_auth(self, async_client: AsyncClient, clean_db):
        """Test retry without authentication."""
        response = await async_client.post("/annotation/batch/retry_failed")
        assert response.status_code == 401


@pytest.mark.asyncio
@pytest.mark.integration
class TestAnnotationStatusTransitions:
    """Test annotation status transitions."""
    
    async def test_annotation_status_flow(
        self, async_client: AsyncClient, admin_headers, clean_db, mongo_db
    ):
        """Test the complete annotation status flow."""
        chunk_data = {
            "chunkId": "status-test-chunk",
            "chunk": "(= (factorial $n) (if (== $n 0) 1 (* $n (factorial (- $n 1)))))",
            "source": "code",
            "isEmbedded": False,
            "status": AnnotationStatus.RAW.value,
            "project": "test_project",
            "repo": "test_repo"
        }
        
        await mongo_db.chunks.insert_one(chunk_data)
        
        # Initial status should be RAW
        chunk = await mongo_db.chunks.find_one({"chunkId": "status-test-chunk"})
        assert chunk["status"] == AnnotationStatus.RAW.value
        
        # Annotate the chunk
        response = await async_client.post(
            "/annotation/status-test-chunk",
            headers=admin_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Status should be ANNOTATED after successful annotation
        assert data["status"] == AnnotationStatus.ANNOTATED.value
        
        # Verify in database
        updated_chunk = await mongo_db.chunks.find_one({"chunkId": "status-test-chunk"})
        assert updated_chunk["status"] == AnnotationStatus.ANNOTATED.value
        assert updated_chunk.get("annotation") is not None
        assert updated_chunk.get("last_annotated_at") is not None
    
    async def test_annotation_updates_timestamp(
        self, async_client: AsyncClient, admin_headers, clean_db, mongo_db
    ):
        """Test that annotation updates last_annotated_at timestamp."""
        chunk_data = {
            "chunkId": "timestamp-test-chunk",
            "chunk": "(= (test $x) $x)",
            "source": "code",
            "isEmbedded": False,
            "status": AnnotationStatus.RAW.value
        }
        
        await mongo_db.chunks.insert_one(chunk_data)
        
        # Annotate
        response = await async_client.post(
            "/annotation/timestamp-test-chunk",
            headers=admin_headers
        )
        
        assert response.status_code == 200
        
        # Verify timestamp was set
        chunk = await mongo_db.chunks.find_one({"chunkId": "timestamp-test-chunk"})
        assert chunk.get("last_annotated_at") is not None
        assert isinstance(chunk["last_annotated_at"], (int, float))


@pytest.mark.asyncio
@pytest.mark.integration
class TestAnnotationEndToEnd:
    """Test complete annotation workflows."""
    
    async def test_annotate_then_batch_retry(
        self, async_client: AsyncClient, admin_headers, clean_db, mongo_db
    ):
        """Test annotating a chunk, then using batch retry."""
        # Create a chunk
        chunk_data = {
            "chunkId": "workflow-chunk-1",
            "chunk": "(= (test-function $x) (* $x 2))",
            "source": "code",
            "isEmbedded": False,
            "status": AnnotationStatus.RAW.value
        }
        
        await mongo_db.chunks.insert_one(chunk_data)
        
        # Step 1: Annotate single chunk
        response1 = await async_client.post(
            "/annotation/workflow-chunk-1",
            headers=admin_headers
        )
        assert response1.status_code == 200
        assert response1.json()["status"] == AnnotationStatus.ANNOTATED.value
        
        # Step 2: Create a failed chunk
        failed_chunk = {
            "chunkId": "workflow-chunk-2",
            "chunk": "(= (another-test $x) $x)",
            "source": "code",
            "isEmbedded": False,
            "status": AnnotationStatus.FAILED_GEN.value,
            "retry_count": 1
        }
        await mongo_db.chunks.insert_one(failed_chunk)
        
        # Step 3: Trigger batch retry
        response2 = await async_client.post(
            "/annotation/batch/retry_failed",
            headers=admin_headers
        )
        assert response2.status_code == 202
    
    async def test_multiple_chunks_annotation(
        self, async_client: AsyncClient, admin_headers, clean_db, mongo_db
    ):
        """Test annotating multiple chunks individually."""
        chunks = [
            {
                "chunkId": f"multi-chunk-{i}",
                "chunk": f"(= (function-{i} $x) (* $x {i}))",
                "source": "code",
                "isEmbedded": False,
                "status": AnnotationStatus.RAW.value
            }
            for i in range(3)
        ]
        
        await mongo_db.chunks.insert_many(chunks)
        
        # Annotate each chunk
        for i in range(3):
            response = await async_client.post(
                f"/annotation/multi-chunk-{i}",
                headers=admin_headers
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == AnnotationStatus.ANNOTATED.value
            # Check for annotation field (description is an alias)
            annotation = data.get("description") or data.get("annotation")
            assert annotation is not None
        
        # Verify all chunks are annotated in database
        for i in range(3):
            chunk = await mongo_db.chunks.find_one({"chunkId": f"multi-chunk-{i}"})
            assert chunk["status"] == AnnotationStatus.ANNOTATED.value
            assert chunk.get("annotation") is not None

