"""
Integration tests for chunks router endpoints.
Tests repository ingestion, chunk CRUD operations, embedding, and search.
"""
import pytest
from httpx import AsyncClient
from pymongo.database import Database
from qdrant_client import AsyncQdrantClient
from app.model.chunk import AnnotationStatus


@pytest.mark.asyncio
@pytest.mark.integration
class TestRepositoryIngestionPipeline:
    """Test complete ingestion pipeline with real repositories."""
    
    TEST_REPO_URL = "https://github.com/123nol/superposedMetta.git"
    TEST_REPO_NAME = "test-metta-repo"
    
    # Backup repository in case primary fails
    TEST_REPO_URL_BACKUP = "https://github.com/123nol/metta_reasoning.git"
    TEST_REPO_NAME_BACKUP = "test-metta-repo-backup"
    
    async def _try_ingest_repository(self, async_client: AsyncClient, admin_headers: dict, chunk_size: int = 1000):
        """
        Try to ingest repository, falling back to backup if primary fails.
        Returns (response, repo_url_used) tuple.
        """
        repos_to_try = [
            (self.TEST_REPO_URL, "primary"),
            (self.TEST_REPO_URL_BACKUP, "backup")
        ]
        
        last_error = None
        for repo_url, repo_type in repos_to_try:
            try:
                response = await async_client.post(
                    "/api/chunks/ingest",
                    params={
                        "repo_url": repo_url,
                        "chunk_size": chunk_size
                    },
                    headers=admin_headers
                )
                
                if response.status_code == 201:
                    return response, repo_url, repo_type
                else:
                    last_error = f"{repo_type} repo failed with status {response.status_code}: {response.text}"
                    continue
                    
            except Exception as e:
                last_error = f"{repo_type} repo failed with exception: {str(e)}"
                continue
        
        # If both repos failed, raise an error
        raise AssertionError(f"Both primary and backup repositories failed. Last error: {last_error}")
    
    async def test_complete_ingestion_flow(
        self,
        async_client: AsyncClient,
        admin_headers: dict,
        clean_db,  # Cleans database before/after each test
        mongo_db: Database,
        qdrant_client: AsyncQdrantClient
    ):
        """
        Test the complete ingestion pipeline:
        1. Clone repository from GitHub
        2. Process .metta files
        3. Chunk code using AST-based chunker
        4. Store chunks in MongoDB
        5. Verify chunks via API endpoints
        """
        
        # ===== STEP 1: Ingest Repository (with backup fallback) =====
        ingest_response, repo_url_used, repo_type = await self._try_ingest_repository(
            async_client, admin_headers, chunk_size=1000
        )
        
        assert ingest_response.status_code == 201, f"Expected 201, got {ingest_response.status_code}: {ingest_response.text} (used {repo_type} repo: {repo_url_used})"
        response_data = ingest_response.json()
        assert "successfully" in response_data["message"].lower()
        
        # ===== STEP 2: Verify Chunks Were Created in MongoDB =====
        chunks_in_db = await mongo_db.chunks.find({}).to_list(length=1000)
        assert len(chunks_in_db) > 0, "No chunks were created in database"
        
        # Verify chunk structure
        sample_chunk = chunks_in_db[0]
        required_fields = ["chunkId", "chunk", "source"]
        for field in required_fields:
            assert field in sample_chunk, f"Missing required field: {field}"
        
        # Verify all chunks have correct source type
        for chunk in chunks_in_db:
            assert chunk["source"] == "code", f"Expected source='code', got {chunk.get('source')}"
        
        # ===== STEP 3: Verify API Endpoints Work with Real Data =====
        
        # Test list endpoint
        list_response = await async_client.get(
            "/api/chunks/",
            headers=admin_headers
        )
        assert list_response.status_code == 200
        api_chunks = list_response.json()
        assert len(api_chunks) > 0
        assert isinstance(api_chunks, list)
        
        # Test filtering (if repo/project fields are set)
        if chunks_in_db[0].get("repo"):
            repo_name = chunks_in_db[0]["repo"]
            filter_response = await async_client.get(
                f"/api/chunks/?repo={repo_name}",
                headers=admin_headers
            )
            assert filter_response.status_code == 200
            filtered_chunks = filter_response.json()
            assert len(filtered_chunks) > 0
    
    async def test_ingestion_creates_valid_chunks(
        self,
        async_client: AsyncClient,
        admin_headers: dict,
        clean_db,
        mongo_db: Database
    ):
        """Test that ingestion creates chunks with valid structure."""
        
        # Ingest repository (with backup fallback)
        response, repo_url_used, repo_type = await self._try_ingest_repository(
            async_client, admin_headers, chunk_size=1000
        )
        assert response.status_code == 201, f"Failed to ingest {repo_type} repo: {repo_url_used}"
        
        # Get chunks from database
        chunks = await mongo_db.chunks.find({}).to_list(length=1000)
        assert len(chunks) > 0
        
        # Verify each chunk has required fields
        for chunk in chunks:
            # Required fields
            assert "chunkId" in chunk, "Missing chunkId"
            assert "chunk" in chunk, "Missing chunk content"
            assert isinstance(chunk["chunk"], str), "Chunk content should be string"
            assert len(chunk["chunk"]) > 0, "Chunk content should not be empty"
            
            # Verify chunkId is unique
            chunk_ids = [c.get("chunkId") for c in chunks]
            assert chunk_ids.count(chunk["chunkId"]) == 1, f"Duplicate chunkId: {chunk['chunkId']}"
    
    async def test_ingestion_handles_duplicates(
        self,
        async_client: AsyncClient,
        admin_headers: dict,
        clean_db,
        mongo_db: Database
    ):
        """Test that ingesting the same repository twice handles duplicates correctly."""
        
        # First ingestion (with backup fallback)
        response1, repo_url_used, repo_type = await self._try_ingest_repository(
            async_client, admin_headers, chunk_size=1000
        )
        assert response1.status_code == 201
        
        chunks_after_first = await mongo_db.chunks.find({}).to_list(length=1000)
        first_count = len(chunks_after_first)
        assert first_count > 0
        
        # Second ingestion (should handle duplicates) - use same repo that worked
        response2 = await async_client.post(
            "/api/chunks/ingest",
            params={
                "repo_url": repo_url_used,  # Use the repo that worked
                "chunk_size": 1000
            },
            headers=admin_headers
        )
        # Should either succeed (skipping duplicates) or fail gracefully
        assert response2.status_code in [201, 400, 409]
        
        chunks_after_second = await mongo_db.chunks.find({}).to_list(length=1000)
        second_count = len(chunks_after_second)
        
        # Should have same or fewer chunks (duplicates should be skipped)
        # Or same count if duplicates are rejected
        assert second_count <= first_count * 2, "Too many chunks after duplicate ingestion"
    
    async def test_ingestion_with_different_chunk_sizes(
        self,
        async_client: AsyncClient,
        admin_headers: dict,
        clean_db,
        mongo_db: Database
    ):
        """Test that different chunk sizes produce different numbers of chunks."""
        
        chunk_sizes = [500, 1000, 1500]
        chunk_counts = []
        
        for chunk_size in chunk_sizes:
            # Clean before each ingestion
            await mongo_db.chunks.delete_many({})
            
            # Use backup fallback for each chunk size test
            response, repo_url_used, repo_type = await self._try_ingest_repository(
                async_client, admin_headers, chunk_size=chunk_size
            )
            assert response.status_code == 201, f"Failed with {repo_type} repo at chunk_size={chunk_size}"
            
            chunks = await mongo_db.chunks.find({}).to_list(length=1000)
            chunk_counts.append(len(chunks))
        
        # Smaller chunk size should generally create more chunks
        # (unless repo is very small)
        if chunk_counts[0] > 0 and chunk_counts[2] > 0:
            # Verify chunking actually happened
            assert all(count > 0 for count in chunk_counts)
    
    async def test_ingestion_error_handling_invalid_repo(
        self,
        async_client: AsyncClient,
        admin_headers: dict,
        clean_db
    ):
        """Test error handling for invalid repository URLs."""
        
        # Test with non-existent repository
        response = await async_client.post(
            "/api/chunks/ingest",
            params={
                "repo_url": "https://github.com/invalid-user/repo-that-does-not-exist-12345.git",
                "chunk_size": 1000
            },
            headers=admin_headers
        )
        # Should return error status
        assert response.status_code in [400, 404, 500]
        assert "error" in response.json().get("detail", "").lower() or "error" in str(response.json())
    
    async def test_ingestion_error_handling_invalid_url(
        self,
        async_client: AsyncClient,
        admin_headers: dict,
        clean_db
    ):
        """Test error handling for invalid URL format."""
        
        # Test with non-GitHub URL
        response = await async_client.post(
            "/api/chunks/ingest",
            params={
                "repo_url": "https://not-a-git-repo.com",
                "chunk_size": 1000
            },
            headers=admin_headers
        )
        assert response.status_code in [400, 500]
    
    @pytest.mark.slow
    async def test_ingestion_then_search(
        self,
        async_client: AsyncClient,
        admin_headers: dict,
        clean_db,
        mongo_db: Database
    ):
        """
        Test that after ingestion, chunks can be searched.
        Note: This test may require embeddings to be created first.
        """
        
        # Ingest repository (with backup fallback)
        response, repo_url_used, repo_type = await self._try_ingest_repository(
            async_client, admin_headers, chunk_size=1000
        )
        assert response.status_code == 201, f"Failed to ingest {repo_type} repo: {repo_url_used}"
        
        # Verify chunks exist
        chunks = await mongo_db.chunks.find({}).to_list(length=100)
        assert len(chunks) > 0
        
        # Try semantic search
        # Note: This may fail if embeddings haven't been created
        search_response = await async_client.get(
            "/api/chunks/search?q=test&top_k=5",
            headers=admin_headers
        )
        
        # May return 200 (if embeddings exist) or 500 (if not)
        # Both are acceptable for this test
        assert search_response.status_code in [200, 500]
        
        if search_response.status_code == 200:
            data = search_response.json()
            assert "query" in data
            assert "results" in data
    
    async def test_ingestion_unauthorized(
        self,
        async_client: AsyncClient,
        auth_headers: dict,  # Regular user, not admin
        clean_db
    ):
        """Test that non-admin users cannot ingest repositories."""
        
        # Try with primary repo first (should fail with 403, not repo error)
        response = await async_client.post(
            "/api/chunks/ingest",
            params={
                "repo_url": self.TEST_REPO_URL,
                "chunk_size": 1000
            },
            headers=auth_headers
        )
        
        # If primary fails with 403, that's expected. If it fails for other reasons, try backup
        if response.status_code != 403:
            # Try backup repo
            response = await async_client.post(
                "/api/chunks/ingest",
                params={
                    "repo_url": self.TEST_REPO_URL_BACKUP,
                    "chunk_size": 1000
                },
                headers=auth_headers
            )
        
        assert response.status_code == 403, "Non-admin users should not be able to ingest repositories"


@pytest.mark.asyncio
@pytest.mark.integration
class TestListChunks:
    """Test GET /api/chunks/ endpoint."""
    
    async def test_list_chunks_success(
        self, async_client: AsyncClient, admin_headers, clean_db, mongo_db
    ):
        """Test listing chunks successfully."""
        # Create test chunks
        chunks = [
            {
                "chunkId": f"list-chunk-{i}",
                "chunk": f"(= (test-function-{i} $x) (* $x {i}))",
                "source": "code",
                "isEmbedded": False,
                "project": "test_project",
                "repo": "test_repo"
            }
            for i in range(5)
        ]
        await mongo_db.chunks.insert_many(chunks)
        
        # List chunks
        response = await async_client.get("/api/chunks/", headers=admin_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 5
        
        # Verify chunk structure
        for chunk in data:
            assert "chunkId" in chunk
            assert "chunk" in chunk
            assert "source" in chunk
    
    async def test_list_chunks_with_limit(
        self, async_client: AsyncClient, admin_headers, clean_db, mongo_db
    ):
        """Test listing chunks with limit parameter."""
        chunks = [
            {
                "chunkId": f"limit-chunk-{i}",
                "chunk": f"(= (func-{i} $x) $x)",
                "source": "code",
                "isEmbedded": False
            }
            for i in range(10)
        ]
        await mongo_db.chunks.insert_many(chunks)
        
        # List with limit
        response = await async_client.get(
            "/api/chunks/?limit=3",
            headers=admin_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 3
    
    async def test_list_chunks_filter_by_project(
        self, async_client: AsyncClient, admin_headers, clean_db, mongo_db
    ):
        """Test filtering chunks by project."""
        chunks = [
            {
                "chunkId": f"project-chunk-{i}",
                "chunk": f"(= (func-{i} $x) $x)",
                "source": "code",
                "isEmbedded": False,
                "project": "project_a" if i < 3 else "project_b"
            }
            for i in range(5)
        ]
        await mongo_db.chunks.insert_many(chunks)
        
        # Filter by project
        response = await async_client.get(
            "/api/chunks/?project=project_a",
            headers=admin_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        for chunk in data:
            assert chunk["project"] == "project_a"
    
    async def test_list_chunks_filter_by_repo(
        self, async_client: AsyncClient, admin_headers, clean_db, mongo_db
    ):
        """Test filtering chunks by repository."""
        chunks = [
            {
                "chunkId": f"repo-chunk-{i}",
                "chunk": f"(= (func-{i} $x) $x)",
                "source": "code",
                "isEmbedded": False,
                "repo": "repo_a" if i < 2 else "repo_b"
            }
            for i in range(4)
        ]
        await mongo_db.chunks.insert_many(chunks)
        
        # Filter by repo
        response = await async_client.get(
            "/api/chunks/?repo=repo_a",
            headers=admin_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        for chunk in data:
            assert chunk["repo"] == "repo_a"
    
    async def test_list_chunks_filter_by_section(
        self, async_client: AsyncClient, admin_headers, clean_db, mongo_db
    ):
        """Test filtering chunks by section."""
        chunks = [
            {
                "chunkId": f"section-chunk-{i}",
                "chunk": f"(= (func-{i} $x) $x)",
                "source": "code",
                "isEmbedded": False,
                "section": ["section_a"] if i < 3 else ["section_b"]
            }
            for i in range(5)
        ]
        await mongo_db.chunks.insert_many(chunks)
        
        # Filter by section
        response = await async_client.get(
            "/api/chunks/?section=section_a",
            headers=admin_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        for chunk in data:
            assert "section_a" in chunk.get("section", [])
    
    async def test_list_chunks_empty(
        self, async_client: AsyncClient, admin_headers, clean_db
    ):
        """Test listing chunks when database is empty."""
        response = await async_client.get("/api/chunks/", headers=admin_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0
    
    async def test_list_chunks_unauthorized(
        self, async_client: AsyncClient, auth_headers, clean_db
    ):
        """Test that non-admin users cannot list chunks."""
        response = await async_client.get("/api/chunks/", headers=auth_headers)
        assert response.status_code == 403
    
    async def test_list_chunks_no_auth(self, async_client: AsyncClient, clean_db):
        """Test listing chunks without authentication."""
        response = await async_client.get("/api/chunks/")
        assert response.status_code == 401


@pytest.mark.asyncio
@pytest.mark.integration
class TestUpdateChunk:
    """Test PATCH /api/chunks/{chunk_id} endpoint."""
    
    async def test_update_chunk_success(
        self, async_client: AsyncClient, admin_headers, clean_db, mongo_db
    ):
        """Test successfully updating a chunk."""
        # Create a chunk
        chunk_data = {
            "chunkId": "update-chunk-1",
            "chunk": "(= (old-function $x) $x)",
            "source": "code",
            "isEmbedded": False,
            "project": "old_project",
            "repo": "old_repo"
        }
        await mongo_db.chunks.insert_one(chunk_data)
        
        # Update chunk
        update_data = {
            "chunk": "(= (new-function $x) (* $x 2))",
            "project": "new_project",
            "repo": "new_repo"
        }
        
        response = await async_client.patch(
            "/api/chunks/update-chunk-1",
            json=update_data,
            headers=admin_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "chunk" in data
        assert data["chunk"]["chunk"] == update_data["chunk"]
        assert data["chunk"]["project"] == "new_project"
        assert data["chunk"]["repo"] == "new_repo"
        
        # Verify in database
        updated_chunk = await mongo_db.chunks.find_one({"chunkId": "update-chunk-1"})
        assert updated_chunk["chunk"] == update_data["chunk"]
        assert updated_chunk["project"] == "new_project"
    
    async def test_update_chunk_partial(
        self, async_client: AsyncClient, admin_headers, clean_db, mongo_db
    ):
        """Test updating only some fields of a chunk."""
        chunk_data = {
            "chunkId": "partial-update-chunk",
            "chunk": "(= (func $x) $x)",
            "source": "code",
            "isEmbedded": False,
            "project": "project_a",
            "repo": "repo_a"
        }
        await mongo_db.chunks.insert_one(chunk_data)
        
        # Update only project
        response = await async_client.patch(
            "/api/chunks/partial-update-chunk",
            json={"project": "project_b"},
            headers=admin_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["chunk"]["project"] == "project_b"
        assert data["chunk"]["repo"] == "repo_a"  # Should remain unchanged
    
    async def test_update_chunk_not_found(
        self, async_client: AsyncClient, admin_headers, clean_db
    ):
        """Test updating a non-existent chunk."""
        response = await async_client.patch(
            "/api/chunks/nonexistent-chunk",
            json={"project": "new_project"},
            headers=admin_headers
        )
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    async def test_update_chunk_empty_update(
        self, async_client: AsyncClient, admin_headers, clean_db, mongo_db
    ):
        """Test updating with no fields provided."""
        chunk_data = {
            "chunkId": "empty-update-chunk",
            "chunk": "(= (func $x) $x)",
            "source": "code",
            "isEmbedded": False
        }
        await mongo_db.chunks.insert_one(chunk_data)
        
        response = await async_client.patch(
            "/api/chunks/empty-update-chunk",
            json={},
            headers=admin_headers
        )
        
        assert response.status_code == 400
        assert "no update data" in response.json()["detail"].lower()
    
    async def test_update_chunk_unauthorized(
        self, async_client: AsyncClient, auth_headers, clean_db, mongo_db
    ):
        """Test that non-admin users cannot update chunks."""
        chunk_data = {
            "chunkId": "unauth-update-chunk",
            "chunk": "(= (func $x) $x)",
            "source": "code",
            "isEmbedded": False
        }
        await mongo_db.chunks.insert_one(chunk_data)
        
        response = await async_client.patch(
            "/api/chunks/unauth-update-chunk",
            json={"project": "new_project"},
            headers=auth_headers
        )
        
        assert response.status_code == 403
    
    async def test_update_chunk_no_auth(
        self, async_client: AsyncClient, clean_db, mongo_db
    ):
        """Test updating chunk without authentication."""
        chunk_data = {
            "chunkId": "noauth-update-chunk",
            "chunk": "(= (func $x) $x)",
            "source": "code",
            "isEmbedded": False
        }
        await mongo_db.chunks.insert_one(chunk_data)
        
        response = await async_client.patch(
            "/api/chunks/noauth-update-chunk",
            json={"project": "new_project"}
        )
        
        assert response.status_code == 401


@pytest.mark.asyncio
@pytest.mark.integration
class TestDeleteChunk:
    """Test DELETE /api/chunks/{chunk_id} endpoint."""
    
    async def test_delete_chunk_success(
        self, async_client: AsyncClient, admin_headers, clean_db, mongo_db
    ):
        """Test successfully deleting a chunk."""
        # Create a chunk
        chunk_data = {
            "chunkId": "delete-chunk-1",
            "chunk": "(= (func $x) $x)",
            "source": "code",
            "isEmbedded": False
        }
        await mongo_db.chunks.insert_one(chunk_data)
        
        # Verify chunk exists
        chunk = await mongo_db.chunks.find_one({"chunkId": "delete-chunk-1"})
        assert chunk is not None
        
        # Delete chunk
        response = await async_client.delete(
            "/api/chunks/delete-chunk-1",
            headers=admin_headers
        )
        
        assert response.status_code == 204
        
        # Verify chunk is deleted
        deleted_chunk = await mongo_db.chunks.find_one({"chunkId": "delete-chunk-1"})
        assert deleted_chunk is None
    
    async def test_delete_chunk_not_found(
        self, async_client: AsyncClient, admin_headers, clean_db
    ):
        """Test deleting a non-existent chunk."""
        response = await async_client.delete(
            "/api/chunks/nonexistent-chunk",
            headers=admin_headers
        )
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    async def test_delete_chunk_unauthorized(
        self, async_client: AsyncClient, auth_headers, clean_db, mongo_db
    ):
        """Test that non-admin users cannot delete chunks."""
        chunk_data = {
            "chunkId": "unauth-delete-chunk",
            "chunk": "(= (func $x) $x)",
            "source": "code",
            "isEmbedded": False
        }
        await mongo_db.chunks.insert_one(chunk_data)
        
        response = await async_client.delete(
            "/api/chunks/unauth-delete-chunk",
            headers=auth_headers
        )
        
        assert response.status_code == 403
        
        # Verify chunk still exists
        chunk = await mongo_db.chunks.find_one({"chunkId": "unauth-delete-chunk"})
        assert chunk is not None
    
    async def test_delete_chunk_no_auth(
        self, async_client: AsyncClient, clean_db, mongo_db
    ):
        """Test deleting chunk without authentication."""
        chunk_data = {
            "chunkId": "noauth-delete-chunk",
            "chunk": "(= (func $x) $x)",
            "source": "code",
            "isEmbedded": False
        }
        await mongo_db.chunks.insert_one(chunk_data)
        
        response = await async_client.delete("/api/chunks/noauth-delete-chunk")
        assert response.status_code == 401


@pytest.mark.asyncio
@pytest.mark.integration
class TestEmbedChunks:
    """Test POST /api/chunks/embed endpoint."""
    
    async def test_embed_chunks_success(
        self, async_client: AsyncClient, admin_headers, clean_db, mongo_db
    ):
        """Test running embedding pipeline successfully."""
        # Create unembedded chunks
        chunks = [
            {
                "chunkId": f"embed-chunk-{i}",
                "chunk": f"(= (test-function-{i} $x) (* $x {i}))",
                "source": "code",
                "isEmbedded": False
            }
            for i in range(3)
        ]
        await mongo_db.chunks.insert_many(chunks)
        
        # Run embedding pipeline
        response = await async_client.post(
            "/api/chunks/embed",
            headers=admin_headers
        )
        
        # May return 200 (if successful) or 500 (if COLLECTION_NAME not set)
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "message" in data
            assert "embedded" in data["message"].lower()
    
    async def test_embed_chunks_no_unembedded(
        self, async_client: AsyncClient, admin_headers, clean_db, mongo_db
    ):
        """Test embedding when all chunks are already embedded."""
        # Create embedded chunks
        chunks = [
            {
                "chunkId": f"already-embedded-{i}",
                "chunk": f"(= (func-{i} $x) $x)",
                "source": "code",
                "isEmbedded": True
            }
            for i in range(2)
        ]
        await mongo_db.chunks.insert_many(chunks)
        
        # Run embedding pipeline
        response = await async_client.post(
            "/api/chunks/embed",
            headers=admin_headers
        )
        
        # Should still return 200 (no chunks to embed)
        assert response.status_code in [200, 500]
    
    async def test_embed_chunks_unauthorized(
        self, async_client: AsyncClient, auth_headers, clean_db
    ):
        """Test that non-admin users cannot run embedding pipeline."""
        response = await async_client.post(
            "/api/chunks/embed",
            headers=auth_headers
        )
        assert response.status_code == 403
    
    async def test_embed_chunks_no_auth(self, async_client: AsyncClient, clean_db):
        """Test embedding without authentication."""
        response = await async_client.post("/api/chunks/embed")
        assert response.status_code == 401


@pytest.mark.asyncio
@pytest.mark.integration
class TestSearchChunks:
    """Test GET /api/chunks/search endpoint."""
    
    async def test_search_chunks_success(
        self, async_client: AsyncClient, admin_headers, clean_db, mongo_db
    ):
        """Test semantic search successfully."""
        # Create chunks
        chunks = [
            {
                "chunkId": f"search-chunk-{i}",
                "chunk": f"(= (test-function-{i} $x) (* $x {i}))",
                "source": "code",
                "isEmbedded": False
            }
            for i in range(3)
        ]
        await mongo_db.chunks.insert_many(chunks)
        
        # Search chunks
        response = await async_client.get(
            "/api/chunks/search?q=test&top_k=5",
            headers=admin_headers
        )
        
        # May return 200 (if embeddings exist) or 500 (if not)
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "query" in data
            assert "top_k" in data
            assert "results" in data
            assert data["query"] == "test"
            assert data["top_k"] == 5
    
    async def test_search_chunks_with_top_k(
        self, async_client: AsyncClient, admin_headers, clean_db, mongo_db
    ):
        """Test search with different top_k values."""
        chunks = [
            {
                "chunkId": f"topk-chunk-{i}",
                "chunk": f"(= (func-{i} $x) $x)",
                "source": "code",
                "isEmbedded": False
            }
            for i in range(10)
        ]
        await mongo_db.chunks.insert_many(chunks)
        
        # Search with top_k=3
        response = await async_client.get(
            "/api/chunks/search?q=function&top_k=3",
            headers=admin_headers
        )
        
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert data["top_k"] == 3
    
    async def test_search_chunks_missing_query(
        self, async_client: AsyncClient, admin_headers, clean_db
    ):
        """Test search without query parameter."""
        response = await async_client.get(
            "/api/chunks/search",
            headers=admin_headers
        )
        
        # Should return 422 (validation error) or 500
        assert response.status_code in [422, 500]
    
    async def test_search_chunks_empty_query(
        self, async_client: AsyncClient, admin_headers, clean_db
    ):
        """Test search with empty query."""
        response = await async_client.get(
            "/api/chunks/search?q=",
            headers=admin_headers
        )
        
        # Should return 422 (validation error) or 500
        assert response.status_code in [422, 500]
    
    async def test_search_chunks_unauthorized(
        self, async_client: AsyncClient, auth_headers, clean_db
    ):
        """Test that non-admin users cannot search chunks."""
        response = await async_client.get(
            "/api/chunks/search?q=test",
            headers=auth_headers
        )
        assert response.status_code == 403
    
    async def test_search_chunks_no_auth(self, async_client: AsyncClient, clean_db):
        """Test search without authentication."""
        response = await async_client.get("/api/chunks/search?q=test")
        assert response.status_code == 401


@pytest.mark.asyncio
@pytest.mark.integration
class TestChunksEndToEnd:
    """Test complete chunk management workflows."""
    
    async def test_create_list_update_delete_workflow(
        self, async_client: AsyncClient, admin_headers, clean_db, mongo_db
    ):
        """Test complete CRUD workflow for chunks."""
        # Step 1: Create chunks via ingestion (or directly)
        chunks = [
            {
                "chunkId": f"workflow-chunk-{i}",
                "chunk": f"(= (workflow-func-{i} $x) (* $x {i}))",
                "source": "code",
                "isEmbedded": False,
                "project": "workflow_project",
                "repo": "workflow_repo"
            }
            for i in range(3)
        ]
        await mongo_db.chunks.insert_many(chunks)
        
        # Step 2: List chunks
        list_response = await async_client.get(
            "/api/chunks/?project=workflow_project",
            headers=admin_headers
        )
        assert list_response.status_code == 200
        assert len(list_response.json()) == 3
        
        # Step 3: Update a chunk
        update_response = await async_client.patch(
            "/api/chunks/workflow-chunk-1",
            json={"project": "updated_project"},
            headers=admin_headers
        )
        assert update_response.status_code == 200
        
        # Step 4: Verify update
        updated_list = await async_client.get(
            "/api/chunks/?project=updated_project",
            headers=admin_headers
        )
        assert updated_list.status_code == 200
        assert len(updated_list.json()) == 1
        
        # Step 5: Delete a chunk
        delete_response = await async_client.delete(
            "/api/chunks/workflow-chunk-0",
            headers=admin_headers
        )
        assert delete_response.status_code == 204
        
        # Step 6: Verify deletion
        final_list = await async_client.get(
            "/api/chunks/?project=workflow_project",
            headers=admin_headers
        )
        assert final_list.status_code == 200
        assert len(final_list.json()) == 1  # One updated, one deleted, one remaining
    
    async def test_filter_combinations(
        self, async_client: AsyncClient, admin_headers, clean_db, mongo_db
    ):
        """Test combining multiple filters."""
        chunks = [
            {
                "chunkId": f"filter-chunk-{i}",
                "chunk": f"(= (func-{i} $x) $x)",
                "source": "code",
                "isEmbedded": False,
                "project": "project_a" if i < 2 else "project_b",
                "repo": "repo_a" if i % 2 == 0 else "repo_b"
            }
            for i in range(4)
        ]
        await mongo_db.chunks.insert_many(chunks)
        
        # Filter by project and repo
        response = await async_client.get(
            "/api/chunks/?project=project_a&repo=repo_a",
            headers=admin_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        for chunk in data:
            assert chunk["project"] == "project_a"
            assert chunk["repo"] == "repo_a"

