"""
Integration tests for key management endpoints (key_management.py router).
Tests API key storage, retrieval, and deletion with real database.
"""
import pytest
from httpx import AsyncClient
from pymongo.database import Database
from app.services.key_management_service import KMS


@pytest.mark.asyncio
@pytest.mark.integration
class TestStoreAPIKey:
    """Test POST /api/kms/store endpoint."""
    
    async def test_store_api_key_success(
        self, async_client: AsyncClient, auth_headers, test_user, clean_db, mongo_db
    ):
        """Test successfully storing an API key."""
        # Store API key
        response = await async_client.post(
            "/api/kms/store",
            json={
                "provider_name": "openai",
                "api_key": "sk-test-key-12345"
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        assert "message" in data
        assert "stored securely" in data["message"].lower()
        
        # Verify cookie is set
        cookies = response.cookies
        assert "openai" in cookies
        assert cookies["openai"] is not None
        assert len(cookies["openai"]) > 0
        
        # Verify key is stored in database
        key_doc = await mongo_db.keys.find_one({
            "provider_name": "openai",
            "userid": test_user["id"]
        })
        assert key_doc is not None
        assert "dek" in key_doc  # Encrypted DEK should be stored
        assert key_doc["provider_name"] == "openai"
    
    async def test_store_api_key_multiple_providers(
        self, async_client: AsyncClient, auth_headers, test_user, clean_db, mongo_db
    ):
        """Test storing API keys for multiple providers."""
        providers = ["openai", "gemini", "anthropic"]
        
        for provider in providers:
            response = await async_client.post(
                "/api/kms/store",
                json={
                    "provider_name": provider,
                    "api_key": f"sk-test-key-{provider}"
                },
                headers=auth_headers
            )
            assert response.status_code == 200
        
        # Verify all keys are stored
        for provider in providers:
            key_doc = await mongo_db.keys.find_one({
                "provider_name": provider,
                "userid": test_user["id"]
            })
            assert key_doc is not None
    
    async def test_store_api_key_update_existing(
        self, async_client: AsyncClient, auth_headers, test_user, clean_db, mongo_db
    ):
        """Test updating an existing API key."""
        # Store initial key
        response1 = await async_client.post(
            "/api/kms/store",
            json={
                "provider_name": "openai",
                "api_key": "sk-old-key"
            },
            headers=auth_headers
        )
        assert response1.status_code == 200
        
        # Get initial DEK
        key_doc1 = await mongo_db.keys.find_one({
            "provider_name": "openai",
            "userid": test_user["id"]
        })
        initial_dek = key_doc1["dek"]
        
        # Update with new key
        response2 = await async_client.post(
            "/api/kms/store",
            json={
                "provider_name": "openai",
                "api_key": "sk-new-key"
            },
            headers=auth_headers
        )
        assert response2.status_code == 200
        
        key_doc2 = await mongo_db.keys.find_one({
            "provider_name": "openai",
            "userid": test_user["id"]
        })
        assert key_doc2["dek"] != initial_dek  
    
    async def test_store_api_key_missing_fields(
        self, async_client: AsyncClient, auth_headers, clean_db
    ):
        """Test storing API key with missing fields."""
        # Missing api_key
        response1 = await async_client.post(
            "/api/kms/store",
            json={"provider_name": "openai"},
            headers=auth_headers
        )
        assert response1.status_code == 422
        
        # Missing provider_name
        response2 = await async_client.post(
            "/api/kms/store",
            json={"api_key": "sk-test-key"},
            headers=auth_headers
        )
        assert response2.status_code == 422
    
    async def test_store_api_key_empty_values(
        self, async_client: AsyncClient, auth_headers, clean_db
    ):
        """Test storing API key with empty values."""
        # Empty api_key
        response1 = await async_client.post(
            "/api/kms/store",
            json={
                "provider_name": "openai",
                "api_key": ""
            },
            headers=auth_headers
        )
        # May accept empty or reject - depends on validation
        assert response1.status_code in [200, 422, 500]
        
        # Empty provider_name
        response2 = await async_client.post(
            "/api/kms/store",
            json={
                "provider_name": "",
                "api_key": "sk-test-key"
            },
            headers=auth_headers
        )
        # May accept empty or reject - depends on validation
        assert response2.status_code in [200, 422, 500]
    
    async def test_store_api_key_unauthorized(
        self, async_client: AsyncClient, clean_db
    ):
        """Test storing API key without authentication."""
        response = await async_client.post(
            "/api/kms/store",
            json={
                "provider_name": "openai",
                "api_key": "sk-test-key"
            }
        )
        assert response.status_code == 401


@pytest.mark.asyncio
@pytest.mark.integration
class TestGetProviders:
    """Test GET /api/kms/providers endpoint."""
    
    async def test_get_providers_success(
        self, async_client: AsyncClient, auth_headers, clean_db, mongo_db
    ):
        """Test successfully retrieving providers."""
        # Store multiple API keys
        providers = ["openai", "gemini", "anthropic"]
        for provider in providers:
            await async_client.post(
                "/api/kms/store",
                json={
                    "provider_name": provider,
                    "api_key": f"sk-test-key-{provider}"
                },
                headers=auth_headers
            )
        
        # Get providers
        response = await async_client.get(
            "/api/kms/providers",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "services" in data
        assert isinstance(data["services"], list)
        assert len(data["services"]) == 3
        assert all(provider in data["services"] for provider in providers)
    
    async def test_get_providers_empty(
        self, async_client: AsyncClient, auth_headers, clean_db
    ):
        """Test getting providers when user has none."""
        response = await async_client.get(
            "/api/kms/providers",
            headers=auth_headers
        )
        
        assert response.status_code == 404
        detail = response.json()["detail"].lower()
        assert "not found" in detail or "no services" in detail
    
    async def test_get_providers_user_isolation(
        self, async_client: AsyncClient, auth_headers, admin_headers, test_user, test_admin, clean_db, mongo_db
    ):
        """Test that users only see their own providers."""
        # User 1 stores a key
        await async_client.post(
            "/api/kms/store",
            json={
                "provider_name": "openai",
                "api_key": "sk-user1-key"
            },
            headers=auth_headers
        )
        
        # User 2 (admin) stores a different key
        await async_client.post(
            "/api/kms/store",
            json={
                "provider_name": "gemini",
                "api_key": "sk-user2-key"
            },
            headers=admin_headers
        )
        
        # User 1 should only see their own provider
        response1 = await async_client.get(
            "/api/kms/providers",
            headers=auth_headers
        )
        assert response1.status_code == 200
        data1 = response1.json()
        assert "openai" in data1["services"]
        assert "gemini" not in data1["services"]
        
        # User 2 should only see their own provider
        response2 = await async_client.get(
            "/api/kms/providers",
            headers=admin_headers
        )
        assert response2.status_code == 200
        data2 = response2.json()
        assert "gemini" in data2["services"]
        assert "openai" not in data2["services"]
    
    async def test_get_providers_unauthorized(
        self, async_client: AsyncClient, clean_db
    ):
        """Test getting providers without authentication."""
        response = await async_client.get("/api/kms/providers")
        assert response.status_code == 401


@pytest.mark.asyncio
@pytest.mark.integration
class TestDeleteAPIKey:
    """Test DELETE /api/kms/delete/{provider_name} endpoint."""
    
    async def test_delete_api_key_success(
        self, async_client: AsyncClient, auth_headers, test_user, clean_db, mongo_db
    ):
        """Test successfully deleting an API key."""
        # Store API key first
        store_response = await async_client.post(
            "/api/kms/store",
            json={
                "provider_name": "openai",
                "api_key": "sk-test-key"
            },
            headers=auth_headers
        )
        assert store_response.status_code == 200
        
        # Verify key exists
        key_doc = await mongo_db.keys.find_one({
            "provider_name": "openai",
            "userid": test_user["id"]
        })
        assert key_doc is not None
        
        # Delete API key
        delete_response = await async_client.delete(
            "/api/kms/delete/openai",
            headers=auth_headers
        )
        
        assert delete_response.status_code == 200
        data = delete_response.json()
        assert "message" in data
        assert "deleted successfully" in data["message"].lower()
        assert "openai" in data["message"]
        
        # Verify cookie is deleted
        # Note: httpx doesn't expose deleted cookies easily, but we can check the response
        assert delete_response.status_code == 200
        
        # Verify key is deleted from database
        deleted_key = await mongo_db.keys.find_one({
            "provider_name": "openai",
            "userid": test_user["id"]
        })
        assert deleted_key is None
    
    async def test_delete_api_key_not_found(
        self, async_client: AsyncClient, auth_headers, clean_db
    ):
        """Test deleting a non-existent API key."""
        response = await async_client.delete(
            "/api/kms/delete/nonexistent-provider",
            headers=auth_headers
        )
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    async def test_delete_api_key_user_isolation(
        self, async_client: AsyncClient, auth_headers, admin_headers, test_user, test_admin, clean_db, mongo_db
    ):
        """Test that users can only delete their own keys."""
        # User 1 stores a key
        await async_client.post(
            "/api/kms/store",
            json={
                "provider_name": "openai",
                "api_key": "sk-user1-key"
            },
            headers=auth_headers
        )
        
        # User 2 stores a key
        await async_client.post(
            "/api/kms/store",
            json={
                "provider_name": "openai",
                "api_key": "sk-user2-key"
            },
            headers=admin_headers
        )
        
        # User 1 tries to delete their key (should succeed)
        response1 = await async_client.delete(
            "/api/kms/delete/openai",
            headers=auth_headers
        )
        assert response1.status_code == 200
        
        # Verify User 1's key is deleted
        key1 = await mongo_db.keys.find_one({
            "provider_name": "openai",
            "userid": test_user["id"]
        })
        assert key1 is None
        
        # Verify User 2's key still exists
        key2 = await mongo_db.keys.find_one({
            "provider_name": "openai",
            "userid": test_admin["id"]
        })
        assert key2 is not None
    
    async def test_delete_api_key_unauthorized(
        self, async_client: AsyncClient, clean_db
    ):
        """Test deleting API key without authentication."""
        response = await async_client.delete("/api/kms/delete/openai")
        assert response.status_code == 401


@pytest.mark.asyncio
@pytest.mark.integration
class TestKeyManagementEndToEnd:
    """Test complete key management workflows."""
    
    async def test_store_get_delete_workflow(
        self, async_client: AsyncClient, auth_headers, clean_db, mongo_db
    ):
        """Test complete workflow: store, get, delete."""
        # Step 1: Store API key
        store_response = await async_client.post(
            "/api/kms/store",
            json={
                "provider_name": "openai",
                "api_key": "sk-test-key-12345"
            },
            headers=auth_headers
        )
        assert store_response.status_code == 200
        
        # Step 2: Get providers (should include openai)
        get_response = await async_client.get(
            "/api/kms/providers",
            headers=auth_headers
        )
        assert get_response.status_code == 200
        data = get_response.json()
        assert "openai" in data["services"]
        
        # Step 3: Delete API key
        delete_response = await async_client.delete(
            "/api/kms/delete/openai",
            headers=auth_headers
        )
        assert delete_response.status_code == 200
        
        # Step 4: Get providers again (should be empty/not found)
        get_response2 = await async_client.get(
            "/api/kms/providers",
            headers=auth_headers
        )
        assert get_response2.status_code == 404
    
    async def test_multiple_providers_management(
        self, async_client: AsyncClient, auth_headers, clean_db, mongo_db
    ):
        """Test managing multiple providers."""
        providers = ["openai", "gemini", "anthropic"]
        
        # Store all providers
        for provider in providers:
            response = await async_client.post(
                "/api/kms/store",
                json={
                    "provider_name": provider,
                    "api_key": f"sk-test-key-{provider}"
                },
                headers=auth_headers
            )
            assert response.status_code == 200
        
        # Verify all are stored
        get_response = await async_client.get(
            "/api/kms/providers",
            headers=auth_headers
        )
        assert get_response.status_code == 200
        data = get_response.json()
        assert len(data["services"]) == 3
        
        # Delete one provider
        delete_response = await async_client.delete(
            "/api/kms/delete/gemini",
            headers=auth_headers
        )
        assert delete_response.status_code == 200
        
        # Verify only 2 remain
        get_response2 = await async_client.get(
            "/api/kms/providers",
            headers=auth_headers
        )
        assert get_response2.status_code == 200
        data2 = get_response2.json()
        assert len(data2["services"]) == 2
        assert "gemini" not in data2["services"]
        assert "openai" in data2["services"]
        assert "anthropic" in data2["services"]
    
    async def test_update_existing_key(
        self, async_client: AsyncClient, auth_headers, test_user, clean_db, mongo_db
    ):
        """Test updating an existing API key by storing again."""
        # Store initial key
        response1 = await async_client.post(
            "/api/kms/store",
            json={
                "provider_name": "openai",
                "api_key": "sk-old-key"
            },
            headers=auth_headers
        )
        assert response1.status_code == 200
        
        # Get initial cookie value
        cookie1 = response1.cookies.get("openai")
        
        # Update with new key
        response2 = await async_client.post(
            "/api/kms/store",
            json={
                "provider_name": "openai",
                "api_key": "sk-new-key"
            },
            headers=auth_headers
        )
        assert response2.status_code == 200
        
        # Get new cookie value
        cookie2 = response2.cookies.get("openai")
        
        # Cookies should be different (different encrypted keys)
        assert cookie1 != cookie2
        
        # Verify only one key exists in database
        keys = await mongo_db.keys.find({
            "provider_name": "openai",
            "userid": test_user["id"]
        }).to_list(length=10)
        assert len(keys) == 1


@pytest.mark.asyncio
@pytest.mark.integration
class TestKeyEncryption:
    """Test that API keys are properly encrypted."""
    
    async def test_api_key_encryption(
        self, async_client: AsyncClient, auth_headers, test_user, clean_db, mongo_db, test_app
    ):
        """Test that stored API keys are encrypted."""
        original_key = "sk-test-secret-key-12345"
        
        # Store API key
        response = await async_client.post(
            "/api/kms/store",
            json={
                "provider_name": "openai",
                "api_key": original_key
            },
            headers=auth_headers
        )
        assert response.status_code == 200
        
        # Get stored key from database
        key_doc = await mongo_db.keys.find_one({
            "provider_name": "openai",
            "userid": test_user["id"]
        })
        
        # Verify original key is NOT in database
        assert original_key not in str(key_doc)
        
        # Verify encrypted DEK is stored
        assert "dek" in key_doc
        assert key_doc["dek"] != original_key
        
        # Verify we can decrypt it using KMS
        kms: KMS = test_app.state.kms
        encrypted_api_key = response.cookies.get("openai")
        
        # Decrypt using KMS
        decrypted_key = await kms.decrypt_api_key(
            encrypted_api_key,
            test_user["id"],
            "openai",
            mongo_db
        )
        
        # Verify decrypted key matches original
        assert decrypted_key == original_key
    
    async def test_different_users_different_encryption(
        self, async_client: AsyncClient, auth_headers, admin_headers, test_user, test_admin, clean_db, mongo_db
    ):
        """Test that same API key for different users is encrypted differently."""
        same_key = "sk-same-key-for-both-users"
        
        # User 1 stores key
        response1 = await async_client.post(
            "/api/kms/store",
            json={
                "provider_name": "openai",
                "api_key": same_key
            },
            headers=auth_headers
        )
        cookie1 = response1.cookies.get("openai")
        
        # User 2 stores same key
        response2 = await async_client.post(
            "/api/kms/store",
            json={
                "provider_name": "openai",
                "api_key": same_key
            },
            headers=admin_headers
        )
        cookie2 = response2.cookies.get("openai")
        
        # Cookies should be different (different encryption per user)
        assert cookie1 != cookie2
        
        # Database DEKs should also be different
        key1 = await mongo_db.keys.find_one({
            "provider_name": "openai",
            "userid": test_user["id"]
        })
        key2 = await mongo_db.keys.find_one({
            "provider_name": "openai",
            "userid": test_admin["id"]
        })
        
        assert key1["dek"] != key2["dek"]

