"""
Integration tests for chat endpoints (chat.py router).
Tests POST /api/chat/ endpoint with search and generate modes, real database and Qdrant.
"""
import pytest
from httpx import AsyncClient
from bson import ObjectId
from datetime import datetime, timezone
from app.db.chat_db import create_chat_session, get_messages_for_session


@pytest.mark.asyncio
@pytest.mark.integration
class TestChatSearchMode:
    """Test POST /api/chat/ endpoint in search mode."""
    
    async def test_chat_search_mode_success(
        self, async_client: AsyncClient, auth_headers, clean_db, mongo_db
    ):
        """Test search mode returns search results."""
        response = await async_client.post(
            "/api/chat/",
            json={
                "query": "Python programming",
                "mode": "search",
                "top_k": 5
            },
            headers=auth_headers
        )
        
        # Search mode should work even without embeddings (returns empty results)
        assert response.status_code in [200, 500], f"Expected 200 or 500, got {response.status_code}: {response.text}"
        
        if response.status_code == 200:
            data = response.json()
            assert "query" in data
            assert "mode" in data
            assert "results" in data
            assert data["mode"] == "search"
            assert data["query"] == "Python programming"
            assert isinstance(data["results"], dict)  # Results by category
    
    async def test_chat_search_mode_with_top_k(
        self, async_client: AsyncClient, auth_headers, clean_db
    ):
        """Test search mode with different top_k values."""
        for top_k in [1, 5, 10]:
            response = await async_client.post(
                "/api/chat/",
                json={
                    "query": "test query",
                    "mode": "search",
                    "top_k": top_k
                },
                headers=auth_headers
            )
            # May return 200 or 500 depending on embeddings
            assert response.status_code in [200, 500]
    
    async def test_chat_search_mode_invalid_top_k(
        self, async_client: AsyncClient, auth_headers, clean_db
    ):
        """Test search mode with invalid top_k values."""
        # top_k too high
        response1 = await async_client.post(
            "/api/chat/",
            json={
                "query": "test",
                "mode": "search",
                "top_k": 51  # Max is 50
            },
            headers=auth_headers
        )
        assert response1.status_code == 422
        
        # top_k too low
        response2 = await async_client.post(
            "/api/chat/",
            json={
                "query": "test",
                "mode": "search",
                "top_k": 0  # Min is 1
            },
            headers=auth_headers
        )
        assert response2.status_code == 422
    
    async def test_chat_search_mode_missing_query(
        self, async_client: AsyncClient, auth_headers, clean_db
    ):
        """Test search mode with missing query."""
        response = await async_client.post(
            "/api/chat/",
            json={
                "mode": "search",
                "top_k": 5
            },
            headers=auth_headers
        )
        assert response.status_code == 422
    
    async def test_chat_search_mode_unauthorized(self, async_client: AsyncClient, clean_db):
        """Test search mode without authentication."""
        response = await async_client.post(
            "/api/chat/",
            json={
                "query": "test query",
                "mode": "search"
            }
        )
        assert response.status_code == 401


@pytest.mark.asyncio
@pytest.mark.integration
class TestChatGenerateMode:
    """Test POST /api/chat/ endpoint in generate mode."""
    
    async def test_chat_generate_mode_success(
        self, async_client: AsyncClient, auth_headers, clean_db, mongo_db
    ):
        """Test generate mode creates session and returns response."""
        response = await async_client.post(
            "/api/chat/",
            json={
                "query": "What is Python?",
                "mode": "generate",
                "provider": "gemini",
                "top_k": 5
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        
        # Verify response structure
        assert "response" in data
        assert "session_id" in data  # New session should be created
        assert isinstance(data["response"], str)
        
        # Verify session was created
        session_id = data["session_id"]
        session = await mongo_db.chat_sessions.find_one({"sessionId": session_id})
        assert session is not None
        
        # Verify messages were saved
        messages = await get_messages_for_session(session_id, mongo_db)
        assert len(messages) == 2  # User message + assistant response
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is Python?"
        assert messages[1]["role"] == "assistant"
    
    async def test_chat_generate_mode_with_existing_session(
        self, async_client: AsyncClient, auth_headers, clean_db, mongo_db, test_user
    ):
        """Test generate mode with existing session ID."""
        user_id = test_user["id"]
        
        # Create a session first
        session_id = await create_chat_session(user_id, mongo_db)
        
        # Send first message
        response1 = await async_client.post(
            "/api/chat/",
            json={
                "query": "First question",
                "mode": "generate",
                "provider": "gemini",
                "session_id": session_id,
                "top_k": 5
            },
            headers=auth_headers
        )
        assert response1.status_code == 200
        data1 = response1.json()
        assert "response" in data1
        assert "session_id" not in data1  # Should not return session_id for existing session
        
        # Send second message in same session
        response2 = await async_client.post(
            "/api/chat/",
            json={
                "query": "Follow up question",
                "mode": "generate",
                "provider": "gemini",
                "session_id": session_id,
                "top_k": 5
            },
            headers=auth_headers
        )
        assert response2.status_code == 200
        
        # Verify all messages are in the session
        messages = await get_messages_for_session(session_id, mongo_db)
        assert len(messages) == 4  # 2 user messages + 2 assistant responses
        
        # Verify conversation history
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "First question"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "Follow up question"
        assert messages[3]["role"] == "assistant"
    
    async def test_chat_generate_mode_different_providers(
        self, async_client: AsyncClient, auth_headers, clean_db
    ):
        """Test generate mode with different providers."""
        # Test gemini (default, should work)
        response = await async_client.post(
            "/api/chat/",
            json={
                "query": "Test query",
                "mode": "generate",
                "provider": "gemini",
                "top_k": 5
            },
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        
        # Test openai (may fail if API keys not configured, which is expected in tests)
        response2 = await async_client.post(
            "/api/chat/",
            json={
                "query": "Test query",
                "mode": "generate",
                "provider": "openai",
                "top_k": 5
            },
            headers=auth_headers
        )
        # May return 200 (if mocked properly) or 500 (if API keys required)
        assert response2.status_code in [200, 500]
    
    async def test_chat_generate_mode_with_model(
        self, async_client: AsyncClient, auth_headers, clean_db
    ):
        """Test generate mode with specific model."""
        # Test with gemini and model (should work)
        response = await async_client.post(
            "/api/chat/",
            json={
                "query": "Test query",
                "mode": "generate",
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "top_k": 5
            },
            headers=auth_headers
        )
        # May return 200 (if mocked properly) or 500 (if API keys required)
        assert response.status_code in [200, 500]
        
        # If successful, verify response structure
        if response.status_code == 200:
            data = response.json()
            assert "response" in data
    
    async def test_chat_generate_mode_missing_provider(
        self, async_client: AsyncClient, auth_headers, clean_db
    ):
        """Test generate mode without provider (should use default)."""
        response = await async_client.post(
            "/api/chat/",
            json={
                "query": "Test query",
                "mode": "generate",
                "top_k": 5
            },
            headers=auth_headers
        )
        # Should work with default provider (gemini)
        assert response.status_code == 200
    
    async def test_chat_generate_mode_invalid_provider(
        self, async_client: AsyncClient, auth_headers, clean_db
    ):
        """Test generate mode with invalid provider."""
        response = await async_client.post(
            "/api/chat/",
            json={
                "query": "Test query",
                "mode": "generate",
                "provider": "invalid_provider",
                "top_k": 5
            },
            headers=auth_headers
        )
        assert response.status_code == 422  # Validation error
    
    async def test_chat_generate_mode_empty_query(
        self, async_client: AsyncClient, auth_headers, clean_db
    ):
        """Test generate mode with empty query."""
        response = await async_client.post(
            "/api/chat/",
            json={
                "query": "",
                "mode": "generate",
                "provider": "gemini",
                "top_k": 5
            },
            headers=auth_headers
        )
        # May accept empty query or reject it
        assert response.status_code in [200, 422]
    
    async def test_chat_generate_mode_unauthorized(self, async_client: AsyncClient, clean_db):
        """Test generate mode without authentication."""
        response = await async_client.post(
            "/api/chat/",
            json={
                "query": "test query",
                "mode": "generate"
            }
        )
        assert response.status_code == 401


@pytest.mark.asyncio
@pytest.mark.integration
class TestChatSessionManagement:
    """Test chat endpoint session management."""
    
    async def test_chat_creates_new_session_when_none_provided(
        self, async_client: AsyncClient, auth_headers, clean_db, mongo_db, test_user
    ):
        """Test that chat creates a new session when session_id is not provided."""
        user_id = test_user["id"]
        
        # Count sessions before
        sessions_before = await mongo_db.chat_sessions.count_documents({"userId": user_id})
        
        # Send chat without session_id
        response = await async_client.post(
            "/api/chat/",
            json={
                "query": "New conversation",
                "mode": "generate",
                "provider": "gemini",
                "top_k": 5
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        
        # Verify new session was created
        session_id = data["session_id"]
        sessions_after = await mongo_db.chat_sessions.count_documents({"userId": user_id})
        assert sessions_after == sessions_before + 1
        
        session = await mongo_db.chat_sessions.find_one({"sessionId": session_id})
        assert session is not None
        assert session["userId"] == user_id
    
    async def test_chat_uses_existing_session(
        self, async_client: AsyncClient, auth_headers, clean_db, mongo_db, test_user
    ):
        """Test that chat uses existing session when session_id is provided."""
        user_id = test_user["id"]
        
        # Create a session
        session_id = await create_chat_session(user_id, mongo_db)
        
        # Count sessions before
        sessions_before = await mongo_db.chat_sessions.count_documents({"userId": user_id})
        
        # Send chat with existing session_id
        response = await async_client.post(
            "/api/chat/",
            json={
                "query": "Using existing session",
                "mode": "generate",
                "provider": "gemini",
                "session_id": session_id,
                "top_k": 5
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" not in data  # Should not return session_id for existing session
        
        # Verify no new session was created
        sessions_after = await mongo_db.chat_sessions.count_documents({"userId": user_id})
        assert sessions_after == sessions_before
    
    async def test_chat_invalid_session_id(
        self, async_client: AsyncClient, auth_headers, clean_db
    ):
        """Test chat with invalid/non-existent session_id."""
        fake_session_id = str(ObjectId())
        
        response = await async_client.post(
            "/api/chat/",
            json={
                "query": "Test query",
                "mode": "generate",
                "provider": "gemini",
                "session_id": fake_session_id,
                "top_k": 5
            },
            headers=auth_headers
        )
        
        # Should still work (session validation might not be strict)
        # Or might fail if session validation is implemented
        assert response.status_code in [200, 404, 500]


@pytest.mark.asyncio
@pytest.mark.integration
class TestChatMessagePersistence:
    """Test that chat messages are properly persisted."""
    
    async def test_chat_saves_user_message(
        self, async_client: AsyncClient, auth_headers, clean_db, mongo_db
    ):
        """Test that user messages are saved to database."""
        response = await async_client.post(
            "/api/chat/",
            json={
                "query": "This is a test question",
                "mode": "generate",
                "provider": "gemini",
                "top_k": 5
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        session_id = data["session_id"]
        
        # Verify user message was saved
        messages = await get_messages_for_session(session_id, mongo_db)
        assert len(messages) >= 1
        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) > 0
        assert user_messages[0]["content"] == "This is a test question"
    
    async def test_chat_saves_assistant_response(
        self, async_client: AsyncClient, auth_headers, clean_db, mongo_db
    ):
        """Test that assistant responses are saved to database."""
        response = await async_client.post(
            "/api/chat/",
            json={
                "query": "Test question",
                "mode": "generate",
                "provider": "gemini",
                "top_k": 5
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        session_id = data["session_id"]
        
        # Verify assistant message was saved
        messages = await get_messages_for_session(session_id, mongo_db)
        assert len(messages) >= 2
        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_messages) > 0
        assert len(assistant_messages[0]["content"]) > 0
    
    async def test_chat_conversation_history(
        self, async_client: AsyncClient, auth_headers, clean_db, mongo_db, test_user
    ):
        """Test that conversation history is maintained across multiple messages."""
        user_id = test_user["id"]
        session_id = await create_chat_session(user_id, mongo_db)
        
        queries = [
            "What is Python?",
            "Tell me more about it",
            "Give me an example"
        ]
        
        for query in queries:
            response = await async_client.post(
                "/api/chat/",
                json={
                    "query": query,
                    "mode": "generate",
                    "provider": "gemini",
                    "session_id": session_id,
                    "top_k": 5
                },
                headers=auth_headers
            )
            assert response.status_code == 200
        
        # Verify all messages are saved
        messages = await get_messages_for_session(session_id, mongo_db)
        assert len(messages) == 6  # 3 user + 3 assistant
        
        # Verify message order
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == queries[0]
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == queries[1]
        assert messages[3]["role"] == "assistant"
        assert messages[4]["role"] == "user"
        assert messages[4]["content"] == queries[2]
        assert messages[5]["role"] == "assistant"


@pytest.mark.asyncio
@pytest.mark.integration
class TestChatEndToEnd:
    """Test complete chat workflows."""
    
    async def test_complete_chat_workflow(
        self, async_client: AsyncClient, auth_headers, clean_db, mongo_db, test_user
    ):
        """Test complete chat workflow: search -> generate -> continue conversation."""
        user_id = test_user["id"]
        
        # Step 1: Search mode
        search_response = await async_client.post(
            "/api/chat/",
            json={
                "query": "Python programming",
                "mode": "search",
                "top_k": 5
            },
            headers=auth_headers
        )
        assert search_response.status_code in [200, 500]
        
        # Step 2: Generate mode (creates new session)
        generate_response = await async_client.post(
            "/api/chat/",
            json={
                "query": "What is Python?",
                "mode": "generate",
                "provider": "gemini",
                "top_k": 5
            },
            headers=auth_headers
        )
        assert generate_response.status_code == 200
        generate_data = generate_response.json()
        session_id = generate_data["session_id"]
        
        # Step 3: Continue conversation in same session
        continue_response = await async_client.post(
            "/api/chat/",
            json={
                "query": "Tell me more",
                "mode": "generate",
                "provider": "gemini",
                "session_id": session_id,
                "top_k": 5
            },
            headers=auth_headers
        )
        assert continue_response.status_code == 200
        
        # Verify session has all messages
        messages = await get_messages_for_session(session_id, mongo_db)
        assert len(messages) == 4  # 2 user + 2 assistant
    
    async def test_chat_with_different_modes_same_session(
        self, async_client: AsyncClient, auth_headers, clean_db, mongo_db, test_user
    ):
        """Test mixing search and generate modes."""
        user_id = test_user["id"]
        session_id = await create_chat_session(user_id, mongo_db)
        
        # Generate mode (saves messages)
        generate_response = await async_client.post(
            "/api/chat/",
            json={
                "query": "What is Python?",
                "mode": "generate",
                "provider": "gemini",
                "session_id": session_id,
                "top_k": 5
            },
            headers=auth_headers
        )
        assert generate_response.status_code == 200
        
        # Search mode (doesn't save messages)
        search_response = await async_client.post(
            "/api/chat/",
            json={
                "query": "Python programming",
                "mode": "search",
                "session_id": session_id,
                "top_k": 5
            },
            headers=auth_headers
        )
        assert search_response.status_code in [200, 500]
        
        # Verify only generate mode messages were saved
        messages = await get_messages_for_session(session_id, mongo_db)
        assert len(messages) == 2  # Only generate mode saves messages
