"""
Integration tests for chat sessions endpoints (chat_sessions.py router).
Tests list_sessions, get_session, and delete_session endpoints with real database.
"""
import pytest
from httpx import AsyncClient
from bson import ObjectId
from datetime import datetime, timezone
from app.db.chat_db import create_chat_session, insert_chat_message


@pytest.mark.asyncio
@pytest.mark.integration
class TestChatSessionsList:
    """Test GET /api/chat/sessions/ endpoint."""
    
    async def test_list_sessions_success_empty(self, async_client: AsyncClient, auth_headers, clean_db):
        """Test listing sessions when user has no sessions."""
        response = await async_client.get(
            "/api/chat/sessions/",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert "total" in data
        assert "page" in data
        assert "limit" in data
        assert "total_pages" in data
        assert "has_next" in data
        assert "has_prev" in data
        
        assert data["sessions"] == []
        assert data["total"] == 0
        assert data["page"] == 1
        assert data["limit"] == 20
        assert data["total_pages"] == 0
        assert data["has_next"] is False
        assert data["has_prev"] is False
    
    async def test_list_sessions_with_data(
        self, async_client: AsyncClient, auth_headers, clean_db, mongo_db, test_user
    ):
        """Test listing sessions when user has sessions."""
        user_id = test_user["id"]
        
        # Create multiple sessions for the user
        session_ids = []
        for i in range(3):
            session_id = await create_chat_session(user_id, mongo_db)
            session_ids.append(session_id)
        
        response = await async_client.get(
            "/api/chat/sessions/",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["sessions"]) == 3
        assert data["total"] == 3
        assert data["page"] == 1
        assert data["limit"] == 20
        assert data["total_pages"] == 1
        assert data["has_next"] is False
        assert data["has_prev"] is False
        
        # Verify session structure
        for session in data["sessions"]:
            assert "sessionId" in session
            assert "createdAt" in session
            assert "userId" in session
            assert session["userId"] == user_id
            assert session["sessionId"] in session_ids
    
    async def test_list_sessions_pagination(
        self, async_client: AsyncClient, auth_headers, clean_db, mongo_db, test_user
    ):
        """Test pagination for listing sessions."""
        user_id = test_user["id"]
        
        # Create 5 sessions
        for i in range(5):
            await create_chat_session(user_id, mongo_db)
        
        # Test first page
        response1 = await async_client.get(
            "/api/chat/sessions/?page=1&limit=2",
            headers=auth_headers
        )
        assert response1.status_code == 200
        data1 = response1.json()
        assert len(data1["sessions"]) == 2
        assert data1["total"] == 5
        assert data1["page"] == 1
        assert data1["limit"] == 2
        assert data1["has_next"] is True
        assert data1["has_prev"] is False
        
        # Test second page
        response2 = await async_client.get(
            "/api/chat/sessions/?page=2&limit=2",
            headers=auth_headers
        )
        assert response2.status_code == 200
        data2 = response2.json()
        assert len(data2["sessions"]) == 2
        assert data2["page"] == 2
        assert data2["has_next"] is True
        assert data2["has_prev"] is True
        
        # Test third page
        response3 = await async_client.get(
            "/api/chat/sessions/?page=3&limit=2",
            headers=auth_headers
        )
        assert response3.status_code == 200
        data3 = response3.json()
        assert len(data3["sessions"]) == 1
        assert data3["page"] == 3
        assert data3["has_next"] is False
        assert data3["has_prev"] is True
        
        # Verify sessions are different across pages
        page1_ids = {s["sessionId"] for s in data1["sessions"]}
        page2_ids = {s["sessionId"] for s in data2["sessions"]}
        page3_ids = {s["sessionId"] for s in data3["sessions"]}
        assert page1_ids.isdisjoint(page2_ids)
        assert page2_ids.isdisjoint(page3_ids)
    
    async def test_list_sessions_only_own_sessions(
        self, async_client: AsyncClient, auth_headers, clean_db, mongo_db, test_user
    ):
        """Test that user only sees their own sessions."""
        user_id = test_user["id"]
        
        # Create another user
        from app.db.users import create_user, UserCreate, UserRole
        other_user_data = UserCreate(
            email="otheruser@test.com",
            password="Password123",
            role=UserRole.USER
        )
        other_user_id = await create_user(other_user_data, mongo_db)
        other_user_id_str = str(other_user_id)
        
        # Create sessions for both users
        await create_chat_session(user_id, mongo_db)
        await create_chat_session(user_id, mongo_db)
        await create_chat_session(other_user_id_str, mongo_db)
        
        # List sessions - should only see own sessions
        response = await async_client.get(
            "/api/chat/sessions/",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["sessions"]) == 2
        for session in data["sessions"]:
            assert session["userId"] == user_id
    
    async def test_list_sessions_invalid_page(
        self, async_client: AsyncClient, auth_headers, clean_db
    ):
        """Test listing sessions with invalid page number."""
        response = await async_client.get(
            "/api/chat/sessions/?page=0",
            headers=auth_headers
        )
        assert response.status_code == 422  # Validation error
    
    async def test_list_sessions_invalid_limit(
        self, async_client: AsyncClient, auth_headers, clean_db
    ):
        """Test listing sessions with invalid limit."""
        # Limit too high
        response1 = await async_client.get(
            "/api/chat/sessions/?limit=101",
            headers=auth_headers
        )
        assert response1.status_code == 422
        
        # Limit too low
        response2 = await async_client.get(
            "/api/chat/sessions/?limit=0",
            headers=auth_headers
        )
        assert response2.status_code == 422
    
    async def test_list_sessions_unauthorized(self, async_client: AsyncClient, clean_db):
        """Test listing sessions without authentication."""
        response = await async_client.get("/api/chat/sessions/")
        assert response.status_code == 401


@pytest.mark.asyncio
@pytest.mark.integration
class TestChatSessionsGet:
    """Test GET /api/chat/sessions/{session_id} endpoint."""
    
    async def test_get_session_success(
        self, async_client: AsyncClient, auth_headers, clean_db, mongo_db, test_user
    ):
        """Test getting a session with messages."""
        user_id = test_user["id"]
        
        # Create a session
        session_id = await create_chat_session(user_id, mongo_db)
        
        # Add messages to the session
        messages = [
            {
                "sessionId": session_id,
                "role": "user",
                "content": "Hello, how are you?",
                "createdAt": datetime.now(timezone.utc)
            },
            {
                "sessionId": session_id,
                "role": "assistant",
                "content": "I'm doing well, thank you!",
                "createdAt": datetime.now(timezone.utc)
            }
        ]
        
        for msg in messages:
            await insert_chat_message(msg, mongo_db)
        
        # Get the session
        response = await async_client.get(
            f"/api/chat/sessions/{session_id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify session structure
        assert "sessionId" in data
        assert "createdAt" in data
        assert "userId" in data
        assert "messages" in data
        
        assert data["sessionId"] == session_id
        assert data["userId"] == user_id
        assert len(data["messages"]) == 2
        
        # Verify messages structure
        for message in data["messages"]:
            assert "messageId" in message
            assert "sessionId" in message
            assert "role" in message
            assert "content" in message
            assert "createdAt" in message
            assert message["sessionId"] == session_id
            assert message["role"] in ["user", "assistant"]
        
        # Verify messages are in chronological order (oldest first)
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][1]["role"] == "assistant"
    
    async def test_get_session_without_messages(
        self, async_client: AsyncClient, auth_headers, clean_db, mongo_db, test_user
    ):
        """Test getting a session that has no messages."""
        user_id = test_user["id"]
        
        # Create a session without messages
        session_id = await create_chat_session(user_id, mongo_db)
        
        # Get the session
        response = await async_client.get(
            f"/api/chat/sessions/{session_id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["sessionId"] == session_id
        assert data["messages"] == []
    
    async def test_get_session_not_found(
        self, async_client: AsyncClient, auth_headers, clean_db
    ):
        """Test getting a non-existent session."""
        fake_session_id = str(ObjectId())
        
        response = await async_client.get(
            f"/api/chat/sessions/{fake_session_id}",
            headers=auth_headers
        )
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    async def test_get_session_other_user(
        self, async_client: AsyncClient, auth_headers, clean_db, mongo_db, test_user
    ):
        """Test that user cannot access another user's session."""
        user_id = test_user["id"]
        
        # Create another user
        from app.db.users import create_user, UserCreate, UserRole
        other_user_data = UserCreate(
            email="otheruser2@test.com",
            password="Password123",
            role=UserRole.USER
        )
        other_user_id = await create_user(other_user_data, mongo_db)
        other_user_id_str = str(other_user_id)
        
        # Create session for other user
        other_session_id = await create_chat_session(other_user_id_str, mongo_db)
        
        # Try to access other user's session
        response = await async_client.get(
            f"/api/chat/sessions/{other_session_id}",
            headers=auth_headers
        )
        
        assert response.status_code == 403
        assert "Access denied" in response.json()["detail"]
    
    async def test_get_session_unauthorized(self, async_client: AsyncClient, clean_db, mongo_db):
        """Test getting a session without authentication."""
        # Create a session
        from app.db.users import create_user, UserCreate, UserRole
        user_data = UserCreate(
            email="unauth@test.com",
            password="Password123",
            role=UserRole.USER
        )
        user_id = await create_user(user_data, mongo_db)
        session_id = await create_chat_session(str(user_id), mongo_db)
        
        # Try to get session without auth
        response = await async_client.get(f"/api/chat/sessions/{session_id}")
        assert response.status_code == 401


@pytest.mark.asyncio
@pytest.mark.integration
class TestChatSessionsDelete:
    """Test DELETE /api/chat/sessions/{session_id} endpoint."""
    
    async def test_delete_session_success(
        self, async_client: AsyncClient, auth_headers, clean_db, mongo_db, test_user
    ):
        """Test successfully deleting a session."""
        user_id = test_user["id"]
        
        # Create a session
        session_id = await create_chat_session(user_id, mongo_db)
        
        # Add some messages
        await insert_chat_message({
            "sessionId": session_id,
            "role": "user",
            "content": "Test message",
            "createdAt": datetime.now(timezone.utc)
        }, mongo_db)
        
        # Verify session exists
        session = await mongo_db.chat_sessions.find_one({"sessionId": session_id})
        assert session is not None
        
        # Delete the session
        response = await async_client.delete(
            f"/api/chat/sessions/{session_id}",
            headers=auth_headers
        )
        
        assert response.status_code == 204
        
        # Verify session is deleted
        deleted_session = await mongo_db.chat_sessions.find_one({"sessionId": session_id})
        assert deleted_session is None
        
        # Note: Messages might still exist (depending on implementation)
        # This is a design decision - sessions deleted but messages might remain
    
    async def test_delete_session_not_found(
        self, async_client: AsyncClient, auth_headers, clean_db
    ):
        """Test deleting a non-existent session."""
        fake_session_id = str(ObjectId())
        
        response = await async_client.delete(
            f"/api/chat/sessions/{fake_session_id}",
            headers=auth_headers
        )
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    async def test_delete_session_other_user(
        self, async_client: AsyncClient, auth_headers, clean_db, mongo_db, test_user
    ):
        """Test that user cannot delete another user's session."""
        user_id = test_user["id"]
        
        # Create another user
        from app.db.users import create_user, UserCreate, UserRole
        other_user_data = UserCreate(
            email="otheruser3@test.com",
            password="Password123",
            role=UserRole.USER
        )
        other_user_id = await create_user(other_user_data, mongo_db)
        other_user_id_str = str(other_user_id)
        
        # Create session for other user
        other_session_id = await create_chat_session(other_user_id_str, mongo_db)
        
        # Try to delete other user's session
        response = await async_client.delete(
            f"/api/chat/sessions/{other_session_id}",
            headers=auth_headers
        )
        
        assert response.status_code == 403
        assert "Access denied" in response.json()["detail"]
        
        # Verify session still exists
        session = await mongo_db.chat_sessions.find_one({"sessionId": other_session_id})
        assert session is not None
    
    async def test_delete_session_unauthorized(self, async_client: AsyncClient, clean_db, mongo_db):
        """Test deleting a session without authentication."""
        # Create a session
        from app.db.users import create_user, UserCreate, UserRole
        user_data = UserCreate(
            email="unauth2@test.com",
            password="Password123",
            role=UserRole.USER
        )
        user_id = await create_user(user_data, mongo_db)
        session_id = await create_chat_session(str(user_id), mongo_db)
        
        # Try to delete session without auth
        response = await async_client.delete(f"/api/chat/sessions/{session_id}")
        assert response.status_code == 401


@pytest.mark.asyncio
@pytest.mark.integration
class TestChatSessionsEndToEnd:
    """Test complete chat sessions workflow."""
    
    async def test_complete_session_lifecycle(
        self, async_client: AsyncClient, auth_headers, clean_db, mongo_db, test_user
    ):
        """Test complete workflow: create session -> add messages -> list -> get -> delete."""
        user_id = test_user["id"]
        
        # Step 1: Create a session (via chat endpoint or directly)
        session_id = await create_chat_session(user_id, mongo_db)
        
        # Step 2: Add messages to the session
        messages_data = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
            {"role": "user", "content": "Tell me more."},
            {"role": "assistant", "content": "Python is known for its simplicity."}
        ]
        
        for msg_data in messages_data:
            await insert_chat_message({
                "sessionId": session_id,
                **msg_data,
                "createdAt": datetime.now(timezone.utc)
            }, mongo_db)
        
        # Step 3: List sessions - should see the new session
        list_response = await async_client.get(
            "/api/chat/sessions/",
            headers=auth_headers
        )
        assert list_response.status_code == 200
        list_data = list_response.json()
        assert list_data["total"] == 1
        assert list_data["sessions"][0]["sessionId"] == session_id
        
        # Step 4: Get the session with messages
        get_response = await async_client.get(
            f"/api/chat/sessions/{session_id}",
            headers=auth_headers
        )
        assert get_response.status_code == 200
        session_data = get_response.json()
        assert session_data["sessionId"] == session_id
        assert len(session_data["messages"]) == 4
        
        # Verify messages are in correct order
        assert session_data["messages"][0]["role"] == "user"
        assert session_data["messages"][1]["role"] == "assistant"
        assert session_data["messages"][2]["role"] == "user"
        assert session_data["messages"][3]["role"] == "assistant"
        
        # Step 5: Delete the session
        delete_response = await async_client.delete(
            f"/api/chat/sessions/{session_id}",
            headers=auth_headers
        )
        assert delete_response.status_code == 204
        
        # Step 6: Verify session is deleted
        get_after_delete = await async_client.get(
            f"/api/chat/sessions/{session_id}",
            headers=auth_headers
        )
        assert get_after_delete.status_code == 404
        
        # Step 7: List sessions - should be empty
        list_after_delete = await async_client.get(
            "/api/chat/sessions/",
            headers=auth_headers
        )
        assert list_after_delete.status_code == 200
        assert list_after_delete.json()["total"] == 0
    
    async def test_multiple_sessions_isolation(
        self, async_client: AsyncClient, auth_headers, clean_db, mongo_db, test_user
    ):
        """Test that multiple sessions are properly isolated."""
        user_id = test_user["id"]
        
        # Create multiple sessions
        session1_id = await create_chat_session(user_id, mongo_db)
        session2_id = await create_chat_session(user_id, mongo_db)
        
        # Add messages to each session
        await insert_chat_message({
            "sessionId": session1_id,
            "role": "user",
            "content": "Session 1 message",
            "createdAt": datetime.now(timezone.utc)
        }, mongo_db)
        
        await insert_chat_message({
            "sessionId": session2_id,
            "role": "user",
            "content": "Session 2 message",
            "createdAt": datetime.now(timezone.utc)
        }, mongo_db)
        
        # Get each session - should only see its own messages
        response1 = await async_client.get(
            f"/api/chat/sessions/{session1_id}",
            headers=auth_headers
        )
        assert response1.status_code == 200
        assert len(response1.json()["messages"]) == 1
        assert response1.json()["messages"][0]["content"] == "Session 1 message"
        
        response2 = await async_client.get(
            f"/api/chat/sessions/{session2_id}",
            headers=auth_headers
        )
        assert response2.status_code == 200
        assert len(response2.json()["messages"]) == 1
        assert response2.json()["messages"][0]["content"] == "Session 2 message"

