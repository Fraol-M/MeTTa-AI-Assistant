"""
Integration tests for authentication endpoints (auth.py router).
Tests signup, login, and refresh endpoints with real database.
"""
import pytest
import asyncio
from httpx import AsyncClient
from jose import jwt
from app.services.auth import get_secret_key, ALGORITHM


@pytest.mark.asyncio
@pytest.mark.integration
class TestAuthSignup:
    """Test POST /api/auth/signup endpoint."""
    
    async def test_signup_success(self, async_client: AsyncClient, clean_db, mongo_db):
        """Test successful user signup."""
        email = "newuser@test.com"
        password = "SecurePassword123!"
        
        response = await async_client.post(
            "/api/auth/signup",
            json={
                "email": email,
                "password": password
            }
        )
        
        assert response.status_code == 201, f"Expected 201, got {response.status_code}: {response.text}"
        response_data = response.json()
        assert "message" in response_data
        assert response_data["message"] == "User created"
        assert "user_id" in response_data
        assert response_data["user_id"] is not None
        
        # Verify user was created in database
        user = await mongo_db.users.find_one({"email": email})
        assert user is not None, "User should be created in database"
        assert user["email"] == email
        assert user["role"] == "user"  # Default role should be USER
        assert "hashed_password" in user
        assert user["hashed_password"] != password  # Password should be hashed
    
    async def test_signup_duplicate_email(self, async_client: AsyncClient, clean_db):
        """Test that signup with duplicate email fails."""
        email = "duplicate@test.com"
        password = "Password123"
        
        # First signup should succeed
        response1 = await async_client.post(
            "/api/auth/signup",
            json={
                "email": email,
                "password": password
            }
        )
        assert response1.status_code == 201
        
        response2 = await async_client.post(
            "/api/auth/signup",
            json={
                "email": email,
                "password": password
            }
        )
        
        assert response2.status_code == 400, f"Expected 400, got {response2.status_code}: {response2.text}"
        response_data = response2.json()
        assert "detail" in response_data
        assert "Email in use" in str(response_data["detail"])
    
    async def test_signup_invalid_email_format(self, async_client: AsyncClient, clean_db):
        """Test signup with invalid email format."""
        response = await async_client.post(
            "/api/auth/signup",
            json={
                "email": "not-an-email",
                "password": "Password123"
            }
        )
        
        assert response.status_code == 422
    
    async def test_signup_missing_fields(self, async_client: AsyncClient, clean_db):
        """Test signup with missing required fields."""
        response1 = await async_client.post(
            "/api/auth/signup",
            json={
                "email": "test@test.com"
            }
        )
        assert response1.status_code == 422
        
        # Missing email
        response2 = await async_client.post(
            "/api/auth/signup",
            json={
                "password": "Password123"
            }
        )
        assert response2.status_code == 422


@pytest.mark.asyncio
@pytest.mark.integration
class TestAuthLogin:
    """Test POST /api/auth/login endpoint."""
    
    async def test_login_success(self, async_client: AsyncClient, clean_db):
        """Test successful login with correct credentials."""
        email = "loginuser@test.com"
        password = "LoginPassword123"
        
        signup_response = await async_client.post(
            "/api/auth/signup",
            json={
                "email": email,
                "password": password
            }
        )
        assert signup_response.status_code == 201
        
        login_response = await async_client.post(
            "/api/auth/login",
            json={
                "email": email,
                "password": password
            }
        )
        
        assert login_response.status_code == 200, f"Expected 200, got {login_response.status_code}: {login_response.text}"
        login_data = login_response.json()
        
        assert "access_token" in login_data
        assert "refresh_token" in login_data
        assert "token_type" in login_data
        assert login_data["token_type"] == "bearer"
        
        access_token = login_data["access_token"]
        refresh_token = login_data["refresh_token"]
        
        decoded_access = jwt.decode(access_token, get_secret_key(), algorithms=[ALGORITHM])
        assert "sub" in decoded_access  # User ID
        assert "role" in decoded_access
        assert "exp" in decoded_access  # Expiration
        assert decoded_access["role"] == "user"
        
        decoded_refresh = jwt.decode(refresh_token, get_secret_key(), algorithms=[ALGORITHM])
        assert "sub" in decoded_refresh
        assert "type" in decoded_refresh
        assert decoded_refresh["type"] == "refresh"
        assert "exp" in decoded_refresh
    
    async def test_login_invalid_password(self, async_client: AsyncClient, clean_db):
        """Test login with incorrect password."""
        email = "wrongpass@test.com"
        password = "CorrectPassword123"
        wrong_password = "WrongPassword123"
        
        # Create user
        await async_client.post(
            "/api/auth/signup",
            json={
                "email": email,
                "password": password
            }
        )
        
        response = await async_client.post(
            "/api/auth/login",
            json={
                "email": email,
                "password": wrong_password
            }
        )
        
        assert response.status_code == 401, f"Expected 401, got {response.status_code}: {response.text}"
        response_data = response.json()
        assert "detail" in response_data
        assert "Invalid credentials" in response_data["detail"]
    
    async def test_login_nonexistent_user(self, async_client: AsyncClient, clean_db):
        """Test login with email that doesn't exist."""
        response = await async_client.post(
            "/api/auth/login",
            json={
                "email": "nonexistent@test.com",
                "password": "AnyPassword123"
            }
        )
        
        assert response.status_code == 401
        response_data = response.json()
        assert "detail" in response_data
        assert "Invalid credentials" in response_data["detail"]
    
    async def test_login_missing_fields(self, async_client: AsyncClient, clean_db):
        """Test login with missing required fields."""
        response1 = await async_client.post(
            "/api/auth/login",
            json={
                "email": "test@test.com"
            }
        )
        assert response1.status_code == 422
        
        response2 = await async_client.post(
            "/api/auth/login",
            json={
                "password": "Password123"
            }
        )
        assert response2.status_code == 422
    
    async def test_login_empty_credentials(self, async_client: AsyncClient, clean_db):
        """Test login with empty email or password."""
        response1 = await async_client.post(
            "/api/auth/login",
            json={
                "email": "",
                "password": "Password123"
            }
        )
        assert response1.status_code in [422, 401]
        
        response2 = await async_client.post(
            "/api/auth/login",
            json={
                "email": "test@test.com",
                "password": ""
            }
        )
        assert response2.status_code in [422, 401]


@pytest.mark.asyncio
@pytest.mark.integration
class TestAuthRefresh:
    """Test POST /api/auth/refresh endpoint."""
    
    async def test_refresh_success(self, async_client: AsyncClient, clean_db):
        """Test successful token refresh."""
        email = "refreshuser@test.com"
        password = "RefreshPassword123"
        
        await async_client.post(
            "/api/auth/signup",
            json={
                "email": email,
                "password": password
            }
        )
        
        login_response = await async_client.post(
            "/api/auth/login",
            json={
                "email": email,
                "password": password
            }
        )
        assert login_response.status_code == 200
        
        login_data = login_response.json()
        old_refresh_token = login_data["refresh_token"]
        old_access_token = login_data["access_token"]
        
        old_access_decoded = jwt.decode(old_access_token, get_secret_key(), algorithms=[ALGORITHM])
        old_refresh_decoded = jwt.decode(old_refresh_token, get_secret_key(), algorithms=[ALGORITHM])
        
        await asyncio.sleep(1)
        
        refresh_response = await async_client.post(
            "/api/auth/refresh",
            json={
                "refresh_token": old_refresh_token
            }
        )
        
        assert refresh_response.status_code == 200, f"Expected 200, got {refresh_response.status_code}: {refresh_response.text}"
        refresh_data = refresh_response.json()
        
        assert "access_token" in refresh_data
        assert "refresh_token" in refresh_data
        assert "token_type" in refresh_data
        assert refresh_data["token_type"] == "bearer"
        
        new_access_token = refresh_data["access_token"]
        new_refresh_token = refresh_data["refresh_token"]
        
        assert new_access_token != old_access_token, "New access token should be different"
        assert new_refresh_token != old_refresh_token, "New refresh token should be different"
        
        new_access_decoded = jwt.decode(new_access_token, get_secret_key(), algorithms=[ALGORITHM])
        new_refresh_decoded = jwt.decode(new_refresh_token, get_secret_key(), algorithms=[ALGORITHM])
        
        assert new_access_decoded["sub"] == old_access_decoded["sub"]
        assert new_refresh_decoded["sub"] == old_refresh_decoded["sub"]
        
        assert new_access_decoded["exp"] > old_access_decoded["exp"]
        assert new_refresh_decoded["exp"] > old_refresh_decoded["exp"]
        
        assert new_refresh_decoded["type"] == "refresh"
    
    async def test_refresh_with_invalid_token(self, async_client: AsyncClient, clean_db):
        """Test refresh with invalid token string."""
        response = await async_client.post(
            "/api/auth/refresh",
            json={
                "refresh_token": "invalid_token_string_12345"
            }
        )
        
        assert response.status_code == 401, f"Expected 401, got {response.status_code}: {response.text}"
        response_data = response.json()
        assert "detail" in response_data
        assert "Invalid refresh token" in response_data["detail"]
    
    async def test_refresh_with_access_token(self, async_client: AsyncClient, clean_db):
        """Test refresh with access token instead of refresh token."""
        email = "accesstoken@test.com"
        password = "Password123"
        
        await async_client.post(
            "/api/auth/signup",
            json={
                "email": email,
                "password": password
            }
        )
        
        login_response = await async_client.post(
            "/api/auth/login",
            json={
                "email": email,
                "password": password
            }
        )
        assert login_response.status_code == 200
        
        login_data = login_response.json()
        access_token = login_data["access_token"]  # Using access token instead of refresh token
        
        # Try to refresh with access token (should fail)
        response = await async_client.post(
            "/api/auth/refresh",
            json={
                "refresh_token": access_token
            }
        )
        
        assert response.status_code == 400, f"Expected 400, got {response.status_code}: {response.text}"
        response_data = response.json()
        assert "detail" in response_data
        assert "Invalid refresh token" in response_data["detail"]
    
    async def test_refresh_missing_token(self, async_client: AsyncClient, clean_db):
        """Test refresh with missing refresh_token field."""
        response = await async_client.post(
            "/api/auth/refresh",
            json={}
        )
        
        assert response.status_code == 422
    
    async def test_refresh_empty_token(self, async_client: AsyncClient, clean_db):
        """Test refresh with empty token string."""
        response = await async_client.post(
            "/api/auth/refresh",
            json={
                "refresh_token": ""
            }
        )
        
        assert response.status_code == 401


@pytest.mark.asyncio
@pytest.mark.integration
class TestAuthEndToEndFlow:
    """Test complete authentication flow from signup to refresh."""
    
    async def test_complete_auth_flow(self, async_client: AsyncClient, clean_db, mongo_db):
        """Test complete flow: signup -> login -> use token -> refresh."""
        email = "flowuser@test.com"
        password = "FlowPassword123"
        
        # Step 1: Signup
        signup_response = await async_client.post(
            "/api/auth/signup",
            json={
                "email": email,
                "password": password
            }
        )
        assert signup_response.status_code == 201
        signup_data = signup_response.json()
        user_id = signup_data["user_id"]
        
        # Verify user exists in database
        from bson import ObjectId
        user = await mongo_db.users.find_one({"_id": ObjectId(user_id)})
        assert user is not None
        assert user["email"] == email
        
        # Step 2: Login
        login_response = await async_client.post(
            "/api/auth/login",
            json={
                "email": email,
                "password": password
            }
        )
        assert login_response.status_code == 200
        login_data = login_response.json()
        access_token = login_data["access_token"]
        refresh_token = login_data["refresh_token"]
        
        # (This would require a protected endpoint - adjust based on your API)
        headers = {"Authorization": f"Bearer {access_token}"}
        
        # Verify token is valid by decoding it
        decoded = jwt.decode(access_token, get_secret_key(), algorithms=[ALGORITHM])
        assert decoded["sub"] == user_id
        assert decoded["role"] == "user"
        
        # Step 4: Refresh tokens
        await asyncio.sleep(1)  # Small delay to ensure different timestamps
        
        refresh_response = await async_client.post(
            "/api/auth/refresh",
            json={
                "refresh_token": refresh_token
            }
        )
        assert refresh_response.status_code == 200
        refresh_data = refresh_response.json()
        
        assert refresh_data["access_token"] != access_token
        assert refresh_data["refresh_token"] != refresh_token
        
        new_access_token = refresh_data["access_token"]
        new_decoded = jwt.decode(new_access_token, get_secret_key(), algorithms=[ALGORITHM])
        assert new_decoded["sub"] == user_id  # Same user
    
    async def test_multiple_refreshes(self, async_client: AsyncClient, clean_db):
        """Test that refresh can be called multiple times."""
        email = "multirefresh@test.com"
        password = "Password123"
        
        await async_client.post(
            "/api/auth/signup",
            json={
                "email": email,
                "password": password
            }
        )
        
        login_response = await async_client.post(
            "/api/auth/login",
            json={
                "email": email,
                "password": password
            }
        )
        assert login_response.status_code == 200
        
        refresh_token = login_response.json()["refresh_token"]
        
        tokens = []
        for i in range(3):
            await asyncio.sleep(1)  
            response = await async_client.post(
                "/api/auth/refresh",
                json={
                    "refresh_token": refresh_token
                }
            )
            assert response.status_code == 200
            data = response.json()
            tokens.append(data["access_token"])
            refresh_token = data["refresh_token"]  # Use new refresh token for next iteration
        
        assert len(tokens) == 3, "Should have 3 tokens"
        for token in tokens:
            decoded = jwt.decode(token, get_secret_key(), algorithms=[ALGORITHM])
            assert "sub" in decoded
            assert "exp" in decoded
        assert len(set(tokens)) >= 2, "At least 2 tokens should be unique"
