"""
Pytest configuration for integration tests.
These tests use REAL services (MongoDB, Qdrant) - not mocks.
"""
import os
import pytest
import asyncio
from typing import AsyncGenerator, Generator
from httpx import AsyncClient
from fastapi import FastAPI
from pymongo import AsyncMongoClient
from pymongo.database import Database
from qdrant_client import AsyncQdrantClient
from sentence_transformers import SentenceTransformer
from bson import ObjectId
from datetime import datetime, timezone


os.environ["JWT_SECRET"] = "test-secret-key-for-integration-tests-only"

# Prevent Git from asking for credentials during tests (prevents hanging)
# These environment variables tell Git to not prompt for username/password
os.environ["GIT_TERMINAL_PROMPT"] = "0"  # Don't prompt in terminal
os.environ["GIT_ASKPASS"] = "echo"  # Don't ask for passwords (echo returns empty)
os.environ["GIT_SSH_COMMAND"] = "ssh -o BatchMode=yes"  # Non-interactive SSH (no password prompts)

mongo_uri = os.getenv("MONGO_URI", "mongodb://mongo:27017")
is_ci = os.getenv("CI", "false").lower() == "true"
if not is_ci and ("localhost" in mongo_uri or "127.0.0.1" in mongo_uri):
    mongo_uri = mongo_uri.replace("localhost", "mongo").replace("127.0.0.1", "mongo")
os.environ["MONGO_URI"] = mongo_uri
os.environ["MONGO_DB"] = "test_integration_db"  
qdrant_host = os.getenv("QDRANT_HOST", "qdrant")
if not is_ci and qdrant_host in ["localhost", "127.0.0.1"]:
    qdrant_host = "qdrant"
os.environ["QDRANT_HOST"] = qdrant_host
os.environ["QDRANT_PORT"] = os.getenv("QDRANT_PORT", "6333")
os.environ["COLLECTION_NAME"] = "test_integration_collection"
os.environ["KEY_ENCRYPTION_KEY"] = "Gs8d0pOL_hhYuwRB_NKiVUh3-2j0PLQkm_i4iZGiEAk="
os.environ["ADMIN_EMAIL"] = "admin@integration.example.com"
os.environ["ADMIN_PASSWORD"] = "admin123"

# Set Redis environment variables before importing app (middleware needs them)
os.environ["REDIS_URL"] = "redis://localhost:6379/0"
os.environ["MAX_REQUESTS"] = "100"
os.environ["WINDOW_SECONDS"] = "60"
os.environ["FRONTEND_URL"] = "http://localhost:5173"

# Mock Redis before importing app (middleware initializes Redis on import)
from unittest.mock import AsyncMock
from redis.asyncio import Redis

# Create a mock Redis client that simulates rate limiting
_mock_redis_client = AsyncMock(spec=Redis)

async def mock_eval(script, num_keys, key, window_seconds):
    """Mock Redis eval - returns a count that's always below limit."""
    return 1  # Always return 1 (below any reasonable limit)

async def mock_ttl(key):
    """Mock Redis ttl - returns remaining time."""
    return 60  # Return 60 seconds

_mock_redis_client.eval = AsyncMock(side_effect=mock_eval)
_mock_redis_client.ttl = AsyncMock(side_effect=mock_ttl)

# Patch Redis.from_url to return our mock (called during middleware initialization)
_original_from_url = Redis.from_url

def _mock_from_url(url, **kwargs):
    """Return the mocked Redis client instead of connecting."""
    return _mock_redis_client

Redis.from_url = staticmethod(_mock_from_url)

from app.main import app
from app.services.key_management_service import KMS
import pytest_asyncio


@pytest_asyncio.fixture(scope="session")
def event_loop():
    """Create a session-scoped event loop for session-scoped async fixtures.
    
    This is required for session-scoped async fixtures like mongo_client, qdrant_client, etc.
    The default pytest-asyncio event_loop is function-scoped, which doesn't work with session-scoped fixtures.
    
    Note: This generates a deprecation warning, but is the recommended approach for session-scoped fixtures
    until pytest-asyncio provides a better solution. The warning is suppressed in pytest.ini.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def mongo_client() -> AsyncGenerator[AsyncMongoClient, None]:
    """Create a real MongoDB client for integration tests."""
    import logging
    logger = logging.getLogger(__name__)
    mongo_uri = os.getenv("MONGO_URI", "mongodb://mongo:27017")
    logger.info(f"Connecting to MongoDB at {mongo_uri}...")
    client = AsyncMongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
    
    # Test connection
    try:
        await client.admin.command("ping")
        logger.info("MongoDB connection successful")
        yield client
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        pytest.skip(f"MongoDB not available at {mongo_uri}: {e}")
    finally:
        await client.close()


@pytest_asyncio.fixture(scope="session")
async def mongo_db(mongo_client: AsyncMongoClient) -> AsyncGenerator[Database, None]:
    """Get the test MongoDB database."""
    db_name = os.getenv("MONGO_DB", "test_integration_db")
    db = mongo_client[db_name]
    
    yield db
    
    # Cleanup: Drop the test database after all tests
    await mongo_client.drop_database(db_name)


@pytest_asyncio.fixture(scope="session")
async def qdrant_client() -> AsyncGenerator[AsyncQdrantClient, None]:
    """Create a real Qdrant client for integration tests."""
    import logging
    logger = logging.getLogger(__name__)
    host = os.getenv("QDRANT_HOST", "qdrant")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    logger.info(f"Connecting to Qdrant at {host}:{port}...")
    
    try:
        client = AsyncQdrantClient(host=host, port=port, timeout=10)
        await client.get_collections()
        logger.info("Qdrant connection successful")
        yield client
    except Exception as e:
        logger.error(f"Qdrant connection failed: {e}")
        pytest.skip(f"Qdrant not available at {host}:{port}: {e}")


@pytest_asyncio.fixture(scope="session")
async def embedding_model() -> SentenceTransformer:
    """Load a real embedding model for integration tests."""
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Loading embedding model (this may take a minute on first run)...")
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        pytest.skip(f"Could not load embedding model: {e}")


@pytest.fixture(scope="session", autouse=True)
def mock_llm_factory():
    """Mock LLMClientFactory.create_client to return a mocked client for all tests.
    
    This ensures integration tests work without API keys even when specific
    models or providers are requested in the chat endpoint.
    """
    from unittest.mock import AsyncMock, patch
    from app.core.clients.llm_clients import LLMClient, LLMProvider
    from app.core.utils.llm_utils import LLMClientFactory
    
    # Create a mock LLM client
    mock_llm = AsyncMock(spec=LLMClient)
    mock_llm.get_model_name.return_value = "test-model"
    mock_llm.generate_text = AsyncMock(return_value="Mocked LLM response for integration tests")
    mock_llm.get_provider.return_value = LLMProvider.GEMINI
    
    # Patch the create_client method to always return our mock
    original_create_client = LLMClientFactory.create_client
    
    def mock_create_client(provider, model_name=None, api_keys=None, retry_cfg=None, **kwargs):
        """Return the mocked LLM client for any provider/model combination."""
        return mock_llm
    
    # Replace the static method
    LLMClientFactory.create_client = staticmethod(mock_create_client)
    
    yield
    
    # Restore original (though not strictly necessary for tests)
    LLMClientFactory.create_client = original_create_client


@pytest_asyncio.fixture(scope="function")
async def clean_db(mongo_db: Database):
    """Clean the database before each test."""
    # Drop all collections before each test
    collections = await mongo_db.list_collection_names()
    for collection_name in collections:
        await mongo_db.drop_collection(collection_name)
    
    yield
    
   


@pytest_asyncio.fixture(scope="function")
async def test_app(mongo_client: AsyncMongoClient, mongo_db: Database, 
                   qdrant_client: AsyncQdrantClient, embedding_model: SentenceTransformer,
                   clean_db) -> FastAPI:  # Add clean_db dependency to ensure DB is cleaned before indexes are created
    """Create a test FastAPI app with real connections.
    
    Note: This fixture is function-scoped to ensure test isolation.
    Each test gets a fresh app state, but connections are reused (session-scoped).
    """
    import logging
    logger = logging.getLogger(__name__)
    from app.rag.embedding.metadata_index import setup_metadata_indexes, create_collection_if_not_exists
    from app.repositories.chunk_repository import ChunkRepository
    from app.db.users import seed_admin
    from unittest.mock import AsyncMock
    from app.core.clients.llm_clients import LLMClient, LLMProvider
    
    # Set app state with real connections (reused from session-scoped fixtures)
    app.state.mongo_client = mongo_client
    app.state.mongo_db = mongo_db
    app.state.qdrant_client = qdrant_client
    app.state.embedding_model = embedding_model
    app.state.kms = KMS(os.environ["KEY_ENCRYPTION_KEY"])
    
    # These operations are idempotent (check if exists before creating)
    collection_name = os.getenv("COLLECTION_NAME", "test_integration_collection")
    await create_collection_if_not_exists(qdrant_client, collection_name)
    await setup_metadata_indexes(qdrant_client, collection_name)
    
    # Seed admin user (idempotent - checks if exists)
    await seed_admin(mongo_db)
    
    # Initialize chunk repository indexes (idempotent)
    chunk_repo = ChunkRepository(mongo_db)
    await chunk_repo._ensure_indexes()
    
    # Mock LLM client for integration tests (no real API keys needed)
    # Note: LLMClientFactory.create_client is already patched by the mock_llm_factory fixture
    from app.core.utils.llm_utils import LLMClientFactory
    mock_llm = LLMClientFactory.create_client(LLMProvider.GEMINI)
    app.state.default_llm_provider = mock_llm
    
    return app


@pytest_asyncio.fixture
async def async_client(test_app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Create an async HTTP client for testing API endpoints."""
    from httpx import ASGITransport
    async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture
async def test_user(mongo_db: Database) -> dict:
    """Create a test user in the database."""
    from app.db.users import create_user, UserCreate, UserRole
    
    user_data = UserCreate(
        email="testuser@integration.example.com",
        password="testpassword123",
        role=UserRole.USER
    )
    
    user_id = await create_user(user_data, mongo_db)
    
    user = await mongo_db.users.find_one({"_id": user_id})
    return {
        "id": str(user["_id"]),
        "email": user["email"],
        "role": user["role"]
    }


@pytest_asyncio.fixture
async def test_admin(mongo_db: Database) -> dict:
    """Create a test admin user in the database."""
    from app.db.users import create_user, UserCreate, UserRole
    
    user_data = UserCreate(
        email="testadmin@integration.example.com",
        password="adminpassword123",
        role=UserRole.ADMIN
    )
    
    user_id = await create_user(user_data, mongo_db)
    
    user = await mongo_db.users.find_one({"_id": user_id})
    return {
        "id": str(user["_id"]),
        "email": user["email"],
        "role": user["role"]
    }


@pytest_asyncio.fixture
async def auth_token(async_client: AsyncClient, test_user: dict) -> str:
    """Get an authentication token for a test user."""
    response = await async_client.post(
        "/api/auth/login",
        json={
            "email": test_user["email"],
            "password": "testpassword123"
        }
    )
    
    if response.status_code != 200:
        # If login fails, create user first
        await async_client.post(
            "/api/auth/signup",
            json={
                "email": test_user["email"],
                "password": "testpassword123"
            }
        )
        response = await async_client.post(
            "/api/auth/login",
            json={
                "email": test_user["email"],
                "password": "testpassword123"
            }
        )
    
    data = response.json()
    return data["access_token"]


@pytest_asyncio.fixture
async def admin_token(async_client: AsyncClient, test_admin: dict) -> str:
    """Get an authentication token for a test admin user."""
    response = await async_client.post(
        "/api/auth/login",
        json={
            "email": test_admin["email"],
            "password": "adminpassword123"
        }
    )
    
    if response.status_code != 200:
        # If login fails, create admin first
        await async_client.post(
            "/api/auth/signup",
            json={
                "email": test_admin["email"],
                "password": "adminpassword123"
            }
        )
        response = await async_client.post(
            "/api/auth/login",
            json={
                "email": test_admin["email"],
                "password": "adminpassword123"
            }
        )
    
    data = response.json()
    return data["access_token"]


@pytest.fixture
def auth_headers(auth_token: str) -> dict:
    """Create authentication headers."""
    return {"Authorization": f"Bearer {auth_token}"}


@pytest.fixture
def admin_headers(admin_token: str) -> dict:
    """Create admin authentication headers."""
    return {"Authorization": f"Bearer {admin_token}"}

