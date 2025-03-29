# web_app/test_auth.py
"""
Unit tests for the UserManager class in auth.py.
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock # For mocking request if needed

# Assuming auth.py is adjacent or path is set correctly
from web_app.auth import UserManager, SECRET_KEY, ALGORITHM # Import constants if needed for token verification
from jose import jwt, JWTError

# --- Test Data ---

# Use a fixed set of users for predictable testing
MOCK_USERS_DATA = {
    "users": [
        {
            "username": "testuser",
            "password": "password123", # Plain password for simple test case
            "is_admin": False,
            "created_at": "2024-01-01T10:00:00Z"
        },
        {
            "username": "admin",
            # Example using bcrypt hash (generate one for testing if needed)
            # $2b$12$YourGeneratedSaltHere.......YourGeneratedHashHere.......
            # For simplicity, using plain password here too, but hash is preferred
            "password": "adminpassword",
            "is_admin": True,
            "created_at": "2024-01-01T10:00:00Z"
        },
        {
            "username": "nouser",
            # Example of user without plain password, only hash
            "password_hash": "$2b$12$dummySalt..............dummyHash...........................",
            "is_admin": False,
            "created_at": "2024-01-01T10:00:00Z"
        }
    ]
}

# --- Fixtures ---

@pytest.fixture
def user_manager() -> UserManager:
    """Provides a UserManager instance initialized with mock data."""
    # Pass the mock data directly to the constructor
    return UserManager(users=MOCK_USERS_DATA)

# --- Test Cases ---

# === Test authenticate() ===

def test_authenticate_valid_user_plain(user_manager: UserManager):
    """Test authentication with correct username and plain password."""
    user_info = user_manager.authenticate("testuser", "password123")
    assert user_info is not None
    assert user_info["username"] == "testuser"
    assert user_info["is_admin"] is False
    # Check that sensitive info is not returned (depends on implementation)
    assert "password" not in user_info
    assert "password_hash" not in user_info

def test_authenticate_valid_admin_plain(user_manager: UserManager):
    """Test authentication for admin user with plain password."""
    user_info = user_manager.authenticate("admin", "adminpassword")
    assert user_info is not None
    assert user_info["username"] == "admin"
    assert user_info["is_admin"] is True

@pytest.mark.skip(reason="Requires bcrypt and valid hash generation for setup")
def test_authenticate_valid_user_hash(user_manager: UserManager):
    """Test authentication with correct username and password matching hash (requires bcrypt)."""
    # Replace 'correct_password_for_nouser' with the actual password for the dummy hash
    user_info = user_manager.authenticate("nouser", "correct_password_for_nouser")
    assert user_info is not None
    assert user_info["username"] == "nouser"
    assert user_info["is_admin"] is False

def test_authenticate_invalid_password(user_manager: UserManager):
    """Test authentication with correct username but wrong password."""
    user_info = user_manager.authenticate("testuser", "wrongpassword")
    assert user_info is None

def test_authenticate_invalid_username(user_manager: UserManager):
    """Test authentication with a username that doesn't exist."""
    user_info = user_manager.authenticate("nonexistent", "password123")
    assert user_info is None

def test_authenticate_empty_credentials(user_manager: UserManager):
    """Test authentication with empty username or password."""
    assert user_manager.authenticate("", "password123") is None
    assert user_manager.authenticate("testuser", "") is None
    assert user_manager.authenticate("", "") is None

# === Test create_access_token() and verification ===

def test_create_access_token_structure(user_manager: UserManager):
    """Test that the created token has the expected structure and data."""
    username = "tokentest"
    is_admin = False
    data = {"sub": username, "is_admin": is_admin}
    token = user_manager.create_access_token(data)

    assert isinstance(token, str)
    assert len(token.split('.')) == 3 # Basic JWT structure check

    # Decode (without verification first for structure check, then with)
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], options={"verify_signature": False, "verify_exp": False})
    assert payload["sub"] == username
    assert payload["is_admin"] == is_admin
    assert "exp" in payload
    assert "iat" in payload

def test_create_access_token_expiration(user_manager: UserManager):
    """Test that the token expires correctly."""
    username = "exp_test"
    data = {"sub": username}
    # Use default expiration from the manager instance
    default_minutes = user_manager.ACCESS_TOKEN_EXPIRE_MINUTES
    token = user_manager.create_access_token(data)

    # Verify expiration is roughly correct
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM]) # Verifies expiration
    expected_exp = datetime.now(timezone.utc) + timedelta(minutes=default_minutes)
    actual_exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
    # Allow a small tolerance for execution time
    assert abs((expected_exp - actual_exp).total_seconds()) < 5

    # Test with custom expiration
    custom_delta = timedelta(seconds=1)
    custom_token = user_manager.create_access_token(data, expires_delta=custom_delta)
    custom_payload = jwt.decode(custom_token, SECRET_KEY, algorithms=[ALGORITHM]) # Verifies expiration
    expected_custom_exp = datetime.now(timezone.utc) + custom_delta
    actual_custom_exp = datetime.fromtimestamp(custom_payload["exp"], tz=timezone.utc)
    assert abs((expected_custom_exp - actual_custom_exp).total_seconds()) < 5


def test_create_access_token_expired(user_manager: UserManager):
    """Test that an expired token fails verification."""
    username = "expired_test"
    data = {"sub": username}
    # Create token that expired 1 minute ago
    expired_delta = timedelta(minutes=-1)
    expired_token = user_manager.create_access_token(data, expires_delta=expired_delta)

    # Decoding should raise ExpiredSignatureError
    with pytest.raises(jwt.ExpiredSignatureError):
        jwt.decode(expired_token, SECRET_KEY, algorithms=[ALGORITHM])


# === Test get_current_user() ===
# Note: Fully unit testing this is harder as it requires a mock Request object.
# These tests mock the request object simply.

def test_get_current_user_valid_token(user_manager: UserManager):
    """Test getting user info from a valid token in cookies."""
    username = "currentuser"
    is_admin = True
    token = user_manager.create_access_token({"sub": username, "is_admin": is_admin})

    mock_request = MagicMock()
    mock_request.cookies.get.return_value = token

    user_info = user_manager.get_current_user(mock_request)

    assert user_info is not None
    assert user_info["username"] == username
    assert user_info["is_admin"] == is_admin
    mock_request.cookies.get.assert_called_once_with("access_token")


def test_get_current_user_no_token(user_manager: UserManager):
    """Test getting user when no token cookie is present."""
    mock_request = MagicMock()
    mock_request.cookies.get.return_value = None # Simulate no cookie

    user_info = user_manager.get_current_user(mock_request)

    assert user_info is None
    mock_request.cookies.get.assert_called_once_with("access_token")

def test_get_current_user_invalid_token(user_manager: UserManager):
    """Test getting user with an invalid/malformed token."""
    mock_request = MagicMock()
    mock_request.cookies.get.return_value = "this.is.not_a_jwt"

    user_info = user_manager.get_current_user(mock_request)

    assert user_info is None

def test_get_current_user_expired_token(user_manager: UserManager):
    """Test getting user with an expired token."""
    expired_token = user_manager.create_access_token({"sub": "user"}, expires_delta=timedelta(minutes=-5))
    mock_request = MagicMock()
    mock_request.cookies.get.return_value = expired_token

    user_info = user_manager.get_current_user(mock_request)

    assert user_info is None