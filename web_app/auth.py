"""Authentication module for Beat Detection Web App.

This module provides functions to authenticate users.
"""

import hashlib
import base64
import os
import logging
from typing import Dict, Any, Optional, List
import time

# FIX: Ensure datetime and timedelta are imported
from datetime import datetime, timedelta, timezone

# Third-party imports
import bcrypt
from jose import jwt
from fastapi import Request, HTTPException, status

# Local imports
# Avoid circular dependency if auth is needed by config; get_users is okay if config loads first.
# Consider passing users directly instead of relying on get_users() inside UserManager.
from web_app.config import get_users  # Keep for now, but consider alternatives

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for JWT (Module level)
# Consider moving these to central config (e.g., Config object) eventually
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "a_very_secret_key_for_development_only")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours (Default)


class UserManager:
    """User manager for the application."""

    def __init__(self, users: Optional[Dict[str, List[Dict[str, Any]]]] = None):
        """Initialize the user manager.

        Args:
            users: Optional dictionary of users for testing. If not provided,
                  users will be loaded from the config file via get_users().
        """
        # Load users - careful with get_users() if it depends on config that might need auth later
        self._users_data = users if users is not None else get_users()

        # FIX: Make the expiration time accessible via the instance
        self.ACCESS_TOKEN_EXPIRE_MINUTES = ACCESS_TOKEN_EXPIRE_MINUTES

    def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate a user.

        Args:
            username: Username
            password: Password

        Returns:
            User data dictionary if authentication successful, None otherwise.
            (Note: Returning sensitive info like password hashes is generally discouraged)
        """
        # Access the loaded user data
        users = self._users_data.get("users", [])

        # Find user by username
        for user_dict in users:
            # Ensure comparison is safe (case-insensitive/normalized?)
            if user_dict.get("username") == username:
                # For TEST_USERS in the test (plain password check - less secure)
                if "password" in user_dict and user_dict["password"] == password:
                    logger.info(
                        f"User '{username}' authenticated via plain password (testing only?)."
                    )
                    # Return a copy without sensitive info if possible
                    return {
                        k: v
                        for k, v in user_dict.items()
                        if k != "password" and k != "password_hash"
                    }

                # Check hashed password using bcrypt if present
                if "password_hash" in user_dict and self._verify_password(
                    password, user_dict["password_hash"]
                ):
                    logger.info(f"User '{username}' authenticated via hashed password.")
                    # Return a copy without sensitive info
                    return {
                        k: v
                        for k, v in user_dict.items()
                        if k != "password" and k != "password_hash"
                    }

                # Check legacy md5 hash if present and bcrypt failed/missing (less secure)
                if "password_md5" in user_dict and self._verify_md5(
                    password, user_dict["password_md5"]
                ):
                    logger.warning(
                        f"User '{username}' authenticated via legacy MD5 hash."
                    )
                    # Return a copy without sensitive info
                    return {
                        k: v
                        for k, v in user_dict.items()
                        if k != "password" and k != "password_md5"
                    }

        logger.warning(f"Authentication failed for username: '{username}'")
        return None

    def get_current_user(self, request: Request) -> Optional[Dict[str, Any]]:
        """Get the current user's info from the access token cookie.

        Args:
            request: FastAPI request

        Returns:
            User data dict (e.g., {"username": "...", "is_admin": ...}) or None if invalid/missing token.
        """
        token = request.cookies.get("access_token")
        if not token:
            return None

        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: Optional[str] = payload.get("sub")
            if not username:
                logger.warning("JWT token payload missing 'sub' (username).")
                return None

            # Validate token expiration implicitly via jwt.decode
            # Return essential, non-sensitive user info
            return {
                "username": username,
                "is_admin": payload.get(
                    "is_admin", False
                ),  # Include admin status from token
            }
        except jwt.ExpiredSignatureError:
            logger.info("JWT token has expired.")
            return None
        except jwt.JWTError as e:
            logger.error(f"Error decoding JWT token: {e}")
            return None
        except Exception as e:
            # Catch unexpected errors during token processing
            logger.error(f"Unexpected error processing authentication token: {e}")
            return None

    def create_access_token(
        self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a JWT access token.

        Args:
            data: Data to encode in the token (e.g., {"sub": username, "is_admin": True})
            expires_delta: Optional expiration time override. If None, uses the default.

        Returns:
            JWT token string.
        """
        to_encode = data.copy()

        # Set expiration time
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            # Use the instance attribute for default expiration
            expire = datetime.now(timezone.utc) + timedelta(
                minutes=self.ACCESS_TOKEN_EXPIRE_MINUTES
            )

        to_encode.update(
            {"exp": expire, "iat": datetime.now(timezone.utc)}
        )  # Add issued-at time

        # Create JWT token
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a plain password against a hashed password (primarily bcrypt)."""
        if not plain_password or not hashed_password:
            return False
        try:
            # Check if hash is likely bcrypt
            if hashed_password.startswith("$2b$"):
                return bcrypt.checkpw(
                    plain_password.encode("utf-8"), hashed_password.encode("utf-8")
                )
            else:
                # If not bcrypt, assume it's not supported by this method
                logger.warning(
                    "Password hash verification attempted on non-bcrypt hash."
                )
                return False
        except ValueError:
            # Handle potential errors if hash is malformed
            logger.error("Error verifying password: Malformed hash?")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during password verification: {e}")
            return False

    def _verify_md5(self, plain_password: str, md5_hash: str) -> bool:
        """Verify a plain password against a legacy MD5 hash."""
        if not plain_password or not md5_hash:
            return False
        # Compare calculated MD5 hash with the stored one
        calculated_hash = hashlib.md5(plain_password.encode("utf-8")).hexdigest()
        return calculated_hash == md5_hash
