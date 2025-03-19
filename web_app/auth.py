"""Authentication module for the beat detection web app.

This module provides functions for user authentication, session management, 
and protecting routes that require authentication.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import RedirectResponse

# Import config module
from web_app.config import get_users, save_users, get_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "a_very_secret_key_for_development_only")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Security scheme for token authentication
security = HTTPBearer()


class AuthManager:
    """Handles authentication-related operations."""
    
    def __init__(self, users: Optional[Dict[str, List[Dict[str, Any]]]] = None):
        """Initialize the auth manager.
        
        Args:
            users: Optional dictionary of users in the format {"users": [{"username": str, "password": str, ...}, ...]}
                  If not provided, users will be loaded from the config file.
        """
        self._users = users
    
    def get_users(self) -> List[Dict[str, Any]]:
        """Get all users from the users file or memory."""
        if self._users is not None:
            return self._users.get("users", [])
        users_data = get_users()
        return users_data.get("users", [])
    
    def save_users(self, users: List[Dict[str, Any]]) -> bool:
        """Save the users list to the users file or memory."""
        if self._users is not None:
            self._users = {"users": users}
            return True
        return save_users({"users": users})
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate a user with username and password."""
        users = self.get_users()
        
        for user in users:
            if user["username"] == username and user["password"] == password:
                return user
        
        return None
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token with the given data."""
        to_encode = data.copy()
        
        # Set expiration time
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        
        # Encode and return token
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    def decode_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Decode a JWT token and return the payload."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except jwt.PyJWTError as e:
            logger.error(f"Error decoding token: {e}")
            return None
    
    def get_current_user(self, request: Request) -> Optional[Dict[str, Any]]:
        """Get the current user from the session."""
        token = request.cookies.get("access_token")
        
        if not token:
            return None
        
        payload = self.decode_token(token)
        if not payload:
            return None
        
        # Return user info from payload
        return {
            "username": payload.get("sub"),
            "is_admin": payload.get("is_admin", False)
        }


# Create a global instance of the auth manager
auth_manager = AuthManager()


async def get_current_user_from_cookie(request: Request) -> Optional[Dict[str, Any]]:
    """Get the current user from the session cookie."""
    return auth_manager.get_current_user(request)


async def require_auth(request: Request) -> Dict[str, Any]:
    """Dependency that requires a valid authentication."""
    user = auth_manager.get_current_user(request)
    
    if not user:
        # Check if this is an AJAX request
        is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
        
        if is_ajax:
            # For AJAX requests, always return 401 Unauthorized
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated"
            )
        else:
            # For web requests, redirect to login page
            redirect_url = f"/login?next={request.url.path}"
            raise HTTPException(
                status_code=status.HTTP_307_TEMPORARY_REDIRECT,
                headers={"Location": redirect_url},
                detail="Not authenticated"
            )
    
    return user 