"""Authentication module for the beat detection web app.

This module provides functions for user authentication, session management, 
and protecting routes that require authentication.
"""

import json
import os
import pathlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import RedirectResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
USERS_FILE = pathlib.Path(__file__).parent / "users.json"
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "a_very_secret_key_for_development_only")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Security scheme for token authentication
security = HTTPBearer()


class AuthManager:
    """Handles authentication-related operations."""
    
    def __init__(self, users_file_path: pathlib.Path = USERS_FILE):
        """Initialize the auth manager with a path to the users file."""
        self.users_file_path = users_file_path
        self._ensure_users_file_exists()
    
    def _ensure_users_file_exists(self) -> None:
        """Ensure the users file exists, creating it if necessary."""
        if not self.users_file_path.exists():
            # Create default users file
            default_users = {
                "users": [
                    {
                        "username": "admin",
                        "password": "admin123",
                        "is_admin": True,
                        "created_at": datetime.now().isoformat()
                    }
                ]
            }
            
            # Write to file
            with open(self.users_file_path, "w") as f:
                json.dump(default_users, f, indent=4)
            
            logger.info(f"Created default users file at {self.users_file_path}")
    
    def get_users(self) -> List[Dict[str, Any]]:
        """Get all users from the users file."""
        try:
            with open(self.users_file_path, "r") as f:
                data = json.load(f)
                return data.get("users", [])
        except Exception as e:
            logger.error(f"Error reading users file: {e}")
            return []
    
    def save_users(self, users: List[Dict[str, Any]]) -> bool:
        """Save the users list to the users file."""
        try:
            with open(self.users_file_path, "w") as f:
                json.dump({"users": users}, f, indent=4)
            return True
        except Exception as e:
            logger.error(f"Error saving users file: {e}")
            return False
    
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
        # Redirect to login page for web requests
        if request.url.path.startswith("/api"):
            # For API requests, return 401 Unauthorized
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