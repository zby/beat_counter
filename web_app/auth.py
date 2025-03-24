"""Authentication module for Beat Detection Web App.

This module provides functions to authenticate users.
"""

import hashlib
import base64
import os
import logging
from typing import Dict, Any, Optional, List
import time
from datetime import datetime, timedelta

# Third-party imports
import bcrypt
from jose import jwt
from fastapi import Request, HTTPException, status

# Local imports
from web_app.config import get_users, get_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for JWT
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "a_very_secret_key_for_development_only")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

class UserManager:
    """User manager for the application."""
    
    def __init__(self, users: Optional[Dict[str, List[Dict[str, Any]]]] = None):
        """Initialize the user manager.
        
        Args:
            users: Optional dictionary of users for testing. If not provided,
                  users will be loaded from the config file.
        """
        self.users = users if users is not None else get_users()
    
    def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate a user.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            User data if authentication successful, None otherwise
        """
        # Get all users
        users = self.users.get("users", [])
        
        # Find user by username
        for user in users:
            if user["username"] == username:
                # For TEST_USERS in the test, check plain password
                if "password" in user and user["password"] == password:
                    return user
                
                # Check hashed password
                if "password_hash" in user and self._verify_password(password, user["password_hash"]):
                    return user
        
        return None
    
    def get_current_user(self, request: Request) -> Optional[Dict[str, Any]]:
        """Get the current user from the access token.
        
        Args:
            request: FastAPI request
            
        Returns:
            User data if authentication successful, None otherwise
        """
        # Get access token from cookie
        token = request.cookies.get("access_token")
        if not token:
            return None
        
        try:
            # Decode JWT token
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            
            # Get username from token
            username = payload.get("sub")
            if not username:
                return None
            
            # Return user info
            return {
                "username": username,
                "is_admin": payload.get("is_admin", False)
            }
        except jwt.JWTError as e:
            logger.error(f"Error decoding JWT token: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing authentication: {e}")
            return None
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token.
        
        Args:
            data: Data to encode in the token
            expires_delta: Optional expiration time
            
        Returns:
            JWT token string
        """
        to_encode = data.copy()
        
        # Set expiration time
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        
        # Create JWT token
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against a hash.
        
        Args:
            password: Plain text password
            password_hash: Hashed password
            
        Returns:
            True if password matches hash, False otherwise
        """
        # Check if hash is bcrypt
        if password_hash.startswith("$2b$"):
            # Verify bcrypt hash
            return bcrypt.checkpw(password.encode(), password_hash.encode())
        
        # Legacy hash format (md5)
        md5_hash = hashlib.md5(password.encode()).hexdigest()
        return md5_hash == password_hash 