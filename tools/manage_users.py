#!/usr/bin/env python3
"""User management script for Beat Detection Web App.

This script provides command-line utilities for managing users.
"""

import argparse
import os
import sys
import secrets
from datetime import datetime

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import config module
from web_app.config import get_users, save_users

def list_users():
    """List all users."""
    users_data = get_users()
    users = users_data.get("users", [])
    
    if not users:
        print("No users found.")
        return
    
    print(f"Found {len(users)} users:")
    for i, user in enumerate(users, 1):
        role = "Admin" if user.get("is_admin") else "User"
        print(f"{i}. {user['username']} ({role})")


def add_user(username, password=None, is_admin=False):
    """Add a new user."""
    users_data = get_users()
    users = users_data.get("users", [])
    
    # Check if user already exists
    for user in users:
        if user["username"] == username:
            print(f"User '{username}' already exists.")
            return False
    
    # Generate password if not provided
    if not password:
        password = secrets.token_urlsafe(8)
        print(f"Generated password: {password}")
    
    # Create new user
    new_user = {
        "username": username,
        "password": password,
        "is_admin": is_admin,
        "created_at": datetime.now().isoformat()
    }
    
    users.append(new_user)
    
    # Save users
    if save_users({"users": users}):
        print(f"User '{username}' added successfully.")
        return True
    else:
        print(f"Failed to add user '{username}'.")
        return False


def delete_user(username):
    """Delete a user."""
    users_data = get_users()
    users = users_data.get("users", [])
    
    # Check if user exists
    user_found = False
    for i, user in enumerate(users):
        if user["username"] == username:
            user_found = True
            del users[i]
            break
    
    if not user_found:
        print(f"User '{username}' not found.")
        return False
    
    # Save users
    if save_users({"users": users}):
        print(f"User '{username}' deleted successfully.")
        return True
    else:
        print(f"Failed to delete user '{username}'.")
        return False


def change_password(username, password):
    """Change a user's password."""
    users_data = get_users()
    users = users_data.get("users", [])
    
    # Check if user exists
    user_found = False
    for user in users:
        if user["username"] == username:
            user_found = True
            user["password"] = password
            break
    
    if not user_found:
        print(f"User '{username}' not found.")
        return False
    
    # Save users
    if save_users({"users": users}):
        print(f"Password for user '{username}' changed successfully.")
        return True
    else:
        print(f"Failed to change password for user '{username}'.")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="User management for Beat Detection Web App")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List users command
    list_parser = subparsers.add_parser("list", help="List all users")
    
    # Add user command
    add_parser = subparsers.add_parser("add", help="Add a new user")
    add_parser.add_argument("username", help="Username for the new user")
    add_parser.add_argument("--password", help="Password for the new user (auto-generated if not provided)")
    add_parser.add_argument("--admin", action="store_true", help="Make the user an admin")
    
    # Delete user command
    delete_parser = subparsers.add_parser("delete", help="Delete a user")
    delete_parser.add_argument("username", help="Username of the user to delete")
    
    # Change password command
    password_parser = subparsers.add_parser("password", help="Change a user's password")
    password_parser.add_argument("username", help="Username of the user")
    password_parser.add_argument("password", help="New password")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_users()
    elif args.command == "add":
        add_user(args.username, args.password, args.admin)
    elif args.command == "delete":
        delete_user(args.username)
    elif args.command == "password":
        change_password(args.username, args.password)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 