#!/usr/bin/env python3
"""
Simple test script to verify authentication functionality
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import backend
    print("Backend imported successfully")
except Exception as e:
    print(f"Backend import failed: {e}")
    sys.exit(1)

def test_user_registration():
    """Test user registration"""
    print("\nTesting User Registration...")
    
    # Test registration
    result = backend.register_user("testuser", 10)  # Use an available ID
    
    if isinstance(result, tuple):
        success, message = result
        if success:
            print(f"Registration successful: {message}")
            return True
        else:
            print(f"Registration failed: {message}")
            return False
    else:
        if result:
            print("Registration successful")
            return True
        else:
            print("Registration failed")
            return False

def test_get_users():
    """Test getting registered users"""
    print("\nTesting Get Users...")
    
    try:
        users = backend.get_registered_users()
        print(f"Found {len(users)} users: {users}")
        return len(users) > 0
    except Exception as e:
        print(f"Get users failed: {e}")
        return False

def main():
    """Run basic tests"""
    print("EEG System Basic Test")
    print("=" * 30)
    
    # Test 1: User Registration
    reg_result = test_user_registration()
    
    # Test 2: Get Users
    users_result = test_get_users()
    
    print("\nTest Results:")
    print(f"Registration: {'PASS' if reg_result else 'FAIL'}")
    print(f"Get Users: {'PASS' if users_result else 'FAIL'}")
    
    if reg_result and users_result:
        print("\nBasic functionality working!")
    else:
        print("\nSome tests failed.")

if __name__ == "__main__":
    main()