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
    print("✅ Backend imported successfully")
except Exception as e:
    print(f"❌ Backend import failed: {e}")
    sys.exit(1)

def test_user_registration():
    """Test user registration"""
    print("\n🧪 Testing User Registration...")
    
    # Test registration
    result = backend.register_user("testuser", 1)
    
    if isinstance(result, tuple):
        success, message = result
        if success:
            print(f"✅ Registration successful: {message}")
            return True
        else:
            print(f"❌ Registration failed: {message}")
            return False
    else:
        if result:
            print("✅ Registration successful")
            return True
        else:
            print("❌ Registration failed")
            return False

def test_get_users():
    """Test getting registered users"""
    print("\n🧪 Testing Get Users...")
    
    try:
        users = backend.get_registered_users()
        print(f"✅ Found {len(users)} users: {users}")
        return len(users) > 0
    except Exception as e:
        print(f"❌ Get users failed: {e}")
        return False

def test_user_info():
    """Test getting user info"""
    print("\n🧪 Testing User Info...")
    
    try:
        users = backend.get_registered_users()
        if users:
            username = users[0]
            info = backend.get_user_info(username)
            if info:
                print(f"✅ User info for {username}:")
                print(f"   Subject ID: {info.get('subject_id')}")
                print(f"   Data segments: {info.get('data_segments')}")
                print(f"   Data exists: {info.get('data_exists')}")
                return True
            else:
                print(f"❌ No info found for user {username}")
                return False
        else:
            print("❌ No users to test")
            return False
    except Exception as e:
        print(f"❌ User info test failed: {e}")
        return False

def test_authentication():
    """Test authentication with sample file"""
    print("\n🧪 Testing Authentication...")
    
    try:
        users = backend.get_registered_users()
        if not users:
            print("❌ No users registered for authentication test")
            return False
        
        username = users[0]
        user_info = backend.get_user_info(username)
        
        if not user_info:
            print(f"❌ No info for user {username}")
            return False
        
        subject_id = user_info.get('subject_id')
        if not subject_id:
            print(f"❌ No subject ID for user {username}")
            return False
        
        # Look for a sample EEG file
        data_dir = Path("data/Filtered_Data")
        if not data_dir.exists():
            print("❌ Data directory not found")
            return False
        
        # Find a file for this subject
        sample_files = list(data_dir.glob(f"s{subject_id:02d}_*.csv"))
        if not sample_files:
            print(f"❌ No sample files found for subject {subject_id}")
            return False
        
        sample_file = sample_files[0]
        print(f"📁 Using sample file: {sample_file}")
        
        # Test authentication
        result = backend.authenticate_with_subject_id(username, subject_id, str(sample_file), 0.8)
        
        if isinstance(result, tuple):
            success, message = result
            print(f"🔐 Authentication result: {success}")
            print(f"📝 Message: {message}")
            return True
        else:
            print(f"🔐 Authentication result: {result}")
            return True
            
    except Exception as e:
        print(f"❌ Authentication test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧠 EEG System Test Suite")
    print("=" * 40)
    
    tests = [
        ("User Registration", test_user_registration),
        ("Get Users", test_get_users),
        ("User Info", test_user_info),
        ("Authentication", test_authentication)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 40)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! System is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()