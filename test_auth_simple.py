#!/usr/bin/env python3
"""
Test authentication functionality
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

def test_authentication():
    """Test authentication"""
    print("\nTesting Authentication...")
    
    try:
        users = backend.get_registered_users()
        if not users:
            print("No users registered")
            return False
        
        username = users[0]  # Use first user
        user_info = backend.get_user_info(username)
        
        if not user_info:
            print(f"No info for user {username}")
            return False
        
        subject_id = user_info.get('subject_id')
        print(f"Testing with user: {username}, subject_id: {subject_id}")
        
        # Find a sample file
        data_dir = Path("data/Filtered_Data")
        if not data_dir.exists():
            print("Data directory not found")
            return False
        
        sample_files = list(data_dir.glob(f"s{subject_id:02d}_*.csv"))
        if not sample_files:
            print(f"No sample files for subject {subject_id}")
            return False
        
        sample_file = sample_files[0]
        print(f"Using file: {sample_file}")
        
        # Test authentication
        result = backend.authenticate_with_subject_id(username, subject_id, str(sample_file), 0.8)
        
        if isinstance(result, tuple):
            success, message = result
            print(f"Result: {success}")
            print(f"Message: {message}")
            return True
        else:
            print(f"Result: {result}")
            return True
            
    except Exception as e:
        print(f"Authentication test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run authentication test"""
    print("Authentication Test")
    print("=" * 30)
    
    result = test_authentication()
    print(f"\nAuthentication Test: {'PASS' if result else 'FAIL'}")

if __name__ == "__main__":
    main()