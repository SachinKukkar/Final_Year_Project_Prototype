#!/usr/bin/env python3
"""
Test authentication with temp file
"""

import sys
import os
import shutil
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import backend
    print("Backend imported successfully")
except Exception as e:
    print(f"Backend import failed: {e}")
    sys.exit(1)

def test_temp_file_auth():
    """Test authentication with temp file"""
    print("\nTesting Authentication with temp file...")
    
    try:
        users = backend.get_registered_users()
        if not users:
            print("No users registered")
            return False
        
        username = users[0]  # Use first user
        user_info = backend.get_user_info(username)
        subject_id = user_info.get('subject_id')
        
        # Find a sample file
        data_dir = Path("data/Filtered_Data")
        sample_files = list(data_dir.glob(f"s{subject_id:02d}_*.csv"))
        if not sample_files:
            print(f"No sample files for subject {subject_id}")
            return False
        
        sample_file = sample_files[0]
        
        # Create temp file (simulate uploaded file)
        temp_file = f"temp_{sample_file.name}"
        shutil.copy(sample_file, temp_file)
        
        print(f"Created temp file: {temp_file}")
        print(f"Testing with user: {username}, subject_id: {subject_id}")
        
        # Test authentication
        result = backend.authenticate_with_subject_id(username, subject_id, temp_file, 0.8)
        
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        if isinstance(result, tuple):
            success, message = result
            print(f"Result: {success}")
            print(f"Message: {message}")
            return success
        else:
            print(f"Result: {result}")
            return result
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run temp file authentication test"""
    print("Temp File Authentication Test")
    print("=" * 40)
    
    result = test_temp_file_auth()
    print(f"\nTemp File Auth Test: {'PASS' if result else 'FAIL'}")

if __name__ == "__main__":
    main()