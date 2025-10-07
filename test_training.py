#!/usr/bin/env python3
"""
Test training functionality
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

def test_training():
    """Test model training"""
    print("\nTesting Model Training...")
    
    # Check users first
    users = backend.get_registered_users()
    print(f"Found {len(users)} users: {users}")
    
    if len(users) < 2:
        print("Need at least 2 users for training")
        return False
    
    # Check user data
    for user in users:
        info = backend.get_user_info(user)
        if info:
            print(f"User {user}: {info.get('data_segments', 0)} segments, data exists: {info.get('data_exists', False)}")
        else:
            print(f"User {user}: No info available")
    
    # Try training
    try:
        print("Starting training...")
        result = backend.train_model()
        
        if result:
            print("Training completed successfully!")
            
            # Check if model files were created
            model_path = Path("assets/model.pth")
            encoder_path = Path("assets/label_encoder.joblib")
            scaler_path = Path("assets/scaler.joblib")
            
            print(f"Model file: {'EXISTS' if model_path.exists() else 'MISSING'}")
            print(f"Encoder file: {'EXISTS' if encoder_path.exists() else 'MISSING'}")
            print(f"Scaler file: {'EXISTS' if scaler_path.exists() else 'MISSING'}")
            
            return True
        else:
            print("Training failed!")
            return False
            
    except Exception as e:
        print(f"Training crashed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run training test"""
    print("EEG Training Test")
    print("=" * 30)
    
    result = test_training()
    
    print(f"\nTraining Test: {'PASS' if result else 'FAIL'}")

if __name__ == "__main__":
    main()