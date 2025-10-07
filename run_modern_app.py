#!/usr/bin/env python3
"""
Launch script for the Modern EEG Biometric Authentication System
"""

import subprocess
import sys
import os
from pathlib import Path
import importlib.util

def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy',
        'torch',
        'scikit-learn',
        'streamlit-option-menu'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall missing packages with:")
        print("   pip install -r modern_requirements.txt")
        return False
    
    print("All required packages are installed!")
    return True

def setup_environment():
    """Setup the environment for the modern app."""
    current_dir = Path(__file__).parent
    
    # Create necessary directories
    directories = [
        'assets/models',
        'logs',
        'exports',
        'temp'
    ]
    
    for directory in directories:
        dir_path = current_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("Environment setup complete!")

def launch_app():
    """Launch the Streamlit application."""
    current_dir = Path(__file__).parent
    app_path = current_dir / "modern_app.py"
    
    if not app_path.exists():
        print("modern_app.py not found!")
        return False
    
    print("Launching Modern EEG Authentication System...")
    print("The app will open in your default web browser")
    print("URL: http://localhost:8501")
    print("\nFeatures available:")
    print("   - Modern web-based interface")
    print("   - Real-time EEG visualization")
    print("   - Multiple ML models (CNN, XGBoost, LightGBM)")
    print("   - Advanced analytics dashboard")
    print("   - Interactive signal analysis")
    print("   - Comprehensive reporting")
    print("\nPress Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        # Launch Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ])
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error launching application: {e}")
        return False
    
    return True

def main():
    """Main function to run the modern EEG app."""
    print("Modern EEG Biometric Authentication System")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\nTo install requirements:")
        print("   pip install -r modern_requirements.txt")
        return
    
    # Setup environment
    setup_environment()
    
    # Launch app
    launch_app()

if __name__ == "__main__":
    main()