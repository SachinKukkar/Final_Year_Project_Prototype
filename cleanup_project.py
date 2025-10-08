"""
Automated cleanup script for EEG Project
Removes unnecessary files while keeping gui_app.py and its dependencies
"""
import os
import shutil
from pathlib import Path

# Files to remove (unnecessary/duplicate files)
FILES_TO_REMOVE = [
    # Duplicate/Alternative Apps
    'final_app.py',
    'fixed_app.py',
    'modern_app.py',
    'modern_backend.py',
    'simple_modern_app.py',
    'working_app.py',
    'run_modern_app.py',
    
    # Test Files
    'test_auth_fixes.py',
    'test_auth_simple.py',
    'test_auth.py',
    'test_eeg_system.py',
    'test_temp_file.py',
    'test_training.py',
    'simple_test.py',
    
    # Alternative/Unused Modules
    'demo_modern_features.py',
    'modern_analytics.py',
    'modern_visualizer.py',
    'metrics.py',
    'utils.py',
    
    # Alternative Documentation
    'README_MODERN.md',
    'IMPROVEMENTS_COMPARISON.md',
    'modern_requirements.txt',
    
    # Setup Files
    'setup.py',
    
    # Settings file (will be regenerated)
    'settings.json',
]

# Essential files that must be kept
ESSENTIAL_FILES = [
    'gui_app.py',
    'backend.py',
    'eeg_processing.py',
    'model_management.py',
    'database.py',
    'config.py',
    'themes.py',
    'main_window.ui',
    'requirements.txt',
    'README.md',
    '.gitignore',
    'dashboard.py',
    'signal_viewer.py',
    'frequency_analyzer.py',
    'performance_analyzer.py',
    'model_comparison.py',
    'settings.py',
]

def create_backup():
    """Create a backup of the project before cleanup."""
    backup_dir = Path('backup_before_cleanup')
    if backup_dir.exists():
        print(f"⚠️  Backup directory already exists: {backup_dir}")
        response = input("Overwrite existing backup? (yes/no): ").lower()
        if response != 'yes':
            print("❌ Backup cancelled. Cleanup aborted.")
            return False
        shutil.rmtree(backup_dir)
    
    print("📦 Creating backup...")
    try:
        # Copy only essential files and directories
        backup_dir.mkdir()
        
        # Copy essential files
        for file in ESSENTIAL_FILES:
            if Path(file).exists():
                shutil.copy2(file, backup_dir / file)
        
        # Copy essential directories
        for dir_name in ['assets', 'data', 'logs']:
            src_dir = Path(dir_name)
            if src_dir.exists():
                shutil.copytree(src_dir, backup_dir / dir_name)
        
        print(f"✅ Backup created: {backup_dir.absolute()}")
        return True
    except Exception as e:
        print(f"❌ Backup failed: {e}")
        return False

def cleanup_files():
    """Remove unnecessary files."""
    print("\n🧹 Starting cleanup...")
    removed_count = 0
    not_found_count = 0
    
    for file in FILES_TO_REMOVE:
        file_path = Path(file)
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"  ✓ Removed: {file}")
                removed_count += 1
            except Exception as e:
                print(f"  ✗ Failed to remove {file}: {e}")
        else:
            not_found_count += 1
    
    print(f"\n📊 Cleanup Summary:")
    print(f"  • Files removed: {removed_count}")
    print(f"  • Files not found: {not_found_count}")
    
    return removed_count

def verify_essential_files():
    """Verify all essential files are present."""
    print("\n🔍 Verifying essential files...")
    missing_files = []
    
    for file in ESSENTIAL_FILES:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("⚠️  Warning: Some essential files are missing:")
        for file in missing_files:
            print(f"  • {file}")
        return False
    else:
        print("✅ All essential files present")
        return True

def cleanup_exports():
    """Clean up exports directory."""
    exports_dir = Path('exports')
    if exports_dir.exists():
        response = input("\n🗑️  Clean up exports directory? (yes/no): ").lower()
        if response == 'yes':
            try:
                shutil.rmtree(exports_dir)
                exports_dir.mkdir()
                print("✅ Exports directory cleaned")
            except Exception as e:
                print(f"❌ Failed to clean exports: {e}")

def main():
    """Main cleanup function."""
    print("=" * 60)
    print("🧠 EEG Project Cleanup Script")
    print("=" * 60)
    print("\nThis script will:")
    print("  1. Create a backup of essential files")
    print("  2. Remove unnecessary/duplicate files")
    print("  3. Verify essential files are present")
    print("\nFiles to be removed:")
    for file in FILES_TO_REMOVE[:5]:
        print(f"  • {file}")
    print(f"  ... and {len(FILES_TO_REMOVE) - 5} more files")
    
    response = input("\n⚠️  Continue with cleanup? (yes/no): ").lower()
    if response != 'yes':
        print("❌ Cleanup cancelled")
        return
    
    # Step 1: Create backup
    if not create_backup():
        print("\n❌ Cleanup aborted due to backup failure")
        return
    
    # Step 2: Cleanup files
    removed_count = cleanup_files()
    
    # Step 3: Verify essential files
    verify_essential_files()
    
    # Step 4: Optional cleanup
    cleanup_exports()
    
    print("\n" + "=" * 60)
    print("✅ Cleanup Complete!")
    print("=" * 60)
    print("\n📋 Next Steps:")
    print("  1. Test the application: python gui_app.py")
    print("  2. If issues occur, restore from: backup_before_cleanup/")
    print("  3. Review CLEANUP_GUIDE.md for details")
    print("\n🎯 Your project is now clean and optimized!")

if __name__ == "__main__":
    main()
