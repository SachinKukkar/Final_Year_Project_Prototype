# EEG Project Cleanup Guide

## 🎯 Working GUI Application: gui_app.py

This document identifies the **essential files** for the working PyQt5 GUI application and lists files that can be safely removed.

---

## ✅ ESSENTIAL FILES (Keep These)

### Core Application Files
- **gui_app.py** - Main PyQt5 GUI application (PRIMARY FILE)
- **backend.py** - Authentication and user management logic
- **eeg_processing.py** - EEG signal processing functions
- **model_management.py** - CNN model architecture
- **database.py** - SQLite database management
- **config.py** - Configuration settings
- **themes.py** - UI theme definitions
- **main_window.ui** - Qt Designer UI layout file
- **requirements.txt** - Python dependencies

### Optional Enhancement Modules (Recommended to Keep)
- **dashboard.py** - Performance dashboard (accessed via hamburger menu)
- **signal_viewer.py** - EEG signal visualization tool
- **frequency_analyzer.py** - Frequency band analysis
- **performance_analyzer.py** - ROC curves and performance metrics
- **model_comparison.py** - Model comparison tool
- **settings.py** - Settings configuration panel

### Essential Directories
- **assets/** - Stores trained models, encoders, scalers, user data
- **data/Filtered_Data/** - EEG CSV data files (s01-s20)
- **logs/** - Application logs (optional but useful)

### Documentation
- **README.md** - Main documentation (from pinned context)
- **.gitignore** - Git ignore rules

---

## ❌ FILES TO REMOVE (Unnecessary/Duplicate)

### Duplicate/Alternative Apps
- **final_app.py** - Streamlit version (different from gui_app.py)
- **fixed_app.py** - Old/fixed version
- **modern_app.py** - Alternative modern version
- **modern_backend.py** - Alternative backend
- **simple_modern_app.py** - Simplified version
- **working_app.py** - Another alternative version
- **run_modern_app.py** - Runner for modern app

### Test Files
- **test_auth_fixes.py**
- **test_auth_simple.py**
- **test_auth.py**
- **test_eeg_system.py**
- **test_temp_file.py**
- **test_training.py**
- **simple_test.py**

### Alternative/Unused Modules
- **demo_modern_features.py**
- **modern_analytics.py**
- **modern_visualizer.py**
- **metrics.py** (if not used by dashboard)
- **utils.py** (check if used first)

### Alternative Documentation
- **README_MODERN.md** - Documentation for modern_app.py
- **IMPROVEMENTS_COMPARISON.md** - Comparison document
- **modern_requirements.txt** - Requirements for modern app

### Setup/Build Files
- **setup.py** - If not needed for installation

### Database Files (Optional Cleanup)
- **eeg_system.db** - Can be regenerated (backup first!)

### Exports Directory
- **exports/** - Can be cleaned periodically

---

## 🔧 How to Clean Up

### Option 1: Manual Cleanup
1. **Backup your project first!**
2. Review the "Files to Remove" list above
3. Delete files one by one or move to a backup folder
4. Test gui_app.py after cleanup

### Option 2: Automated Cleanup Script
Run the provided `cleanup_project.py` script (see below)

---

## 📋 Minimal Working Structure

After cleanup, your project should look like this:

```
EEG_Project_Final_Year_DEMO-main/
├── gui_app.py                 # ⭐ MAIN APPLICATION
├── backend.py
├── eeg_processing.py
├── model_management.py
├── database.py
├── config.py
├── themes.py
├── main_window.ui
├── requirements.txt
├── README.md
├── .gitignore
│
├── dashboard.py               # Optional enhancements
├── signal_viewer.py
├── frequency_analyzer.py
├── performance_analyzer.py
├── model_comparison.py
├── settings.py
│
├── assets/                    # Generated files
│   ├── models/
│   ├── model.pth
│   ├── label_encoder.joblib
│   ├── scaler.joblib
│   ├── users.json
│   └── data_*.npy
│
├── data/
│   └── Filtered_Data/         # EEG CSV files
│       ├── s01_ex01_s01.csv
│       └── ...
│
└── logs/                      # Optional logs
```

---

## 🚀 Running the Application

After cleanup:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the GUI application
python gui_app.py
```

---

## ⚠️ Important Notes

1. **Backup First**: Always backup your project before deleting files
2. **Test After Cleanup**: Run `python gui_app.py` to ensure everything works
3. **Database**: The `eeg_system.db` will be recreated automatically if deleted
4. **Model Files**: Keep `assets/` directory - it contains trained models
5. **Data Files**: Keep `data/Filtered_Data/` - contains EEG training data

---

## 🔍 Verification Checklist

After cleanup, verify these features work:

- [ ] GUI launches successfully
- [ ] User registration works
- [ ] Model training works
- [ ] Authentication works
- [ ] Dashboard opens (hamburger menu)
- [ ] Signal viewer works
- [ ] Frequency analyzer works
- [ ] Theme toggle works (light/dark)
- [ ] Settings panel opens

---

## 📞 Support

If you encounter issues after cleanup:
1. Check that all essential files are present
2. Verify `requirements.txt` dependencies are installed
3. Check `logs/` directory for error messages
4. Restore from backup if needed
