# gui_app.py - Quick Reference Guide

## 🎯 Overview

**gui_app.py** is the main PyQt5-based GUI application for EEG Biometric Authentication System.

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python gui_app.py
```

---

## 🎨 Features

### 1. User Registration
- **Username Input**: Enter unique username
- **Subject ID Selection**: Choose from 1-20 (based on available EEG data)
- **Register Button**: Registers user with EEG data from `data/Filtered_Data/`
- **De-register Button**: Removes user and their data

### 2. Model Training
- **Train Button**: Trains CNN model on all registered users
- Requires at least 2 registered users
- Shows progress in status bar
- Saves model to `assets/model.pth`

### 3. Authentication
- **Browse Button**: Select EEG CSV file for authentication
- **Authenticate Button**: Verifies user identity
- Shows confidence score and result

### 4. Hamburger Menu (☰)
Access advanced tools:
- **👥 Show Users**: View all registered users
- **📈 Signal Viewer**: Visualize EEG signals
- **🌊 Frequency Analyzer**: Analyze frequency bands
- **📉 Performance Analysis**: View ROC curves and metrics
- **🔬 Model Comparison**: Compare different ML models
- **⚙️ Settings**: Configure system parameters

### 5. Dashboard (📊)
- View system statistics
- Authentication success rates
- User analytics
- Export reports

### 6. Theme Toggle (🌙/☀️)
- Switch between dark and light themes
- Persistent across sessions

---

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+R` | Register User |
| `Ctrl+T` | Train Model |
| `Ctrl+A` | Authenticate |
| `Ctrl+D` | Open Dashboard |
| `Ctrl+S` | Open Settings |
| `F1` | Show Help |
| `Escape` | Close Dialogs |

---

## 📁 File Dependencies

### Core Backend Files
```
gui_app.py
├── backend.py              # Authentication logic
│   ├── eeg_processing.py   # Signal processing
│   ├── model_management.py # CNN model
│   └── database.py         # SQLite database
├── themes.py               # UI themes
├── config.py               # Configuration
└── main_window.ui          # UI layout
```

### Optional Enhancement Modules
```
gui_app.py (hamburger menu)
├── dashboard.py            # Performance dashboard
├── signal_viewer.py        # Signal visualization
├── frequency_analyzer.py   # Frequency analysis
├── performance_analyzer.py # ROC curves
├── model_comparison.py     # Model comparison
└── settings.py             # Settings panel
```

---

## 🗂️ Data Structure

### Input Data Format
```
data/Filtered_Data/
├── s01_ex01_s01.csv  # Subject 1, Exercise 1, Session 1
├── s01_ex02_s01.csv  # Subject 1, Exercise 2, Session 1
└── ...
```

**CSV Format:**
- Columns: `P4`, `Cz`, `F8`, `T7` (EEG channels)
- No headers
- Numeric values only

### Generated Files
```
assets/
├── model.pth              # Trained CNN model
├── label_encoder.joblib   # Label encoder
├── scaler.joblib          # Feature scaler
├── users.json             # User registry
└── data_<username>.npy    # User EEG data
```

---

## 🔧 Configuration (config.py)

### EEG Processing
- **CHANNELS**: `['P4', 'Cz', 'F8', 'T7']`
- **WINDOW_SIZE**: `256` samples
- **STEP_SIZE**: `128` samples (50% overlap)
- **SAMPLING_RATE**: `256` Hz

### Model Training
- **BATCH_SIZE**: `32`
- **LEARNING_RATE**: `0.001`
- **NUM_EPOCHS**: `50`
- **PATIENCE**: `5` (early stopping)
- **DROPOUT_RATE**: `0.5`

### Authentication
- **AUTH_THRESHOLD**: `0.90` (90% confidence)
- **MAJORITY_VOTE_THRESHOLD**: `0.5` (>50% segments must match)

---

## 🏗️ Architecture

### CNN Model (EEG_CNN_Improved)
```
Input: (batch, 256, 4) - 256 samples × 4 channels
├── Conv Block 1: 4 → 32 channels
├── Conv Block 2: 32 → 64 channels
├── Conv Block 3: 64 → 128 channels
├── Conv Block 4: 128 → 256 channels
├── Flatten
├── FC Layer: → 256
├── FC Layer: → 128
└── Output: → num_classes
```

### Authentication Flow
```
1. Load EEG file
2. Segment into windows (256 samples)
3. Normalize with scaler
4. Predict each segment
5. Majority voting
6. Return result + confidence
```

---

## 🎯 User Workflow

### First Time Setup
1. **Launch**: `python gui_app.py`
2. **Register Users**:
   - Enter username (e.g., "Alice")
   - Select Subject ID (e.g., 1)
   - Click "Register User"
   - Repeat for at least 2 users
3. **Train Model**:
   - Click "Train Model on All Registered Users"
   - Wait for training to complete (~5 minutes)

### Authentication
1. **Browse**: Select EEG file (e.g., `s01_ex05.csv`)
2. **Enter Details**:
   - Username: "Alice"
   - Subject ID: 1
3. **Authenticate**: Click "Authenticate"
4. **Result**: View confidence score and access decision

---

## 🐛 Troubleshooting

### Issue: "No users registered"
**Solution**: Register at least 2 users before training

### Issue: "Model not trained"
**Solution**: Click "Train Model" after registering users

### Issue: "Subject ID mismatch"
**Solution**: Ensure Subject ID matches the registered user's ID

### Issue: "No valid EEG data segments"
**Solution**: Check CSV file format (4 columns: P4, Cz, F8, T7)

### Issue: Dashboard/Tools not opening
**Solution**: Check if optional modules are present:
- `dashboard.py`
- `signal_viewer.py`
- `frequency_analyzer.py`
- etc.

---

## 📊 Database Schema

### users table
```sql
- id: INTEGER PRIMARY KEY
- username: TEXT UNIQUE
- subject_id: INTEGER
- data_segments: INTEGER
- created_at: TIMESTAMP
- updated_at: TIMESTAMP
```

### auth_logs table
```sql
- id: INTEGER PRIMARY KEY
- username: TEXT
- success: BOOLEAN
- confidence: REAL
- reason: TEXT
- timestamp: TIMESTAMP
```

---

## 🔒 Security Features

1. **Biometric Authentication**: Uses unique EEG patterns
2. **Confidence Thresholds**: Rejects low-confidence predictions
3. **Majority Voting**: Multiple segments must agree
4. **Subject ID Validation**: Prevents ID spoofing
5. **File Validation**: Checks EEG data quality

---

## 📈 Performance Metrics

### Expected Performance
- **Training Accuracy**: >95%
- **Authentication Confidence**: >90%
- **Processing Time**: <2 seconds per file
- **Model Size**: ~5-10 MB

### Monitoring
- View statistics in Dashboard (📊)
- Check `logs/` directory for detailed logs
- Use Performance Analyzer for ROC curves

---

## 🎨 UI Customization

### Themes
- **Dark Theme** (default): Professional dark mode
- **Light Theme**: Clean light mode
- Toggle with 🌙/☀️ button

### Styling
- Themes defined in `themes.py`
- Uses Qt stylesheets
- Customizable colors and fonts

---

## 📝 Code Structure

### Main Window Class
```python
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        # Load UI from main_window.ui
        # Setup icons, shortcuts, toolbar
        # Connect button signals
    
    def register_clicked(self):
        # Handle user registration
    
    def train_clicked(self):
        # Handle model training
    
    def authenticate_clicked(self):
        # Handle authentication
    
    def show_dashboard(self):
        # Open dashboard window
```

---

## 🔄 Update & Maintenance

### Adding New Users
1. Add EEG data files to `data/Filtered_Data/`
2. Register user via GUI
3. Retrain model

### Updating Model
1. Modify `model_management.py`
2. Retrain with new architecture
3. Test authentication

### Backup
```bash
# Backup essential files
cp -r assets/ assets_backup/
cp users.json users_backup.json
cp eeg_system.db eeg_system_backup.db
```

---

## 📞 Support

For issues or questions:
1. Check `logs/` directory for errors
2. Review this reference guide
3. Check `CLEANUP_GUIDE.md` for file structure
4. Verify all dependencies are installed

---

## 🎓 Learning Resources

### Understanding the Code
- **gui_app.py**: PyQt5 GUI implementation
- **backend.py**: Core authentication algorithms
- **model_management.py**: Deep learning model
- **eeg_processing.py**: Signal processing techniques

### Key Concepts
- **EEG Biometrics**: Using brain signals for identification
- **CNN**: Convolutional Neural Networks for pattern recognition
- **Windowing**: Segmenting signals for analysis
- **Majority Voting**: Ensemble decision making

---

**Version**: 1.0  
**Last Updated**: 2024  
**Author**: Sachin Kukkar
