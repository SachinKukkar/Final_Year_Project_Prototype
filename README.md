# EEG Biometric Authentication System

🧠 A modern web-based EEG signal processing and biometric authentication system using deep learning.

## ✨ Features

- **Modern Web Interface**: Streamlit-based dashboard with real-time analytics
- **EEG Biometric Authentication**: CNN-based subject identification from EEG signals
- **User Management**: Register and manage users with EEG data
- **Model Training**: Automated deep learning model training with progress tracking
- **Real-time Authentication**: Upload EEG files for instant identity verification
- **Performance Analytics**: Comprehensive metrics and visualizations
- **Multi-Channel Support**: Processes P4, Cz, F8, T7 EEG channels

## 🚀 Quick Setup & Installation

### Prerequisites
- Python 3.7 or higher
- Windows/Linux/macOS

### 1. Clone Repository
```bash
git clone https://github.com/SachinKukkar/EEG_Project_Final_Year_DEMO.git
cd EEG_Project_Final_Year_DEMO-main
```

### 2. Install Dependencies
```bash
pip install -r modern_requirements.txt
```

### 3. Initialize Database
```bash
python database.py
```

## 🎯 Running the Application

### Start the Web Application
```bash
streamlit run final_app.py
```

The application will open in your browser at `http://localhost:8501`

## 📱 Using the Application

### 1. Dashboard
- View system overview and statistics
- Monitor registered users and model status

### 2. User Registration
- Navigate to "👤 User Management"
- Upload EEG data files (.csv format)
- Files should be named: `subject_XX_*.csv` (where XX is user ID)
- System automatically extracts subject ID and processes data

### 3. Model Training
- Go to "🧠 Model Training"
- Click "Start Training" to train the CNN model
- Monitor training progress with real-time metrics
- Model automatically saves when training completes

### 4. Authentication
- Visit "🔐 Authentication"
- Upload an EEG file for verification
- System processes the file and returns:
  - Predicted user identity
  - Confidence score
  - Authentication result

### 5. Analytics
- Check "📊 Analytics" for detailed performance metrics
- View training history and model performance

## 📁 Project Structure

```
EEG_Project_Final_Year_DEMO-main/
├── final_app.py              # Main Streamlit application
├── backend.py                # Core processing and authentication
├── database.py               # SQLite database management
├── model_management.py       # CNN model architecture
├── eeg_processing.py         # EEG signal processing
├── config.py                 # Configuration settings
├── utils.py                  # Utility functions
├── modern_requirements.txt   # Dependencies
├── assets/                   # Model files and data
├── data/                     # EEG datasets
└── logs/                     # Application logs
```

## 🔧 Configuration

Edit `config.py` to customize:
- EEG processing parameters
- Model hyperparameters
- Authentication thresholds
- File paths and database settings

## 📊 EEG Data Format

### File Naming Convention
```
subject_01_session1.csv
subject_02_trial1.csv
subject_XX_description.csv
```

### CSV Format
- Columns: P4, Cz, F8, T7 (EEG channels)
- Rows: Time samples
- No headers required
- Numeric values only

### Example Data Structure
```csv
-12.5,8.3,15.2,-3.7
-11.8,9.1,14.6,-4.2
-13.2,7.9,16.1,-3.1
...
```

## 🧪 Testing

### Test Individual Components
```bash
# Test user registration
python test_registration.py

# Test model training
python test_training.py

# Test authentication
python test_authentication.py

# Test with temporary files
python test_temp_file.py
```

## 🛠️ Technical Specifications

### Model Architecture
- 4-layer CNN with batch normalization
- Dropout regularization (0.3)
- Multi-class classification
- PyTorch implementation
- Early stopping with patience=5

### Data Processing
- Window size: 256 samples
- Step size: 128 samples (50% overlap)
- Channels: P4, Cz, F8, T7
- Preprocessing: Z-score normalization

### Performance Metrics
- Training accuracy: >95%
- Authentication confidence: >90%
- Real-time processing capability

## 🚨 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
pip install --upgrade -r modern_requirements.txt
```

**2. Database Issues**
```bash
python database.py  # Reinitialize database
```

**3. Model Training Fails**
- Ensure at least 2 users are registered
- Check EEG data format and file naming
- Verify sufficient data samples

**4. Authentication Errors**
- Ensure model is trained first
- Check uploaded file format
- Verify file naming convention

**5. Unicode/Encoding Errors**
- Run with: `python -X utf8 final_app.py`
- Or set environment: `set PYTHONIOENCODING=utf-8`

## 📋 Dependencies

### Core Libraries
- `streamlit>=1.28.0` - Web interface
- `torch>=1.9.0` - Deep learning
- `pandas>=1.3.0` - Data processing
- `numpy>=1.21.0` - Numerical computing
- `plotly>=5.0.0` - Interactive visualizations
- `scikit-learn>=1.0.0` - Machine learning utilities

See `modern_requirements.txt` for complete list.

## 🎯 Usage Examples

### 1. Register Multiple Users
```python
# Upload files named:
# subject_01_baseline.csv
# subject_02_baseline.csv
# subject_03_baseline.csv
```

### 2. Train Model
```python
# After registering users, click "Start Training"
# Monitor progress in real-time
# Model saves automatically when complete
```

### 3. Authenticate User
```python
# Upload test file: subject_01_test.csv
# System returns: "User 1 authenticated with 95% confidence"
```

## 🔒 Security Features

- Biometric authentication using unique EEG patterns
- Secure file handling with temporary storage
- Database encryption for user data
- Confidence thresholds for authentication

## 📈 Performance Optimization

- Efficient data preprocessing with vectorized operations
- GPU acceleration support for model training
- Optimized memory usage for large datasets
- Real-time processing capabilities

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

## 📄 License

This project is for educational and research purposes.

## 👨‍💻 Author

**Sachin Kukkar** - Final Year Project Demo

---

## 🆘 Support

For issues or questions:
1. Check troubleshooting section above
2. Review test scripts for examples
3. Ensure all dependencies are installed
4. Verify EEG data format and naming

**Happy EEG Processing! 🧠✨**