# 🚀 Streamlit Version - Quick Start

## 📌 Overview

This is the **Streamlit version** of the EEG Biometric Authentication System, converted from the PyQt5 GUI application.

---

## 🎯 Installation

### 1. Install Dependencies
```bash
pip install streamlit
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

---

## ✨ Features

All features from `gui_app.py` are available:

### 🏠 Home
- System overview
- Quick statistics
- Model status

### 👤 User Management
- **Register User**: Add new users with Subject ID
- **De-register User**: Remove users with confirmation
- **View Users**: See all registered users and their details

### 🧠 Model Training
- Train CNN model on registered users
- Progress tracking
- Requires at least 2 users

### 🔐 Authentication
- Upload EEG CSV file
- Enter username and Subject ID
- Get authentication result with confidence score

### 📊 Dashboard
- System statistics
- Authentication success rates
- User analytics
- Model information

---

## 🎨 Differences from PyQt5 Version

| Feature | PyQt5 (gui_app.py) | Streamlit (streamlit_app.py) |
|---------|-------------------|------------------------------|
| **UI Framework** | PyQt5 | Streamlit |
| **Installation** | Complex | Simple (`pip install streamlit`) |
| **Running** | `python gui_app.py` | `streamlit run streamlit_app.py` |
| **Interface** | Desktop app | Web browser |
| **Theme Toggle** | Built-in button | Streamlit settings |
| **Keyboard Shortcuts** | Yes | No |
| **Hamburger Menu** | Yes | Sidebar navigation |
| **File Upload** | File dialog | Drag & drop |

---

## 📋 Usage

### Register Users
1. Go to **👤 User Management** → **Register User** tab
2. Enter username (e.g., "Alice")
3. Select Subject ID (1-20)
4. Click **Register User**

### Train Model
1. Go to **🧠 Model Training**
2. Ensure at least 2 users are registered
3. Click **🚀 Start Training**
4. Wait for completion

### Authenticate
1. Go to **🔐 Authentication**
2. Upload EEG CSV file
3. Enter username and Subject ID
4. Click **🔐 Authenticate**

---

## 🔧 Configuration

Same configuration as PyQt5 version:
- Edit `config.py` for settings
- Uses same backend (`backend.py`)
- Same model architecture
- Same database

---

## 🐛 Troubleshooting

### Issue: "Module not found: streamlit"
```bash
pip install streamlit
```

### Issue: Port already in use
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Issue: File upload not working
- Ensure file is CSV format
- Check file size (Streamlit has 200MB default limit)

---

## 🎯 Advantages of Streamlit Version

✅ **Easier to Install**: Just `pip install streamlit`  
✅ **Web-based**: Access from any browser  
✅ **Responsive**: Works on mobile devices  
✅ **No UI file needed**: No `.ui` file required  
✅ **Simpler deployment**: Easy to deploy to cloud  
✅ **Auto-reload**: Changes reflect immediately  

---

## 📝 Notes

- Both versions use the same backend
- Data and models are shared
- Can run both versions simultaneously (different ports)
- Streamlit version is recommended for web deployment

---

## 🚀 Deployment

### Deploy to Streamlit Cloud (Free)
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Deploy!

### Deploy to Heroku
```bash
# Create Procfile
echo "web: streamlit run streamlit_app.py --server.port $PORT" > Procfile

# Deploy
git push heroku main
```

---

**Status**: ✅ Ready to use  
**Recommended for**: Web deployment, demos, cloud hosting
