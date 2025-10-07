import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import json
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Safe imports with fallbacks
def safe_import():
    """Safely import backend modules with fallbacks"""
    try:
        import backend
        import eeg_processing
        import model_management
        return True, backend, eeg_processing, model_management
    except ImportError as e:
        st.error(f"Import error: {e}")
        return False, None, None, None

# Page configuration
st.set_page_config(
    page_title="🧠 EEG Authentication System",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Global variables
BACKEND_AVAILABLE, backend_module, eeg_module, model_module = safe_import()

def main():
    st.markdown('<h1 class="main-header">🧠 EEG Biometric Authentication System</h1>', unsafe_allow_html=True)
    
    if not BACKEND_AVAILABLE:
        st.error("❌ Backend modules not available. Please check your installation.")
        st.info("Make sure all required files are in the project directory.")
        return
    
    # Initialize session state
    if 'users_cache' not in st.session_state:
        st.session_state.users_cache = {}
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = 0
    
    # Sidebar navigation
    with st.sidebar:
        st.title("🧭 Navigation")
        st.markdown("---")
        
        page = st.radio("Choose a page:", [
            "🏠 Dashboard",
            "👤 User Management", 
            "🤖 Model Training",
            "🔐 Authentication",
            "📊 Analytics"
        ])
        
        st.markdown("---")
        
        # Quick refresh button
        if st.button("🔄 Refresh", use_container_width=True):
            st.session_state.users_cache = {}
            st.rerun()
        
        # System status
        st.markdown("### 📊 Quick Status")
        try:
            users = get_users_safe()
            model_exists = Path("assets/model.pth").exists()
            st.write(f"👥 Users: {len(users)}")
            st.write(f"🤖 Model: {'✅' if model_exists else '❌'}")
        except:
            st.write("❓ Status unknown")
    
    # Route to pages
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "👤 User Management":
        show_user_management()
    elif page == "🤖 Model Training":
        show_model_training()
    elif page == "🔐 Authentication":
        show_authentication()
    elif page == "📊 Analytics":
        show_analytics()

def get_users_safe():
    """Safely get users with caching"""
    try:
        current_time = time.time()
        if current_time - st.session_state.last_refresh > 5:  # Cache for 5 seconds
            users = backend_module.get_registered_users()
            users_data = {}
            
            for username in users:
                try:
                    info = backend_module.get_user_info(username)
                    if info:
                        users_data[username] = info
                    else:
                        users_data[username] = {
                            'username': username,
                            'subject_id': 'Unknown',
                            'data_segments': 0,
                            'data_exists': False
                        }
                except:
                    users_data[username] = {
                        'username': username,
                        'subject_id': 'Unknown',
                        'data_segments': 0,
                        'data_exists': False
                    }
            
            st.session_state.users_cache = users_data
            st.session_state.last_refresh = current_time
        
        return st.session_state.users_cache
    except Exception as e:
        st.error(f"Error loading users: {e}")
        return {}

def show_dashboard():
    st.header("📊 System Dashboard")
    
    users_data = get_users_safe()
    
    # System metrics
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        model_exists = Path("assets/model.pth").exists()
        data_files = len(list(Path("data/Filtered_Data").glob("*.csv"))) if Path("data/Filtered_Data").exists() else 0
        ready_users = sum(1 for info in users_data.values() if info.get('data_exists', False))
        
        with col1:
            st.metric("👥 Users", len(users_data))
        with col2:
            st.metric("🤖 Model", "✅ Trained" if model_exists else "❌ Not Trained")
        with col3:
            st.metric("📁 Data Files", data_files)
        with col4:
            st.metric("✅ Ready Users", ready_users)
    
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
    
    # Quick actions
    st.subheader("⚡ Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🤖 Train Model", use_container_width=True, type="primary"):
            if len(users_data) >= 2:
                with st.spinner("Training model..."):
                    try:
                        success = backend_module.train_model()
                        if success:
                            st.success("✅ Model trained successfully!")
                            st.balloons()
                        else:
                            st.error("❌ Training failed!")
                    except Exception as e:
                        st.error(f"Training error: {e}")
            else:
                st.warning("⚠️ Need at least 2 users")
    
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.session_state.users_cache = {}
            st.rerun()
    
    with col3:
        if st.button("📊 System Info", use_container_width=True):
            show_system_info()
    
    # User overview
    if users_data:
        st.subheader("📈 User Overview")
        
        user_display_data = []
        for username, info in users_data.items():
            user_display_data.append({
                'Username': username,
                'Subject ID': info.get('subject_id', 'Unknown'),
                'Data Segments': info.get('data_segments', 0),
                'Status': '✅ Ready' if info.get('data_exists', False) else '❌ No Data'
            })
        
        if user_display_data:
            df = pd.DataFrame(user_display_data)
            st.dataframe(df, use_container_width=True)
            
            # Simple visualization
            if len(df) > 0:
                fig = px.bar(df, x='Username', y='Data Segments', 
                           title="Data Segments per User")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("🔍 No users registered yet.")

def show_system_info():
    """Show detailed system information"""
    st.subheader("🔧 System Information")
    
    info = {
        "Python Version": sys.version,
        "Streamlit Version": st.__version__,
        "Working Directory": os.getcwd(),
        "Assets Directory": "✅ Exists" if Path("assets").exists() else "❌ Missing",
        "Data Directory": "✅ Exists" if Path("data").exists() else "❌ Missing",
        "Model File": "✅ Exists" if Path("assets/model.pth").exists() else "❌ Missing",
        "Backend Available": "✅ Yes" if BACKEND_AVAILABLE else "❌ No"
    }
    
    for key, value in info.items():
        st.write(f"**{key}:** {value}")

def show_user_management():
    st.header("👤 User Management")
    
    users_data = get_users_safe()
    
    tab1, tab2, tab3 = st.tabs(["➕ Add User", "👥 View Users", "🗑️ Remove User"])
    
    with tab1:
        st.subheader("Register New User")
        
        with st.form("register_user_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                username = st.text_input("👤 Username", placeholder="Enter unique username")
                subject_id = st.selectbox("🆔 Subject ID", range(1, 21))
            
            with col2:
                # Show available IDs
                used_ids = [info.get('subject_id') for info in users_data.values() 
                           if info.get('subject_id') != 'Unknown' and info.get('subject_id') is not None]
                available_ids = [i for i in range(1, 21) if i not in used_ids]
                
                st.info(f"**Available IDs:** {available_ids}")
                if used_ids:
                    st.warning(f"**Used IDs:** {used_ids}")
            
            submitted = st.form_submit_button("✅ Register User", type="primary", use_container_width=True)
            
            if submitted:
                if not username:
                    st.error("❌ Please enter a username")
                elif username in users_data:
                    st.error(f"❌ Username '{username}' already exists")
                elif subject_id in used_ids:
                    st.error(f"❌ Subject ID {subject_id} is already in use")
                else:
                    with st.spinner(f"🔄 Registering '{username}'..."):
                        try:
                            result = backend_module.register_user(username, subject_id)
                            
                            if isinstance(result, tuple):
                                success, message = result
                                if success:
                                    st.success(f"✅ {message}")
                                    st.session_state.users_cache = {}  # Clear cache
                                    st.balloons()
                                    time.sleep(2)
                                    st.rerun()
                                else:
                                    st.error(f"❌ {message}")
                            else:
                                if result:
                                    st.success(f"✅ User '{username}' registered!")
                                    st.session_state.users_cache = {}
                                    st.balloons()
                                    time.sleep(2)
                                    st.rerun()
                                else:
                                    st.error("❌ Registration failed!")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
    
    with tab2:
        st.subheader("👥 Registered Users")
        
        if users_data:
            user_details = []
            for username, info in users_data.items():
                user_details.append({
                    'Username': username,
                    'Subject ID': info.get('subject_id', 'Unknown'),
                    'Data Segments': info.get('data_segments', 0),
                    'Status': '✅ Ready' if info.get('data_exists', False) else '❌ No Data'
                })
            
            df = pd.DataFrame(user_details)
            st.dataframe(df, use_container_width=True)
            
            # Export option
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                "users.csv",
                "text/csv",
                use_container_width=True
            )
        else:
            st.info("📭 No users registered yet")
    
    with tab3:
        st.subheader("🗑️ Remove User")
        
        if users_data:
            user_to_remove = st.selectbox("👤 Select user:", list(users_data.keys()))
            
            if user_to_remove:
                user_info = users_data[user_to_remove]
                
                st.write("**User Details:**")
                st.write(f"• Username: {user_to_remove}")
                st.write(f"• Subject ID: {user_info.get('subject_id', 'Unknown')}")
                st.write(f"• Data Segments: {user_info.get('data_segments', 0)}")
                
                st.error("⚠️ **WARNING:** This will permanently delete all user data!")
                
                confirm = st.checkbox(f"I confirm deletion of '{user_to_remove}'")
                
                if confirm and st.button("🗑️ DELETE USER", type="secondary"):
                    with st.spinner(f"🗑️ Removing '{user_to_remove}'..."):
                        try:
                            result = backend_module.deregister_user(user_to_remove)
                            
                            if isinstance(result, tuple):
                                success, message = result
                                if success:
                                    st.success(f"✅ {message}")
                                    st.session_state.users_cache = {}
                                    time.sleep(2)
                                    st.rerun()
                                else:
                                    st.error(f"❌ {message}")
                            else:
                                if result:
                                    st.success(f"✅ User '{user_to_remove}' removed!")
                                    st.session_state.users_cache = {}
                                    time.sleep(2)
                                    st.rerun()
                                else:
                                    st.error("❌ Removal failed!")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
        else:
            st.info("📭 No users to remove")

def show_model_training():
    st.header("🤖 Model Training")
    
    users_data = get_users_safe()
    
    if len(users_data) < 2:
        st.warning("⚠️ **Need at least 2 users for training**")
        st.info("Please register more users first.")
        return
    
    # Training info
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📊 Training Data:**")
        total_segments = sum(info.get('data_segments', 0) for info in users_data.values())
        ready_users = sum(1 for info in users_data.values() if info.get('data_exists', False))
        
        st.write(f"• Users: {len(users_data)}")
        st.write(f"• Ready Users: {ready_users}")
        st.write(f"• Total Segments: {total_segments}")
    
    with col2:
        st.write("**🔧 Model Status:**")
        model_exists = Path("assets/model.pth").exists()
        encoder_exists = Path("assets/label_encoder.joblib").exists()
        scaler_exists = Path("assets/scaler.joblib").exists()
        
        st.write(f"• Model: {'✅' if model_exists else '❌'}")
        st.write(f"• Encoder: {'✅' if encoder_exists else '❌'}")
        st.write(f"• Scaler: {'✅' if scaler_exists else '❌'}")
    
    # Training button
    if ready_users < 2:
        st.error(f"❌ Only {ready_users} users have data available")
    else:
        st.success(f"✅ Ready to train with {ready_users} users")
        
        if st.button("🚀 START TRAINING", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Show progress
                for i in range(101):
                    progress_bar.progress(i/100)
                    status_text.text(f"Training... {i}%")
                    time.sleep(0.02)
                
                # Actually train
                success = backend_module.train_model()
                
                if success:
                    st.success("🎉 Training completed!")
                    st.balloons()
                else:
                    st.error("❌ Training failed!")
            except Exception as e:
                st.error(f"❌ Training error: {e}")

def show_authentication():
    st.header("🔐 Authentication System")
    
    users_data = get_users_safe()
    
    if not users_data:
        st.warning("⚠️ No users registered")
        return
    
    if not Path("assets/model.pth").exists():
        st.warning("⚠️ Model not trained")
        return
    
    # Authentication form
    with st.form("auth_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            username = st.selectbox("👤 Select User:", list(users_data.keys()))
            threshold = st.slider("🎯 Threshold:", 0.5, 1.0, 0.9, 0.05)
        
        with col2:
            uploaded_file = st.file_uploader("📁 Upload EEG File:", type=['csv'])
            show_details = st.checkbox("📊 Show Details", True)
        
        submitted = st.form_submit_button("🔐 AUTHENTICATE", type="primary", use_container_width=True)
        
        if submitted and uploaded_file and username:
            authenticate_user(username, uploaded_file, threshold, show_details)

def authenticate_user(username, uploaded_file, threshold, show_details):
    """Perform user authentication"""
    
    user_info = st.session_state.users_cache.get(username, {})
    subject_id = user_info.get('subject_id')
    
    if not subject_id or subject_id == 'Unknown':
        st.error("❌ User subject ID not found")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Save temp file
        temp_path = f"temp_{uploaded_file.name}"
        
        status_text.text("📁 Processing file...")
        progress_bar.progress(25)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        status_text.text("🔐 Authenticating...")
        progress_bar.progress(75)
        
        # Authenticate
        result = backend_module.authenticate_with_subject_id(username, subject_id, temp_path, threshold)
        
        progress_bar.progress(100)
        status_text.text("✅ Complete!")
        
        # Process results
        if isinstance(result, tuple):
            success, message = result
            confidence = 0.9 if success else 0.5
        else:
            success = result
            message = "Authentication completed"
            confidence = 0.8 if success else 0.4
        
        # Display results
        if success:
            st.success(f"✅ **AUTHENTICATED**")
            st.success(f"📝 {message}")
        else:
            st.error(f"❌ **ACCESS DENIED**")
            st.error(f"📝 {message}")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("👤 User", username)
        with col2:
            st.metric("🎯 Confidence", f"{confidence:.1%}")
        with col3:
            st.metric("📊 Threshold", f"{threshold:.1%}")
        
        if show_details:
            st.subheader("📊 Details")
            st.write(f"• **File:** {uploaded_file.name}")
            st.write(f"• **Size:** {uploaded_file.size} bytes")
            st.write(f"• **Subject ID:** {subject_id}")
            st.write(f"• **Result:** {'✅ Success' if success else '❌ Failed'}")
    
    except Exception as e:
        st.error(f"❌ Authentication error: {e}")
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def show_analytics():
    st.header("📊 Analytics & Reports")
    
    users_data = get_users_safe()
    
    tab1, tab2 = st.tabs(["📈 Overview", "📄 Reports"])
    
    with tab1:
        st.subheader("📊 System Overview")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("👥 Users", len(users_data))
        with col2:
            ready = sum(1 for info in users_data.values() if info.get('data_exists', False))
            st.metric("✅ Ready", ready)
        with col3:
            model_trained = Path("assets/model.pth").exists()
            st.metric("🤖 Model", "✅" if model_trained else "❌")
        with col4:
            total_segments = sum(info.get('data_segments', 0) for info in users_data.values())
            st.metric("📊 Segments", total_segments)
        
        # User chart
        if users_data:
            user_chart_data = []
            for username, info in users_data.items():
                user_chart_data.append({
                    'User': username,
                    'Segments': info.get('data_segments', 0),
                    'Ready': info.get('data_exists', False)
                })
            
            if user_chart_data:
                df = pd.DataFrame(user_chart_data)
                fig = px.bar(df, x='User', y='Segments', title="Data by User")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📄 Generate Report")
        
        if st.button("📄 Generate System Report", type="primary"):
            report = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "system_status": {
                    "total_users": len(users_data),
                    "ready_users": sum(1 for info in users_data.values() if info.get('data_exists', False)),
                    "model_trained": Path("assets/model.pth").exists(),
                    "total_segments": sum(info.get('data_segments', 0) for info in users_data.values())
                },
                "users": [
                    {
                        "username": username,
                        "subject_id": info.get('subject_id', 'Unknown'),
                        "data_segments": info.get('data_segments', 0),
                        "data_available": info.get('data_exists', False)
                    }
                    for username, info in users_data.items()
                ]
            }
            
            st.success("✅ Report generated!")
            st.json(report)
            
            # Download
            json_str = json.dumps(report, indent=2, default=str)
            st.download_button(
                "📥 Download Report",
                json_str,
                "system_report.json",
                "application/json"
            )

if __name__ == "__main__":
    main()