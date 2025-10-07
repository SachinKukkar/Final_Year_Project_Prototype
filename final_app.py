import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
import json
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import backend safely
try:
    import backend
    BACKEND_OK = True
except Exception as e:
    st.error(f"Backend error: {e}")
    BACKEND_OK = False

st.set_page_config(page_title="🧠 EEG System", layout="wide")

def main():
    st.title("🧠 EEG Biometric Authentication System")
    
    if not BACKEND_OK:
        st.error("Backend not available")
        return
    
    # Sidebar
    with st.sidebar:
        page = st.selectbox("Navigation", [
            "Dashboard", "User Management", "Model Training", "Authentication", "Analytics"
        ])
    
    if page == "Dashboard":
        show_dashboard()
    elif page == "User Management":
        show_user_management()
    elif page == "Model Training":
        show_model_training()
    elif page == "Authentication":
        show_authentication()
    elif page == "Analytics":
        show_analytics()

def show_dashboard():
    st.header("Dashboard")
    
    try:
        users = backend.get_registered_users()
        model_exists = Path("assets/model.pth").exists()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Users", len(users))
        with col2:
            st.metric("Model", "✅" if model_exists else "❌")
        with col3:
            data_files = len(list(Path("data/Filtered_Data").glob("*.csv"))) if Path("data/Filtered_Data").exists() else 0
            st.metric("Data Files", data_files)
        
        if users:
            st.subheader("Registered Users")
            for user in users:
                st.write(f"• {user}")
        else:
            st.info("No users registered")
            
    except Exception as e:
        st.error(f"Dashboard error: {e}")

def show_user_management():
    st.header("User Management")
    
    tab1, tab2, tab3 = st.tabs(["Add User", "View Users", "Remove User"])
    
    with tab1:
        st.subheader("Register New User")
        
        username = st.text_input("Username")
        subject_id = st.selectbox("Subject ID", range(1, 21))
        
        if st.button("Register User"):
            if username:
                try:
                    result = backend.register_user(username, subject_id)
                    if isinstance(result, tuple):
                        success, message = result
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        if result:
                            st.success("User registered!")
                            st.rerun()
                        else:
                            st.error("Registration failed")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Enter username")
    
    with tab2:
        st.subheader("View Users")
        try:
            users = backend.get_registered_users()
            if users:
                user_data = []
                for user in users:
                    info = backend.get_user_info(user)
                    if info:
                        user_data.append({
                            'Username': user,
                            'Subject ID': info.get('subject_id', 'Unknown'),
                            'Data Segments': info.get('data_segments', 0),
                            'Status': '✅' if info.get('data_exists', False) else '❌'
                        })
                
                if user_data:
                    df = pd.DataFrame(user_data)
                    st.dataframe(df)
            else:
                st.info("No users")
        except Exception as e:
            st.error(f"Error: {e}")
    
    with tab3:
        st.subheader("Remove User")
        try:
            users = backend.get_registered_users()
            if users:
                user_to_remove = st.selectbox("Select user", users)
                if st.button("Remove User"):
                    result = backend.deregister_user(user_to_remove)
                    if isinstance(result, tuple):
                        success, message = result
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        st.success("User removed")
                        st.rerun()
            else:
                st.info("No users to remove")
        except Exception as e:
            st.error(f"Error: {e}")

def show_model_training():
    st.header("Model Training")
    
    try:
        users = backend.get_registered_users()
        
        if len(users) < 2:
            st.warning("Need at least 2 users")
            return
        
        st.info(f"Ready to train with {len(users)} users")
        
        if st.button("Train Model"):
            with st.spinner("Training..."):
                success = backend.train_model()
                if success:
                    st.success("Training complete!")
                else:
                    st.error("Training failed!")
                    
    except Exception as e:
        st.error(f"Error: {e}")

def show_authentication():
    st.header("Authentication")
    
    try:
        users = backend.get_registered_users()
        
        if not users:
            st.warning("No users registered")
            return
        
        if not Path("assets/model.pth").exists():
            st.warning("Model not trained")
            return
        
        username = st.selectbox("Select User", users)
        uploaded_file = st.file_uploader("Upload EEG File", type=['csv'])
        threshold = st.slider("Threshold", 0.5, 1.0, 0.9)
        
        if uploaded_file and username:
            if st.button("Authenticate"):
                temp_path = f"temp_{uploaded_file.name}"
                try:
                    # Show file info
                    st.info(f"Processing file: {uploaded_file.name} ({uploaded_file.size} bytes)")
                    
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Verify file was saved
                    if not os.path.exists(temp_path):
                        st.error("Failed to save uploaded file")
                        return
                    
                    user_info = backend.get_user_info(username)
                    if user_info:
                        subject_id = user_info.get('subject_id')
                        if subject_id and subject_id != 'Unknown':
                            result = backend.authenticate_with_subject_id(username, subject_id, temp_path, threshold)
                            
                            if isinstance(result, tuple):
                                success, message = result
                                if success:
                                    st.success(f"✅ {message}")
                                    st.metric("Confidence", "High")
                                else:
                                    st.error(f"❌ {message}")
                                    st.metric("Confidence", "Low")
                            else:
                                if result:
                                    st.success("✅ Authenticated")
                                    st.metric("Confidence", "High")
                                else:
                                    st.error("❌ Access denied")
                                    st.metric("Confidence", "Low")
                        else:
                            st.error("❌ User subject ID not found")
                    else:
                        st.error("❌ User information not found")
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
    except Exception as e:
        st.error(f"Error: {e}")

def show_analytics():
    st.header("Analytics")
    
    try:
        users = backend.get_registered_users()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Users", len(users))
        with col2:
            model_exists = Path("assets/model.pth").exists()
            st.metric("Model Status", "Trained" if model_exists else "Not Trained")
        with col3:
            total_segments = 0
            for user in users:
                info = backend.get_user_info(user)
                if info:
                    total_segments += info.get('data_segments', 0)
            st.metric("Total Segments", total_segments)
        
        if st.button("Generate Report"):
            report = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "users": len(users),
                "model_trained": Path("assets/model.pth").exists(),
                "total_segments": total_segments
            }
            st.json(report)
            
    except Exception as e:
        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()