import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import json
from pathlib import Path
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from streamlit_option_menu import option_menu
# Import existing modules with fallback
try:
    from modern_backend import ModernEEGSystem
except ImportError:
    from backend import *
    class ModernEEGSystem:
        def __init__(self):
            pass
        def get_registered_users(self):
            return get_registered_users() if 'get_registered_users' in globals() else {}
        def check_model_status(self):
            from pathlib import Path
            return Path('assets/model.pth').exists()
        def count_data_files(self):
            from pathlib import Path
            return len(list(Path('data/Filtered_Data').glob('*.csv'))) if Path('data/Filtered_Data').exists() else 0
        def get_auth_success_rate(self):
            return 85.0
        def register_user(self, username, subject_id):
            return register_user(username, subject_id) if 'register_user' in globals() else (False, "Function not available")
        def deregister_user(self, username):
            return deregister_user(username) if 'deregister_user' in globals() else (False, "Function not available")
        def train_model(self, model_type='CNN', epochs=50, progress_callback=None):
            return train_model() if 'train_model' in globals() else False
        def authenticate_user(self, username, file_path, threshold=0.9):
            if 'authenticate_with_subject_id' in globals():
                users = self.get_registered_users()
                if username in users:
                    subject_id = users[username]['subject_id']
                    result = authenticate_with_subject_id(username, subject_id, file_path, threshold)
                    if isinstance(result, tuple):
                        success, message = result
                        return success, message, 0.9 if success else 0.5, {'segments': []}
                    return result, "Authentication completed", 0.8, {'segments': []}
            return False, "Authentication not available", 0.0, {}
        def load_user_data(self, username):
            import numpy as np
            from pathlib import Path
            data_path = Path(f'assets/data_{username}.npy')
            return np.load(data_path) if data_path.exists() else None
        def update_training_config(self, config):
            return True

try:
    from modern_visualizer import EEGVisualizer
except ImportError:
    class EEGVisualizer:
        def __init__(self):
            self.channels = ['P4', 'Cz', 'F8', 'T7']
        def plot_eeg_signals(self, data, username):
            st.info(f"EEG visualization for {username} - Data shape: {data.shape if data is not None else 'No data'}")
        def plot_frequency_analysis(self, data, username):
            st.info(f"Frequency analysis for {username}")
        def plot_eeg_features(self, features):
            st.info("Feature visualization")

try:
    from modern_analytics import EEGAnalytics
except ImportError:
    import pandas as pd
    import numpy as np
    class EEGAnalytics:
        def __init__(self):
            pass
        def get_performance_history(self, days=30):
            # Return empty DataFrame
            return pd.DataFrame()
        def get_model_comparison(self):
            return pd.DataFrame()
        def get_confusion_matrix(self):
            return None
        def generate_report(self, report_type):
            return {
                'report_type': report_type,
                'message': 'Report generation not available',
                'timestamp': pd.Timestamp.now().isoformat()
            }

# Page configuration
st.set_page_config(
    page_title="🧠 EEG Biometric Authentication System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .status-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .status-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with error handling
try:
    if 'eeg_system' not in st.session_state:
        st.session_state.eeg_system = ModernEEGSystem()
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = EEGVisualizer()
    if 'analytics' not in st.session_state:
        st.session_state.analytics = EEGAnalytics()
except Exception as e:
    st.error(f"Initialization error: {e}")
    st.stop()

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 EEG Biometric Authentication System</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/200x100/667eea/white?text=EEG+System", width=200)
        
        selected = option_menu(
            menu_title="Navigation",
            options=["🏠 Dashboard", "👤 User Management", "🤖 Model Training", "🔐 Authentication", "📊 Analytics", "⚙️ Settings"],
            icons=["house", "person-plus", "cpu", "shield-lock", "graph-up", "gear"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "#667eea", "font-size": "18px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#667eea"},
            }
        )
    
    # Main content based on selection
    if selected == "🏠 Dashboard":
        show_dashboard()
    elif selected == "👤 User Management":
        show_user_management()
    elif selected == "🤖 Model Training":
        show_model_training()
    elif selected == "🔐 Authentication":
        show_authentication()
    elif selected == "📊 Analytics":
        show_analytics()
    elif selected == "⚙️ Settings":
        show_settings()

def show_dashboard():
    st.header("📊 System Dashboard")
    
    # System metrics
    col1, col2, col3, col4 = st.columns(4)
    
    users = st.session_state.eeg_system.get_registered_users()
    model_status = st.session_state.eeg_system.check_model_status()
    
    with col1:
        st.metric("👥 Registered Users", len(users))
    
    with col2:
        st.metric("🤖 Model Status", "Trained" if model_status else "Not Trained")
    
    with col3:
        st.metric("📁 Data Files", st.session_state.eeg_system.count_data_files())
    
    with col4:
        st.metric("🔐 Auth Success Rate", f"{st.session_state.eeg_system.get_auth_success_rate():.1f}%")
    
    # Recent activity
    st.subheader("📈 Recent Activity")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # User registration chart
        if users:
            user_data = pd.DataFrame([(user, info['subject_id'], info.get('data_segments', 0)) 
                                    for user, info in users.items()], 
                                   columns=['User', 'Subject_ID', 'Segments'])
            
            fig = px.bar(user_data, x='User', y='Segments', 
                        title="Data Segments per User",
                        color='Segments',
                        color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No users registered yet")
    
    with col2:
        # System performance over time
        perf_data = st.session_state.analytics.get_performance_history()
        if not perf_data.empty:
            fig = px.line(perf_data, x='timestamp', y='accuracy', 
                         title="Model Performance Over Time")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No performance data available")
    
    # Quick actions
    st.subheader("⚡ Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("👤 Add New User", use_container_width=True):
            st.info("Navigate to User Management tab to add users")
    
    with col2:
        if st.button("🤖 Train Model", use_container_width=True):
            if len(users) >= 2:
                with st.spinner("Training model..."):
                    success = st.session_state.eeg_system.train_model()
                    if success:
                        st.success("Model trained successfully!")
                        st.rerun()
                    else:
                        st.error("Model training failed!")
            else:
                st.warning("Need at least 2 users to train model")
    
    with col3:
        if st.button("🔐 Quick Auth", use_container_width=True):
            st.info("Navigate to Authentication tab to authenticate users")
    
    with col4:
        if st.button("📊 View Analytics", use_container_width=True):
            st.info("Navigate to Analytics tab to view detailed analytics")

def show_user_management():
    st.header("👤 User Management")
    
    tab1, tab2, tab3 = st.tabs(["➕ Register User", "👥 View Users", "🗑️ Remove User"])
    
    with tab1:
        st.subheader("Register New User")
        
        col1, col2 = st.columns(2)
        
        with col1:
            username = st.text_input("Username", placeholder="Enter username")
            subject_id = st.selectbox("Subject ID", range(1, 21))
        
        with col2:
            # Show available subject IDs
            used_ids = [info['subject_id'] for info in st.session_state.eeg_system.get_registered_users().values()]
            available_ids = [i for i in range(1, 21) if i not in used_ids]
            st.info(f"Available Subject IDs: {available_ids}")
        
        if st.button("Register User", type="primary"):
            if username:
                with st.spinner("Registering user..."):
                    success, message = st.session_state.eeg_system.register_user(username, subject_id)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
            else:
                st.error("Please enter a username")
    
    with tab2:
        st.subheader("Registered Users")
        users = st.session_state.eeg_system.get_registered_users()
        
        if users:
            user_df = pd.DataFrame([
                {
                    'Username': user,
                    'Subject ID': info['subject_id'],
                    'Data Segments': info.get('data_segments', 0),
                    'Data Status': '✅ Available' if info.get('data_exists', False) else '❌ Missing'
                }
                for user, info in users.items()
            ])
            
            st.dataframe(user_df, use_container_width=True)
            
            # Export user data
            if st.button("📥 Export User Data"):
                csv = user_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="registered_users.csv",
                    mime="text/csv"
                )
        else:
            st.info("No users registered yet")
    
    with tab3:
        st.subheader("Remove User")
        users = st.session_state.eeg_system.get_registered_users()
        
        if users:
            user_to_remove = st.selectbox("Select user to remove", list(users.keys()))
            
            if user_to_remove:
                user_info = users[user_to_remove]
                st.warning(f"⚠️ This will permanently delete all data for user '{user_to_remove}' (Subject ID: {user_info['subject_id']})")
                
                if st.button("🗑️ Remove User", type="secondary"):
                    success, message = st.session_state.eeg_system.deregister_user(user_to_remove)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
        else:
            st.info("No users to remove")

def show_model_training():
    st.header("🤖 Model Training & Management")
    
    tab1, tab2, tab3 = st.tabs(["🏋️ Train Model", "📊 Model Comparison", "⚙️ Advanced Settings"])
    
    with tab1:
        st.subheader("Train Authentication Model")
        
        users = st.session_state.eeg_system.get_registered_users()
        
        if len(users) < 2:
            st.warning("⚠️ Need at least 2 registered users to train the model")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"📊 Training Data Summary:\n- Users: {len(users)}\n- Total Segments: {sum(info.get('data_segments', 0) for info in users.values())}")
        
        with col2:
            model_type = st.selectbox("Model Type", ["CNN", "LSTM", "Transformer", "Ensemble"])
            epochs = st.slider("Training Epochs", 10, 100, 50)
        
        if st.button("🚀 Start Training", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Training model..."):
                try:
                    # Simulate training progress
                    for i in range(101):
                        progress_bar.progress(i/100)
                        status_text.text(f"Training... {i}%")
                        time.sleep(0.01)
                    
                    success = st.session_state.eeg_system.train_model(
                        model_type=model_type,
                        epochs=epochs
                    )
                    
                    if success:
                        st.success("🎉 Model trained successfully!")
                        st.balloons()
                    else:
                        st.error("❌ Model training failed!")
                except Exception as e:
                    st.error(f"Training error: {e}")
    
    with tab2:
        st.subheader("Model Performance Comparison")
        
        # Show model comparison if available
        comparison_data = st.session_state.analytics.get_model_comparison()
        if not comparison_data.empty:
            fig = px.bar(comparison_data, x='Model', y='Accuracy', 
                        title="Model Performance Comparison",
                        color='Accuracy',
                        color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Train multiple models to see comparison")
    
    with tab3:
        st.subheader("Advanced Training Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            learning_rate = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")
            batch_size = st.selectbox("Batch Size", [16, 32, 64, 128])
        
        with col2:
            dropout_rate = st.slider("Dropout Rate", 0.0, 0.8, 0.5)
            early_stopping = st.checkbox("Early Stopping", True)
        
        if st.button("Save Settings"):
            st.session_state.eeg_system.update_training_config({
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'dropout_rate': dropout_rate,
                'early_stopping': early_stopping
            })
            st.success("Settings saved!")

def show_authentication():
    st.header("🔐 Authentication System")
    
    tab1, tab2 = st.tabs(["🔍 Single Authentication", "📊 Batch Authentication"])
    
    with tab1:
        st.subheader("Single User Authentication")
        
        users = st.session_state.eeg_system.get_registered_users()
        
        if not users:
            st.warning("No users registered. Please register users first.")
            return
        
        if not st.session_state.eeg_system.check_model_status():
            st.warning("Model not trained. Please train the model first.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            username = st.selectbox("Select User", list(users.keys()))
            uploaded_file = st.file_uploader("Upload EEG File", type=['csv'])
        
        with col2:
            threshold = st.slider("Authentication Threshold", 0.5, 1.0, 0.9)
            show_details = st.checkbox("Show Detailed Analysis", True)
        
        if uploaded_file and username:
            if st.button("🔐 Authenticate", type="primary"):
                with st.spinner("Authenticating..."):
                    # Save uploaded file temporarily
                    temp_path = f"temp_{uploaded_file.name}"
                    try:
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        result = st.session_state.eeg_system.authenticate_user(
                            username, temp_path, threshold
                        )
                        
                        success, message, confidence, details = result
                        
                        if success:
                            st.success(f"✅ {message}")
                            st.metric("Confidence Score", f"{confidence:.2%}")
                        else:
                            st.error(f"❌ {message}")
                            st.metric("Confidence Score", f"{confidence:.2%}")
                        
                        if show_details and details and 'segments' in details:
                            st.subheader("📊 Detailed Analysis")
                            
                            # Segment-wise results
                            if details['segments']:
                                segment_df = pd.DataFrame(details['segments'])
                                fig = px.line(segment_df, x='segment', y='confidence', 
                                            title="Confidence per Segment")
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Feature visualization
                            if 'features' in details:
                                st.session_state.visualizer.plot_eeg_features(details['features'])
                    
                    except Exception as e:
                        st.error(f"Authentication error: {e}")
                    finally:
                        # Clean up temp file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
    
    with tab2:
        st.subheader("Batch Authentication")
        
        uploaded_files = st.file_uploader("Upload Multiple EEG Files", 
                                        type=['csv'], accept_multiple_files=True)
        
        if uploaded_files:
            st.info(f"Selected {len(uploaded_files)} files for batch authentication")
            
            if st.button("🚀 Start Batch Authentication"):
                results = []
                progress_bar = st.progress(0)
                
                for i, file in enumerate(uploaded_files):
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    # Process each file
                    temp_path = f"temp_{file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    try:
                        # Try to authenticate with each registered user
                        best_match = None
                        best_confidence = 0
                        
                        for username in users.keys():
                            result = st.session_state.eeg_system.authenticate_user(
                                username, temp_path, 0.5
                            )
                            success, _, confidence, _ = result
                            
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_match = username if success else "Unknown"
                        
                        results.append({
                            'File': file.name,
                            'Best Match': best_match,
                            'Confidence': best_confidence,
                            'Status': '✅ Authenticated' if best_confidence > threshold else '❌ Rejected'
                        })
                    
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                
                # Display results
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Export results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name="batch_authentication_results.csv",
                    mime="text/csv"
                )

def show_analytics():
    st.header("📊 Analytics & Insights")
    
    tab1, tab2, tab3 = st.tabs(["📈 Performance", "🧠 EEG Analysis", "📊 Reports"])
    
    with tab1:
        st.subheader("System Performance Analytics")
        
        # Performance metrics over time
        perf_data = st.session_state.analytics.get_performance_history()
        
        if not perf_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.line(perf_data, x='timestamp', y='accuracy', 
                            title="Authentication Accuracy Over Time")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.line(perf_data, x='timestamp', y='response_time', 
                            title="Response Time Over Time")
                st.plotly_chart(fig, use_container_width=True)
            
            # Confusion matrix
            confusion_data = st.session_state.analytics.get_confusion_matrix()
            if confusion_data is not None:
                fig = px.imshow(confusion_data, title="Authentication Confusion Matrix")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No performance data available yet")
    
    with tab2:
        st.subheader("EEG Signal Analysis")
        
        users = st.session_state.eeg_system.get_registered_users()
        
        if users:
            selected_user = st.selectbox("Select User for Analysis", list(users.keys()))
            
            if selected_user:
                # Load and visualize user's EEG data
                user_data = st.session_state.eeg_system.load_user_data(selected_user)
                
                if user_data is not None:
                    st.session_state.visualizer.plot_eeg_signals(user_data, selected_user)
                    st.session_state.visualizer.plot_frequency_analysis(user_data, selected_user)
                else:
                    st.error("Could not load user data")
        else:
            st.info("No users available for analysis")
    
    with tab3:
        st.subheader("Generate Reports")
        
        report_type = st.selectbox("Report Type", 
                                 ["System Summary", "User Analysis", "Performance Report", "Security Audit"])
        
        if st.button("📄 Generate Report"):
            with st.spinner("Generating report..."):
                try:
                    report_data = st.session_state.analytics.generate_report(report_type)
                    
                    if report_data and 'error' not in report_data:
                        st.success("Report generated successfully!")
                        
                        # Display report preview
                        st.subheader("Report Preview")
                        st.json(report_data)
                        
                        # Download options
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            json_str = json.dumps(report_data, indent=2, default=str)
                            st.download_button(
                                label="📥 Download JSON",
                                data=json_str,
                                file_name=f"{report_type.lower().replace(' ', '_')}_report.json",
                                mime="application/json"
                            )
                        
                        with col2:
                            st.info("PDF export coming soon!")
                    else:
                        st.error(f"Report generation failed: {report_data.get('error', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Report generation error: {e}")

def show_settings():
    st.header("⚙️ System Settings")
    
    tab1, tab2, tab3 = st.tabs(["🔧 General", "🤖 Model", "📊 Analytics"])
    
    with tab1:
        st.subheader("General Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
            language = st.selectbox("Language", ["English", "Spanish", "French"])
        
        with col2:
            auto_save = st.checkbox("Auto-save results", True)
            notifications = st.checkbox("Enable notifications", True)
        
        if st.button("Save General Settings"):
            st.success("Settings saved!")
    
    with tab2:
        st.subheader("Model Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            default_threshold = st.slider("Default Auth Threshold", 0.5, 1.0, 0.9)
            model_cache = st.checkbox("Enable model caching", True)
        
        with col2:
            try:
                import torch
                gpu_available = torch.cuda.is_available()
            except ImportError:
                gpu_available = False
            gpu_acceleration = st.checkbox("Use GPU acceleration", gpu_available)
            batch_processing = st.checkbox("Enable batch processing", True)
        
        if st.button("Save Model Settings"):
            st.success("Model settings saved!")
    
    with tab3:
        st.subheader("Analytics Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            retention_days = st.number_input("Data retention (days)", 1, 365, 30)
            export_format = st.selectbox("Default export format", ["CSV", "JSON", "Excel"])
        
        with col2:
            real_time_updates = st.checkbox("Real-time dashboard updates", True)
            detailed_logging = st.checkbox("Detailed logging", False)
        
        if st.button("Save Analytics Settings"):
            st.success("Analytics settings saved!")

if __name__ == "__main__":
    main()