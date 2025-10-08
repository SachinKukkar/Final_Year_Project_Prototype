import streamlit as st
import backend
import os

# Page config
st.set_page_config(
    page_title="🧠 EEG Biometric Authentication",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state
if 'selected_file' not in st.session_state:
    st.session_state.selected_file = None
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# Custom CSS
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: 500;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🧠 EEG Biometric Authentication System")

# Sidebar
with st.sidebar:
    st.header("⚙️ Menu")
    page = st.radio("Navigate", ["🏠 Home", "👤 User Management", "🧠 Model Training", "🔐 Authentication", "📊 Dashboard"])

# Home Page
if page == "🏠 Home":
    st.header("Welcome to EEG Biometric Authentication")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Registered Users", len(backend.get_registered_users()))
    with col2:
        model_status = "✅ Trained" if os.path.exists('assets/model.pth') else "❌ Not Trained"
        st.metric("Model Status", model_status)
    with col3:
        data_files = len([f for f in os.listdir('data/Filtered_Data') if f.endswith('.csv')]) if os.path.exists('data/Filtered_Data') else 0
        st.metric("Data Files", data_files)
    
    st.info("👈 Use the sidebar to navigate between different sections")

# User Management
elif page == "👤 User Management":
    st.header("👤 User Management")
    
    tab1, tab2, tab3 = st.tabs(["Register User", "De-register User", "View Users"])
    
    with tab1:
        st.subheader("Register New User")
        username = st.text_input("Username", key="reg_username")
        subject_id = st.number_input("Subject ID", min_value=1, max_value=20, value=1, key="reg_subject")
        
        if st.button("Register User", type="primary"):
            if not username:
                st.error("❌ Please provide a username")
            else:
                with st.spinner(f"Registering {username}..."):
                    result = backend.register_user(username, subject_id)
                    if isinstance(result, tuple):
                        success, message = result
                        if success:
                            st.success(f"✅ {message}")
                        else:
                            st.error(f"❌ {message}")
    
    with tab2:
        st.subheader("De-register User")
        dereg_username = st.text_input("Username to Remove", key="dereg_username")
        
        if dereg_username:
            user_info = backend.get_user_info(dereg_username)
            if user_info:
                st.info(f"""
                **User Details:**
                - Subject ID: {user_info['subject_id']}
                - Data Segments: {user_info.get('data_segments', 'Unknown')}
                - Data File: {'✅ Exists' if user_info['data_exists'] else '❌ Missing'}
                """)
                
                st.warning("⚠️ **WARNING:** This action cannot be undone! All EEG training data will be permanently deleted.")
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("🗑️ Delete User", type="primary", use_container_width=True):
                        with st.spinner(f"De-registering {dereg_username}..."):
                            result = backend.deregister_user(dereg_username)
                            if isinstance(result, tuple):
                                success, message = result
                                if success:
                                    st.success(f"✅ {message}")
                                    st.rerun()
                                else:
                                    st.error(f"❌ {message}")
            else:
                st.error(f"❌ User '{dereg_username}' is not registered")
    
    with tab3:
        st.subheader("Registered Users")
        users = backend.get_registered_users()
        if users:
            for user in users:
                user_info = backend.get_user_info(user)
                if user_info:
                    with st.expander(f"👤 {user}"):
                        st.write(f"**Subject ID:** {user_info['subject_id']}")
                        st.write(f"**Data Segments:** {user_info.get('data_segments', 'Unknown')}")
                        st.write(f"**Data File:** {'✅ Exists' if user_info['data_exists'] else '❌ Missing'}")
        else:
            st.info("No users registered yet")

# Model Training
elif page == "🧠 Model Training":
    st.header("🧠 Model Training")
    
    users = backend.get_registered_users()
    st.info(f"**Registered Users:** {len(users)}")
    
    if len(users) < 2:
        st.warning("⚠️ Need at least 2 registered users to train the model")
    else:
        st.success(f"✅ Ready to train with {len(users)} users")
        
        if st.button("🚀 Start Training", type="primary"):
            with st.spinner("Training model... This may take a few minutes"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Initializing training...")
                progress_bar.progress(10)
                
                success = backend.train_model()
                
                progress_bar.progress(100)
                
                if success:
                    st.success("✅ Training complete! Model saved successfully.")
                else:
                    st.error("❌ Training failed. Check console for errors.")

# Authentication
elif page == "🔐 Authentication":
    st.header("🔐 Authentication")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload EEG File")
        uploaded_file = st.file_uploader("Choose EEG CSV file", type=['csv'])
        
        if uploaded_file:
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.selected_file = temp_path
            st.success(f"✅ File loaded: {uploaded_file.name}")
    
    with col2:
        st.subheader("User Details")
        auth_username = st.text_input("Username", key="auth_username")
        auth_subject_id = st.number_input("Subject ID", min_value=1, max_value=20, value=1, key="auth_subject")
    
    if st.button("🔐 Authenticate", type="primary"):
        if not auth_username:
            st.error("❌ Please provide a username")
        elif not st.session_state.selected_file:
            st.error("❌ Please upload an EEG file")
        else:
            with st.spinner(f"Authenticating {auth_username}..."):
                result = backend.authenticate_with_subject_id(
                    auth_username, 
                    auth_subject_id, 
                    st.session_state.selected_file
                )
                
                if isinstance(result, tuple):
                    is_auth, reason = result
                    if is_auth:
                        st.success(f"✅ {reason}")
                    else:
                        st.error(f"❌ {reason}")
                
                # Clean up temp file
                if os.path.exists(st.session_state.selected_file):
                    try:
                        os.remove(st.session_state.selected_file)
                    except:
                        pass

# Dashboard
elif page == "📊 Dashboard":
    st.header("📊 System Dashboard")
    
    from database import db
    
    # System Statistics
    st.subheader("System Statistics")
    auth_stats = db.get_auth_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Attempts", auth_stats['total_attempts'])
    with col2:
        st.metric("Successful", auth_stats['successful'])
    with col3:
        st.metric("Success Rate", f"{auth_stats['success_rate']:.1%}")
    with col4:
        st.metric("Avg Confidence", f"{auth_stats['avg_confidence']:.3f}")
    
    # User Analytics
    st.subheader("User Analytics")
    db_users = db.get_users()
    
    if db_users:
        for username, user_data in db_users.items():
            with st.expander(f"👤 {username}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Subject ID:** {user_data['subject_id']}")
                    st.write(f"**Data Segments:** {user_data['data_segments']}")
                with col2:
                    data_file = f"assets/data_{username}.npy"
                    file_exists = os.path.exists(data_file)
                    st.write(f"**Data File:** {'✅ Exists' if file_exists else '❌ Missing'}")
                    if file_exists and user_data['data_segments'] > 0:
                        total_samples = user_data['data_segments'] * 256
                        duration = total_samples / 256
                        st.write(f"**Duration:** {duration:.1f} seconds")
    else:
        st.info("No users in database yet")
    
    # Model Information
    st.subheader("Model Information")
    model_exists = os.path.exists('assets/model.pth')
    if model_exists:
        model_size = os.path.getsize('assets/model.pth') / (1024*1024)
        st.success(f"✅ Model trained (Size: {model_size:.2f} MB)")
    else:
        st.warning("❌ Model not trained yet")

# Footer
st.markdown("---")
st.markdown("🧠 **EEG Biometric Authentication System** | Built with Streamlit")
