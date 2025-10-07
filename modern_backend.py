import os
import json
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
from pathlib import Path
import logging
import time
from datetime import datetime
import sqlite3
from typing import Dict, List, Tuple, Optional, Any

from eeg_processing import get_subject_files, load_and_segment_csv
from model_management import EEG_CNN_Improved

class ModernEEGSystem:
    """Enhanced EEG processing system with modern features."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.absolute()
        self.assets_dir = self.base_dir / 'assets'
        self.data_dir = self.base_dir / 'data' / 'Filtered_Data'
        self.models_dir = self.assets_dir / 'models'
        
        # Create directories
        self.assets_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # File paths
        self.users_path = self.assets_dir / 'users.json'
        self.config_path = self.assets_dir / 'config.json'
        self.db_path = self.assets_dir / 'eeg_system.db'
        
        # Initialize database
        self._init_database()
        
        # Load configuration
        self.config = self._load_config()
        
        # Setup logging
        self._setup_logging()
        
        # Available models
        self.available_models = {
            'CNN': self._train_cnn_model,
            'LSTM': self._train_lstm_model,
            'Transformer': self._train_transformer_model,
            'RandomForest': self._train_rf_model,
            'XGBoost': self._train_xgb_model,
            'LightGBM': self._train_lgb_model,
            'Ensemble': self._train_ensemble_model
        }
    
    def _init_database(self):
        """Initialize SQLite database for storing system data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                subject_id INTEGER NOT NULL,
                data_segments INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Authentication logs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS auth_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                success BOOLEAN NOT NULL,
                confidence REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                details TEXT
            )
        ''')
        
        # Model performance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_type TEXT NOT NULL,
                accuracy REAL,
                precision_score REAL,
                recall REAL,
                f1_score REAL,
                training_time REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_config(self) -> Dict:
        """Load system configuration."""
        default_config = {
            'channels': ['P4', 'Cz', 'F8', 'T7'],
            'window_size': 256,
            'step_size': 128,
            'sampling_rate': 256,
            'auth_threshold': 0.90,
            'batch_size': 32,
            'learning_rate': 0.001,
            'num_epochs': 50,
            'dropout_rate': 0.5,
            'early_stopping': True,
            'patience': 5
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults
                    default_config.update(config)
            except Exception as e:
                logging.warning(f"Could not load config: {e}")
        
        return default_config
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = self.base_dir / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'eeg_system_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
    
    def register_user(self, username: str, subject_id: int) -> Tuple[bool, str]:
        """Register a new user with enhanced validation."""
        logging.info(f"Registering user: {username} (Subject ID: {subject_id})")
        
        try:
            # Check if user already exists
            existing_users = self.get_registered_users()
            
            if username in existing_users:
                return False, f"Username '{username}' already exists"
            
            # Check if subject ID is already used
            for user, info in existing_users.items():
                if info['subject_id'] == subject_id:
                    return False, f"Subject ID {subject_id} already used by '{user}'"
            
            # Get subject files
            subject_files = get_subject_files(str(self.data_dir), subject_id)
            if not subject_files:
                return False, f"No data files found for Subject ID {subject_id}"
            
            # Process EEG data
            all_segments = []
            for file_path in subject_files:
                segments = load_and_segment_csv(file_path)
                if segments:
                    all_segments.extend(segments)
            
            if not all_segments:
                return False, f"Could not extract valid data segments for Subject ID {subject_id}"
            
            # Save user data
            user_data = np.array(all_segments)
            data_path = self.assets_dir / f'data_{username}.npy'
            np.save(data_path, user_data)
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO users (username, subject_id, data_segments)
                VALUES (?, ?, ?)
            ''', (username, subject_id, len(all_segments)))
            conn.commit()
            conn.close()
            
            # Update JSON file for backward compatibility
            users = {}
            if self.users_path.exists():
                with open(self.users_path, 'r') as f:
                    users = json.load(f)
            
            users[username] = subject_id
            with open(self.users_path, 'w') as f:
                json.dump(users, f, indent=4)
            
            message = f"User '{username}' registered successfully with {len(all_segments)} data segments"
            logging.info(message)
            return True, message
            
        except Exception as e:
            error_msg = f"Registration failed: {str(e)}"
            logging.error(error_msg)
            return False, error_msg
    
    def deregister_user(self, username: str) -> Tuple[bool, str]:
        """Remove a registered user."""
        logging.info(f"Deregistering user: {username}")
        
        try:
            users = self.get_registered_users()
            
            if username not in users:
                return False, f"User '{username}' not found"
            
            # Remove from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM users WHERE username = ?', (username,))
            conn.commit()
            conn.close()
            
            # Remove data file
            data_path = self.assets_dir / f'data_{username}.npy'
            if data_path.exists():
                data_path.unlink()
            
            # Update JSON file
            if self.users_path.exists():
                with open(self.users_path, 'r') as f:
                    json_users = json.load(f)
                
                if username in json_users:
                    del json_users[username]
                    with open(self.users_path, 'w') as f:
                        json.dump(json_users, f, indent=4)
            
            message = f"User '{username}' removed successfully"
            logging.info(message)
            return True, message
            
        except Exception as e:
            error_msg = f"Deregistration failed: {str(e)}"
            logging.error(error_msg)
            return False, error_msg
    
    def get_registered_users(self) -> Dict[str, Dict]:
        """Get all registered users with their information."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT username, subject_id, data_segments FROM users')
            rows = cursor.fetchall()
            conn.close()
            
            users = {}
            for username, subject_id, data_segments in rows:
                data_path = self.assets_dir / f'data_{username}.npy'
                users[username] = {
                    'subject_id': subject_id,
                    'data_segments': data_segments,
                    'data_exists': data_path.exists()
                }
            
            return users
            
        except Exception as e:
            logging.error(f"Error getting users: {e}")
            return {}
    
    def train_model(self, model_type: str = 'CNN', epochs: int = None, 
                   progress_callback=None) -> bool:
        """Train the specified model type."""
        logging.info(f"Training {model_type} model")
        
        try:
            users = self.get_registered_users()
            if len(users) < 2:
                logging.error("Need at least 2 users to train model")
                return False
            
            # Load training data
            all_data, all_labels = [], []
            for username, info in users.items():
                data_path = self.assets_dir / f'data_{username}.npy'
                if data_path.exists():
                    user_data = np.load(data_path)
                    all_data.append(user_data)
                    all_labels.extend([username] * len(user_data))
            
            if not all_data:
                logging.error("No training data available")
                return False
            
            X = np.concatenate(all_data)
            y = np.array(all_labels)
            
            # Train the specified model
            start_time = time.time()
            
            if model_type in self.available_models:
                success, metrics = self.available_models[model_type](X, y, epochs, progress_callback)
                
                if success:
                    training_time = time.time() - start_time
                    
                    # Save performance metrics
                    self._save_model_performance(model_type, metrics, training_time)
                    
                    logging.info(f"{model_type} model trained successfully in {training_time:.2f}s")
                    return True
                else:
                    logging.error(f"{model_type} model training failed")
                    return False
            else:
                logging.error(f"Unknown model type: {model_type}")
                return False
                
        except Exception as e:
            logging.error(f"Model training failed: {e}")
            return False
    
    def _train_cnn_model(self, X, y, epochs=None, progress_callback=None):
        """Train CNN model."""
        try:
            # Prepare data
            scaler = StandardScaler()
            X_reshaped = X.reshape(-1, X.shape[-1])
            scaler.fit(X_reshaped)
            X_scaled = scaler.transform(X_reshaped).reshape(X.shape)
            
            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(y)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Create datasets and loaders
            train_dataset = EEGDataset(X_train, y_train)
            val_dataset = EEGDataset(X_val, y_val)
            train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False)
            
            # Initialize model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = EEG_CNN_Improved(num_classes=len(encoder.classes_)).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.config['learning_rate'])
            
            # Training loop
            num_epochs = epochs or self.config['num_epochs']
            best_val_acc = 0.0
            patience_counter = 0
            
            for epoch in range(num_epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                
                val_acc = correct / total
                
                # Progress callback
                if progress_callback:
                    progress_callback(
                        (epoch + 1) / num_epochs,
                        f"Epoch {epoch+1}/{num_epochs} - Val Acc: {val_acc:.4f}"
                    )
                
                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), self.models_dir / 'cnn_model.pth')
                    joblib.dump(encoder, self.models_dir / 'cnn_encoder.joblib')
                    joblib.dump(scaler, self.models_dir / 'cnn_scaler.joblib')
                else:
                    patience_counter += 1
                    if self.config['early_stopping'] and patience_counter >= self.config['patience']:
                        break
            
            # Calculate final metrics
            metrics = {
                'accuracy': best_val_acc,
                'precision_score': best_val_acc,  # Simplified for now
                'recall': best_val_acc,
                'f1_score': best_val_acc
            }
            
            return True, metrics
            
        except Exception as e:
            logging.error(f"CNN training failed: {e}")
            return False, {}
    
    def _train_rf_model(self, X, y, epochs=None, progress_callback=None):
        """Train Random Forest model."""
        try:
            # Flatten features for traditional ML
            X_flat = X.reshape(X.shape[0], -1)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_flat, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            # Save model
            joblib.dump(model, self.models_dir / 'rf_model.joblib')
            
            metrics = {
                'accuracy': accuracy,
                'precision_score': accuracy,
                'recall': accuracy,
                'f1_score': accuracy
            }
            
            if progress_callback:
                progress_callback(1.0, f"Random Forest trained - Accuracy: {accuracy:.4f}")
            
            return True, metrics
            
        except Exception as e:
            logging.error(f"Random Forest training failed: {e}")
            return False, {}
    
    def _train_xgb_model(self, X, y, epochs=None, progress_callback=None):
        """Train XGBoost model."""
        try:
            # Flatten features
            X_flat = X.reshape(X.shape[0], -1)
            
            # Encode labels
            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(y)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_flat, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Train model
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            # Save model
            joblib.dump(model, self.models_dir / 'xgb_model.joblib')
            joblib.dump(encoder, self.models_dir / 'xgb_encoder.joblib')
            
            metrics = {
                'accuracy': accuracy,
                'precision_score': accuracy,
                'recall': accuracy,
                'f1_score': accuracy
            }
            
            if progress_callback:
                progress_callback(1.0, f"XGBoost trained - Accuracy: {accuracy:.4f}")
            
            return True, metrics
            
        except Exception as e:
            logging.error(f"XGBoost training failed: {e}")
            return False, {}
    
    def _train_lgb_model(self, X, y, epochs=None, progress_callback=None):
        """Train LightGBM model."""
        try:
            # Similar implementation to XGBoost
            X_flat = X.reshape(X.shape[0], -1)
            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(y)
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_flat, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            joblib.dump(model, self.models_dir / 'lgb_model.joblib')
            joblib.dump(encoder, self.models_dir / 'lgb_encoder.joblib')
            
            metrics = {
                'accuracy': accuracy,
                'precision_score': accuracy,
                'recall': accuracy,
                'f1_score': accuracy
            }
            
            if progress_callback:
                progress_callback(1.0, f"LightGBM trained - Accuracy: {accuracy:.4f}")
            
            return True, metrics
            
        except Exception as e:
            logging.error(f"LightGBM training failed: {e}")
            return False, {}
    
    def _train_lstm_model(self, X, y, epochs=None, progress_callback=None):
        """Train LSTM model (placeholder)."""
        # Implementation would go here
        return False, {}
    
    def _train_transformer_model(self, X, y, epochs=None, progress_callback=None):
        """Train Transformer model (placeholder)."""
        # Implementation would go here
        return False, {}
    
    def _train_ensemble_model(self, X, y, epochs=None, progress_callback=None):
        """Train ensemble of multiple models."""
        # Implementation would go here
        return False, {}
    
    def _save_model_performance(self, model_type: str, metrics: Dict, training_time: float):
        """Save model performance to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO model_performance 
                (model_type, accuracy, precision_score, recall, f1_score, training_time)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                model_type,
                metrics.get('accuracy', 0),
                metrics.get('precision_score', 0),
                metrics.get('recall', 0),
                metrics.get('f1_score', 0),
                training_time
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Error saving model performance: {e}")
    
    def authenticate_user(self, username: str, file_path: str, 
                         threshold: float = None) -> Tuple[bool, str, float, Dict]:
        """Authenticate a user with detailed results."""
        threshold = threshold or self.config['auth_threshold']
        
        try:
            # Check if user exists
            users = self.get_registered_users()
            if username not in users:
                return False, f"User '{username}' not registered", 0.0, {}
            
            # Load model (try CNN first, fallback to others)
            model_path = self.models_dir / 'cnn_model.pth'
            encoder_path = self.models_dir / 'cnn_encoder.joblib'
            scaler_path = self.models_dir / 'cnn_scaler.joblib'
            
            if not all(p.exists() for p in [model_path, encoder_path, scaler_path]):
                return False, "Model not trained", 0.0, {}
            
            # Load model components
            encoder = joblib.load(encoder_path)
            scaler = joblib.load(scaler_path)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = EEG_CNN_Improved(num_classes=len(encoder.classes_)).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            # Process EEG file
            segments = load_and_segment_csv(file_path)
            if not segments:
                return False, "No valid EEG segments found", 0.0, {}
            
            # Authenticate segments
            predictions = []
            confidences = []
            segment_results = []
            
            with torch.no_grad():
                for i, segment in enumerate(segments):
                    segment_scaled = scaler.transform(segment)
                    segment_tensor = torch.tensor(segment_scaled, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    outputs = model(segment_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                    
                    confidence_val = confidence.item()
                    confidences.append(confidence_val)
                    
                    predicted_user = encoder.inverse_transform([predicted_idx.item()])[0]
                    is_match = predicted_user == username and confidence_val >= threshold
                    predictions.append(is_match)
                    
                    segment_results.append({
                        'segment': i,
                        'confidence': confidence_val,
                        'predicted_user': predicted_user,
                        'match': is_match
                    })
            
            # Determine final result
            positive_votes = sum(predictions)
            total_segments = len(predictions)
            avg_confidence = np.mean(confidences)
            max_confidence = np.max(confidences)
            
            success = positive_votes > total_segments / 2
            
            if success:
                message = f"Authentication successful: {positive_votes}/{total_segments} segments matched"
            else:
                message = f"Authentication failed: {positive_votes}/{total_segments} segments matched"
            
            # Log authentication attempt
            self._log_authentication(username, success, avg_confidence, message)
            
            details = {
                'segments': segment_results,
                'total_segments': total_segments,
                'positive_votes': positive_votes,
                'avg_confidence': avg_confidence,
                'max_confidence': max_confidence
            }
            
            return success, message, avg_confidence, details
            
        except Exception as e:
            error_msg = f"Authentication error: {str(e)}"
            logging.error(error_msg)
            return False, error_msg, 0.0, {}
    
    def _log_authentication(self, username: str, success: bool, confidence: float, details: str):
        """Log authentication attempt to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO auth_logs (username, success, confidence, details)
                VALUES (?, ?, ?, ?)
            ''', (username, success, confidence, details))
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Error logging authentication: {e}")
    
    def check_model_status(self) -> bool:
        """Check if any model is trained and available."""
        model_files = [
            'cnn_model.pth',
            'rf_model.joblib',
            'xgb_model.joblib',
            'lgb_model.joblib'
        ]
        
        return any((self.models_dir / model_file).exists() for model_file in model_files)
    
    def count_data_files(self) -> int:
        """Count available data files."""
        return len(list(self.data_dir.glob('*.csv')))
    
    def get_auth_success_rate(self) -> float:
        """Get authentication success rate from logs."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT success FROM auth_logs')
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return 0.0
            
            success_count = sum(1 for (success,) in results if success)
            return (success_count / len(results)) * 100
            
        except Exception as e:
            logging.error(f"Error calculating success rate: {e}")
            return 0.0
    
    def load_user_data(self, username: str) -> Optional[np.ndarray]:
        """Load EEG data for a specific user."""
        try:
            data_path = self.assets_dir / f'data_{username}.npy'
            if data_path.exists():
                return np.load(data_path)
            return None
        except Exception as e:
            logging.error(f"Error loading user data: {e}")
            return None
    
    def update_training_config(self, new_config: Dict):
        """Update training configuration."""
        self.config.update(new_config)
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving config: {e}")

class EEGDataset(Dataset):
    """PyTorch dataset for EEG data."""
    
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )