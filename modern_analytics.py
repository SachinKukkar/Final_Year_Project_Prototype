import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go

class EEGAnalytics:
    """Advanced analytics system for EEG authentication."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.absolute()
        self.assets_dir = self.base_dir / 'assets'
        self.db_path = self.assets_dir / 'eeg_system.db'
        
    def get_performance_history(self, days: int = 30) -> pd.DataFrame:
        """Get system performance history."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get authentication logs
            query = '''
                SELECT 
                    DATE(timestamp) as date,
                    AVG(CASE WHEN success = 1 THEN confidence ELSE 0 END) as avg_confidence,
                    COUNT(*) as total_attempts,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_attempts,
                    (SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) as success_rate
                FROM auth_logs 
                WHERE timestamp >= datetime('now', '-{} days')
                GROUP BY DATE(timestamp)
                ORDER BY date
            '''.format(days)
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['date'])
                df['accuracy'] = df['success_rate'] / 100.0
                df['response_time'] = np.random.normal(0.5, 0.1, len(df))  # Simulated for now
            
            return df
            
        except Exception as e:
            logging.error(f"Error getting performance history: {e}")
            return pd.DataFrame()
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Get comparison of different models."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT 
                    model_type as Model,
                    MAX(accuracy) as Accuracy,
                    MAX(precision_score) as Precision,
                    MAX(recall) as Recall,
                    MAX(f1_score) as F1_Score,
                    AVG(training_time) as Training_Time,
                    COUNT(*) as Training_Runs
                FROM model_performance 
                GROUP BY model_type
                ORDER BY accuracy DESC
            '''
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            return df
            
        except Exception as e:
            logging.error(f"Error getting model comparison: {e}")
            return pd.DataFrame()
    
    def get_confusion_matrix(self, days: int = 30) -> Optional[np.ndarray]:
        """Get confusion matrix for recent authentications."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT username, success
                FROM auth_logs 
                WHERE timestamp >= datetime('now', '-{} days')
            '''.format(days)
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                return None
            
            # Create binary classification matrix (simplified)
            y_true = [1] * len(df)  # All should be authentic
            y_pred = df['success'].astype(int).tolist()
            
            cm = confusion_matrix(y_true, y_pred)
            return cm
            
        except Exception as e:
            logging.error(f"Error getting confusion matrix: {e}")
            return None
    
    def get_user_analytics(self, username: str) -> Dict[str, Any]:
        """Get detailed analytics for a specific user."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # User authentication history
            auth_query = '''
                SELECT 
                    timestamp,
                    success,
                    confidence,
                    details
                FROM auth_logs 
                WHERE username = ?
                ORDER BY timestamp DESC
                LIMIT 100
            '''
            
            auth_df = pd.read_sql_query(auth_query, conn, params=(username,))
            
            # User registration info
            user_query = '''
                SELECT 
                    username,
                    subject_id,
                    data_segments,
                    created_at
                FROM users 
                WHERE username = ?
            '''
            
            user_df = pd.read_sql_query(user_query, conn, params=(username,))
            conn.close()
            
            analytics = {
                'username': username,
                'registration_info': user_df.to_dict('records')[0] if not user_df.empty else {},
                'auth_history': auth_df.to_dict('records'),
                'total_attempts': len(auth_df),
                'successful_attempts': auth_df['success'].sum() if not auth_df.empty else 0,
                'success_rate': (auth_df['success'].mean() * 100) if not auth_df.empty else 0,
                'avg_confidence': auth_df['confidence'].mean() if not auth_df.empty else 0,
                'last_auth': auth_df['timestamp'].iloc[0] if not auth_df.empty else None
            }
            
            return analytics
            
        except Exception as e:
            logging.error(f"Error getting user analytics: {e}")
            return {}
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # User statistics
            user_stats = pd.read_sql_query('''
                SELECT 
                    COUNT(*) as total_users,
                    AVG(data_segments) as avg_segments_per_user,
                    MIN(created_at) as first_user_registered,
                    MAX(created_at) as last_user_registered
                FROM users
            ''', conn)
            
            # Authentication statistics
            auth_stats = pd.read_sql_query('''
                SELECT 
                    COUNT(*) as total_attempts,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_attempts,
                    AVG(confidence) as avg_confidence,
                    MIN(timestamp) as first_auth,
                    MAX(timestamp) as last_auth
                FROM auth_logs
            ''', conn)
            
            # Model statistics
            model_stats = pd.read_sql_query('''
                SELECT 
                    COUNT(DISTINCT model_type) as total_models_trained,
                    MAX(accuracy) as best_accuracy,
                    AVG(training_time) as avg_training_time
                FROM model_performance
            ''', conn)
            
            conn.close()
            
            stats = {
                'users': user_stats.to_dict('records')[0] if not user_stats.empty else {},
                'authentication': auth_stats.to_dict('records')[0] if not auth_stats.empty else {},
                'models': model_stats.to_dict('records')[0] if not model_stats.empty else {},
                'system_uptime': self._calculate_system_uptime(),
                'data_quality_score': self._calculate_data_quality_score()
            }
            
            return stats
            
        except Exception as e:
            logging.error(f"Error getting system statistics: {e}")
            return {}
    
    def _calculate_system_uptime(self) -> str:
        """Calculate system uptime based on first user registration."""
        try:
            conn = sqlite3.connect(self.db_path)
            result = pd.read_sql_query('''
                SELECT MIN(created_at) as first_activity
                FROM users
            ''', conn)
            conn.close()
            
            if not result.empty and result['first_activity'].iloc[0]:
                first_activity = pd.to_datetime(result['first_activity'].iloc[0])
                uptime = datetime.now() - first_activity
                return str(uptime.days) + " days"
            
            return "0 days"
            
        except Exception as e:
            logging.error(f"Error calculating uptime: {e}")
            return "Unknown"
    
    def _calculate_data_quality_score(self) -> float:
        """Calculate overall data quality score."""
        try:
            # This is a simplified quality score based on various factors
            conn = sqlite3.connect(self.db_path)
            
            # Factor 1: User data completeness
            user_completeness = pd.read_sql_query('''
                SELECT AVG(CASE WHEN data_segments > 0 THEN 1.0 ELSE 0.0 END) as completeness
                FROM users
            ''', conn)
            
            # Factor 2: Authentication consistency
            auth_consistency = pd.read_sql_query('''
                SELECT AVG(confidence) as consistency
                FROM auth_logs
                WHERE success = 1
            ''', conn)
            
            conn.close()
            
            completeness_score = user_completeness['completeness'].iloc[0] if not user_completeness.empty else 0
            consistency_score = auth_consistency['consistency'].iloc[0] if not auth_consistency.empty else 0
            
            # Weighted average
            quality_score = (completeness_score * 0.6 + consistency_score * 0.4) * 100
            return round(quality_score, 2)
            
        except Exception as e:
            logging.error(f"Error calculating data quality score: {e}")
            return 0.0
    
    def generate_report(self, report_type: str) -> Dict[str, Any]:
        """Generate comprehensive reports."""
        try:
            if report_type == "System Summary":
                return self._generate_system_summary_report()
            elif report_type == "User Analysis":
                return self._generate_user_analysis_report()
            elif report_type == "Performance Report":
                return self._generate_performance_report()
            elif report_type == "Security Audit":
                return self._generate_security_audit_report()
            else:
                return {"error": f"Unknown report type: {report_type}"}
                
        except Exception as e:
            logging.error(f"Error generating report: {e}")
            return {"error": str(e)}
    
    def _generate_system_summary_report(self) -> Dict[str, Any]:
        """Generate system summary report."""
        stats = self.get_system_statistics()
        performance = self.get_performance_history()
        model_comparison = self.get_model_comparison()
        
        report = {
            "report_type": "System Summary",
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_users": stats.get('users', {}).get('total_users', 0),
                "total_auth_attempts": stats.get('authentication', {}).get('total_attempts', 0),
                "overall_success_rate": self._calculate_overall_success_rate(),
                "system_uptime": stats.get('system_uptime', '0 days'),
                "data_quality_score": stats.get('data_quality_score', 0)
            },
            "performance_trends": {
                "avg_daily_attempts": performance['total_attempts'].mean() if not performance.empty else 0,
                "best_day_accuracy": performance['accuracy'].max() if not performance.empty else 0,
                "worst_day_accuracy": performance['accuracy'].min() if not performance.empty else 0
            },
            "model_performance": model_comparison.to_dict('records') if not model_comparison.empty else [],
            "recommendations": self._generate_recommendations(stats, performance)
        }
        
        return report
    
    def _generate_user_analysis_report(self) -> Dict[str, Any]:
        """Generate user analysis report."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get all users
            users_df = pd.read_sql_query('SELECT username FROM users', conn)
            
            user_analyses = []
            for username in users_df['username']:
                user_analytics = self.get_user_analytics(username)
                user_analyses.append(user_analytics)
            
            conn.close()
            
            report = {
                "report_type": "User Analysis",
                "generated_at": datetime.now().isoformat(),
                "total_users": len(user_analyses),
                "user_details": user_analyses,
                "user_statistics": {
                    "most_active_user": max(user_analyses, key=lambda x: x['total_attempts'])['username'] if user_analyses else None,
                    "highest_success_rate": max(user_analyses, key=lambda x: x['success_rate'])['username'] if user_analyses else None,
                    "avg_success_rate": np.mean([u['success_rate'] for u in user_analyses]) if user_analyses else 0
                }
            }
            
            return report
            
        except Exception as e:
            logging.error(f"Error generating user analysis report: {e}")
            return {"error": str(e)}
    
    def _generate_performance_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        performance_data = self.get_performance_history(90)  # Last 90 days
        model_comparison = self.get_model_comparison()
        
        report = {
            "report_type": "Performance Report",
            "generated_at": datetime.now().isoformat(),
            "time_period": "Last 90 days",
            "performance_metrics": {
                "avg_accuracy": performance_data['accuracy'].mean() if not performance_data.empty else 0,
                "accuracy_trend": self._calculate_trend(performance_data['accuracy']) if not performance_data.empty else "stable",
                "avg_response_time": performance_data['response_time'].mean() if not performance_data.empty else 0,
                "total_authentications": performance_data['total_attempts'].sum() if not performance_data.empty else 0
            },
            "model_comparison": model_comparison.to_dict('records') if not model_comparison.empty else [],
            "performance_issues": self._identify_performance_issues(performance_data),
            "optimization_suggestions": self._generate_optimization_suggestions(performance_data, model_comparison)
        }
        
        return report
    
    def _generate_security_audit_report(self) -> Dict[str, Any]:
        """Generate security audit report."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Failed authentication attempts
            failed_attempts = pd.read_sql_query('''
                SELECT 
                    username,
                    COUNT(*) as failed_count,
                    MAX(timestamp) as last_failed_attempt
                FROM auth_logs 
                WHERE success = 0 
                GROUP BY username
                ORDER BY failed_count DESC
            ''', conn)
            
            # Suspicious patterns
            suspicious_activity = pd.read_sql_query('''
                SELECT 
                    username,
                    DATE(timestamp) as date,
                    COUNT(*) as attempts_per_day
                FROM auth_logs 
                GROUP BY username, DATE(timestamp)
                HAVING attempts_per_day > 10
                ORDER BY attempts_per_day DESC
            ''', conn)
            
            conn.close()
            
            report = {
                "report_type": "Security Audit",
                "generated_at": datetime.now().isoformat(),
                "security_metrics": {
                    "total_failed_attempts": failed_attempts['failed_count'].sum() if not failed_attempts.empty else 0,
                    "users_with_failures": len(failed_attempts) if not failed_attempts.empty else 0,
                    "suspicious_activity_detected": len(suspicious_activity) if not suspicious_activity.empty else 0
                },
                "failed_attempts_by_user": failed_attempts.to_dict('records') if not failed_attempts.empty else [],
                "suspicious_patterns": suspicious_activity.to_dict('records') if not suspicious_activity.empty else [],
                "security_recommendations": self._generate_security_recommendations(failed_attempts, suspicious_activity)
            }
            
            return report
            
        except Exception as e:
            logging.error(f"Error generating security audit report: {e}")
            return {"error": str(e)}
    
    def _calculate_overall_success_rate(self) -> float:
        """Calculate overall system success rate."""
        try:
            conn = sqlite3.connect(self.db_path)
            result = pd.read_sql_query('''
                SELECT 
                    (SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) as success_rate
                FROM auth_logs
            ''', conn)
            conn.close()
            
            return result['success_rate'].iloc[0] if not result.empty else 0.0
            
        except Exception as e:
            logging.error(f"Error calculating success rate: {e}")
            return 0.0
    
    def _calculate_trend(self, data: pd.Series) -> str:
        """Calculate trend direction."""
        if len(data) < 2:
            return "insufficient_data"
        
        # Simple linear trend
        x = np.arange(len(data))
        slope = np.polyfit(x, data, 1)[0]
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "declining"
        else:
            return "stable"
    
    def _identify_performance_issues(self, performance_data: pd.DataFrame) -> List[str]:
        """Identify potential performance issues."""
        issues = []
        
        if performance_data.empty:
            issues.append("No performance data available")
            return issues
        
        # Check for low accuracy
        if performance_data['accuracy'].mean() < 0.8:
            issues.append("Average accuracy below 80%")
        
        # Check for declining trend
        if self._calculate_trend(performance_data['accuracy']) == "declining":
            issues.append("Accuracy showing declining trend")
        
        # Check for high response times
        if performance_data['response_time'].mean() > 1.0:
            issues.append("High average response time (>1 second)")
        
        # Check for inconsistent performance
        if performance_data['accuracy'].std() > 0.1:
            issues.append("High accuracy variance indicating inconsistent performance")
        
        return issues
    
    def _generate_recommendations(self, stats: Dict, performance: pd.DataFrame) -> List[str]:
        """Generate system recommendations."""
        recommendations = []
        
        # User-based recommendations
        total_users = stats.get('users', {}).get('total_users', 0)
        if total_users < 5:
            recommendations.append("Consider registering more users to improve model robustness")
        
        # Performance-based recommendations
        if not performance.empty:
            avg_accuracy = performance['accuracy'].mean()
            if avg_accuracy < 0.9:
                recommendations.append("Consider retraining models with more data or different algorithms")
            
            if performance['response_time'].mean() > 0.5:
                recommendations.append("Optimize model inference for faster response times")
        
        # Data quality recommendations
        quality_score = stats.get('data_quality_score', 0)
        if quality_score < 80:
            recommendations.append("Improve data quality by reviewing EEG signal preprocessing")
        
        return recommendations
    
    def _generate_optimization_suggestions(self, performance_data: pd.DataFrame, 
                                         model_comparison: pd.DataFrame) -> List[str]:
        """Generate optimization suggestions."""
        suggestions = []
        
        if not model_comparison.empty:
            best_model = model_comparison.loc[model_comparison['Accuracy'].idxmax(), 'Model']
            suggestions.append(f"Consider using {best_model} as primary model (highest accuracy)")
            
            if len(model_comparison) > 1:
                suggestions.append("Implement ensemble methods combining top-performing models")
        
        if not performance_data.empty:
            if performance_data['response_time'].mean() > 0.5:
                suggestions.append("Implement model quantization or pruning for faster inference")
                suggestions.append("Consider using GPU acceleration if available")
        
        suggestions.append("Implement continuous learning to adapt to new data patterns")
        suggestions.append("Set up automated model retraining schedules")
        
        return suggestions
    
    def _generate_security_recommendations(self, failed_attempts: pd.DataFrame, 
                                         suspicious_activity: pd.DataFrame) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        if not failed_attempts.empty:
            max_failures = failed_attempts['failed_count'].max()
            if max_failures > 10:
                recommendations.append("Implement account lockout after multiple failed attempts")
                recommendations.append("Add rate limiting for authentication requests")
        
        if not suspicious_activity.empty:
            recommendations.append("Implement anomaly detection for unusual authentication patterns")
            recommendations.append("Add logging and alerting for suspicious activities")
        
        recommendations.append("Regular security audits and penetration testing")
        recommendations.append("Implement multi-factor authentication for admin access")
        recommendations.append("Regular backup and recovery testing")
        
        return recommendations
    
    def export_analytics_data(self, format_type: str = "csv") -> str:
        """Export analytics data in specified format."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get all relevant data
            users_df = pd.read_sql_query('SELECT * FROM users', conn)
            auth_logs_df = pd.read_sql_query('SELECT * FROM auth_logs', conn)
            model_perf_df = pd.read_sql_query('SELECT * FROM model_performance', conn)
            
            conn.close()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format_type.lower() == "csv":
                # Export as separate CSV files
                export_dir = self.assets_dir / "exports"
                export_dir.mkdir(exist_ok=True)
                
                users_df.to_csv(export_dir / f"users_{timestamp}.csv", index=False)
                auth_logs_df.to_csv(export_dir / f"auth_logs_{timestamp}.csv", index=False)
                model_perf_df.to_csv(export_dir / f"model_performance_{timestamp}.csv", index=False)
                
                return str(export_dir)
            
            elif format_type.lower() == "json":
                # Export as JSON
                export_data = {
                    "users": users_df.to_dict('records'),
                    "auth_logs": auth_logs_df.to_dict('records'),
                    "model_performance": model_perf_df.to_dict('records'),
                    "exported_at": datetime.now().isoformat()
                }
                
                export_path = self.assets_dir / f"analytics_export_{timestamp}.json"
                with open(export_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                return str(export_path)
            
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
        except Exception as e:
            logging.error(f"Error exporting analytics data: {e}")
            return ""