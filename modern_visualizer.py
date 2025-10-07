import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple

class EEGVisualizer:
    """Advanced EEG visualization system for Streamlit."""
    
    def __init__(self):
        self.channels = ['P4', 'Cz', 'F8', 'T7']
        self.sampling_rate = 256
        self.colors = px.colors.qualitative.Set1
    
    def plot_eeg_signals(self, data: np.ndarray, username: str, max_segments: int = 5):
        """Plot EEG signals with interactive features."""
        st.subheader(f"🧠 EEG Signals for {username}")
        
        # Limit segments for performance
        n_segments = min(len(data), max_segments)
        
        # Create subplot for each channel
        fig = make_subplots(
            rows=len(self.channels), cols=1,
            subplot_titles=self.channels,
            shared_xaxis=True,
            vertical_spacing=0.02
        )
        
        # Time axis
        time_axis = np.arange(data.shape[1]) / self.sampling_rate
        
        for segment_idx in range(n_segments):
            for ch_idx, channel in enumerate(self.channels):
                fig.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=data[segment_idx, :, ch_idx],
                        name=f"Segment {segment_idx+1}" if ch_idx == 0 else None,
                        line=dict(color=self.colors[segment_idx % len(self.colors)]),
                        showlegend=(ch_idx == 0),
                        opacity=0.7
                    ),
                    row=ch_idx+1, col=1
                )
        
        fig.update_layout(
            height=800,
            title=f"EEG Time Series - {username}",
            xaxis_title="Time (seconds)",
            hovermode='x unified'
        )
        
        # Update y-axis labels
        for i, channel in enumerate(self.channels):
            fig.update_yaxes(title_text=f"{channel} (μV)", row=i+1, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Signal statistics
        self._show_signal_statistics(data, username)
    
    def _show_signal_statistics(self, data: np.ndarray, username: str):
        """Show statistical information about the signals."""
        st.subheader("📊 Signal Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Basic statistics
            stats_data = []
            for ch_idx, channel in enumerate(self.channels):
                channel_data = data[:, :, ch_idx].flatten()
                stats_data.append({
                    'Channel': channel,
                    'Mean (μV)': f"{np.mean(channel_data):.2f}",
                    'Std (μV)': f"{np.std(channel_data):.2f}",
                    'Min (μV)': f"{np.min(channel_data):.2f}",
                    'Max (μV)': f"{np.max(channel_data):.2f}"
                })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
        
        with col2:
            # Signal quality metrics
            quality_data = []
            for ch_idx, channel in enumerate(self.channels):
                channel_data = data[:, :, ch_idx]
                
                # Calculate signal-to-noise ratio (simplified)
                signal_power = np.mean(channel_data**2)
                noise_estimate = np.var(np.diff(channel_data, axis=1))
                snr = 10 * np.log10(signal_power / noise_estimate) if noise_estimate > 0 else 0
                
                # Calculate zero-crossing rate
                zcr = np.mean([np.sum(np.diff(np.sign(segment)) != 0) for segment in channel_data])
                
                quality_data.append({
                    'Channel': channel,
                    'SNR (dB)': f"{snr:.2f}",
                    'Zero Crossings': f"{zcr:.0f}",
                    'RMS (μV)': f"{np.sqrt(np.mean(channel_data**2)):.2f}"
                })
            
            quality_df = pd.DataFrame(quality_data)
            st.dataframe(quality_df, use_container_width=True)
    
    def plot_frequency_analysis(self, data: np.ndarray, username: str):
        """Plot frequency domain analysis."""
        st.subheader(f"🌊 Frequency Analysis for {username}")
        
        # Calculate power spectral density for each channel
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=self.channels,
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Frequency bands of interest
        bands = {
            'Delta (0.5-4 Hz)': (0.5, 4),
            'Theta (4-8 Hz)': (4, 8),
            'Alpha (8-13 Hz)': (8, 13),
            'Beta (13-30 Hz)': (13, 30),
            'Gamma (30-50 Hz)': (30, 50)
        }
        
        band_powers = {band: [] for band in bands.keys()}
        
        for ch_idx, channel in enumerate(self.channels):
            # Average across segments
            channel_data = np.mean(data[:, :, ch_idx], axis=0)
            
            # Calculate PSD
            freqs, psd = signal.welch(channel_data, fs=self.sampling_rate, nperseg=256)
            
            # Plot PSD
            row = ch_idx // 2 + 1
            col = ch_idx % 2 + 1
            
            fig.add_trace(
                go.Scatter(
                    x=freqs,
                    y=10 * np.log10(psd),  # Convert to dB
                    name=channel,
                    line=dict(color=self.colors[ch_idx])
                ),
                row=row, col=col
            )
            
            # Calculate band powers
            for band_name, (low, high) in bands.items():
                band_mask = (freqs >= low) & (freqs <= high)
                band_power = np.trapz(psd[band_mask], freqs[band_mask])
                band_powers[band_name].append(band_power)
        
        fig.update_layout(
            height=600,
            title=f"Power Spectral Density - {username}",
            showlegend=False
        )
        
        # Update axes
        for i in range(1, 5):
            row = (i-1) // 2 + 1
            col = (i-1) % 2 + 1
            fig.update_xaxes(title_text="Frequency (Hz)", row=row, col=col)
            fig.update_yaxes(title_text="Power (dB)", row=row, col=col)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Band power analysis
        self._show_band_power_analysis(band_powers, username)
    
    def _show_band_power_analysis(self, band_powers: Dict, username: str):
        """Show frequency band power analysis."""
        st.subheader("📊 Frequency Band Analysis")
        
        # Create band power dataframe
        band_data = []
        for band, powers in band_powers.items():
            for ch_idx, power in enumerate(powers):
                band_data.append({
                    'Band': band,
                    'Channel': self.channels[ch_idx],
                    'Power': power,
                    'Relative Power': power / sum(powers) * 100
                })
        
        band_df = pd.DataFrame(band_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Band power heatmap
            pivot_df = band_df.pivot(index='Band', columns='Channel', values='Power')
            
            fig = px.imshow(
                pivot_df.values,
                x=pivot_df.columns,
                y=pivot_df.index,
                color_continuous_scale='viridis',
                title="Band Power Heatmap"
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Relative band power
            fig = px.bar(
                band_df,
                x='Band',
                y='Relative Power',
                color='Channel',
                title="Relative Band Power Distribution",
                barmode='group'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def plot_eeg_features(self, features: Dict):
        """Plot extracted EEG features."""
        st.subheader("🔍 Feature Analysis")
        
        if 'time_features' in features:
            self._plot_time_features(features['time_features'])
        
        if 'frequency_features' in features:
            self._plot_frequency_features(features['frequency_features'])
        
        if 'connectivity_features' in features:
            self._plot_connectivity_features(features['connectivity_features'])
    
    def _plot_time_features(self, time_features: Dict):
        """Plot time domain features."""
        st.write("⏱️ **Time Domain Features**")
        
        feature_names = list(time_features.keys())
        feature_values = list(time_features.values())
        
        fig = px.bar(
            x=feature_names,
            y=feature_values,
            title="Time Domain Features",
            labels={'x': 'Feature', 'y': 'Value'}
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def _plot_frequency_features(self, freq_features: Dict):
        """Plot frequency domain features."""
        st.write("🌊 **Frequency Domain Features**")
        
        # Create radar chart for frequency features
        categories = list(freq_features.keys())
        values = list(freq_features.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Frequency Features'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values)]
                )),
            showlegend=True,
            title="Frequency Domain Features",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _plot_connectivity_features(self, conn_features: Dict):
        """Plot connectivity features."""
        st.write("🔗 **Connectivity Features**")
        
        if 'correlation_matrix' in conn_features:
            corr_matrix = conn_features['correlation_matrix']
            
            fig = px.imshow(
                corr_matrix,
                x=self.channels,
                y=self.channels,
                color_continuous_scale='RdBu',
                title="Channel Correlation Matrix"
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def plot_authentication_results(self, results: Dict):
        """Plot authentication results visualization."""
        st.subheader("🔐 Authentication Analysis")
        
        if 'segments' in results:
            segments_data = results['segments']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Confidence per segment
                segment_df = pd.DataFrame(segments_data)
                
                fig = px.line(
                    segment_df,
                    x='segment',
                    y='confidence',
                    title="Confidence per Segment",
                    markers=True
                )
                
                # Add threshold line
                fig.add_hline(
                    y=0.9,  # Default threshold
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Threshold"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Match/no-match distribution
                match_counts = segment_df['match'].value_counts()
                
                fig = px.pie(
                    values=match_counts.values,
                    names=['No Match', 'Match'] if False in match_counts.index else ['Match'],
                    title="Segment Match Distribution"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Segments", results.get('total_segments', 0))
        
        with col2:
            st.metric("Positive Votes", results.get('positive_votes', 0))
        
        with col3:
            st.metric("Average Confidence", f"{results.get('avg_confidence', 0):.2%}")
        
        with col4:
            st.metric("Max Confidence", f"{results.get('max_confidence', 0):.2%}")
    
    def plot_model_comparison(self, comparison_data: pd.DataFrame):
        """Plot model performance comparison."""
        st.subheader("🔬 Model Performance Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy comparison
            fig = px.bar(
                comparison_data,
                x='Model',
                y='Accuracy',
                title="Model Accuracy Comparison",
                color='Accuracy',
                color_continuous_scale='viridis'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Training time comparison
            if 'Training_Time' in comparison_data.columns:
                fig = px.bar(
                    comparison_data,
                    x='Model',
                    y='Training_Time',
                    title="Training Time Comparison",
                    color='Training_Time',
                    color_continuous_scale='plasma'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics table
        st.subheader("📊 Detailed Metrics")
        st.dataframe(comparison_data, use_container_width=True)
    
    def plot_system_performance(self, performance_data: pd.DataFrame):
        """Plot system performance over time."""
        st.subheader("📈 System Performance Over Time")
        
        if performance_data.empty:
            st.info("No performance data available")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy over time
            fig = px.line(
                performance_data,
                x='timestamp',
                y='accuracy',
                title="Authentication Accuracy Over Time",
                markers=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Success rate over time
            if 'success_rate' in performance_data.columns:
                fig = px.line(
                    performance_data,
                    x='timestamp',
                    y='success_rate',
                    title="Success Rate Over Time",
                    markers=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def create_3d_brain_plot(self, data: np.ndarray, title: str = "3D Brain Activity"):
        """Create a 3D visualization of brain activity (simplified)."""
        st.subheader("🧠 3D Brain Activity Visualization")
        
        # Simplified 3D representation using channel positions
        # In a real implementation, you'd use actual electrode positions
        channel_positions = {
            'P4': [1, -1, 0],
            'Cz': [0, 0, 1],
            'F8': [1, 1, 0],
            'T7': [-1, 0, 0]
        }
        
        # Calculate average activity per channel
        avg_activity = np.mean(np.abs(data), axis=(0, 1))
        
        x, y, z = [], [], []
        colors = []
        
        for i, channel in enumerate(self.channels):
            pos = channel_positions[channel]
            x.append(pos[0])
            y.append(pos[1])
            z.append(pos[2])
            colors.append(avg_activity[i])
        
        fig = go.Figure(data=go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers+text',
            marker=dict(
                size=20,
                color=colors,
                colorscale='viridis',
                showscale=True,
                colorbar=dict(title="Activity Level")
            ),
            text=self.channels,
            textposition="middle center"
        ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)