# ============================================
# APLIKASI STREAMLIT - SIMULASI TANGKI AIR
# Praktikum Pemodelan dan Simulasi - Modul 4
# ============================================

import numpy as np
from scipy.integrate import solve_ivp
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math

# ====================
# 1. KONFIGURASI & SETUP
# ====================

st.set_page_config(
    page_title="Simulasi Tangki Air",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class TankConfig:
    """Konfigurasi parameter tangki air"""
    
    # Dimensi tangki
    tank_height: float = 2.0      # meter
    tank_radius: float = 1.0       # meter
    
    # Debit aliran
    inlet_flow_rate: float = 0.05  # m³/detik
    outlet_flow_rate: float = 0.03  # m³/detik
    
    # Kondisi awal
    initial_height: float = 0.0     # meter
    
    # Parameter simulasi
    simulation_time: float = 300.0  # detik
    time_step: float = 1.0           # detik
    
    # Atribut yang dihitung
    tank_area: float = field(init=False, default=None)
    max_volume: float = field(init=False, default=None)
    
    def __post_init__(self):
        """Validasi konfigurasi dan hitung atribut turunan"""
        self.tank_area = math.pi * (self.tank_radius ** 2)
        self.max_volume = self.tank_area * self.tank_height
        if self.inlet_flow_rate <= 0 and self.outlet_flow_rate <= 0:
            st.warning("Peringatan: Kedua debit bernilai nol, tidak ada aliran!")
        if self.initial_height > self.tank_height:
            st.warning("Peringatan: Ketinggian awal melebihi tinggi tangki!")
    
    def copy(self):
        """Buat salinan konfigurasi"""
        params = {k: v for k, v in self.__dict__.items() 
                  if k not in ['tank_area', 'max_volume']}
        return TankConfig(**params)
    
    def update_parameter(self, parameter_name: str, value: float):
        """Update satu parameter dan hitung ulang atribut turunan"""
        if parameter_name in self.__annotations__:
            setattr(self, parameter_name, value)
            self.__post_init__()
        else:
            raise ValueError(f"Parameter {parameter_name} tidak valid")


# ====================
# 2. MODEL FISIKA
# ====================

class PhysicsModel:
    """Model fisika untuk aliran air dalam tangki"""
    
    def __init__(self, config: TankConfig):
        self.config = config
    
    def net_flow_rate(self, height: float) -> float:
        """Hitung debit bersih (inlet - outlet) dengan mempertimbangkan batas tangki"""
        
        # Jika tangki penuh, inlet berhenti (air meluap)
        if height >= self.config.tank_height:
            q_in = 0.0
        else:
            q_in = self.config.inlet_flow_rate
        
        # Jika tangki kosong, outlet berhenti (tidak ada air)
        if height <= 0:
            q_out = 0.0
        else:
            q_out = self.config.outlet_flow_rate
        
        return q_in - q_out
    
    def height_change_rate(self, height: float) -> float:
        """Hitung laju perubahan ketinggian air dh/dt"""
        net_flow = self.net_flow_rate(height)
        if self.config.tank_area > 0:
            return net_flow / self.config.tank_area
        return 0.0


# ====================
# 3. SISTEM PERSAMAAN DIFERENSIAL
# ====================

class DifferentialEquations:
    """Sistem persamaan diferensial untuk simulasi kontinu"""
    
    def __init__(self, physics_model: PhysicsModel):
        self.physics = physics_model
        self.config = physics_model.config
    
    def system_equations(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Sistem persamaan diferensial:
        y = [height]
        
        Returns:
        dy/dt = [dh/dt]
        """
        height = y[0]
        
        # Batasi height dalam rentang [0, tank_height] untuk stabilitas numerik
        height = np.clip(height, 0, self.config.tank_height)
        
        # Hitung laju perubahan ketinggian
        dh_dt = self.physics.height_change_rate(height)
        
        return np.array([dh_dt])
    
    def get_initial_conditions(self) -> np.ndarray:
        """Kondisi awal sistem"""
        return np.array([self.config.initial_height])


# ====================
# 4. SIMULATOR UTAMA
# ====================

class WaterTankSimulator:
    """Simulator utama proses pengisian/pengosongan tangki"""
    
    def __init__(self, config: TankConfig):
        self.config = config
        self.physics = PhysicsModel(config)
        self.equations = DifferentialEquations(self.physics)
        
        # Results storage
        self.time_history = None
        self.height_history = None
        self.volume_history = None
        self.results = None
    
    def run_simulation(self) -> Dict:
        """Jalankan simulasi"""
        # Setup time
        t_span = (0, self.config.simulation_time)
        t_eval = np.arange(0, self.config.simulation_time + self.config.time_step, 
                           self.config.time_step)
        
        # Initial conditions
        y0 = self.equations.get_initial_conditions()
        
        # Solve ODE system
        solution = solve_ivp(
            fun=self.equations.system_equations,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6,
            atol=1e-9,
            dense_output=True
        )
        
        # Store results
        self.time_history = solution.t
        self.height_history = np.clip(solution.y[0], 0, self.config.tank_height)
        self.volume_history = self.height_history * self.config.tank_area
        
        # Calculate metrics
        self.results = self._calculate_metrics()
        
        return self.results
    
    def _calculate_metrics(self) -> Dict:
        """Hitung metrik kualitas simulasi"""
        if self.time_history is None:
            raise ValueError("Jalankan simulasi terlebih dahulu")
        
        # Waktu untuk mencapai penuh/kosong
        time_to_full = self._get_time_to_height(self.config.tank_height)
        time_to_empty = self._get_time_to_height(0)
        
        # Volume maksimum teoritis
        max_volume = self.config.tank_area * self.config.tank_height
        
        metrics = {
            # Time metrics
            'time_to_full': time_to_full,
            'time_to_empty': time_to_empty,
            'final_time': self.time_history[-1],
            
            # Height metrics
            'max_height': np.max(self.height_history),
            'min_height': np.min(self.height_history),
            'final_height': self.height_history[-1],
            'height_percentage': (self.height_history[-1] / self.config.tank_height) * 100,
            
            # Volume metrics
            'max_volume': np.max(self.volume_history),
            'min_volume': np.min(self.volume_history),
            'final_volume': self.volume_history[-1],
            'volume_percentage': (self.volume_history[-1] / max_volume) * 100,
            
            # Flow metrics
            'total_inflow': self.config.inlet_flow_rate * self.time_history[-1],
            'total_outflow': self.config.outlet_flow_rate * self.time_history[-1],
            'net_flow': (self.config.inlet_flow_rate - self.config.outlet_flow_rate) * self.time_history[-1],
        }
        
        return metrics
    
    def _get_time_to_height(self, target_height: float) -> Optional[float]:
        """Waktu untuk mencapai ketinggian tertentu"""
        if target_height == 0:
            # Mencari waktu ketika height mendekati 0 (dengan toleransi)
            indices = np.where(self.height_history <= 0.01)[0]
        else:
            indices = np.where(self.height_history >= target_height - 0.01)[0]
        
        if len(indices) > 0:
            return self.time_history[indices[0]]
        return None


# ====================
# 5. VISUALISASI dengan PLOTLY
# ====================

class PlotlyVisualization:
    """Kelas untuk visualisasi hasil simulasi dengan Plotly"""
    
    @staticmethod
    def plot_height_profile(simulator: WaterTankSimulator):
        """Plot profil ketinggian air"""
        fig = go.Figure()
        
        time = simulator.time_history
        height = simulator.height_history
        config = simulator.config
        
        # Tambah garis ketinggian
        fig.add_trace(go.Scatter(
            x=time, 
            y=height,
            mode='lines',
            name='Ketinggian Air',
            line=dict(color='blue', width=3),
            fill='tozeroy',
            fillcolor='rgba(0,0,255,0.1)',
            hovertemplate='Waktu: %{x:.1f} detik<br>Ketinggian: %{y:.2f} m<extra></extra>'
        ))
        
        # Tambah garis referensi
        fig.add_hline(y=config.tank_height, line_dash="dash", 
                     line_color="red", opacity=0.7,
                     annotation_text=f"Tinggi Maks ({config.tank_height} m)")
        
        # Tambah garis waktu ke penuh/kosong
        if simulator.results['time_to_full']:
            t_full = simulator.results['time_to_full']
            fig.add_vline(x=t_full, line_dash="dot", line_color="green", opacity=0.7,
                         annotation_text=f"Penuh: {t_full:.1f} detik")
        
        if simulator.results['time_to_empty']:
            t_empty = simulator.results['time_to_empty']
            fig.add_vline(x=t_empty, line_dash="dot", line_color="orange", opacity=0.7,
                         annotation_text=f"Kosong: {t_empty:.1f} detik")
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='Profil Ketinggian Air dalam Tangki',
                font=dict(size=20, family="Arial, sans-serif", color="darkblue")
            ),
            xaxis_title="Waktu (detik)",
            yaxis_title="Ketinggian Air (m)",
            hovermode="x unified",
            showlegend=True,
            height=500,
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def plot_volume_profile(simulator: WaterTankSimulator):
        """Plot profil volume air"""
        fig = go.Figure()
        
        time = simulator.time_history
        volume = simulator.volume_history
        config = simulator.config
        max_volume = config.tank_area * config.tank_height
        
        # Tambah garis volume
        fig.add_trace(go.Scatter(
            x=time, 
            y=volume,
            mode='lines',
            name='Volume Air',
            line=dict(color='green', width=3),
            fill='tozeroy',
            fillcolor='rgba(0,255,0,0.1)',
            hovertemplate='Waktu: %{x:.1f} detik<br>Volume: %{y:.2f} m³<extra></extra>'
        ))
        
        # Tambah garis referensi
        fig.add_hline(y=max_volume, line_dash="dash", 
                     line_color="red", opacity=0.7,
                     annotation_text=f"Volume Maks ({max_volume:.2f} m³)")
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='Profil Volume Air dalam Tangki',
                font=dict(size=20, family="Arial, sans-serif", color="darkgreen")
            ),
            xaxis_title="Waktu (detik)",
            yaxis_title="Volume Air (m³)",
            hovermode="x unified",
            showlegend=True,
            height=500,
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def plot_comparison_chart(simulators: List[WaterTankSimulator], 
                               labels: List[str]):
        """Plot perbandingan beberapa simulasi"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Perbandingan Profil Ketinggian', 
                           'Perbandingan Profil Volume',
                           'Debit Aliran', 
                           'Waktu Pengisian/Pengosongan'),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Plot 1: Height comparison
        for i, sim in enumerate(simulators):
            fig.add_trace(
                go.Scatter(
                    x=sim.time_history,
                    y=sim.height_history,
                    mode='lines',
                    name=labels[i],
                    line=dict(color=colors[i % len(colors)], width=2),
                    showlegend=True if i == 0 else False,
                    hovertemplate='Waktu: %{x:.1f} detik<br>Ketinggian: %{y:.2f} m<extra>' + labels[i] + '</extra>'
                ),
                row=1, col=1
            )
        
        # Plot 2: Volume comparison
        for i, sim in enumerate(simulators):
            fig.add_trace(
                go.Scatter(
                    x=sim.time_history,
                    y=sim.volume_history,
                    mode='lines',
                    name=labels[i],
                    line=dict(color=colors[i % len(colors)], width=2),
                    showlegend=False,
                    hovertemplate='Waktu: %{x:.1f} detik<br>Volume: %{y:.2f} m³<extra>' + labels[i] + '</extra>'
                ),
                row=1, col=2
            )
        
        # Plot 3: Flow rates (bar chart)
        for i, sim in enumerate(simulators):
            config = sim.config
            fig.add_trace(
                go.Bar(
                    name=labels[i],
                    x=['Inlet', 'Outlet'],
                    y=[config.inlet_flow_rate, config.outlet_flow_rate],
                    marker_color=colors[i % len(colors)],
                    opacity=0.7,
                    showlegend=False,
                    hovertemplate='Debit: %{y:.3f} m³/detik<extra>' + labels[i] + '</extra>'
                ),
                row=2, col=1
            )
        
        # Plot 4: Time to full/empty
        metrics = ['time_to_full', 'time_to_empty']
        metric_labels = ['Waktu ke Penuh', 'Waktu ke Kosong']
        
        for i, sim in enumerate(simulators):
            values = [sim.results[metric] or 0 for metric in metrics]
            fig.add_trace(
                go.Bar(
                    name=labels[i],
                    x=metric_labels,
                    y=values,
                    marker_color=colors[i % len(colors)],
                    opacity=0.7,
                    showlegend=False,
                    hovertemplate='Nilai: %{y:.1f} detik<extra>' + labels[i] + '</extra>'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            barmode='group',
            hovermode="closest",
            template="plotly_white"
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Waktu (detik)", row=1, col=1)
        fig.update_xaxes(title_text="Waktu (detik)", row=1, col=2)
        fig.update_yaxes(title_text="Ketinggian (m)", row=1, col=1)
        fig.update_yaxes(title_text="Volume (m³)", row=1, col=2)
        fig.update_yaxes(title_text="Debit (m³/detik)", row=2, col=1)
        fig.update_yaxes(title_text="Waktu (detik)", row=2, col=2)
        
        return fig


# ====================
# 6. ANALISIS SENSITIVITAS
# ====================

class SensitivityAnalysis:
    """Analisis sensitivitas parameter"""
    
    @staticmethod
    def analyze_parameter_sensitivity(base_config: TankConfig,
                                      parameter_name: str,
                                      values: List[float]) -> Dict:
        """Analisis sensitivitas untuk satu parameter"""
        results = []
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, value in enumerate(values):
            status_text.text(f"Menganalisis {parameter_name} = {value:.3f}...")
            
            # Create new config with modified parameter
            config = base_config.copy()
            config.update_parameter(parameter_name, value)
            
            # Run simulation
            simulator = WaterTankSimulator(config)
            metrics = simulator.run_simulation()
            
            results.append({
                'value': value,
                'simulator': simulator,
                'metrics': metrics
            })
            
            # Update progress
            progress_bar.progress((i + 1) / len(values))
        
        status_text.empty()
        progress_bar.empty()
        
        return {
            'parameter': parameter_name,
            'results': results
        }


# ====================
# 7. FUNGSI STREAMLIT
# ====================

def create_sidebar():
    """Buat sidebar untuk input parameter"""
    st.sidebar.title("⚙️ Parameter Tangki Air")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📏 Dimensi Tangki")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        tank_height = st.number_input("Tinggi Tangki (m)", 
                                      min_value=0.5, max_value=10.0, value=2.0, step=0.1)
    with col2:
        tank_radius = st.number_input("Radius Tangki (m)", 
                                      min_value=0.3, max_value=5.0, value=1.0, step=0.1)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("💧 Debit Aliran")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        inlet_flow_rate = st.number_input("Debit Inlet (m³/detik)", 
                                          min_value=0.0, max_value=1.0, value=0.05, step=0.01,
                                          format="%.3f")
    with col2:
        outlet_flow_rate = st.number_input("Debit Outlet (m³/detik)", 
                                           min_value=0.0, max_value=1.0, value=0.03, step=0.01,
                                           format="%.3f")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⏱️ Kondisi Awal & Simulasi")
    
    initial_height = st.sidebar.slider("Ketinggian Awal (m)", 
                                       0.0, tank_height, 0.0, 0.1)
    
    simulation_time = st.sidebar.slider("Waktu Simulasi (detik)", 
                                        30, 3600, 300, 30)
    
    # Hitung informasi tangki
    tank_area = math.pi * (tank_radius ** 2)
    max_volume = tank_area * tank_height
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Informasi Tangki")
    st.sidebar.info(f"""
    **Luas Penampang:** {tank_area:.2f} m²
    **Volume Maks:** {max_volume:.2f} m³
    **Kapasitas:** {max_volume * 1000:.0f} liter
    """)
    
    # Buat konfigurasi
    config = TankConfig(
        tank_height=tank_height,
        tank_radius=tank_radius,
        inlet_flow_rate=inlet_flow_rate,
        outlet_flow_rate=outlet_flow_rate,
        initial_height=initial_height,
        simulation_time=float(simulation_time)
    )
    
    return config


def display_results(simulator, results):
    """Tampilkan hasil simulasi dalam metric cards"""
    
    # Hitung debit bersih
    net_flow = simulator.config.inlet_flow_rate - simulator.config.outlet_flow_rate
    
    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="⏱️ Waktu ke Penuh",
            value=f"{results['time_to_full']:.1f} detik" if results['time_to_full'] else "Tidak tercapai",
            delta=f"{results['time_to_full']/60:.2f} menit" if results['time_to_full'] else None,
            delta_color="normal"
        )
        st.metric(
            label="📏 Ketinggian Akhir",
            value=f"{results['final_height']:.2f} m",
            delta=f"{results['height_percentage']:.1f}% dari maks"
        )
    
    with col2:
        st.metric(
            label="⏱️ Waktu ke Kosong",
            value=f"{results['time_to_empty']:.1f} detik" if results['time_to_empty'] else "Tidak tercapai",
            delta=f"{results['time_to_empty']/60:.2f} menit" if results['time_to_empty'] else None,
            delta_color="inverse"
        )
        st.metric(
            label="💧 Volume Akhir",
            value=f"{results['final_volume']:.2f} m³",
            delta=f"{results['volume_percentage']:.1f}% dari maks"
        )
    
    with col3:
        st.metric(
            label="📈 Ketinggian Maks",
            value=f"{results['max_height']:.2f} m"
        )
        st.metric(
            label="📈 Volume Maks",
            value=f"{results['max_volume']:.2f} m³"
        )
    
    with col4:
        st.metric(
            label="🔄 Debit Bersih",
            value=f"{net_flow:.3f} m³/detik",
            delta="Mengisi" if net_flow > 0 else "Mengosongkan" if net_flow < 0 else "Stabil"
        )
        st.metric(
            label="⏱️ Waktu Simulasi",
            value=f"{simulator.config.simulation_time:.0f} detik"
        )
    
    # Status bar
    if net_flow > 0:
        if results['time_to_full']:
            st.success(f"✅ Tangki akan PENUH dalam {results['time_to_full']:.1f} detik")
        else:
            st.info(f"ℹ️ Tangki menuju PENUH (saat ini {results['height_percentage']:.1f}%)")
    elif net_flow < 0:
        if results['time_to_empty']:
            st.warning(f"⚠️ Tangki akan KOSONG dalam {results['time_to_empty']:.1f} detik")
        else:
            st.info(f"ℹ️ Tangki menuju KOSONG (sisa {results['height_percentage']:.1f}%)")
    else:
        st.info(f"ℹ️ Ketinggian air STABIL pada {results['final_height']:.2f} m")


def main():
    """Aplikasi utama Streamlit"""
    
    # Header
    st.title("💧 Simulasi Kontinu Sistem Tangki Air")
    st.markdown("""
    Aplikasi ini mensimulasikan proses **pengisian dan pengosongan tangki air** secara kontinu.
    Sesuaikan parameter di sidebar dan lihat hasil simulasi secara real-time.
    """)
    
    # Sidebar untuk input parameter
    config = create_sidebar()
    
    # Jalankan simulasi
    with st.spinner("⏳ Menjalankan simulasi..."):
        simulator = WaterTankSimulator(config)
        results = simulator.run_simulation()
    
    # Tampilkan hasil
    st.success("✅ Simulasi selesai!")
    display_results(simulator, results)
    
    # Tab untuk visualisasi
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Profil Ketinggian", 
        "📊 Profil Volume", 
        "🔍 Analisis Sensitivitas", 
        "📋 Data & Parameter"
    ])
    
    with tab1:
        st.subheader("Profil Ketinggian Air")
        fig_height = PlotlyVisualization.plot_height_profile(simulator)
        st.plotly_chart(fig_height, use_container_width=True)
        
        # Informasi tambahan
        with st.expander("ℹ️ Penjelasan Profil Ketinggian"):
            st.markdown("""
            **Rumus perubahan ketinggian:**
            - **Pengisian saja:** `h(t) = h₀ + (Q_in/A) × t` (linear naik)
            - **Pengosongan saja:** `h(t) = h₀ - (Q_out/A) × t` (linear turun)
            - **Bersamaan:** `h(t) = h₀ + ((Q_in - Q_out)/A) × t`
            
            **Keterangan:**
            - `h₀` = ketinggian awal (m)
            - `Q_in` = debit inlet (m³/detik)
            - `Q_out` = debit outlet (m³/detik)
            - `A` = luas penampang tangki (m²)
            """)
    
    with tab2:
        st.subheader("Profil Volume Air")
        fig_volume = PlotlyVisualization.plot_volume_profile(simulator)
        st.plotly_chart(fig_volume, use_container_width=True)
        
        # Progress bars
        st.subheader("📊 Status Kapasitas")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            vol_percent = results['volume_percentage']
            st.progress(min(vol_percent/100, 1.0), 
                       text=f"**Volume Saat Ini:** {results['final_volume']:.2f} m³ ({vol_percent:.1f}%)")
            
            if vol_percent >= 95:
                st.success("✅ Tangki hampir penuh")
            elif vol_percent >= 75:
                st.info("ℹ️ Kapasitas mencukupi")
            elif vol_percent >= 50:
                st.warning("⚠️ Kapasitas menengah")
            else:
                st.error("❌ Kapasitas rendah")
        
        with col2:
            if results['time_to_full']:
                fill_progress = min(simulator.time_history[-1] / results['time_to_full'], 1.0)
                st.progress(fill_progress, 
                           text=f"**Progres Pengisian:** {fill_progress*100:.1f}%")
        
        with col3:
            if results['time_to_empty']:
                empty_progress = min(simulator.time_history[-1] / results['time_to_empty'], 1.0)
                st.progress(empty_progress, 
                           text=f"**Progres Pengosongan:** {empty_progress*100:.1f}%")
    
    with tab3:
        st.subheader("🔍 Analisis Sensitivitas Parameter")
        
        # Pilih parameter untuk analisis sensitivitas
        param_options = {
            "Radius Tangki (m)": "tank_radius",
            "Tinggi Tangki (m)": "tank_height", 
            "Debit Inlet (m³/detik)": "inlet_flow_rate",
            "Debit Outlet (m³/detik)": "outlet_flow_rate"
        }
        
        selected_param = st.selectbox(
            "Pilih parameter untuk analisis sensitivitas:",
            list(param_options.keys())
        )
        
        param_name = param_options[selected_param]
        
        # Buat range nilai berdasarkan parameter
        base_val = getattr(config, param_name)
        
        if "radius" in param_name or "tinggi" in param_name:
            # Untuk dimensi, buat range di sekitar nilai dasar
            values = [
                base_val * 0.5,
                base_val * 0.75,
                base_val,
                base_val * 1.25,
                base_val * 1.5,
                base_val * 1.75,
                base_val * 2.0
            ]
        else:
            # Untuk debit, buat range dari 0 sampai 2× nilai dasar
            values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        
        # Pilih metrik yang akan dianalisis
        metric_options = {
            "Waktu ke Penuh": "time_to_full",
            "Waktu ke Kosong": "time_to_empty",
            "Ketinggian Akhir": "final_height",
            "Volume Akhir": "final_volume"
        }
        
        selected_metric = st.selectbox(
            "Pilih metrik yang dianalisis:",
            list(metric_options.keys())
        )
        metric_name = metric_options[selected_metric]
        
        # Jalankan analisis
        if st.button("🚀 Jalankan Analisis Sensitivitas", type="primary"):
            with st.spinner(f"Menjalankan analisis untuk {selected_param}..."):
                analysis = SensitivityAnalysis.analyze_parameter_sensitivity(
                    config, param_name, values
                )
                
                # Buat dataframe hasil
                analysis_data = []
                for result in analysis['results']:
                    val = result['metrics'][metric_name]
                    analysis_data.append({
                        'Nilai': result['value'],
                        selected_metric: val if val is not None else 0
                    })
                
                df_analysis = pd.DataFrame(analysis_data)
                
                # Tampilkan tabel
                st.dataframe(df_analysis.style.format({
                    selected_metric: '{:.2f}'
                }), use_container_width=True)
                
                # Buat grafik sensitivitas
                fig_sens = go.Figure()
                
                fig_sens.add_trace(go.Scatter(
                    x=df_analysis['Nilai'],
                    y=df_analysis[selected_metric],
                    mode='lines+markers',
                    name=selected_metric,
                    line=dict(color='red', width=3),
                    marker=dict(size=10, color='blue'),
                    hovertemplate=f'{selected_param}: %{{x:.2f}}<br>{selected_metric}: %{{y:.2f}}<extra></extra>'
                ))
                
                fig_sens.update_layout(
                    title=f"Sensitivitas {selected_param} terhadap {selected_metric}",
                    xaxis_title=selected_param,
                    yaxis_title=selected_metric,
                    hovermode="x unified",
                    template="plotly_white",
                    height=500
                )
                
                st.plotly_chart(fig_sens, use_container_width=True)
    
    with tab4:
        st.subheader("📋 Data Simulasi")
        
        # Buat dataframe dari hasil
        data = {
            'Waktu (detik)': simulator.time_history,
            'Ketinggian Air (m)': simulator.height_history,
            'Volume Air (m³)': simulator.volume_history,
            'Persentase Ketinggian (%)': (simulator.height_history / config.tank_height) * 100,
            'Persentase Volume (%)': (simulator.volume_history / (config.tank_area * config.tank_height)) * 100
        }
        
        df = pd.DataFrame(data)
        
        # Tampilkan tabel
        st.dataframe(
            df.style.format({
                'Waktu (detik)': '{:.1f}',
                'Ketinggian Air (m)': '{:.2f}',
                'Volume Air (m³)': '{:.2f}',
                'Persentase Ketinggian (%)': '{:.1f}%',
                'Persentase Volume (%)': '{:.1f}%'
            }),
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data sebagai CSV",
            data=csv,
            file_name=f"data_tangki_{config.tank_height}m_{config.tank_radius}m.csv",
            mime="text/csv"
        )
        
        # Tampilkan parameter simulasi
        with st.expander("📋 Parameter Simulasi Lengkap"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Dimensi Tangki:**")
                st.write(f"- Tinggi: {config.tank_height} m")
                st.write(f"- Radius: {config.tank_radius} m")
                st.write(f"- Luas Penampang: {config.tank_area:.2f} m²")
                st.write(f"- Volume Maks: {config.max_volume:.2f} m³")
                st.write(f"- Kapasitas: {config.max_volume * 1000:.0f} liter")
            
            with col2:
                st.markdown("**Parameter Aliran:**")
                st.write(f"- Debit Inlet: {config.inlet_flow_rate} m³/detik")
                st.write(f"- Debit Outlet: {config.outlet_flow_rate} m³/detik")
                st.write(f"- Debit Bersih: {config.inlet_flow_rate - config.outlet_flow_rate:.3f} m³/detik")
                st.write(f"- Ketinggian Awal: {config.initial_height} m")
                st.write(f"- Waktu Simulasi: {config.simulation_time} detik")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>© 2026 - Simulasi Kontinu Sistem Tangki Air | Praktikum Pemodelan dan Simulasi</p>
        <p style='font-size: 0.8em; color: gray;'>Institut Teknologi Del</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()