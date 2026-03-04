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

@dataclass
class TankConfig:
    """Konfigurasi parameter water tank"""
    
    # Parameter geometri tank (silinder vertikal)
    tank_height: float = 3.0          # m
    tank_diameter: float = 2.0         # m
    tank_area: float = field(init=False, default=None)  # m²
    
    # Parameter pipa
    inlet_diameter: float = 0.15       # m (diperbesar)
    inlet_area: float = field(init=False, default=None)  # m²
    outlet_diameter: float = 0.15      # m (diperbesar)
    outlet_area: float = field(init=False, default=None) # m²
    
    # Parameter aliran
    inlet_velocity: float = 2.0        # m/s (dipercepat)
    outlet_velocity: float = 2.0       # m/s (dipercepat)
    
    # Parameter fisik
    initial_height: float = 0.5         # m
    g: float = 9.81                     # m/s²
    rho_water: float = 1000.0           # kg/m³
    
    # Koefisien losses
    discharge_coeff: float = 0.6         # koefisien discharge
    
    # Parameter kontrol
    pump_on: bool = True
    valve_open: bool = True
    
    # Parameter simulasi (diperbesar)
    simulation_time: float = 600.0       # detik (10 menit)
    time_step: float = 1.0                # detik
    
    def __post_init__(self):
        """Validasi konfigurasi dan hitung atribut turunan"""
        self.tank_area = math.pi * (self.tank_diameter/2)**2
        self.inlet_area = math.pi * (self.inlet_diameter/2)**2
        self.outlet_area = math.pi * (self.outlet_diameter/2)**2
        self.max_volume = self.tank_area * self.tank_height
        
        if self.initial_height > self.tank_height:
            st.warning("⚠️ Peringatan: Ketinggian awal melebihi tinggi tank")
        if self.initial_height < 0:
            st.warning("⚠️ Peringatan: Ketinggian awal tidak boleh negatif")
    
    def copy(self):
        """Buat salinan konfigurasi"""
        params = {k: v for k, v in self.__dict__.items() 
                 if k not in ['tank_area', 'inlet_area', 'outlet_area', 'max_volume']}
        new_config = TankConfig(**params)
        return new_config
    
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

class WaterTankPhysics:
    """Model fisika untuk water tank"""
    
    def __init__(self, config: TankConfig):
        self.config = config
    
    def inlet_flow_rate(self, water_height: float) -> float:
        """Hitung laju aliran masuk (m³/s)"""
        if not self.config.pump_on:
            return 0.0
            
        if water_height >= self.config.tank_height - 0.01:
            return 0.0
            
        return self.config.inlet_area * self.config.inlet_velocity
    
    def outlet_flow_rate(self, water_height: float) -> float:
        """Hitung laju aliran keluar berdasarkan ketinggian air (m³/s)"""
        if not self.config.valve_open:
            return 0.0
            
        if water_height <= 0.01:
            return 0.0
            
        # Model Torricelli: v = Cd * sqrt(2gh)
        outlet_velocity = self.config.discharge_coeff * np.sqrt(2 * self.config.g * max(0, water_height))
        return self.config.outlet_area * outlet_velocity
    
    def net_flow_rate(self, water_height: float) -> float:
        """Hitung laju aliran neto (m³/s)"""
        Q_in = self.inlet_flow_rate(water_height)
        Q_out = self.outlet_flow_rate(water_height)
        return Q_in - Q_out
    
    def height_to_volume(self, height: float) -> float:
        """Konversi ketinggian ke volume"""
        return height * self.config.tank_area

# ====================
# 3. SISTEM PERSAMAAN DIFERENSIAL
# ====================

class TankDifferentialEquations:
    """Sistem persamaan diferensial untuk simulasi kontinu water tank"""
    
    def __init__(self, physics_model: WaterTankPhysics):
        self.physics = physics_model
        self.config = physics_model.config
    
    def system_equations(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Sistem persamaan diferensial:
        y = [water_height]
        
        Returns:
        dy/dt = [dh/dt]
        """
        h = y[0]  # ketinggian air
        
        # Batasi ketinggian antara 0 dan tinggi maksimum
        h = np.clip(h, 0, self.config.tank_height)
        
        # Hitung laju aliran neto
        Q_net = self.physics.net_flow_rate(h)
        
        # Perubahan ketinggian: dh/dt = Q_net / A_tank
        if self.config.tank_area > 0:
            dh_dt = Q_net / self.config.tank_area
        else:
            dh_dt = 0.0
            
        # Jika ketinggian mencapai batas, set laju perubahan ke 0
        if (h >= self.config.tank_height - 0.01 and dh_dt > 0) or (h <= 0.01 and dh_dt < 0):
            dh_dt = 0.0
            
        return np.array([dh_dt])
    
    def get_initial_conditions(self) -> np.ndarray:
        """Kondisi awal sistem"""
        return np.array([self.config.initial_height])

# ====================
# 4. SIMULATOR UTAMA
# ====================

class WaterTankSimulator:
    """Simulator utama water tank"""
    
    def __init__(self, config: TankConfig):
        self.config = config
        self.physics = WaterTankPhysics(config)
        self.equations = TankDifferentialEquations(self.physics)
        
        # Results storage
        self.time_history = None
        self.height_history = None
        self.volume_history = None
        self.inflow_history = None
        self.outflow_history = None
        self.netflow_history = None
        self.results = None
    
    def run_simulation(self) -> Dict:
        """Jalankan simulasi"""
        # Setup time
        t_span = (0, self.config.simulation_time)
        t_eval = np.arange(0, self.config.simulation_time, self.config.time_step)
        
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
        self.height_history = solution.y[0]
        self.volume_history = self.height_history * self.config.tank_area
        
        # Calculate flow histories
        self.inflow_history = np.zeros_like(self.time_history)
        self.outflow_history = np.zeros_like(self.time_history)
        self.netflow_history = np.zeros_like(self.time_history)
        
        for i, h in enumerate(self.height_history):
            self.inflow_history[i] = self.physics.inlet_flow_rate(h)
            self.outflow_history[i] = self.physics.outlet_flow_rate(h)
            self.netflow_history[i] = self.inflow_history[i] - self.outflow_history[i]
        
        # Calculate metrics
        self.results = self._calculate_metrics()
        
        return self.results
    
    def _calculate_metrics(self) -> Dict:
        """Hitung metrik kinerja water tank"""
        if self.time_history is None:
            raise ValueError("Jalankan simulasi terlebih dahulu")
        
        # Cari waktu mencapai penuh (ketinggian >= 95% dari maks)
        full_indices = np.where(self.height_history >= 0.95 * self.config.tank_height)[0]
        time_to_full = float(self.time_history[full_indices[0]]) if len(full_indices) > 0 else self.config.simulation_time
        
        # Cari waktu mencapai kosong (ketinggian <= 5% dari maks)
        empty_indices = np.where(self.height_history <= 0.05 * self.config.tank_height)[0]
        time_to_empty = float(self.time_history[empty_indices[0]]) if len(empty_indices) > 0 else self.config.simulation_time
        
        # Cari waktu mencapai setengah
        half_indices = np.where(self.height_history >= self.config.tank_height/2)[0]
        time_to_half = float(self.time_history[half_indices[0]]) if len(half_indices) > 0 else self.config.simulation_time
        
        metrics = {
            # Time metrics
            'time_to_empty': time_to_empty,
            'time_to_full': time_to_full,
            'time_to_half': time_to_half,
            
            # Height metrics
            'max_height': float(np.max(self.height_history)),
            'min_height': float(np.min(self.height_history)),
            'final_height': float(self.height_history[-1]),
            'avg_height': float(np.mean(self.height_history)),
            
            # Volume metrics
            'max_volume': float(np.max(self.volume_history)),
            'min_volume': float(np.min(self.volume_history)),
            'final_volume': float(self.volume_history[-1]),
            
            # Flow metrics (konversi ke m³/jam untuk readability)
            'max_inflow': float(np.max(self.inflow_history) * 3600),
            'max_outflow': float(np.max(self.outflow_history) * 3600),
            'avg_inflow': float(np.mean(self.inflow_history) * 3600),
            'avg_outflow': float(np.mean(self.outflow_history) * 3600),
            
            # Total volume (integral)
            'total_inflow': float(np.trapezoid(self.inflow_history, self.time_history)),
            'total_outflow': float(np.trapezoid(self.outflow_history, self.time_history)),
            'net_volume_change': float(self.volume_history[-1] - self.volume_history[0]),
        }
        
        return metrics

# ====================
# 5. VISUALISASI dengan PLOTLY
# ====================

class PlotlyWaterTankViz:
    """Kelas untuk visualisasi water tank dengan Plotly"""
    
    @staticmethod
    def plot_height_profile(simulator: WaterTankSimulator, title_prefix=""):
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
            hovertemplate='Waktu: %{x:.1f} detik<br>Ketinggian: %{y:.2f} m<extra></extra>'
        ))
        
        # Tambah area fill
        fig.add_trace(go.Scatter(
            x=time,
            y=height,
            fill='tozeroy',
            mode='none',
            name='Volume Air',
            fillcolor='rgba(0,0,255,0.1)',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Tambah garis referensi
        fig.add_hline(y=config.tank_height, line_dash="dash", 
                     line_color="red", opacity=0.7,
                     annotation_text=f"Tinggi Maks ({config.tank_height:.1f} m)")
        
        fig.add_hline(y=0, line_dash="dash", 
                     line_color="green", opacity=0.7,
                     annotation_text="Tank Kosong")
        
        # Update layout
        title = f'{title_prefix} - Profil Ketinggian Air' if title_prefix else 'Profil Ketinggian Air dalam Tank'
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18, family="Arial, sans-serif", color="darkblue")
            ),
            xaxis_title="Waktu (detik)",
            yaxis_title="Ketinggian Air (m)",
            hovermode="x unified",
            showlegend=True,
            height=400,
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def plot_flow_rates(simulator: WaterTankSimulator):
        """Plot laju aliran masuk dan keluar"""
        fig = go.Figure()
        
        time = simulator.time_history
        inflow = simulator.inflow_history * 3600  # m³/h
        outflow = simulator.outflow_history * 3600  # m³/h
        netflow = simulator.netflow_history * 3600  # m³/h
        
        fig.add_trace(go.Scatter(
            x=time, y=inflow,
            mode='lines', name='Aliran Masuk',
            line=dict(color='green', width=2.5),
            hovertemplate='Waktu: %{x:.1f} detik<br>Q_in: %{y:.2f} m³/h<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=time, y=outflow,
            mode='lines', name='Aliran Keluar',
            line=dict(color='red', width=2.5),
            hovertemplate='Waktu: %{x:.1f} detik<br>Q_out: %{y:.2f} m³/h<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=time, y=netflow,
            mode='lines', name='Aliran Neto',
            line=dict(color='purple', width=2, dash='dot'),
            hovertemplate='Waktu: %{x:.1f} detik<br>Q_net: %{y:.2f} m³/h<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text='Profil Laju Aliran',
                font=dict(size=18, family="Arial, sans-serif", color="darkblue")
            ),
            xaxis_title="Waktu (detik)",
            yaxis_title="Laju Aliran (m³/jam)",
            hovermode="x unified",
            showlegend=True,
            height=400,
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def plot_volume_profile(simulator: WaterTankSimulator):
        """Plot profil volume"""
        fig = go.Figure()
        
        time = simulator.time_history
        volume = simulator.volume_history
        config = simulator.config
        
        fig.add_trace(go.Scatter(
            x=time, y=volume,
            mode='lines', name='Volume Air',
            line=dict(color='purple', width=3),
            fill='tozeroy',
            fillcolor='rgba(128,0,128,0.1)',
            hovertemplate='Waktu: %{x:.1f} detik<br>Volume: %{y:.2f} m³<extra></extra>'
        ))
        
        fig.add_hline(y=config.max_volume, line_dash="dash", 
                     line_color="red", opacity=0.7,
                     annotation_text=f"Volume Maks ({config.max_volume:.2f} m³)")
        
        fig.update_layout(
            title=dict(
                text='Profil Volume Air dalam Tank',
                font=dict(size=18, family="Arial, sans-serif", color="darkblue")
            ),
            xaxis_title="Waktu (detik)",
            yaxis_title="Volume (m³)",
            hovermode="x unified",
            showlegend=True,
            height=400,
            template="plotly_white"
        )
        
        return fig

# ====================
# 6. ANALISIS SKENARIO
# ====================

class ScenarioAnalysis:
    """Analisis berbagai skenario water tank"""
    
    @staticmethod
    def analyze_filling_only(base_config: TankConfig):
        """Analisis skenario hanya pengisian"""
        config = base_config.copy()
        config.valve_open = False
        config.pump_on = True
        
        simulator = WaterTankSimulator(config)
        simulator.run_simulation()
        return simulator
    
    @staticmethod
    def analyze_emptying_only(base_config: TankConfig):
        """Analisis skenario hanya pengosongan"""
        config = base_config.copy()
        config.pump_on = False
        config.valve_open = True
        
        simulator = WaterTankSimulator(config)
        simulator.run_simulation()
        return simulator
    
    @staticmethod
    def analyze_simultaneous(base_config: TankConfig):
        """Analisis skenario pengisian dan pengosongan bersamaan"""
        config = base_config.copy()
        config.pump_on = True
        config.valve_open = True
        
        simulator = WaterTankSimulator(config)
        simulator.run_simulation()
        return simulator

# ====================
# 7. APLIKASI STREAMLIT
# ====================

def create_sidebar():
    """Buat sidebar untuk input parameter"""
    st.sidebar.title("⚙️ Parameter Water Tank")
    
    st.sidebar.subheader("📐 Geometri Tank")
    tank_height = st.sidebar.slider("Tinggi Tank (m)", 1.0, 5.0, 3.0, 0.5)
    tank_diameter = st.sidebar.slider("Diameter Tank (m)", 0.5, 4.0, 2.0, 0.1)
    
    st.sidebar.subheader("🔧 Parameter Pipa")
    inlet_diameter = st.sidebar.slider("Diameter Pipa Inlet (m)", 0.05, 0.3, 0.15, 0.01)
    outlet_diameter = st.sidebar.slider("Diameter Pipa Outlet (m)", 0.05, 0.3, 0.15, 0.01)
    
    st.sidebar.subheader("💧 Parameter Aliran")
    inlet_velocity = st.sidebar.slider("Kecepatan Aliran Masuk (m/s)", 0.5, 3.0, 2.0, 0.1)
    outlet_velocity = st.sidebar.slider("Kecepatan Aliran Keluar (m/s)", 0.5, 3.0, 2.0, 0.1)
    discharge_coeff = st.sidebar.slider("Koefisien Discharge", 0.3, 0.9, 0.6, 0.05)
    
    st.sidebar.subheader("⏱️ Kondisi Awal")
    initial_height = st.sidebar.slider("Ketinggian Awal Air (m)", 0.0, tank_height, 0.5, 0.1)
    
    st.sidebar.subheader("🎮 Kontrol Operasi")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        pump_on = st.checkbox("Pompa Menyala", value=True)
    with col2:
        valve_open = st.checkbox("Valve Terbuka", value=True)
    
    st.sidebar.subheader("⏲️ Parameter Simulasi")
    simulation_time = st.sidebar.slider("Waktu Simulasi (detik)", 100, 1200, 600, 50)
    
    # Buat konfigurasi
    config = TankConfig(
        tank_height=tank_height,
        tank_diameter=tank_diameter,
        inlet_diameter=inlet_diameter,
        outlet_diameter=outlet_diameter,
        inlet_velocity=inlet_velocity,
        outlet_velocity=outlet_velocity,
        initial_height=initial_height,
        discharge_coeff=discharge_coeff,
        pump_on=pump_on,
        valve_open=valve_open,
        simulation_time=float(simulation_time)
    )
    
    return config

def display_results(simulator, results):
    """Tampilkan hasil simulasi dalam metric cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Waktu ke Penuh", f"{results['time_to_full']:.1f} detik")
        st.metric("Ketinggian Akhir", f"{results['final_height']:.2f} m")
    
    with col2:
        st.metric("Waktu ke Kosong", f"{results['time_to_empty']:.1f} detik")
        st.metric("Volume Akhir", f"{results['final_volume']:.2f} m³")
    
    with col3:
        st.metric("Max Aliran Masuk", f"{results['max_inflow']:.2f} m³/jam")
        st.metric("Max Aliran Keluar", f"{results['max_outflow']:.2f} m³/jam")
    
    with col4:
        st.metric("Total Air Masuk", f"{results['total_inflow']:.2f} m³")
        st.metric("Total Air Keluar", f"{results['total_outflow']:.2f} m³")

def main():
    """Aplikasi utama Streamlit"""
    st.set_page_config(
        page_title="Simulasi Water Tank",
        page_icon="💧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("💧 Simulasi Kontinu Water Tank System")
    st.markdown("""
    Aplikasi ini mensimulasikan sistem water tank (pam air) secara kontinu menggunakan model fisika fluida.
    Semua parameter dapat disesuaikan di sidebar.
    """)
    
    # Sidebar untuk input parameter
    config = create_sidebar()
    
    # Informasi debit
    col1, col2 = st.columns(2)
    with col1:
        debit_inlet = config.inlet_area * config.inlet_velocity * 3600
        st.info(f"💧 **Debit Inlet**: {debit_inlet:.2f} m³/jam")
    with col2:
        debit_outlet = config.outlet_area * config.outlet_velocity * 3600
        st.info(f"💧 **Debit Outlet**: {debit_outlet:.2f} m³/jam")
    
    # Jalankan simulasi
    with st.spinner("Menjalankan simulasi..."):
        simulator = WaterTankSimulator(config)
        results = simulator.run_simulation()
    
    st.success("✅ Simulasi selesai!")
    
    # Tampilkan hasil
    display_results(simulator, results)
    
    # Tab untuk visualisasi
    tab1, tab2, tab3 = st.tabs([
        "📈 Profil Ketinggian", 
        "📊 Metrik & Aliran", 
        "🔄 Analisis Skenario"
    ])
    
    with tab1:
        st.subheader("Profil Ketinggian Air")
        
        # Plot height profile
        fig_height = PlotlyWaterTankViz.plot_height_profile(simulator)
        st.plotly_chart(fig_height, use_container_width=True)
        
        # Informasi tambahan
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Ketinggian Maks", f"{results['max_height']:.2f} m")
        with col2:
            st.metric("Ketinggian Min", f"{results['min_height']:.2f} m")
        with col3:
            st.metric("Ketinggian Rata-rata", f"{results['avg_height']:.2f} m")
    
    with tab2:
        st.subheader("Laju Aliran")
        
        # Plot flow rates
        fig_flow = PlotlyWaterTankViz.plot_flow_rates(simulator)
        st.plotly_chart(fig_flow, use_container_width=True)
        
        # Plot volume
        st.subheader("Profil Volume")
        fig_volume = PlotlyWaterTankViz.plot_volume_profile(simulator)
        st.plotly_chart(fig_volume, use_container_width=True)
        
        # Tabel metrik
        st.subheader("Metrik Kinerja Detail")
        metrics_df = pd.DataFrame([
            {"Metrik": "Waktu ke Penuh (detik)", "Nilai": f"{results['time_to_full']:.1f}"},
            {"Metrik": "Waktu ke Kosong (detik)", "Nilai": f"{results['time_to_empty']:.1f}"},
            {"Metrik": "Waktu ke Setengah (detik)", "Nilai": f"{results['time_to_half']:.1f}"},
            {"Metrik": "Ketinggian Maks (m)", "Nilai": f"{results['max_height']:.2f}"},
            {"Metrik": "Ketinggian Min (m)", "Nilai": f"{results['min_height']:.2f}"},
            {"Metrik": "Volume Maks (m³)", "Nilai": f"{results['max_volume']:.2f}"},
            {"Metrik": "Volume Min (m³)", "Nilai": f"{results['min_volume']:.2f}"},
            {"Metrik": "Max Aliran Masuk (m³/jam)", "Nilai": f"{results['max_inflow']:.2f}"},
            {"Metrik": "Max Aliran Keluar (m³/jam)", "Nilai": f"{results['max_outflow']:.2f}"},
            {"Metrik": "Total Air Masuk (m³)", "Nilai": f"{results['total_inflow']:.2f}"},
            {"Metrik": "Total Air Keluar (m³)", "Nilai": f"{results['total_outflow']:.2f}"},
        ])
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    with tab3:
        st.subheader("Analisis Berbagai Skenario")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("**Skenario 1: Hanya Pengisian**")
            if st.button("Jalankan Skenario 1", key="btn1"):
                with st.spinner("Menjalankan simulasi pengisian..."):
                    filling_sim = ScenarioAnalysis.analyze_filling_only(config)
                    fig = PlotlyWaterTankViz.plot_height_profile(filling_sim, "Pengisian")
                    st.plotly_chart(fig, use_container_width=True)
                    st.success(f"✅ Waktu mencapai penuh: {filling_sim.results['time_to_full']:.1f} detik")
                    st.metric("Volume Akhir", f"{filling_sim.results['final_volume']:.2f} m³")
        
        with col2:
            st.warning("**Skenario 2: Hanya Pengosongan**")
            if st.button("Jalankan Skenario 2", key="btn2"):
                with st.spinner("Menjalankan simulasi pengosongan..."):
                    emptying_config = config.copy()
                    emptying_config.initial_height = config.tank_height
                    emptying_sim = ScenarioAnalysis.analyze_emptying_only(emptying_config)
                    fig = PlotlyWaterTankViz.plot_height_profile(emptying_sim, "Pengosongan")
                    st.plotly_chart(fig, use_container_width=True)
                    st.success(f"✅ Waktu mencapai kosong: {emptying_sim.results['time_to_empty']:.1f} detik")
                    st.metric("Volume Akhir", f"{emptying_sim.results['final_volume']:.2f} m³")
        
        with col3:
            st.success("**Skenario 3: Pengisian & Pengosongan**")
            if st.button("Jalankan Skenario 3", key="btn3"):
                with st.spinner("Menjalankan simulasi bersamaan..."):
                    simultaneous_sim = ScenarioAnalysis.analyze_simultaneous(config)
                    fig = PlotlyWaterTankViz.plot_height_profile(simultaneous_sim, "Bersamaan")
                    st.plotly_chart(fig, use_container_width=True)
                    st.success(f"✅ Ketinggian akhir: {simultaneous_sim.results['final_height']:.2f} m")
                    st.metric("Volume Akhir", f"{simultaneous_sim.results['final_volume']:.2f} m³")
    
    # Footer
    st.markdown("---")
    st.markdown("© 2026 - Simulasi Water Tank System | Praktikum MODSIM")

if __name__ == "__main__":
    main()