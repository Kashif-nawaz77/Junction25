import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# --- 1. SYSTEM CONSTRAINTS AND CONSTANTS (From Junction_HSY_task_description.pptx) ---
TUNNEL_CONSTANTS = {
    "L1_MIN": 0.5,  # Minimum safe tunnel level [m]
    "L1_MAX": 30.0,  # Maximum operating tunnel level [m]
    "L2_LEVEL": 40.0, # Water level at WWTP [m]
    "TIMESTEP_MINUTES": 15,
    "TIME_STEPS_PER_HOUR": 4,
    "NORDPOOL_AREA": "FI", # Finland
}

# --- 2. DATA UTILITY FUNCTIONS ---

def load_and_clean_data(file_name="data/hsy_data.csv"):
    """Loads historical HSY operations data and cleans headers/types."""
    # Skip the unit row (row 1) so the main headers are used.
    df = pd.read_csv(file_name, skiprows=[1])

    # Normalize timestamps: convert time dots to colons -> "15.11.2024 0:00:00"
    df['Time stamp'] = df['Time stamp'].astype(str).str.replace(
        r'(\d{1,2})\.(\d{1,2})\.(\d{4})\s+(\d{1,2})\.(\d{1,2})\.(\d{1,2})',
        r'\1.\2.\3 \4:\5:\6',
        regex=True
    )

    # Parse with explicit format (day-first) and coerce any bad strings to NaT
    df['Time stamp'] = pd.to_datetime(df['Time stamp'],
                                  format='%d.%m.%Y %H:%M:%S',
                                  dayfirst=True,
                                  errors='coerce')

    # Drop rows where parsing failed
    df = df.loc[df['Time stamp'].notna()].copy()

    def clean_col_name(col):
        # Cleans column names for Python access
        return col.lower().replace(' ', '_').replace(':', '').replace('.', '').replace('-', '_').split('(')[0].strip()

    df.columns = [clean_col_name(col) for col in df.columns]
    df = df.set_index('time_stamp')
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    return df

# --- 3. PUMP AND TUNNEL PHYSICS ---

def get_level_volume_interpolator(volume_file="data/tunnel_volume_table.csv"):
    """
    Creates Level-to-Volume and Volume-to-Level interpolation functions.
    Critical for the MPC mass balance (dV/dt).
    """
    try:
        df_vol = pd.read_csv(volume_file)
        # Assumes the first column is Level L1 m and the second is Volume V m³
        level = df_vol.iloc[:, 0].values
        volume = df_vol.iloc[:, 1].values
        
        # V = f(L1)
        interp_v_l = interp1d(level, volume, kind='linear', fill_value='extrapolate')
        # L1 = f(V) -> Needed for constraints in MPC
        interp_l_v = interp1d(volume, level, kind='linear', fill_value='extrapolate')
        
        return interp_v_l, interp_l_v
    except FileNotFoundError:
        print(f"Error: Volume file '{volume_file}' not found. Using simple linear model for physics.")
        # Fallback: simple V=f(L1) based on a rough tunnel cross-section
        interp_v_l = lambda L1: 10000 * L1 + 350
        interp_l_v = lambda V: (V - 350) / 10000
        return interp_v_l, interp_l_v


def calculate_power(Q_total_m3h, L1_m):
    """
    Simulates total power consumption (P) based on total flow (Q) and current level (L1).
    P = (rho * g * Q * H) / (eta * 1000)
    Total Head (H) = L2 - L1 + H_friction
    """
    rho = 1000      # density of water [kg/m³]
    g = 9.81        # gravity [m/s²]
    L2 = TUNNEL_CONSTANTS["L2_LEVEL"]
    
    H_static = L2 - L1_m # Static head [m]
    
    # Placeholder for a rough constant friction loss
    H_friction = 1.0 # [m]
    H_total = H_static + H_friction 
    
    # Q_total must be in m³/s for power calculation
    Q_m3_s = Q_total_m3h / 3600 
    
    # Total Efficiency (eta): Simplified constant based on pump curves (~80%)
    # For a complex model, this must be P = f(Q, H) modeled from the pump curves.
    ETA_TOTAL = 0.80 
    
    P_kW = (rho * g * Q_m3_s * H_total) / (ETA_TOTAL) / 1000
    return P_kW