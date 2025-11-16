import numpy as np
import pandas as pd
from gekko import GEKKO
from utils.helpers import TUNNEL_CONSTANTS, get_level_volume_interpolator, calculate_power

class OptimizationAgent:
    def __init__(self, current_L1, price_forecast, inflow_forecast, horizon=96):
        self.L1_start = current_L1
        self.prices = price_forecast.values # EUR/kWh
        self.inflow_15min = inflow_forecast.values
        self.horizon = horizon
        self.dt_h = TUNNEL_CONSTANTS["TIMESTEP_MINUTES"] / 60.0 # Time step in hours (0.25h)
        self.V_interp, self.L_interp = get_level_volume_interpolator()

    def solve_mpc(self):
        """
        Solves the Model Predictive Control problem using GEKKO.
        Objective: Minimize total energy cost.
        """
        m = GEKKO(remote=False)
        m.time = np.linspace(0, (self.horizon - 1) * self.dt_h, self.horizon)

        # --- MODEL PARAMETERS ---
        L1_min = TUNNEL_CONSTANTS["L1_MIN"]
        L1_max = TUNNEL_CONSTANTS["L1_MAX"]
        
        # GEKKO requires piecewise linear functions (m.pwl) for interpolation
        # For simplicity and quick implementation, we use a simple linear approximation
        # for L1 from V in the constraint, and integrate V directly.
        # This is the V to L1 conversion based on the volume table (V_min, V_max corresponding to L1_min, L1_max)
        
        V_start = self.V_interp(self.L1_start)
        V_min = self.V_interp(L1_min)
        V_max = self.V_interp(L1_max)
        
        # --- TIME-VARYING PARAMETERS (FORECASTS) ---
        # Convert inflow from m³/15min to m³/h for consistency with F2
        F1_h = self.inflow_15min * TUNNEL_CONSTANTS["TIME_STEPS_PER_HOUR"]
        
        Price_t = m.Param(value=self.prices, name='Price_t') # EUR/kWh
        F1_t = m.Param(value=F1_h, name='F1_t') # m³/h
        
        # --- MODEL VARIABLES ---
        # V (Tunnel Volume) - State variable
        V = m.Var(value=V_start, lb=V_min, ub=V_max, name='V')
        
        # F2 (Total Pumped Flow) - Manipulated Variable (Controller Output)
        # Units: m³/h. Max flow set to 10000 m³/h (based on max historical + safety)
        F2 = m.MV(value=F1_h[0], lb=100, ub=10000, name='F2') 
        F2.DCOST = 0.01 # Minimize change in F2 (smooth flow requirement)
        F2.STATUS = 1 # Allow the optimizer to change F2

        # L1 (Tunnel Level) - Calculated variable (must be non-linear L1=f(V))
        # For the prototype, we use a simple linear inverse relationship to approximate L1(V) for the MPC.
        # In the final model, use m.pwl() for an accurate L1(V)
        L1 = m.Intermediate(L1_min + (L1_max - L1_min) * (V - V_min) / (V_max - V_min), name='L1')

        # P (Total Power) - Calculated
        P = m.Intermediate(calculate_power(F2, L1), name='P') 
        
        # --- SYSTEM DYNAMICS (Differential Equation) ---
        # dV/dt = F1 - F2
        # F1 and F2 are in m³/h, V is in m³, dt is in hours
        m.Equation(V.dt() == F1_t - F2)
        
        # --- CONSTRAINTS ---
        # 1. Level Constraints enforced by V's bounds (V_min, V_max)

        # 2. Level Target: Tunnel must be emptied (L1 < 0.5m) at the end of the horizon 
        # (This is a simplified version of the daily empty requirement)
        m.fix(L1, val=0.5, pos=self.horizon - 1) 

        # --- OBJECTIVE FUNCTION ---
        # Minimize total energy cost: Integral of (P * Price_t) over the time horizon
        # Cost = P * Price_t * dt_h
        m.Obj(m.integral(P * Price_t))
        
        # --- SOLVER CONFIGURATION ---
        m.options.IMODE = 6      # MPC Mode (optimization over time)
        m.options.SOLVER = 3     # IPOPT solver (robust non-linear solver)
        m.options.CV_TYPE = 1    
        
        try:
            m.solve(disp=False)
            
            # Post-processing: Use the accurate L1 interpolation for final results
            final_L1_values = self.L_interp(V.value)
            
            results = pd.DataFrame({
                'L1_m_Optimal': final_L1_values,
                'V_m3_Optimal': V.value,
                'F2_m3/h_Optimal': F2.value,
                'P_kW_Optimal': P.value,
                'Price_EUR/kWh': Price_t.value,
                'Inflow_m3/h': F1_t.value,
            }, index=inflow_forecast.index)
            
            return results
        
        except Exception as e:
            print(f"MPC Solver failed: {e}")
            return None