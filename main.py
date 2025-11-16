import pandas as pd
from agents.price_agent import PriceAgent
from agents.inflow_agent import InflowAgent
from agents.optimization_agent import OptimizationAgent
from utils.helpers import load_and_clean_data

def run_optimization_cycle():
    """
    Runs a single 24-hour optimization cycle based on current state and 
    forecasts from the multi-agent system.
    """
    print("--- Starting Intelligent Flow Optimization Cycle ---")

    # 1. Initialization and Current State
    try:
        hist_df = load_and_clean_data(file_name="data/hsy_data.csv")
        current_L1 = hist_df['water_level_in_tunnel_l2'].iloc[-1]
        print(f"Current Tunnel Level (L1): {current_L1:.2f} m")
    except Exception as e:
        print(f"Error initializing data. Ensure 'data/hsy_data.csv' exists: {e}")
        return

    # --- 2. Forecasting Phase (Agents) ---
    print("\n--- 2. Forecasting ---")

    # Price Agent: Get 24-hour (96 step) price forecast
    price_agent = PriceAgent()
    price_forecast = price_agent.fetch_nordpool_prices(n_steps=96)
    
    # Inflow Agent: Get 24-hour (96 step) inflow forecast
    inflow_agent = InflowAgent()
    inflow_forecast = inflow_agent.forecast_inflow(n_steps=96)

    if price_forecast is None or inflow_forecast is None:
        print("Forecast failure. Cannot proceed with optimization.")
        return

    # --- 3. Optimization Phase (MPC Agent) ---
    print("\n--- 3. Optimization (Model Predictive Control) ---")
    
    optimizer = OptimizationAgent(
        current_L1=current_L1,
        price_forecast=price_forecast,
        inflow_forecast=inflow_forecast,
        horizon=len(price_forecast)
    )
    
    optimal_f2_profile = optimizer.solve_mpc()

    # --- 4. Results ---
    print("\n--- 4. Optimal Pumping Strategy ---")
    if optimal_f2_profile is not None:
        # The first step is the instruction for the next 15 minutes
        next_action = optimal_f2_profile.iloc[0]
        
        print(f"\nImmediate Pumping Action (Next 15 min at {next_action.name.strftime('%H:%M')}):")
        print(f"  Target Pump Flow (F2): {next_action['F2_m3/h_Optimal']:.0f} m³/h")
        print(f"  Predicted L1 Change: {current_L1:.2f} m -> {next_action['L1_m_Optimal']:.2f} m")
        print(f"  Energy Cost Rate: {next_action['P_kW_Optimal']:.1f} kW @ {next_action['Price_EUR/kWh'] * 100:.2f} EUR/100kWh")


        print("\nFull 5-Hour Optimal Profile (First 20 steps):")
        # Display the key decision variables
        print(optimal_f2_profile[['L1_m_Optimal', 'F2_m3/h_Optimal', 'Price_EUR/kWh', 'P_kW_Optimal']].head(20).to_markdown())
        
    # 

if __name__ == "__main__":
    run_optimization_cycle()