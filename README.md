# Junction25
# 🧠 Flow-Forward: The Autonomous Pump Optimizer

> **"Making every kilowatt count: The Nord Pool grid's smartest drain."**

***

## 🚀 Overview & Implementation Idea

Flow-Forward is a **Multi-Agent AI** system designed to minimize the energy consumption and cost of wastewater pumping, addressing the challenge of volatile **Nord Pool Day-Ahead Prices** and unpredictable flow.

### Implementation Core

The core solution is built on **Model Predictive Control (MPC)** running a **Mixed-Integer Non-Linear Program (MINLP)**. This powerful setup allows the system to make complex operational decisions every 15 minutes by explicitly handling **discrete hardware states**.

### What We Did & Achieved

1.  **Forecasting Layer:** We deployed dedicated agents to forecast external inputs:
    * **Price Agent:** Predicts 24-hour **Nord Pool Day-Ahead Prices**.
    * **Inflow Agent:** Predicts **Inflow ($F_1$)** based on weather data.

2.  **Discrete Optimization:** The central optimizer uses the forecasts and continuous sensor data (Tunnel Level $L_1$) to select the exact combination of pumps. This solves the discrete problem: **Which of the 5 available pumps should run?**

3.  **Constraint Guarantee:** The model ensures **$L_1$ safety** (level stays between $0.5\text{ m}$ and $8.0\text{ m}$), guarantees smooth output flow, and adheres to the **Control 5 Pumps** capacity limit.

**Result:** A validated, 24-hour optimal pumping schedule that intelligently shifts massive energy consumption to the lowest-cost periods while maintaining full compliance with HSY's stringent process and safety requirements.