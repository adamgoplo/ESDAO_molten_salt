import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP


"""
One-tank stratified thermal energy storage - Solar Salt
Waste heat source: ~400°C inlet, ~200°C outlet target
Model: N-node 1D energy balance (explicit Euler)
Based on: Koffi et al. (multi-node model), MATLAB example from lecture
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from solar_salt_properties import density, specific_heat, thermal_conductivity

# ── PARAMETERS ────────────────────────────────────────────────────────────────
V_tank  = 5.0       # Tank volume [m3]
H_tank  = 4.0       # Tank height [m]
T_init  = 200.0     # Initial tank temperature [°C] - cold state
T_in    = 400.0     # Waste heat inlet temperature [°C]
T_amb   = 25.0      # Ambient temperature [°C]
T_ref   = 25.0      # Reference temperature for exergy [°C]
T_ref_K = T_ref + 273.15

N       = 10        # Number of layers (nodes)
U_wall  = 0.5       # Heat loss coefficient tank wall [W/m2·K]
m_dot   = 2.0       # Mass flow rate during charging [kg/s]

# Time settings
dt      = 60.0      # Time step [s]
t_total = 8 * 3600  # Total simulation time [s] (8 hours)
steps   = int(t_total / dt)

# ── GEOMETRY ──────────────────────────────────────────────────────────────────
D_tank  = np.sqrt(4 * V_tank / (H_tank * np.pi))
A_cross = np.pi * D_tank**2 / 4     # Cross-sectional area [m2]
A_wall  = np.pi * D_tank * H_tank   # Lateral wall area [m2]
dz      = H_tank / N                # Layer height [m]

# ── INITIAL CONDITIONS ────────────────────────────────────────────────────────
T = np.full((steps, N), T_init)     # Temperature matrix [steps x nodes]
T[0, 0] = T_in                      # Top node starts at inlet temperature

# ── MAIN SIMULATION LOOP ─────────────────────────────────────────────────────
for i in range(steps - 1):
    for j in range(N):
        T_j   = T[i, j]
        rho   = density(T_j)
        cp    = specific_heat(T_j)
        k     = thermal_conductivity(T_j)
        m_lay = rho * V_tank / N    # Mass of each layer [kg]

        # Heat loss through wall
        Q_loss = U_wall * (A_wall / N) * (T_j - T_amb)

        # Conduction to adjacent layers
        cond_top    = k * A_cross / dz * (T[i, j-1] - T_j) if j > 0   else 0.0
        cond_bottom = k * A_cross / dz * (T[i, j+1] - T_j) if j < N-1 else 0.0

        # Advection: hot fluid enters at top (node 0), flows downward
        adv = m_dot * cp * (T_in - T_j) if j == 0 else m_dot * cp * (T[i, j-1] - T_j)

        dT = dt / (m_lay * cp) * (adv + cond_top + cond_bottom - Q_loss)
        T[i+1, j] = T_j + dT

# ── EXERGY ANALYSIS ───────────────────────────────────────────────────────────
def exergy_layer(T_C, m_kg):
    """Specific stored exergy in a layer [J]"""
    T_K = T_C + 273.15
    cp  = specific_heat(T_C)
    return m_kg * cp * ((T_K - T_ref_K) - T_ref_K * np.log(T_K / T_ref_K))

# Total stored exergy over time
t_axis = np.arange(steps) * dt / 3600  # [hours]
exergy_total = np.zeros(steps)
for i in range(steps):
    for j in range(N):
        rho   = density(T[i, j])
        m_lay = rho * V_tank / N
        exergy_total[i] += exergy_layer(T[i, j], m_lay)

# ── RESULTS ───────────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)

# Plot 1: Temperature profiles in each node over time
fig, ax = plt.subplots(figsize=(10, 5))
for j in range(N):
    ax.plot(t_axis, T[:, j], label=f"Layer {j+1}")
ax.axhline(200, color="red", linestyle="--", label="T_out target (200°C)")
ax.set_xlabel("Time [h]")
ax.set_ylabel("Temperature [°C]")
ax.set_title("Stratified Tank – Temperature per Layer (Solar Salt)")
ax.legend(loc="right", fontsize=7)
ax.grid(True)
plt.tight_layout()
plt.savefig("results/temperature_profiles.png", dpi=150)
plt.show()

# Plot 2: Stored exergy over time
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t_axis, exergy_total / 1e6, color="darkorange")
ax.set_xlabel("Time [h]")
ax.set_ylabel("Stored Exergy [MJ]")
ax.set_title("Total Stored Exergy in Tank")
ax.grid(True)
plt.tight_layout()
plt.savefig("results/exergy_stored.png", dpi=150)
plt.show()

# Save temperatures to CSV
np.savetxt(
    "results/temperature_matrix.csv",
    T,
    delimiter=",",
    header=",".join([f"Layer_{j+1}" for j in range(N)])
)

print(f"Simulation complete. Final avg temperature: {T[-1].mean():.1f} °C")
print(f"Final stored exergy: {exergy_total[-1]/1e6:.2f} MJ")
