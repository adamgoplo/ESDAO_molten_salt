"""
Closed one-tank stratified molten salt TES with heat exchangers
Charging : HX in top layers transfers heat FROM hot HTF TO salt
Discharging: HX in top layers transfers heat FROM salt TO cold HTF
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from solar_salt_properties import density, specific_heat, thermal_conductivity
from plotting import plot_all




# ── PARAMETERS ────────────────────────────────────────────────────────────────
V_tank   = 5.0       # Tank volume [m3]
H_tank   = 4.0       # Tank height [m]
T_init   = 200.0     # Initial tank temperature [°C] - cold state
T_HTF_chg  = 450.0   # Hot HTF inlet temperature during charging [°C]
T_HTF_dis  = 180.0   # Cold HTF inlet temperature during discharging [°C]
T_out_min  = 220.0   # Minimum useful HTF outlet temp during discharge [°C]
T_amb    = 25.0      # Ambient temperature [°C]
T_ref    = 25.0      # Reference temperature for exergy [°C]
T_ref_K  = T_ref + 273.15

N        = 10        # Number of layers
N_HX     = 5        # Number of top layers with HX (charging from top)
UA_HX    = 500.0     # Total UA of heat exchanger [W/K]
UA_HX_layer = UA_HX / N_HX  # UA per layer [W/K]
U_wall   = 0.5       # Wall heat loss coefficient [W/m2·K]

dt         = 60.0    # Time step [s]
t_charge   = 8 * 3600
t_max_dis  = 8 * 3600
steps_chg  = int(t_charge / dt)
steps_dis  = int(t_max_dis / dt)

# ── GEOMETRY ──────────────────────────────────────────────────────────────────
D_tank   = np.sqrt(4 * V_tank / (H_tank * np.pi))
A_cross  = np.pi * D_tank**2 / 4
A_wall   = np.pi * D_tank * H_tank
dz       = H_tank / N

# ── SIMULATION ────────────────────────────────────────────────────────────────
def simulate(T_start, steps, mode):
    T = np.zeros((steps, N))
    T[0] = T_start.copy()
    Q_HX_total = np.zeros(steps)  # track HX heat transfer per timestep
    stop = steps

    for i in range(steps - 1):
        T_new = T[i].copy()
        Q_HX_step = 0.0

        for j in range(N):
            T_j  = T[i, j]
            rho  = density(T_j)
            cp   = specific_heat(T_j)
            k    = thermal_conductivity(T_j)
            m_lay = rho * V_tank / N

            # Wall heat loss
            Q_loss = U_wall * (A_wall / N) * (T_j - T_amb)

            # Conduction between adjacent layers
            cond_top    = k * A_cross / dz * (T[i, j-1] - T_j) if j > 0 else 0.0
            cond_bottom = k * A_cross / dz * (T[i, j+1] - T_j) if j < N-1 else 0.0

            # Heat exchanger term 
            Q_HX = 0.0
            if mode == 'charge' and j < N_HX:
                # HX in top layers: hot HTF heats salt
                Q_HX = UA_HX_layer * (T_HTF_chg - T_j)
                Q_HX_step += Q_HX
            elif mode == 'discharge' and j < N_HX:
                # HX in top layers: salt heats cold HTF
                Q_HX = UA_HX_layer * (T_HTF_dis - T_j)
                Q_HX_step += Q_HX

            dT = dt / (m_lay * cp) * (Q_HX + cond_top + cond_bottom - Q_loss)
            T_new[j] = T_j + dT

        T[i+1] = T_new
        Q_HX_total[i] = Q_HX_step

        # Stop discharge when HX becomes ineffective (salt too cold)
        if mode == 'discharge' and T[i+1, 0] < T_out_min:
            stop = i + 2
            print(f"Discharge stopped at t={stop*dt/3600:.2f} h "
                  f"(T_top={T[i+1,0]:.1f}°C < {T_out_min}°C)")
            break

    return T[:stop], stop, Q_HX_total[:stop]


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    T_start = np.full(N, T_init)

    # 1. Charging
    print("=== CHARGING ===")
    T_chg, steps_chg_done, Q_chg = simulate(T_start, steps_chg, 'charge')

    # 2. Storage (no HX, only losses)
    print("=== STORAGE ===")
    T_stor, steps_stor_done, Q_stor = simulate(T_chg[-1], steps_chg, 'storage')

    # 3. Discharging
    print("=== DISCHARGING ===")
    T_dis, steps_dis_done, Q_dis = simulate(T_stor[-1], steps_dis, 'discharge')

    # 4. Plot all results
    plot_all(T_chg, T_stor, T_dis, Q_chg, Q_dis, dt, N, H_tank, V_tank)