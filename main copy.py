"""
Closed one-tank stratified molten salt TES with heat exchangers.

Layer indexing convention:
  j = 0  →  TOP  of tank  (hot zone, where HX sits)
  j = N-1 → BOTTOM of tank (cold zone)

Charging :  HX in top layers (j < N_HX) adds heat FROM hot HTF TO salt
Discharging: HX in top layers (j < N_HX) extracts heat FROM salt TO cold HTF
Storage :   No HX active, only wall losses and inter-layer conduction

Irreversibilities modelled:
  1. HX heat transfer across finite ΔT   (charging and discharging)
  2. Wall heat loss to ambient            (all three phases)
  3. Axial conduction between layers      (all three phases)

Neglected irreversibilities (justification in report):
  - Pump/fan work (assumed negligible compared to thermal flows)
  - Radiation losses (insulation assumed opaque)
  - Mixing at thermocline (numerical diffusion kept small by fine layering)
"""

import numpy as np
from solar_salt_properties import density, specific_heat, thermal_conductivity
from plotting import plot_all


# ── PARAMETERS ────────────────────────────────────────────────────────────────
V_tank      = 5.0       # Tank volume [m³]
H_tank      = 4.0       # Tank height [m]
T_init      = 200.0     # Initial (cold) tank temperature [°C]
T_HTF_chg   = 450.0     # Hot HTF temperature during charging [°C]
T_HTF_dis   = 180.0     # Cold HTF inlet temperature during discharging [°C]
T_out_min   = 220.0     # Min useful salt temp at top during discharge [°C]
T_amb       = 25.0      # Ambient temperature [°C]
T_ref       = 25.0      # Dead-state temperature for exergy [°C]
T_ref_K     = T_ref + 273.15

N           = 10        # Number of vertical layers
N_HX        = 5         # Top layers that contain the heat exchanger
UA_HX       = 500.0     # Total HX conductance [W/K]
UA_HX_layer = UA_HX / N_HX   # HX conductance per layer [W/K]
U_wall      = 0.5       # Overall wall heat-loss coefficient [W/m²K]

dt          = 60.0      # Time step [s]
t_charge    = 8 * 3600  # Charging period [s]
t_storage   = 4 * 3600  # Idle storage period [s]
t_max_dis   = 8 * 3600  # Max discharge period [s]

steps_chg  = int(t_charge  / dt)
steps_stor = int(t_storage / dt)
steps_dis  = int(t_max_dis / dt)

# ── GEOMETRY ──────────────────────────────────────────────────────────────────
D_tank  = np.sqrt(4 * V_tank / (H_tank * np.pi))
A_cross = np.pi * D_tank**2 / 4        # Cross-sectional area [m²]
A_wall  = np.pi * D_tank * H_tank      # Lateral wall area [m²]
dz      = H_tank / N                   # Layer height [m]


# ── EXERGY PROFILE ────────────────────────────────────────────────────────────
def compute_exergy_profile(T_mat):
    """
    Total stored thermo-mechanical exergy [J] at each timestep.
    T_mat : shape (steps, N), temperatures in °C
    """
    Ex = np.zeros(T_mat.shape[0])
    for i in range(T_mat.shape[0]):
        ex_total = 0.0
        for j in range(N):
            T_C = T_mat[i, j]
            T_K = T_C + 273.15
            m_j = density(T_C) * (V_tank / N)
            cp  = specific_heat(T_C)
            ex_total += m_j * cp * ((T_K - T_ref_K) - T_ref_K * np.log(T_K / T_ref_K))
        Ex[i] = ex_total
    return Ex  # [J]


# ── SIMULATION ────────────────────────────────────────────────────────────────
def simulate(T_start, steps, mode):
    """
    Explicit Euler time integration of the stratified tank.

    Returns
    -------
    T       : ndarray (actual_steps, N), temperatures [°C]
    stop    : int, timesteps actually completed
    Q_HX    : ndarray (actual_steps,), net HX power [W]  +ve = into salt
    I_wall  : ndarray (actual_steps,), wall irreversibility rate [W]
    I_cond  : ndarray (actual_steps,), conduction irreversibility rate [W]
    """
    T      = np.zeros((steps, N))
    T[0]   = T_start.copy()
    Q_HX   = np.zeros(steps)
    I_wall = np.zeros(steps)
    I_cond = np.zeros(steps)
    stop   = steps

    T_amb_K = T_amb + 273.15

    for i in range(steps - 1):
        T_new       = T[i].copy()
        Q_HX_step   = 0.0
        I_wall_step = 0.0
        I_cond_step = 0.0

        for j in range(N):
            T_j   = T[i, j]
            T_j_K = T_j + 273.15
            rho   = density(T_j)
            cp    = specific_heat(T_j)
            k     = thermal_conductivity(T_j)
            m_lay = rho * (V_tank / N)

            # ── Wall heat loss ─────────────────────────────────────────────
            Q_loss = U_wall * (A_wall / N) * (T_j - T_amb)  # [W]
            # Gouy-Stodola: I = T0 * S_gen = T0 * Q_loss * (1/T0 - 1/T_j)
            I_wall_step += T_ref_K * Q_loss * (1.0/T_amb_K - 1.0/T_j_K)

            # ── Axial conduction ───────────────────────────────────────────
            cond_above = k * A_cross / dz * (T[i, j-1] - T_j) if j > 0   else 0.0
            cond_below = k * A_cross / dz * (T[i, j+1] - T_j) if j < N-1 else 0.0

            # Count each interface once (layer j → layer j+1 below it)
            if j < N - 1:
                T_below_K = T[i, j+1] + 273.15
                Q_cond_ij = k * A_cross / dz * (T_j - T[i, j+1])  # +ve if j hotter
                if abs(T_j_K - T_below_K) > 1e-9:
                    I_cond_step += T_ref_K * Q_cond_ij * (1.0/T_below_K - 1.0/T_j_K)

            # ── Heat exchanger ─────────────────────────────────────────────
            Q_hx = 0.0
            if mode == 'charge' and j < N_HX:
                Q_hx       = UA_HX_layer * (T_HTF_chg - T_j)   # +ve into salt
                Q_HX_step += Q_hx
            elif mode == 'discharge' and j < N_HX:
                Q_hx       = UA_HX_layer * (T_HTF_dis - T_j)   # -ve out of salt
                Q_HX_step += Q_hx

            dT = (dt / (m_lay * cp)) * (Q_hx + cond_above + cond_below - Q_loss)
            T_new[j] = T_j + dT

        T[i+1]    = T_new
        Q_HX[i]   = Q_HX_step
        I_wall[i] = I_wall_step
        I_cond[i] = I_cond_step

        if mode == 'discharge' and T[i+1, 0] < T_out_min:
            stop = i + 2
            print(f"  Discharge stopped at t = {stop * dt / 3600:.2f} h  "
                  f"(T_top = {T[i+1, 0]:.1f} °C < {T_out_min} °C)")
            break

    return T[:stop], stop, Q_HX[:stop], I_wall[:stop], I_cond[:stop]


# ── HX IRREVERSIBILITY ────────────────────────────────────────────────────────
def compute_HX_irreversibility(T_mat, mode, steps):
    """
    Gouy-Stodola irreversibility from finite ΔT across the HX [J].
      I = T_ref * integral( Q_layer * (1/T_salt - 1/T_HTF) ) dt
    """
    T_HTF_K = (T_HTF_chg if mode == 'charge' else T_HTF_dis) + 273.15
    I_total = 0.0
    for i in range(steps):
        for j in range(N_HX):
            T_j_K = T_mat[i, j] + 273.15
            if mode == 'charge':
                Q_layer = UA_HX_layer * (T_HTF_chg - T_mat[i, j])
            else:
                Q_layer = UA_HX_layer * (T_HTF_dis - T_mat[i, j])
            dS_gen   = Q_layer * (1.0/T_j_K - 1.0/T_HTF_K)
            I_total += T_ref_K * dS_gen * dt
    return I_total  # [J], ≥ 0


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    T_start = np.full(N, T_init)

    # 1. Charging
    print("=== CHARGING ===")
    T_chg,  steps_chg_done,  Q_chg,  I_wall_chg,  I_cond_chg  = simulate(
        T_start, steps_chg, 'charge')

    # 2. Idle storage
    print("=== STORAGE ===")
    T_stor, steps_stor_done, Q_stor, I_wall_stor, I_cond_stor = simulate(
        T_chg[-1], steps_stor, 'storage')

    # 3. Discharging
    print("=== DISCHARGING ===")
    T_dis,  steps_dis_done,  Q_dis,  I_wall_dis,  I_cond_dis  = simulate(
        T_stor[-1], steps_dis, 'discharge')

    # ── EXERGY PROFILES ───────────────────────────────────────────────────────
    Ex_chg  = compute_exergy_profile(T_chg)
    Ex_stor = compute_exergy_profile(T_stor)
    Ex_dis  = compute_exergy_profile(T_dis)

    Ex_charged    = Ex_chg[-1]  - Ex_chg[0]   # exergy gained during charging  [J]
    Ex_discharged = Ex_dis[0]   - Ex_dis[-1]  # exergy recovered during discharge [J]
    Ex_stor_loss  = Ex_stor[0]  - Ex_stor[-1] # exergy lost during idle storage [J]

    # ── IRREVERSIBILITIES [J] ─────────────────────────────────────────────────
    I_HX_chg  = compute_HX_irreversibility(T_chg,  'charge',    steps_chg_done)
    I_HX_dis  = compute_HX_irreversibility(T_dis,  'discharge', steps_dis_done)

    I_wall_chg_J  = np.sum(I_wall_chg)  * dt
    I_wall_stor_J = np.sum(I_wall_stor) * dt
    I_wall_dis_J  = np.sum(I_wall_dis)  * dt
    I_cond_chg_J  = np.sum(I_cond_chg)  * dt
    I_cond_stor_J = np.sum(I_cond_stor) * dt
    I_cond_dis_J  = np.sum(I_cond_dis)  * dt

    I_wall_total = I_wall_chg_J + I_wall_stor_J + I_wall_dis_J
    I_cond_total = I_cond_chg_J + I_cond_stor_J + I_cond_dis_J
    I_HX_total   = I_HX_chg + I_HX_dis

    # ── EXERGY BALANCE CHECK ──────────────────────────────────────────────────
    # Ex_charged = Ex_discharged + Ex_residual + Ex_stor_loss
    #            + I_HX_chg + I_HX_dis
    #            + I_wall_(chg+stor+dis) + I_cond_(chg+stor+dis)
    #
    # Ex_residual = exergy still in tank above cold initial state at end of cycle
    Ex_residual = Ex_dis[-1] - Ex_chg[0]

    balance_RHS = (Ex_discharged
                   + Ex_residual
                   + Ex_stor_loss
                   + I_HX_chg + I_HX_dis
                   + I_wall_chg_J + I_wall_stor_J + I_wall_dis_J
                   + I_cond_chg_J + I_cond_stor_J + I_cond_dis_J)
    balance_err = abs(Ex_charged - balance_RHS) / max(abs(Ex_charged), 1.0) * 100

    # ── EFFICIENCIES ──────────────────────────────────────────────────────────
    eta_ex = Ex_discharged / Ex_charged * 100
    E_in   = np.sum(Q_chg)         * dt
    E_out  = np.sum(np.abs(Q_dis)) * dt
    eta_en = E_out / E_in * 100

    # ── PRINT RESULTS ─────────────────────────────────────────────────────────
    MWh = 3.6e6
    print("\n=== EXERGY ANALYSIS ===")
    print(f"  Exergy charged (input):       {Ex_charged    / MWh:8.4f} MWh")
    print(f"  Exergy discharged (output):   {Ex_discharged / MWh:8.4f} MWh")
    print(f"  Exergy residual in tank:      {Ex_residual   / MWh:8.4f} MWh")
    print(f"  Exergy loss (idle storage):   {Ex_stor_loss  / MWh:8.4f} MWh")
    print()
    print("  --- Irreversibilities by source ---")
    print(f"  I_HX       charging:          {I_HX_chg      / MWh:8.4f} MWh")
    print(f"  I_HX       discharging:       {I_HX_dis      / MWh:8.4f} MWh")
    print(f"  I_wall     charging:          {I_wall_chg_J  / MWh:8.4f} MWh")
    print(f"  I_wall     storage:           {I_wall_stor_J / MWh:8.4f} MWh")
    print(f"  I_wall     discharging:       {I_wall_dis_J  / MWh:8.4f} MWh")
    print(f"  I_cond     charging:          {I_cond_chg_J  / MWh:8.4f} MWh")
    print(f"  I_cond     storage:           {I_cond_stor_J / MWh:8.4f} MWh")
    print(f"  I_cond     discharging:       {I_cond_dis_J  / MWh:8.4f} MWh")
    print(f"  Total irreversibility:        {(I_HX_total+I_wall_total+I_cond_total)/MWh:8.4f} MWh")
    print()
    print(f"  Exergy balance closure error: {balance_err:8.2f} %  (target < 1 %)")
    print()
    print(f"  Exergetic efficiency:         {eta_ex:8.1f} %")
    print(f"  Energetic efficiency:         {eta_en:8.1f} %")

    # ── PLOT ──────────────────────────────────────────────────────────────────
    plot_all(T_chg, T_stor, T_dis, Q_chg, Q_dis, dt, N, H_tank, V_tank,
             Ex_chg=Ex_chg, Ex_stor=Ex_stor, Ex_dis=Ex_dis)