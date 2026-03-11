"""
Closed one-tank stratified molten salt TES with heat exchangers.

Layer indexing convention:
  j = 0   → TOP    of tank (hot zone, HX located here)
  j = N-1 → BOTTOM of tank (cold zone)

Improvements over basic model:
  1. NTU-effectiveness HTF model  : HTF temperature evolves layer-by-layer;
                                    outlet temperature tracked each timestep.
  2. Buoyancy / mixing enforcement: After each Euler step, any unstable
                                    temperature inversion (cold-over-hot) is
                                    resolved by merging the offending layers.
  3. Finer vertical resolution    : N increased from 10 → 30 layers for a
                                    sharper thermocline and less numerical
                                    diffusion.
  4. End-cap heat losses          : Top (j=0) and bottom (j=N-1) layers also
                                    lose heat through the flat end plates.
  5. Lagrangian (fixed-mass) layers: Each layer tracks a fixed mass of salt;
                                    the effective volume per layer varies with
                                    temperature-dependent density, so the
                                    conduction length dz_j is updated each step.

Irreversibilities modelled:
  1. HX heat transfer across finite ΔT  (charge & discharge)
  2. Wall + end-cap heat loss to ambient (all phases)
  3. Axial conduction between layers     (all phases)
  4. Buoyancy mixing                     (tracked separately)

Neglected (justified in report):
  - Pump/fan work (≪ thermal flows)
  - Radiation losses (well-insulated tank)
"""

import numpy as np
from solar_salt_properties import density, specific_heat, thermal_conductivity
from plotting import plot_all


# ── PARAMETERS ────────────────────────────────────────────────────────────────
V_tank      = 5.0        # Tank total volume [m³]
H_tank      = 4.0        # Tank height [m]
T_init      = 200.0      # Initial cold-state temperature [°C]
T_HTF_chg   = 450.0      # HTF inlet temperature during charging [°C]
T_HTF_dis   = 180.0      # HTF inlet temperature during discharging [°C]
T_out_min   = 220.0      # Min useful top-layer salt temp for discharge [°C]
T_amb       = 25.0       # Ambient temperature [°C]
T_ref       = 25.0       # Dead-state temperature for exergy [°C]
T_ref_K     = T_ref + 273.15

# ── IMPROVEMENT 3: finer layering ─────────────────────────────────────────────
N           = 30         # Number of vertical layers (was 10)
N_HX        = 15         # Top layers with HX (keep same fraction: N//2)

UA_HX       = 500.0      # Total HX conductance [W/K]
UA_HX_layer = UA_HX / N_HX    # Per-layer HX conductance [W/K]
U_wall      = 0.5        # Lateral wall heat-loss coefficient [W/m²K]

# ── IMPROVEMENT 1: HTF flow parameters ────────────────────────────────────────
m_dot_HTF   = 2.0        # HTF mass flow rate [kg/s]
cp_HTF      = 2300.0     # HTF specific heat (e.g. Therminol VP-1) [J/kgK]
# HTF flows top-to-bottom during charge (hot in at j=0),
# bottom-to-top during discharge (cold in at j=N-1, heating from bottom HX).
# NOTE: since our HX is in the TOP layers only, during discharge the cold HTF
# enters at j=N_HX-1 (bottom of HX zone) and exits at j=0 (top).

dt          = 60.0       # Time step [s]
t_charge    = 8  * 3600
t_storage   = 4  * 3600
t_max_dis   = 8  * 3600

steps_chg  = int(t_charge  / dt)
steps_stor = int(t_storage / dt)
steps_dis  = int(t_max_dis / dt)

# ── GEOMETRY ──────────────────────────────────────────────────────────────────
D_tank  = np.sqrt(4 * V_tank / (H_tank * np.pi))
A_cross = np.pi * D_tank**2 / 4     # Cross-sectional area [m²]
A_lat   = np.pi * D_tank * H_tank   # Total lateral wall area [m²]

# ── IMPROVEMENT 5: fixed-mass layers ─────────────────────────────────────────
# Compute initial mass of each layer from the uniform cold-state density
rho_init   = density(T_init)
m_layer    = rho_init * V_tank / N   # Fixed mass per layer [kg] — never changes


def layer_height(T_j):
    """Instantaneous height of a layer given its temperature [m]."""
    return m_layer / (density(T_j) * A_cross)


# ── EXERGY PROFILE ────────────────────────────────────────────────────────────
def compute_exergy_profile(T_mat):
    """
    Total stored thermo-mechanical exergy [J] at every saved timestep.
    T_mat : shape (steps, N), °C
    """
    Ex = np.zeros(T_mat.shape[0])
    for i in range(T_mat.shape[0]):
        ex = 0.0
        for j in range(N):
            T_C = T_mat[i, j]
            T_K = T_C + 273.15
            cp  = specific_heat(T_C)
            ex += m_layer * cp * ((T_K - T_ref_K) - T_ref_K * np.log(T_K / T_ref_K))
        Ex[i] = ex
    return Ex  # [J]


# ── NTU-EFFECTIVENESS HX HEAT TRANSFER ───────────────────────────────────────
def hx_heat_transfer(T_salt_layer, T_HTF_in, mode):
    """
    NTU-effectiveness model for a single HX layer.

    The HTF stream has finite capacity rate C_HTF = m_dot * cp_HTF.
    The salt layer has effectively infinite capacity (large thermal mass
    relative to one timestep), so we use the single-stream NTU formula:
        ε = 1 - exp(-NTU),   NTU = UA / C_HTF

    Parameters
    ----------
    T_salt_layer : float, current salt temperature in this layer [°C]
    T_HTF_in     : float, HTF temperature entering this layer [°C]
    mode         : 'charge' or 'discharge'

    Returns
    -------
    Q_hx      : float, heat transferred TO salt [W]  (+ve = salt gains heat)
    T_HTF_out : float, HTF temperature leaving this layer [°C]
    """
    C_HTF = m_dot_HTF * cp_HTF                     # [W/K]
    NTU   = UA_HX_layer / C_HTF
    eps   = 1.0 - np.exp(-NTU)

    # Q_max is based on inlet temperature difference
    Q_max = C_HTF * abs(T_HTF_in - T_salt_layer)   # [W], always ≥ 0

    if mode == 'charge':
        # HTF hotter than salt → heat flows HTF→salt
        Q_hx      = eps * C_HTF * (T_HTF_in - T_salt_layer)   # +ve if T_HTF > T_salt
        T_HTF_out = T_HTF_in - Q_hx / C_HTF
    else:
        # discharge: salt hotter than HTF → heat flows salt→HTF
        Q_hx      = eps * C_HTF * (T_HTF_in - T_salt_layer)   # -ve if T_salt > T_HTF
        T_HTF_out = T_HTF_in - Q_hx / C_HTF

    return Q_hx, T_HTF_out


# ── BUOYANCY MIXING ───────────────────────────────────────────────────────────
def enforce_stratification(T_arr):
    """
    Resolve temperature inversions (cold-over-hot) by mass-weighted mixing.
    Scans top→bottom; merges any pair where T[j] < T[j+1] (upper colder
    than lower).  Repeats until the profile is monotonically non-increasing.

    Returns
    -------
    T_stable : corrected temperature array [°C]
    I_mix    : irreversibility from mixing [J]  (T_ref * ΔS_mix)
    """
    T_stable = T_arr.copy()
    I_mix    = 0.0
    changed  = True
    while changed:
        changed = False
        for j in range(N - 1):
            if T_stable[j] < T_stable[j + 1]:   # inversion: upper colder
                # Mass-weighted average (equal masses here)
                T_avg = 0.5 * (T_stable[j] + T_stable[j + 1])
                # Irreversibility from mixing two streams at T_j and T_{j+1}
                cp_j   = specific_heat(T_stable[j])
                cp_jp1 = specific_heat(T_stable[j + 1])
                T_j_K  = T_stable[j]     + 273.15
                T_jp1_K= T_stable[j + 1] + 273.15
                T_avg_K= T_avg           + 273.15
                # ΔS = m*cp*ln(T_avg/T_j) + m*cp*ln(T_avg/T_{j+1})
                dS = (m_layer * cp_j   * np.log(T_avg_K / T_j_K) +
                      m_layer * cp_jp1 * np.log(T_avg_K / T_jp1_K))
                I_mix   += T_ref_K * dS   # Gouy-Stodola [J per mixing event × dt already outside]
                T_stable[j]     = T_avg
                T_stable[j + 1] = T_avg
                changed = True
    return T_stable, I_mix


# ── SIMULATION ────────────────────────────────────────────────────────────────
def simulate(T_start, steps, mode):
    """
    Explicit Euler + NTU-HX + buoyancy correction.

    Returns
    -------
    T           : (actual_steps, N) temperatures [°C]
    stop        : int
    Q_HX_arr    : (actual_steps,) total HX power [W], +ve into salt
    T_HTF_out_arr:(actual_steps,) HTF outlet temperature [°C]  (0 if no HX)
    I_wall_arr  : (actual_steps,) wall loss irreversibility rate [W]
    I_cond_arr  : (actual_steps,) conduction irreversibility rate [W]
    I_mix_arr   : (actual_steps,) buoyancy mixing irreversibility [J/step]
    """
    T             = np.zeros((steps, N))
    T[0]          = T_start.copy()
    Q_HX_arr      = np.zeros(steps)
    T_HTF_out_arr = np.zeros(steps)
    I_wall_arr    = np.zeros(steps)
    I_cond_arr    = np.zeros(steps)
    I_mix_arr     = np.zeros(steps)
    stop          = steps

    T_amb_K = T_amb + 273.15

    for i in range(steps - 1):
        T_cur       = T[i]
        T_new       = T_cur.copy()
        Q_HX_step   = 0.0
        I_wall_step = 0.0
        I_cond_step = 0.0

        # ── IMPROVEMENT 5: compute current layer heights from density ──────
        dz_j = np.array([layer_height(T_cur[j]) for j in range(N)])   # [m]

        # Pre-compute Q_hx per layer: HTF marches through HX zone layer-by-layer
        Q_hx_per_layer  = np.zeros(N)
        T_HTF_out_final = 0.0

        if mode == 'charge':
            T_htf = T_HTF_chg             # enters at top (j=0)
            for j in range(N_HX):
                Q_hx, T_htf        = hx_heat_transfer(T_cur[j], T_htf, 'charge')
                Q_hx_per_layer[j]  = Q_hx
                Q_HX_step         += Q_hx
            T_HTF_out_final = T_htf       # exits at j=N_HX-1

        elif mode == 'discharge':
            T_htf = T_HTF_dis             # enters at bottom of HX zone (j=N_HX-1)
            for j in range(N_HX - 1, -1, -1):
                Q_hx, T_htf        = hx_heat_transfer(T_cur[j], T_htf, 'discharge')
                Q_hx_per_layer[j]  = Q_hx
                Q_HX_step         += Q_hx
            T_HTF_out_final = T_htf       # exits at j=0

        T_HTF_out_arr[i] = T_HTF_out_final

        # ── LAYER ENERGY BALANCE ───────────────────────────────────────────
        for j in range(N):
            T_j   = T_cur[j]
            T_j_K = T_j + 273.15
            cp    = specific_heat(T_j)
            k     = thermal_conductivity(T_j)

            # ── IMPROVEMENT 4: end-cap losses for top and bottom layers ───
            Q_loss_lat = U_wall * (A_lat / N) * (T_j - T_amb)   # lateral [W]
            Q_loss_cap = 0.0
            if j == 0 or j == N - 1:
                Q_loss_cap = U_wall * A_cross * (T_j - T_amb)   # end cap [W]
            Q_loss = Q_loss_lat + Q_loss_cap

            # Wall irreversibility (Gouy-Stodola)
            I_wall_step += T_ref_K * Q_loss * (1.0/T_amb_K - 1.0/T_j_K)

            # ── IMPROVEMENT 5: use variable dz_j for conduction length ────
            dz_above = 0.5 * (dz_j[j-1] + dz_j[j]) if j > 0   else dz_j[j]
            dz_below = 0.5 * (dz_j[j]   + dz_j[j+1]) if j < N-1 else dz_j[j]

            cond_above = k * A_cross / dz_above * (T_cur[j-1] - T_j) if j > 0   else 0.0
            cond_below = k * A_cross / dz_below * (T_cur[j+1] - T_j) if j < N-1 else 0.0

            # Conduction irreversibility (count each interface once)
            if j < N - 1:
                T_below_K = T_cur[j+1] + 273.15
                Q_cond_ij = k * A_cross / dz_below * (T_j - T_cur[j+1])
                if abs(T_j_K - T_below_K) > 1e-9:
                    I_cond_step += T_ref_K * Q_cond_ij * (1.0/T_below_K - 1.0/T_j_K)

            Q_hx = Q_hx_per_layer[j]

            dT = (dt / (m_layer * cp)) * (Q_hx + cond_above + cond_below - Q_loss)
            T_new[j] = T_j + dT

        # ── IMPROVEMENT 2: enforce stable stratification ──────────────────
        T_new, I_mix_step = enforce_stratification(T_new)

        T[i+1]          = T_new
        Q_HX_arr[i]     = Q_HX_step
        I_wall_arr[i]   = I_wall_step
        I_cond_arr[i]   = I_cond_step
        I_mix_arr[i]    = I_mix_step   # already [J] from mixing function

        # Early stop: discharge top layer too cold
        if mode == 'discharge' and T[i+1, 0] < T_out_min:
            stop = i + 2
            print(f"  Discharge stopped at t = {stop * dt / 3600:.2f} h  "
                  f"(T_top = {T[i+1, 0]:.1f} °C < {T_out_min} °C)")
            break

    return (T[:stop], stop,
            Q_HX_arr[:stop], T_HTF_out_arr[:stop],
            I_wall_arr[:stop], I_cond_arr[:stop], I_mix_arr[:stop])


# ── HX IRREVERSIBILITY (post-processing, uses actual NTU Q per layer) ─────────
def compute_HX_irreversibility(T_mat, mode, steps):
    """
    Gouy-Stodola irreversibility from HX finite-ΔT heat transfer [J].
    Re-walks the HTF layer-by-layer to recover the local HTF temperature
    at each layer, then computes:
        I = T_ref * Q_layer * (1/T_salt - 1/T_HTF_mean)  per layer per step
    where T_HTF_mean is the average of HTF inlet and outlet for that layer.
    """
    I_total = 0.0
    for i in range(steps):
        if mode == 'charge':
            T_htf_in = T_HTF_chg          # enters at top (j=0)
            for j in range(N_HX):
                T_j_K        = T_mat[i, j] + 273.15
                Q_layer, T_htf_out = hx_heat_transfer(T_mat[i, j], T_htf_in, 'charge')
                T_htf_mean_K = 0.5 * (T_htf_in + T_htf_out) + 273.15
                dS_gen       = Q_layer * (1.0/T_j_K - 1.0/T_htf_mean_K)
                I_total     += T_ref_K * dS_gen * dt
                T_htf_in     = T_htf_out  # outlet becomes inlet for next layer
        else:
            T_htf_in = T_HTF_dis          # enters at bottom of HX zone (j=N_HX-1)
            for j in range(N_HX - 1, -1, -1):
                T_j_K        = T_mat[i, j] + 273.15
                Q_layer, T_htf_out = hx_heat_transfer(T_mat[i, j], T_htf_in, 'discharge')
                T_htf_mean_K = 0.5 * (T_htf_in + T_htf_out) + 273.15
                dS_gen       = Q_layer * (1.0/T_j_K - 1.0/T_htf_mean_K)
                I_total     += T_ref_K * dS_gen * dt
                T_htf_in     = T_htf_out
    return I_total  # [J], ≥ 0


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    T_start = np.full(N, T_init)

    # 1. Charging
    print("=== CHARGING ===")
    (T_chg,  steps_chg_done,  Q_chg,  T_HTF_out_chg,
     I_wall_chg,  I_cond_chg,  I_mix_chg)  = simulate(T_start,     steps_chg,  'charge')

    # 2. Idle storage
    print("=== STORAGE ===")
    (T_stor, steps_stor_done, Q_stor, T_HTF_out_stor,
     I_wall_stor, I_cond_stor, I_mix_stor) = simulate(T_chg[-1],   steps_stor, 'storage')

    # 3. Discharging
    print("=== DISCHARGING ===")
    (T_dis,  steps_dis_done,  Q_dis,  T_HTF_out_dis,
     I_wall_dis,  I_cond_dis,  I_mix_dis)  = simulate(T_stor[-1],  steps_dis,  'discharge')

    # ── EXERGY PROFILES ───────────────────────────────────────────────────────
    Ex_chg  = compute_exergy_profile(T_chg)
    Ex_stor = compute_exergy_profile(T_stor)
    Ex_dis  = compute_exergy_profile(T_dis)

    Ex_charged    = Ex_chg[-1]  - Ex_chg[0]
    Ex_discharged = Ex_dis[0]   - Ex_dis[-1]
    Ex_stor_loss  = Ex_stor[0]  - Ex_stor[-1]

    # ── IRREVERSIBILITIES [J] ─────────────────────────────────────────────────
    I_HX_chg  = compute_HX_irreversibility(T_chg,  'charge',    steps_chg_done)
    I_HX_dis  = compute_HX_irreversibility(T_dis,  'discharge', steps_dis_done)

    I_wall_chg_J  = np.sum(I_wall_chg)  * dt
    I_wall_stor_J = np.sum(I_wall_stor) * dt
    I_wall_dis_J  = np.sum(I_wall_dis)  * dt
    I_cond_chg_J  = np.sum(I_cond_chg)  * dt
    I_cond_stor_J = np.sum(I_cond_stor) * dt
    I_cond_dis_J  = np.sum(I_cond_dis)  * dt
    I_mix_chg_J   = np.sum(I_mix_chg)   # already [J] per step, no × dt
    I_mix_stor_J  = np.sum(I_mix_stor)
    I_mix_dis_J   = np.sum(I_mix_dis)

    I_wall_total = I_wall_chg_J + I_wall_stor_J + I_wall_dis_J
    I_cond_total = I_cond_chg_J + I_cond_stor_J + I_cond_dis_J
    I_mix_total  = I_mix_chg_J  + I_mix_stor_J  + I_mix_dis_J
    I_HX_total   = I_HX_chg + I_HX_dis

    # ── EXERGY BALANCE ────────────────────────────────────────────────────────
    Ex_residual  = Ex_dis[-1] - Ex_chg[0]
    balance_RHS  = (Ex_discharged + Ex_residual + Ex_stor_loss
                    + I_HX_chg + I_HX_dis
                    + I_wall_chg_J + I_wall_stor_J + I_wall_dis_J
                    + I_cond_chg_J + I_cond_stor_J + I_cond_dis_J
                    + I_mix_chg_J  + I_mix_stor_J  + I_mix_dis_J)
    balance_err  = abs(Ex_charged - balance_RHS) / max(abs(Ex_charged), 1.0) * 100

    # ── EFFICIENCIES ──────────────────────────────────────────────────────────
    eta_ex = Ex_discharged / Ex_charged * 100
    E_in   = np.sum(Q_chg)         * dt
    E_out  = np.sum(np.abs(Q_dis)) * dt
    eta_en = E_out / E_in * 100

    # ── HTF OUTLET TEMPERATURE SUMMARY ────────────────────────────────────────
    # Only meaningful during active HX steps (Q_chg/Q_dis ≠ 0)
    T_HTF_out_chg_mean = np.mean(T_HTF_out_chg[Q_chg  != 0]) if np.any(Q_chg  != 0) else 0.0
    T_HTF_out_dis_mean = np.mean(np.abs(T_HTF_out_dis[Q_dis != 0])) if np.any(Q_dis != 0) else 0.0

    # ── PRINT ─────────────────────────────────────────────────────────────────
    MWh = 3.6e6
    print("\n=== HTF OUTLET TEMPERATURES ===")
    print(f"  Mean HTF outlet (charging):    {T_HTF_out_chg_mean:.1f} °C  "
          f"(in: {T_HTF_chg:.0f} °C)")
    print(f"  Mean HTF outlet (discharging): {T_HTF_out_dis_mean:.1f} °C  "
          f"(in: {T_HTF_dis:.0f} °C)")

    print("\n=== EXERGY ANALYSIS ===")
    print(f"  Exergy charged (input):        {Ex_charged    / MWh:8.4f} MWh")
    print(f"  Exergy discharged (output):    {Ex_discharged / MWh:8.4f} MWh")
    print(f"  Exergy residual in tank:       {Ex_residual   / MWh:8.4f} MWh")
    print(f"  Exergy loss (idle storage):    {Ex_stor_loss  / MWh:8.4f} MWh")
    print()
    print("  --- Irreversibilities by source ---")
    print(f"  I_HX       charging:           {I_HX_chg      / MWh:8.4f} MWh")
    print(f"  I_HX       discharging:        {I_HX_dis      / MWh:8.4f} MWh")
    print(f"  I_wall     charging:           {I_wall_chg_J  / MWh:8.4f} MWh")
    print(f"  I_wall     storage:            {I_wall_stor_J / MWh:8.4f} MWh")
    print(f"  I_wall     discharging:        {I_wall_dis_J  / MWh:8.4f} MWh")
    print(f"  I_cond     charging:           {I_cond_chg_J  / MWh:8.4f} MWh")
    print(f"  I_cond     storage:            {I_cond_stor_J / MWh:8.4f} MWh")
    print(f"  I_cond     discharging:        {I_cond_dis_J  / MWh:8.4f} MWh")
    print(f"  I_mix      charging:           {I_mix_chg_J   / MWh:8.4f} MWh")
    print(f"  I_mix      storage:            {I_mix_stor_J  / MWh:8.4f} MWh")
    print(f"  I_mix      discharging:        {I_mix_dis_J   / MWh:8.4f} MWh")
    print(f"  Total irreversibility:         "
          f"{(I_HX_total+I_wall_total+I_cond_total+I_mix_total)/MWh:8.4f} MWh")
    print()
    print(f"  Exergy balance closure error:  {balance_err:8.2f} %  (target < 1 %)")
    print()
    print(f"  Exergetic efficiency:          {eta_ex:8.1f} %")
    print(f"  Energetic efficiency:          {eta_en:8.1f} %")

    # ── PLOT ──────────────────────────────────────────────────────────────────
    plot_all(T_chg, T_stor, T_dis, Q_chg, Q_dis, dt, N, H_tank, V_tank,
             Ex_chg=Ex_chg, Ex_stor=Ex_stor, Ex_dis=Ex_dis)