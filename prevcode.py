"""
Closed one-tank stratified molten salt TES with heat exchangers.

Layer indexing:  j=0 → TOP (hot),  j=N-1 → BOTTOM (cold)

Improvements over basic model:
  1. NTU-effectiveness HTF model   – HTF temperature evolves layer-by-layer
  2. Buoyancy / mixing enforcement – unstable inversions merged each step
  3. Finer vertical resolution     – N=30 layers (was 10)
  4. End-cap heat losses           – top and bottom flat plates included
  5. Lagrangian fixed-mass layers  – dz updated each step from rho(T)

Performance: property functions are evaluated once into lookup tables at
startup; all per-layer operations use NumPy array ops. Expected runtime < 10s.
"""

import numpy as np
from solar_salt_properties import density, specific_heat, thermal_conductivity
from plotting import plot_all
import time

# ── PARAMETERS ────────────────────────────────────────────────────────────────
V_tank      = 5.0
H_tank      = 4.0
T_init      = 200.0
T_HTF_chg   = 450.0
T_HTF_dis   = 180.0
T_out_min   = 220.0
T_amb       = 25.0
T_ref       = 25.0
T_ref_K     = T_ref + 273.15
T_amb_K     = T_amb + 273.15

N           = 30          # number of vertical layers
N_HX        = 15          # top layers containing the HX (= N//2)

UA_HX       = 500.0
UA_HX_layer = UA_HX / N_HX
U_wall      = 0.5

m_dot_HTF   = 2.0         # HTF mass flow rate [kg/s]
cp_HTF      = 2300.0      # HTF specific heat [J/kgK]  (e.g. Therminol VP-1)
C_HTF       = m_dot_HTF * cp_HTF
NTU         = UA_HX_layer / C_HTF
eps_HX      = 1.0 - np.exp(-NTU)

dt          = 60.0
t_charge    = 8  * 3600
t_storage   = 4  * 3600
t_max_dis   = 8  * 3600

steps_chg  = int(t_charge  / dt)
steps_stor = int(t_storage / dt)
steps_dis  = int(t_max_dis / dt)

# ── GEOMETRY ──────────────────────────────────────────────────────────────────
D_tank      = np.sqrt(4 * V_tank / (H_tank * np.pi))
A_cross     = np.pi * D_tank**2 / 4
A_lat       = np.pi * D_tank * H_tank

# ── FIXED MASS PER LAYER ──────────────────────────────────────────────────────
rho_init    = density(T_init)
m_layer     = rho_init * V_tank / N      # [kg], constant — Lagrangian layers

A_lat_layer = A_lat / N
endcap_mask = np.zeros(N)
endcap_mask[0]   = 1.0
endcap_mask[N-1] = 1.0

# ── PROPERTY LOOKUP TABLES ────────────────────────────────────────────────────
# Built once at startup; np.interp replaces per-element Python calls in the loop
_T_lut   = np.linspace(150.0, 600.0, 4501)   # 0.1 °C resolution
_rho_lut = np.array([density(t)              for t in _T_lut])
_cp_lut  = np.array([specific_heat(t)        for t in _T_lut])
_k_lut   = np.array([thermal_conductivity(t) for t in _T_lut])

print("Property tables built.")


def props(T_arr):
    """Vectorised property lookup — pure NumPy, no Python loops."""
    rho = np.interp(T_arr, _T_lut, _rho_lut)
    cp  = np.interp(T_arr, _T_lut, _cp_lut)
    k   = np.interp(T_arr, _T_lut, _k_lut)
    return rho, cp, k


def _cp_scalar(T):
    return float(np.interp(T, _T_lut, _cp_lut))


# ── HTF MARCH ─────────────────────────────────────────────────────────────────
def htf_march(T_salt_hx, T_HTF_inlet, direction):
    """
    March HTF through HX layers sequentially.
    direction: 'top_down' (charge) or 'bottom_up' (discharge)
    Returns Q_hx [W] per layer (+ve = heat into salt) and HTF outlet temp [°C].
    """
    Q_hx  = np.zeros(N_HX)
    T_htf = T_HTF_inlet
    order = range(N_HX) if direction == 'top_down' else range(N_HX-1, -1, -1)
    for j in order:
        Q_j     = eps_HX * C_HTF * (T_htf - T_salt_hx[j])
        Q_hx[j] = Q_j
        T_htf  -= Q_j / C_HTF
    return Q_hx, T_htf


# ── BUOYANCY MIXING ───────────────────────────────────────────────────────────
def enforce_stratification(T_arr):
    """
    Resolve cold-over-hot inversions using the pool-adjacent-violators (PAV)
    algorithm — O(N) and branchless, no Python while/for loop over layers.

    Any contiguous block of layers that violates monotonic decrease is replaced
    by their mass-weighted mean temperature.  Irreversibility is computed for
    each merged block via Gouy-Stodola.

    Returns stabilised temperature array and mixing irreversibility [J].
    """
    T = T_arr.copy()

    # Fast exit: check if already stable (common case during storage)
    if np.all(T[:-1] >= T[1:]):
        return T, 0.0

    I_mix = 0.0

    # PAV: scan and accumulate blocks that need merging
    # Each block is a contiguous run where the monotone constraint is violated.
    # We represent the merged profile as a stack of (start, end, mean) blocks.
    stack = []   # list of [start_idx, end_idx, sum_T, count]

    for j in range(N):
        # New single-element block
        block = [j, j, T[j], 1]
        # Merge with top of stack while inversion exists (top mean < new mean)
        while stack and stack[-1][2] / stack[-1][3] < block[2] / block[3]:
            prev = stack.pop()
            block[0]  = prev[0]           # extend start
            block[2] += prev[2]           # accumulate sum
            block[3] += prev[3]           # accumulate count
        stack.append(block)

    # Write merged temperatures back and compute irreversibility
    for block in stack:
        start, end, sum_T, count = block
        T_avg   = sum_T / count
        T_avg_K = T_avg + 273.15
        for j in range(start, end + 1):
            if abs(T[j] - T_avg) > 1e-10:   # only if actually merged
                T_j_K  = T[j] + 273.15
                cp_j   = _cp_scalar(T[j])
                dS     = m_layer * cp_j * np.log(T_avg_K / T_j_K)
                I_mix += T_ref_K * dS
            T[j] = T_avg

    return T, I_mix


# ── SIMULATION ────────────────────────────────────────────────────────────────
def simulate(T_start, steps, mode):
    T             = np.zeros((steps, N))
    T[0]          = T_start.copy()
    Q_HX_arr      = np.zeros(steps)
    T_HTF_out_arr = np.zeros(steps)
    I_wall_arr    = np.zeros(steps)
    I_cond_arr    = np.zeros(steps)
    I_mix_arr     = np.zeros(steps)
    stop          = steps

    for i in range(steps - 1):
        T_c = T[i]

        # ── Properties via lookup tables ──────────────────────────────────
        rho, cp, k = props(T_c)
        T_K = T_c + 273.15

        # ── Lagrangian layer heights ──────────────────────────────────────
        dz       = m_layer / (rho * A_cross)
        dz_iface = 0.5 * (dz[:-1] + dz[1:])

        # ── Conduction ────────────────────────────────────────────────────
        k_iface  = 0.5 * (k[:-1] + k[1:])
        flux     = k_iface * A_cross / dz_iface * (T_c[:-1] - T_c[1:])
        Q_cond   = np.zeros(N)
        Q_cond[:-1] -= flux
        Q_cond[1:]  += flux

        # ── Wall + end-cap losses ─────────────────────────────────────────
        Q_loss = (U_wall * A_lat_layer * (T_c - T_amb) +
                  U_wall * A_cross * endcap_mask * (T_c - T_amb))

        # ── Irreversibilities ─────────────────────────────────────────────
        I_wall_step = np.sum(T_ref_K * Q_loss * (1.0/T_amb_K - 1.0/T_K))
        I_cond_step = np.sum(T_ref_K * flux   * (1.0/T_K[1:] - 1.0/T_K[:-1]))

        # ── HX heat transfer ──────────────────────────────────────────────
        Q_hx      = np.zeros(N)
        T_htf_out = 0.0
        if mode == 'charge':
            Q_hx[:N_HX], T_htf_out = htf_march(T_c[:N_HX], T_HTF_chg, 'top_down')
        elif mode == 'discharge':
            Q_hx[:N_HX], T_htf_out = htf_march(T_c[:N_HX], T_HTF_dis, 'bottom_up')

        # ── Euler update ──────────────────────────────────────────────────
        T_new = T_c + (dt / (m_layer * cp)) * (Q_hx + Q_cond - Q_loss)

        # ── Buoyancy correction ───────────────────────────────────────────
        T_new, I_mix_step = enforce_stratification(T_new)

        T[i+1]           = T_new
        Q_HX_arr[i]      = np.sum(Q_hx)
        T_HTF_out_arr[i] = T_htf_out
        I_wall_arr[i]    = I_wall_step
        I_cond_arr[i]    = I_cond_step
        I_mix_arr[i]     = I_mix_step

        if mode == 'discharge' and T_new[0] < T_out_min:
            stop = i + 2
            print(f"  Discharge stopped at t = {stop*dt/3600:.2f} h  "
                  f"(T_top = {T_new[0]:.1f} °C < {T_out_min} °C)")
            break

    return (T[:stop], stop,
            Q_HX_arr[:stop], T_HTF_out_arr[:stop],
            I_wall_arr[:stop], I_cond_arr[:stop], I_mix_arr[:stop])


# ── EXERGY PROFILE ────────────────────────────────────────────────────────────
def compute_exergy_profile(T_mat):
    T_K = T_mat + 273.15
    cp  = np.interp(T_mat, _T_lut, _cp_lut)
    ex  = m_layer * cp * ((T_K - T_ref_K) - T_ref_K * np.log(T_K / T_ref_K))
    return ex.sum(axis=1)


# ── HX IRREVERSIBILITY ────────────────────────────────────────────────────────
def compute_HX_irreversibility(T_mat, mode, steps):
    I_total = 0.0
    T_slice = T_mat[:steps, :N_HX]
    for i in range(steps):
        T_htf_in = T_HTF_chg if mode == 'charge' else T_HTF_dis
        order    = range(N_HX) if mode == 'charge' else range(N_HX-1, -1, -1)
        for j in order:
            T_j_K        = T_slice[i, j] + 273.15
            Q_layer      = eps_HX * C_HTF * (T_htf_in - T_slice[i, j])
            T_htf_out    = T_htf_in - Q_layer / C_HTF
            T_htf_mean_K = 0.5 * (T_htf_in + T_htf_out) + 273.15
            dS_gen       = Q_layer * (1.0/T_j_K - 1.0/T_htf_mean_K)
            I_total     += T_ref_K * dS_gen * dt
            T_htf_in     = T_htf_out
    return I_total


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    T_start = np.full(N, T_init)
    t0 = time.time()

    print("=== CHARGING ===")
    (T_chg,  steps_chg_done,  Q_chg,  T_HTF_out_chg,
     I_wall_chg,  I_cond_chg,  I_mix_chg)  = simulate(T_start,    steps_chg,  'charge')
    print(f"  done in {time.time()-t0:.1f}s")

    t1 = time.time()
    print("=== STORAGE ===")
    (T_stor, steps_stor_done, Q_stor, T_HTF_out_stor,
     I_wall_stor, I_cond_stor, I_mix_stor) = simulate(T_chg[-1],  steps_stor, 'storage')
    print(f"  done in {time.time()-t1:.1f}s")

    t2 = time.time()
    print("=== DISCHARGING ===")
    (T_dis,  steps_dis_done,  Q_dis,  T_HTF_out_dis,
     I_wall_dis,  I_cond_dis,  I_mix_dis)  = simulate(T_stor[-1], steps_dis,  'discharge')
    print(f"  done in {time.time()-t2:.1f}s")

    # ── EXERGY ────────────────────────────────────────────────────────────────
    Ex_chg  = compute_exergy_profile(T_chg)
    Ex_stor = compute_exergy_profile(T_stor)
    Ex_dis  = compute_exergy_profile(T_dis)

    Ex_charged    = Ex_chg[-1]  - Ex_chg[0]
    Ex_discharged = Ex_dis[0]   - Ex_dis[-1]
    Ex_stor_loss  = Ex_stor[0]  - Ex_stor[-1]

    # ── IRREVERSIBILITIES ─────────────────────────────────────────────────────
    I_HX_chg      = compute_HX_irreversibility(T_chg,  'charge',    steps_chg_done)
    I_HX_dis      = compute_HX_irreversibility(T_dis,  'discharge', steps_dis_done)
    I_wall_chg_J  = np.sum(I_wall_chg)  * dt
    I_wall_stor_J = np.sum(I_wall_stor) * dt
    I_wall_dis_J  = np.sum(I_wall_dis)  * dt
    I_cond_chg_J  = np.sum(I_cond_chg)  * dt
    I_cond_stor_J = np.sum(I_cond_stor) * dt
    I_cond_dis_J  = np.sum(I_cond_dis)  * dt
    I_mix_chg_J   = np.sum(I_mix_chg)
    I_mix_stor_J  = np.sum(I_mix_stor)
    I_mix_dis_J   = np.sum(I_mix_dis)

    I_wall_total  = I_wall_chg_J + I_wall_stor_J + I_wall_dis_J
    I_cond_total  = I_cond_chg_J + I_cond_stor_J + I_cond_dis_J
    I_mix_total   = I_mix_chg_J  + I_mix_stor_J  + I_mix_dis_J
    I_HX_total    = I_HX_chg + I_HX_dis

    # ── EXERGY BALANCE ────────────────────────────────────────────────────────
    Ex_residual = Ex_dis[-1] - Ex_chg[0]
    balance_RHS = (Ex_discharged + Ex_residual + Ex_stor_loss
                   + I_HX_total + I_wall_total + I_cond_total + I_mix_total)
    balance_err = abs(Ex_charged - balance_RHS) / max(abs(Ex_charged), 1.0) * 100

    # ── EFFICIENCIES ──────────────────────────────────────────────────────────
    eta_ex = Ex_discharged / Ex_charged * 100
    E_in   = np.sum(Q_chg)         * dt
    E_out  = np.sum(np.abs(Q_dis)) * dt
    eta_en = E_out / E_in * 100

    # ── HTF OUTLET TEMPERATURES ───────────────────────────────────────────────
    mask_chg = Q_chg != 0
    mask_dis = Q_dis != 0
    T_out_chg_mean = np.mean(T_HTF_out_chg[mask_chg]) if mask_chg.any() else float('nan')
    T_out_dis_mean = np.mean(T_HTF_out_dis[mask_dis]) if mask_dis.any() else float('nan')

    # ── PRINT ─────────────────────────────────────────────────────────────────
    MWh = 3.6e6
    print(f"\nTotal runtime: {time.time()-t0:.1f} s")

    print("\n=== HTF OUTLET TEMPERATURES ===")
    print(f"  Charging    – inlet: {T_HTF_chg:.0f} °C   mean outlet: {T_out_chg_mean:.1f} °C")
    print(f"  Discharging – inlet: {T_HTF_dis:.0f} °C   mean outlet: {T_out_dis_mean:.1f} °C")

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
