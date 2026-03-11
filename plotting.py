"""
plotting.py  –  Molten Salt Stratified Tank – all plots
Usage: from plotting import plot_all
       plot_all(T_chg, T_stor, T_dis, Q_chg, Q_dis, dt, N, H_tank, V_tank,
                Ex_chg=..., Ex_stor=..., Ex_dis=...,
                m_dot_chg=..., m_dot_dis=...,
                P_target_chg=..., P_target_dis=...,
                output_dir="results")
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.colors import BoundaryNorm
import os


# ── THERMOCLINE METRIC HELPERS ────────────────────────────────────────────────
def _thermocline_metrics(T_mat, H_tank, N, T_hot_ref, T_cold_ref,
                          lo_frac=0.10, hi_frac=0.90):
    """
    For each timestep compute:
      - thickness  : height span [m] of the 10%–90% transition zone
      - center     : height [m] of the 50% isotherm
      - MIX_number : Rosen & Dincer mixing parameter (0=perfect, 1=fully mixed)

    Layer j=0 is TOP; z_centers[0] = H_tank − dz/2.
    """
    dz       = H_tank / N
    z_top    = np.linspace(H_tank - dz / 2, dz / 2, N)   # j=0 at top
    z_bottom = z_top[::-1]                                 # j=N-1 at bottom (flipped)

    steps    = T_mat.shape[0]
    thickness = np.full(steps, np.nan)
    center    = np.full(steps, np.nan)
    mix_num   = np.full(steps, np.nan)

    for i in range(steps):
        T_row = T_mat[i]          # j=0 top → j=N-1 bottom
        T_flip = T_row[::-1]      # bottom → top for interpolation

        T_hot  = T_hot_ref
        T_cold = T_cold_ref
        span   = T_hot - T_cold
        if span < 1.0:
            continue

        T_lo = T_cold + lo_frac * span
        T_hi = T_cold + hi_frac * span
        T_mid = T_cold + 0.50 * span

        # Interpolate crossings (T increases with z_bottom when stratified)
        # np.interp requires x increasing; T_flip may not be monotone after mixing
        # Use a simple scan: find first crossing from bottom upward
        def _interp_crossing(T_target):
            for j in range(N - 1):
                t1, t2 = T_flip[j], T_flip[j + 1]
                if (t1 <= T_target <= t2) or (t2 <= T_target <= t1):
                    if abs(t2 - t1) < 1e-9:
                        return z_bottom[j]
                    frac = (T_target - t1) / (t2 - t1)
                    return z_bottom[j] + frac * dz
            return np.nan

        z_lo  = _interp_crossing(T_lo)
        z_hi  = _interp_crossing(T_hi)
        z_cen = _interp_crossing(T_mid)

        if not np.isnan(z_lo) and not np.isnan(z_hi):
            thickness[i] = abs(z_hi - z_lo)
        center[i] = z_cen

        # MIX number: mean absolute deviation from ideal step profile
        # Ideal: T_hot above z_cen (if known), T_cold below
        if not np.isnan(z_cen):
            T_ideal = np.where(z_bottom >= z_cen, T_hot, T_cold)
            mix_num[i] = np.mean(np.abs(T_flip - T_ideal)) / span

    return thickness, center, mix_num


# ── 1. TEMPERATURE vs TIME ────────────────────────────────────────────────────
def plot_temp_vs_time(T_chg, T_stor, T_dis, dt, N, output_dir):
    steps_chg  = T_chg.shape[0]
    steps_stor = T_stor.shape[0]
    steps_dis  = T_dis.shape[0]
    t_chg  = np.arange(steps_chg)  * dt / 3600
    t_stor = np.arange(steps_stor) * dt / 3600
    t_dis  = np.arange(steps_dis)  * dt / 3600

    layers = [0, N//4, N//2, 3*N//4, N-1]
    labels = [f"L{j+1} (top)" if j == 0
              else f"L{j+1} (bot)" if j == N-1
              else f"L{j+1}"
              for j in layers]
    cmap   = plt.cm.get_cmap("plasma", len(layers))
    colors = [cmap(i) for i in range(len(layers))]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle("Layer Temperature Evolution", fontsize=14, fontweight="bold")

    for ax, (T_mat, t_h, title) in zip(
        axes,
        [(T_chg, t_chg, "Charging"), (T_stor, t_stor, "Storage"),
         (T_dis, t_dis, "Discharging")]
    ):
        for idx, (j, lbl) in enumerate(zip(layers, labels)):
            ax.plot(t_h, T_mat[:, j], label=lbl, color=colors[idx], linewidth=2)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Time [h]")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Temperature [°C]")
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=len(layers),
               bbox_to_anchor=(0.5, -0.05), frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "temp_vs_time.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


# ── 2. VERTICAL TEMPERATURE PROFILE ──────────────────────────────────────────
def plot_tank_profile(T_chg, T_stor, T_dis, N, H_tank, output_dir):
    dz       = H_tank / N
    z_centers = np.linspace(H_tank - dz / 2, dz / 2, N)

    snapshots   = {
        "Initial":       T_chg[0],
        "End Charge":    T_chg[-1],
        "End Storage":   T_stor[-1],
        "End Discharge": T_dis[-1],
    }
    snap_colors = ["steelblue", "tomato", "orange", "mediumseagreen"]

    fig, ax = plt.subplots(figsize=(6, 8))
    for (label, T_snap), c in zip(snapshots.items(), snap_colors):
        ax.plot(T_snap, z_centers, "o-", label=label, color=c,
                linewidth=2.5, markersize=6)

    T_min_all = min(T.min() for T in snapshots.values()) - 20
    T_max_all = max(T.max() for T in snapshots.values()) + 20
    ax.axhline(0,      color="gray", linewidth=1.5, linestyle="--", alpha=0.5)
    ax.axhline(H_tank, color="gray", linewidth=1.5, linestyle="--", alpha=0.5)
    ax.set_xlim(T_min_all, T_max_all)
    ax.set_ylim(-0.2, H_tank + 0.2)
    ax.set_xlabel("Temperature [°C]", fontsize=12)
    ax.set_ylabel("Tank Height [m]",  fontsize=12)
    ax.set_title("Vertical Temp Profile in Tank", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, alpha=0.3)
    ax.text(T_min_all + 5, H_tank + 0.05, "TOP",    fontsize=9, color="gray")
    ax.text(T_min_all + 5, -0.15,          "BOTTOM", fontsize=9, color="gray")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tank_geometry_profile.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


# ── 3. HX POWER vs TIME ───────────────────────────────────────────────────────
def plot_hx_power(Q_chg, Q_dis, dt, output_dir):
    P_chg = Q_chg / 1000
    P_dis = np.abs(Q_dis) / 1000

    trapz  = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    E_chg  = trapz(Q_chg,          dx=dt) / 3.6e6
    E_dis  = trapz(np.abs(Q_dis),  dx=dt) / 3.6e6

    t_chg = np.arange(len(P_chg)) * dt / 3600
    t_dis = np.arange(len(P_dis)) * dt / 3600

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle("Heat Exchanger Power", fontsize=14, fontweight="bold")

    ax1.fill_between(t_chg, P_chg, alpha=0.3, color="tomato")
    ax1.plot(t_chg, P_chg, color="tomato", linewidth=2)
    ax1.set_title(f"Charging  |  E = {E_chg:.2f} MWh", fontsize=12)
    ax1.set_xlabel("Time [h]")
    ax1.set_ylabel("Power [kW]")
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(t_dis, P_dis, alpha=0.3, color="steelblue")
    ax2.plot(t_dis, P_dis, color="steelblue", linewidth=2)
    ax2.set_title(f"Discharging  |  E = {E_dis:.2f} MWh", fontsize=12)
    ax2.set_xlabel("Time [h]")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hx_power.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    return E_chg, E_dis


# ── 4. EXERGY vs TIME ─────────────────────────────────────────────────────────
def plot_exergy(Ex_chg, Ex_stor, Ex_dis, dt, output_dir):
    ex_chg_kwh  = Ex_chg  / 3.6e6
    ex_stor_kwh = Ex_stor / 3.6e6
    ex_dis_kwh  = Ex_dis  / 3.6e6

    t_chg_h  = np.arange(len(Ex_chg))  * dt / 3600
    t_stor_h = np.arange(len(Ex_stor)) * dt / 3600 + t_chg_h[-1]
    t_dis_h  = np.arange(len(Ex_dis))  * dt / 3600 + t_stor_h[-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_chg_h,  ex_chg_kwh,  color="tomato",     linewidth=2, label="Charging")
    ax.plot(t_stor_h, ex_stor_kwh, color="orange",      linewidth=2, label="Storage")
    ax.plot(t_dis_h,  ex_dis_kwh,  color="steelblue",   linewidth=2, label="Discharging")
    ax.fill_between(t_chg_h,  ex_chg_kwh,  alpha=0.15, color="tomato")
    ax.fill_between(t_stor_h, ex_stor_kwh, alpha=0.15, color="orange")
    ax.fill_between(t_dis_h,  ex_dis_kwh,  alpha=0.15, color="steelblue")

    ax.annotate(f"Peak: {ex_chg_kwh[-1]:.2f} kWh",
                xy=(t_chg_h[-1], ex_chg_kwh[-1]),
                xytext=(t_chg_h[-1] - 1.5, ex_chg_kwh[-1] + 0.5),
                arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9)
    ax.annotate(f"End: {ex_dis_kwh[-1]:.2f} kWh",
                xy=(t_dis_h[-1], ex_dis_kwh[-1]),
                xytext=(t_dis_h[-1] - 2.0, ex_dis_kwh[-1] + 0.5),
                arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9)

    eta_ex = (Ex_dis[0] - Ex_dis[-1]) / (Ex_chg[-1] - Ex_chg[0]) * 100
    ax.set_title(f"Tank Exergy Over Full Cycle   |   η_ex = {eta_ex:.1f}%",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Exergy stored in tank [kWh]")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exergy_vs_time.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    return eta_ex


# ── 5. THERMOCLINE HEATMAP  (the star plot) ───────────────────────────────────
def plot_thermocline_heatmap(T_chg, T_stor, T_dis, dt, N, H_tank, output_dir):
    """
    Full-cycle 2D pcolormesh: x = time [h], y = tank height [m], color = T [°C].
    Thermocline boundaries (10 %/90 % isotherms) are overlaid as white lines.
    """
    dz = H_tank / N
    # z coordinates at layer centres, bottom = 0
    z_bot = np.linspace(dz / 2, H_tank - dz / 2, N)   # bottom → top

    # ── Build continuous time axis ─────────────────────────────────────────
    t_chg_h  = np.arange(T_chg.shape[0])  * dt / 3600
    t_stor_h = np.arange(T_stor.shape[0]) * dt / 3600 + t_chg_h[-1]
    t_dis_h  = np.arange(T_dis.shape[0])  * dt / 3600 + t_stor_h[-1]

    t_all = np.concatenate([t_chg_h, t_stor_h[1:], t_dis_h[1:]])

    # T_mat shape: (time, N) with j=0 top; flip to bottom-up for plotting
    T_all  = np.vstack([T_chg, T_stor[1:], T_dis[1:]])
    T_plot = T_all[:, ::-1].T       # shape (N, time), row 0 = bottom

    # ── Thermocline boundary tracking (10 % / 90 % isotherms) ─────────────
    T_hot_ref  = T_all.max()
    T_cold_ref = T_all.min()
    span       = T_hot_ref - T_cold_ref
    T_lo_iso   = T_cold_ref + 0.10 * span
    T_hi_iso   = T_cold_ref + 0.90 * span

    def _iso_height(T_row_flippped, T_target):
        """Height [m] where temperature first crosses T_target (bottom→top)."""
        T_f = T_row_flippped
        for j in range(N - 1):
            t1, t2 = T_f[j], T_f[j + 1]
            if (t1 - T_target) * (t2 - T_target) <= 0 and abs(t2 - t1) > 1e-9:
                frac = (T_target - t1) / (t2 - t1)
                return z_bot[j] + frac * dz
        return np.nan

    z_lo_line = np.array([_iso_height(T_plot[:, k], T_lo_iso) for k in range(len(t_all))])
    z_hi_line = np.array([_iso_height(T_plot[:, k], T_hi_iso) for k in range(len(t_all))])

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(15, 6))

    # Temperature heatmap
    pcm = ax.pcolormesh(t_all, z_bot, T_plot,
                        cmap="inferno", shading="gouraud",
                        vmin=T_cold_ref, vmax=T_hot_ref)

    cbar = fig.colorbar(pcm, ax=ax, pad=0.01, fraction=0.025)
    cbar.set_label("Temperature [°C]", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # 10 %/90 % thermocline boundaries
    valid = ~np.isnan(z_lo_line)
    ax.plot(t_all[valid], z_lo_line[valid],
            color="white", lw=1.5, ls="--", label="10 % isotherm")
    valid = ~np.isnan(z_hi_line)
    ax.plot(t_all[valid], z_hi_line[valid],
            color="cyan",  lw=1.5, ls="--", label="90 % isotherm")

    # Shade thermocline zone between the two isotherms
    # (fill_betweenx works in axis coords; do it manually with polygon)
    for k in range(len(t_all) - 1):
        lo1, lo2 = z_lo_line[k], z_lo_line[k + 1]
        hi1, hi2 = z_hi_line[k], z_hi_line[k + 1]
        if not any(np.isnan([lo1, lo2, hi1, hi2])):
            ax.fill(
                [t_all[k], t_all[k+1], t_all[k+1], t_all[k]],
                [lo1,      lo2,        hi2,         hi1],
                color="white", alpha=0.08
            )

    # Phase boundary lines
    t_end_chg  = t_chg_h[-1]
    t_end_stor = t_stor_h[-1]
    ax.axvline(t_end_chg,  color="lime",    lw=1.8, ls=":", alpha=0.9,
               label=f"End charge ({t_end_chg:.1f} h)")
    ax.axvline(t_end_stor, color="yellow",  lw=1.8, ls=":", alpha=0.9,
               label=f"End storage ({t_end_stor:.1f} h)")

    # HX zone marker on y-axis
    ax.axhspan(H_tank / 2, H_tank, color="white", alpha=0.04,
               linewidth=0, label="HX zone (top half)")

    ax.set_xlabel("Time [h]", fontsize=12)
    ax.set_ylabel("Tank height [m]", fontsize=12)
    ax.set_title("Thermocline Evolution — Full Cycle", fontsize=14, fontweight="bold")
    ax.set_xlim(t_all[0], t_all[-1])
    ax.set_ylim(0, H_tank)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.7)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "thermocline_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  thermocline_heatmap.png")


# ── 6. THERMOCLINE METRICS over time ─────────────────────────────────────────
def plot_thermocline_metrics(T_chg, T_stor, T_dis, dt, N, H_tank, output_dir):
    """
    Three metrics plotted continuously across the full cycle:
      • Thermocline thickness  [m]  (10%–90% isotherm span)
      • Thermocline centre     [m]  (50% isotherm height from bottom)
      • MIX number             [–]  (0 = perfect stratification)
    """
    T_hot_ref  = max(T_chg.max(), T_stor.max(), T_dis.max())
    T_cold_ref = min(T_chg.min(), T_stor.min(), T_dis.min())

    def _metrics(T_mat):
        return _thermocline_metrics(T_mat, H_tank, N, T_hot_ref, T_cold_ref)

    thk_c, cen_c, mix_c = _metrics(T_chg)
    thk_s, cen_s, mix_s = _metrics(T_stor)
    thk_d, cen_d, mix_d = _metrics(T_dis)

    t_chg_h  = np.arange(T_chg.shape[0])  * dt / 3600
    t_stor_h = np.arange(T_stor.shape[0]) * dt / 3600 + t_chg_h[-1]
    t_dis_h  = np.arange(T_dis.shape[0])  * dt / 3600 + t_stor_h[-1]

    t_all = np.concatenate([t_chg_h, t_stor_h[1:], t_dis_h[1:]])
    thk   = np.concatenate([thk_c,   thk_s[1:],    thk_d[1:]])
    cen   = np.concatenate([cen_c,   cen_s[1:],    cen_d[1:]])
    mix   = np.concatenate([mix_c,   mix_s[1:],    mix_d[1:]])

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Thermocline Quality Metrics — Full Cycle",
                 fontsize=14, fontweight="bold")

    # ── Thickness ─────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(t_all, thk, color="#e05c00", linewidth=2)
    ax.fill_between(t_all, thk, alpha=0.2, color="#e05c00")
    ax.set_ylabel("Thickness [m]", fontsize=11)
    ax.set_title("Thermocline Thickness  (10%–90% isotherm span)",
                 fontsize=10, style="italic")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # ── Centre height ──────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(t_all, cen, color="#0077bb", linewidth=2)
    ax.axhline(H_tank / 2, color="gray", ls="--", lw=1, alpha=0.6,
               label="Tank mid-point")
    ax.set_ylabel("Centre height [m]", fontsize=11)
    ax.set_title("50 % Isotherm Height  (thermocline centre)",
                 fontsize=10, style="italic")
    ax.set_ylim(0, H_tank)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── MIX number ─────────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(t_all, mix, color="#009944", linewidth=2)
    ax.fill_between(t_all, mix, alpha=0.2, color="#009944")
    ax.set_ylabel("MIX number [–]", fontsize=11)
    ax.set_title("MIX Number  (0 = perfect stratification, 1 = fully mixed)",
                 fontsize=10, style="italic")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time [h]", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Phase boundaries on all axes
    for ax in axes:
        ax.axvline(t_chg_h[-1],  color="lime",   lw=1.5, ls=":", alpha=0.8)
        ax.axvline(t_stor_h[-1], color="yellow", lw=1.5, ls=":", alpha=0.8)

    # Phase labels on top panel
    axes[0].text(t_chg_h[-1]  * 0.5, axes[0].get_ylim()[1] * 0.9,
                 "Charging",    ha="center", fontsize=9, color="tomato")
    axes[0].text((t_chg_h[-1] + t_stor_h[-1]) * 0.5, axes[0].get_ylim()[1] * 0.9,
                 "Storage",     ha="center", fontsize=9, color="orange")
    axes[0].text((t_stor_h[-1] + t_dis_h[-1]) * 0.5, axes[0].get_ylim()[1] * 0.9,
                 "Discharging", ha="center", fontsize=9, color="steelblue")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "thermocline_metrics.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  thermocline_metrics.png")


# ── 7. MASS FLOW RATE vs TIME ─────────────────────────────────────────────────
def plot_mass_flow(m_dot_chg, m_dot_dis, dt, P_target_chg, P_target_dis,
                   Q_chg, Q_dis, output_dir):
    """
    Two-panel: mass flow and actual delivered power vs time for each active phase.
    Shows how m_dot is modulated to track P_target.
    """
    t_chg = np.arange(len(m_dot_chg)) * dt / 3600
    t_dis = np.arange(len(m_dot_dis)) * dt / 3600

    P_chg_kW = Q_chg / 1e3
    P_dis_kW = np.abs(Q_dis) / 1e3
    P_tgt_c  = P_target_chg / 1e3
    P_tgt_d  = P_target_dis / 1e3

    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex="col")
    fig.suptitle("Dynamic Mass-Flow Control", fontsize=14, fontweight="bold")

    # ── Charging ──────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(t_chg, m_dot_chg, color="tomato", linewidth=2)
    ax.fill_between(t_chg, m_dot_chg, alpha=0.15, color="tomato")
    ax.set_ylabel("ṁ  [kg/s]", fontsize=11)
    ax.set_title("Charging — mass flow rate", fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t_chg, P_chg_kW, color="tomato", linewidth=2, label="Actual")
    ax.axhline(P_tgt_c, color="black", ls="--", lw=1.5, label=f"Target {P_tgt_c:.0f} kW")
    ax.fill_between(t_chg, P_chg_kW, alpha=0.15, color="tomato")
    ax.set_ylabel("Power [kW]", fontsize=11)
    ax.set_xlabel("Time [h]", fontsize=11)
    ax.set_title("Charging — HX power", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Discharging ────────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(t_dis, m_dot_dis, color="steelblue", linewidth=2)
    ax.fill_between(t_dis, m_dot_dis, alpha=0.15, color="steelblue")
    ax.set_title("Discharging — mass flow rate", fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t_dis, P_dis_kW, color="steelblue", linewidth=2, label="Actual")
    ax.axhline(P_tgt_d, color="black", ls="--", lw=1.5, label=f"Target {P_tgt_d:.0f} kW")
    ax.fill_between(t_dis, P_dis_kW, alpha=0.15, color="steelblue")
    ax.set_xlabel("Time [h]", fontsize=11)
    ax.set_title("Discharging — HX power", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mass_flow_control.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  mass_flow_control.png")


# ── MASTER CALL ───────────────────────────────────────────────────────────────
def plot_all(T_chg, T_stor, T_dis, Q_chg, Q_dis, dt, N, H_tank, V_tank,
             Ex_chg=None, Ex_stor=None, Ex_dis=None,
             m_dot_chg=None, m_dot_dis=None,
             P_target_chg=None, P_target_dis=None,
             output_dir="results"):

    os.makedirs(output_dir, exist_ok=True)

    # Legacy plots
    plot_temp_vs_time(T_chg, T_stor, T_dis, dt, N, output_dir)
    plot_tank_profile(T_chg, T_stor, T_dis, N, H_tank, output_dir)
    E_chg, E_dis = plot_hx_power(Q_chg, Q_dis, dt, output_dir)

    print(f"[plotting] Saved to '{output_dir}/':")
    print(f"  temp_vs_time.png | tank_geometry_profile.png | hx_power.png")
    print(f"  E_chg={E_chg:.3f} MWh  |  E_dis={E_dis:.3f} MWh  |  "
          f"eta={E_dis/E_chg*100:.1f}%")

    if Ex_chg is not None:
        eta_ex = plot_exergy(Ex_chg, Ex_stor, Ex_dis, dt, output_dir)
        print(f"  exergy_vs_time.png | eta_ex={eta_ex:.1f}%")

    # ── New: thermocline plots ─────────────────────────────────────────────
    plot_thermocline_heatmap(T_chg, T_stor, T_dis, dt, N, H_tank, output_dir)
    plot_thermocline_metrics(T_chg, T_stor, T_dis, dt, N, H_tank, output_dir)

    # ── New: dynamic mass-flow plots ──────────────────────────────────────
    if m_dot_chg is not None and m_dot_dis is not None:
        plot_mass_flow(m_dot_chg, m_dot_dis, dt,
                       P_target_chg or 0.0, P_target_dis or 0.0,
                       Q_chg, Q_dis, output_dir)