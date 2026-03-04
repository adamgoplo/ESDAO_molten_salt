"""
plotting.py  –  Molten Salt Stratified Tank – all plots
Usage: from plotting import plot_all
       plot_all(T_chg, T_stor, T_dis, Q_chg, Q_dis, dt, N, H_tank, V_tank, output_dir="results")
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def plot_all(T_chg, T_stor, T_dis, Q_chg, Q_dis, dt, N, H_tank, V_tank,
             output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    steps_chg = T_chg.shape[0]
    steps_stor = T_stor.shape[0]
    steps_dis  = T_dis.shape[0]

    t_chg  = np.arange(steps_chg)  * dt / 3600
    t_stor = np.arange(steps_stor) * dt / 3600
    t_dis  = np.arange(steps_dis)  * dt / 3600

    dz = H_tank / N
    z_centers = np.linspace(H_tank - dz / 2, dz / 2, N)   # top → bottom

    layers = [0, N//4, N//2, 3*N//4, N-1]
    labels = [f"L{j+1} (top)" if j == 0
              else f"L{j+1} (bot)" if j == N-1
              else f"L{j+1}"
              for j in layers]

    cmap   = plt.cm.get_cmap("plasma", len(layers))
    colors = [cmap(i) for i in range(len(layers))]

    # ── 1. TEMPERATURE vs TIME  (3 subplots) ─────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle("Layer Temperature Evolution", fontsize=14, fontweight="bold")

    for ax, (T_mat, t_h, title) in zip(
        axes,
        [(T_chg, t_chg, "Charging"), (T_stor, t_stor, "Storage"), (T_dis, t_dis, "Discharging")]
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
    plt.savefig(os.path.join(output_dir, "temp_vs_time.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── 2. TANK GEOMETRY – vertical temperature profile ───────────────────────
    snapshots = {
        "Initial":       T_chg[0],
        "End Charge":    T_chg[-1],
        "End Storage":   T_stor[-1],
        "End Discharge": T_dis[-1],
    }
    snap_colors = ["steelblue", "tomato", "orange", "mediumseagreen"]

    fig, ax = plt.subplots(figsize=(6, 8))
    for (label, T_snap), c in zip(snapshots.items(), snap_colors):
        ax.plot(T_snap, z_centers, "o-", label=label, color=c, linewidth=2.5,
                markersize=6)

    # Draw tank outline
    T_min_all = min(T.min() for T in snapshots.values()) - 20
    T_max_all = max(T.max() for T in snapshots.values()) + 20
    ax.axhline(0,        color="gray", linewidth=1.5, linestyle="--", alpha=0.5)
    ax.axhline(H_tank,   color="gray", linewidth=1.5, linestyle="--", alpha=0.5)
    ax.set_xlim(T_min_all, T_max_all)
    ax.set_ylim(-0.2, H_tank + 0.2)
    ax.set_xlabel("Temperature [°C]", fontsize=12)
    ax.set_ylabel("Tank Height [m]",  fontsize=12)
    ax.set_title("Vertical Temp Profile in Tank", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, alpha=0.3)
    # Annotate top/bottom
    ax.text(T_min_all + 5, H_tank + 0.05, "TOP", fontsize=9, color="gray")
    ax.text(T_min_all + 5, -0.15,          "BOTTOM", fontsize=9, color="gray")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tank_geometry_profile.png"), dpi=150,
                bbox_inches="tight")
    plt.close()

    # ── 3. HX POWER vs TIME ───────────────────────────────────────────────────
    P_chg = Q_chg / 1000        # W → kW
    P_dis = np.abs(Q_dis) / 1000

    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    E_chg = trapz(Q_chg, dx=dt) / 3.6e6   # MWh
    E_dis = trapz(np.abs(Q_dis), dx=dt) / 3.6e6

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
    plt.savefig(os.path.join(output_dir, "hx_power.png"), dpi=150,
                bbox_inches="tight")
    plt.close()

    print(f"[plotting] Saved to '{output_dir}/':")
    print(f"  temp_vs_time.png | tank_geometry_profile.png | hx_power.png")
    print(f"  E_chg={E_chg:.3f} MWh  |  E_dis={E_dis:.3f} MWh  |  eta={E_dis/E_chg*100:.1f}%")