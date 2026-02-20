import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

def main():
    # ─── Physical constants & ambient conditions ─────────────────────────────────
    P_AMB     = 100900.0 # Ambient pressure [Pa]
    T_AMB     = 298.15   # Ambient temperature [K]
    RHO_AIR   = 1.18     # Air density [kg/m^3]
    RHO_WATER = 998.0    # Water density [kg/m^3]
    G         = 9.81     # Gravitational acceleration [m/s^2]

    # ─── File paths ──────────────────────────────────────────────────────────────
    INPUT_CSV  = "./data/pressure_vs_theta.csv"
    OUTPUT_CSV = "./outputs/text/pressure_vs_theta.csv"
    OUTPUT_FILE = "./outputs/text/summary.txt"

    # ─── Ensure figures output directory exists ──────────────────────────────────
    FIG_DIR = os.path.join("outputs", "figures")
    os.makedirs(FIG_DIR, exist_ok=True)

    # ─── Read CSV ────────────────────────────────────────────────────────────────
    theta_deg_list = []
    P_inf_raw_list = []
    P_0_raw_list   = []
    P_raw_list     = []

    with open(INPUT_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            theta_deg_list.append(float(row["theta_deg"]))
            P_inf_raw_list.append(float(row["P_inf"]))
            P_0_raw_list.append(float(row["P_0"]))
            P_raw_list.append(float(row["P"]))

    N = len(theta_deg_list)
    print(f"Read {N} data points from '{INPUT_CSV}'.")

    # ─── Convert water column heights to differential pressures ──────────────────
    # The manometer readings are heights (in cm of water):
    #     ΔP = ρ_water · g · Δh
    #     P_actual = P_ref - ρ_water · g · h_reading
    #     P_surface - P_freestream  =  ρ_water · g · (h_surface - h_inf)
    #     q_inf = P_0 - P_inf = ρ_water · g · (h_0 - h_inf)
    SCALE = 1e-2  # Measurements taken in cm (convert to meters)

    # ─── Compute differential pressures ─────────────────────────────────────────
    # For each data point we have a corresponding P_inf and P_0 reading,
    # so we compute per-row to account for any drift.
    delta_P_list = [] # P_surface - P_freestream  [Pa]
    q_inf_list   = [] # dynamic pressure  [Pa]
    U_inf_list   = [] # freestream velocity [m/s]
    Cp_list      = [] # pressure coefficient

    for i in range(N):
        h_inf = P_inf_raw_list[i] * SCALE # freestream static tap height [m]
        h_0   = P_0_raw_list[i]   * SCALE # stagnation tap height [m]
        h_p   = P_raw_list[i]     * SCALE # surface tap height [m]

        # (P_surface - P_inf) in Pa
        # Higher reading = higher pressure, so:
        delta_P = RHO_WATER * G * (h_p - h_inf)

        # Dynamic pressure  q = P_total - P_static = rho_w * g * (h_0 - h_inf)
        q_inf = RHO_WATER * G * (h_0 - h_inf)

        if q_inf <= 0:
            print(f"WARNING row {i}: q_inf = {q_inf:.2f} Pa <= 0  (h_inf={h_inf:.4f}, h_0={h_0:.4f})")
            # Use absolute value as fallback but flag it
            q_inf = abs(q_inf) if abs(q_inf) > 1e-6 else 1e-6

        U_inf = (2.0 * q_inf / RHO_AIR) ** 0.5 # Bernoulli: q = 0.5 * rho * U^2

        Cp = delta_P / q_inf # Cp = (P - P_inf) / q_inf  = (P - P_inf) / (0.5 * rho * U^2)

        delta_P_list.append(delta_P)
        q_inf_list.append(q_inf)
        U_inf_list.append(U_inf)
        Cp_list.append(Cp)

    # ─── Convert lists to numpy arrays for convenience ──────────────────────────
    theta_deg = np.array(theta_deg_list)
    theta_rad = np.deg2rad(theta_deg)
    Cp_exp    = np.array(Cp_list)
    dP_exp    = np.array(delta_P_list)
    q_inf_arr = np.array(q_inf_list)

    # ─── Inviscid (potential flow) theory for a cylinder ─────────────────────────
    # Cp_inviscid = 1 - 4 sin^2(theta)
    theta_theory = np.linspace(0, 180, 500)
    theta_theory_rad = np.deg2rad(theta_theory)
    Cp_inviscid = 1.0 - 4.0 * np.sin(theta_theory_rad) ** 2

    # ─── Figure 1: Cp vs theta ──────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(theta_deg, Cp_exp, 'o', markersize=5, label='Experimental')
    ax1.plot(theta_theory, Cp_inviscid, '-', linewidth=1.5, label='Inviscid theory ($1 - 4\\sin^2\\theta$)')
    ax1.set_xlabel(r'$\theta$ [deg]')
    ax1.set_ylabel(r'$C_p$')
    ax1.set_title(r'Pressure Coefficient $C_p$ vs Angular Position $\theta$')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(os.path.join(FIG_DIR, "Cp_vs_theta.png"), dpi=300)
    print(f"Saved {os.path.join(FIG_DIR, 'Cp_vs_theta.png')}")

    # ─── Figure 2: Cd vs theta (cumulative drag coefficient) ────────────────────
    #
    # The drag force on the cylinder (per unit span) is:
    #     D = R ∫₀²π P·cos(θ) dθ
    #
    # We define the drag coefficient as:
    #     C_d = D / (q_inf · d)  =  D / (q_inf · 2R)
    #
    # Substituting and non-dimensionalising with Cp = (P - P_inf) / q_inf :
    #     C_d = (1/2) ∫₀²π Cp · cos(θ) dθ
    #
    # (The P_inf contribution integrates to zero over a closed surface.)
    #
    # We approximate the integral cumulatively using the trapezoidal rule so we
    # can plot C_d as a function of the upper integration limit θ.

    # Sort by angle to ensure proper integration order
    sort_idx  = np.argsort(theta_deg)
    theta_sorted     = theta_deg[sort_idx]
    theta_sorted_rad = theta_rad[sort_idx]
    Cp_sorted        = Cp_exp[sort_idx]

    # Integrand: Cp(θ) · cos(θ)
    integrand_exp = Cp_sorted * np.cos(theta_sorted_rad)

    # Cumulative trapezoidal integration: (1/2) ∫₀^θ Cp·cos(θ') dθ'
    Cd_cumulative_exp = np.zeros(len(theta_sorted))
    for j in range(1, len(theta_sorted)):
        dtheta = theta_sorted_rad[j] - theta_sorted_rad[j - 1]
        Cd_cumulative_exp[j] = Cd_cumulative_exp[j - 1] + 0.5 * (integrand_exp[j] + integrand_exp[j - 1]) * dtheta
    Cd_cumulative_exp *= 0.5  # the 1/2 prefactor from C_d = (1/2) ∫ Cp cos(θ) dθ

    # Inviscid theory: Cp_inv = 1 - 4sin²θ  →  Cd = 0 (d'Alembert's paradox)
    # Cumulative for plotting:
    integrand_inv = (1.0 - 4.0 * np.sin(theta_theory_rad) ** 2) * np.cos(theta_theory_rad)
    Cd_cumulative_inv = np.zeros(len(theta_theory))
    for j in range(1, len(theta_theory)):
        dtheta = theta_theory_rad[j] - theta_theory_rad[j - 1]
        Cd_cumulative_inv[j] = Cd_cumulative_inv[j - 1] + 0.5 * (integrand_inv[j] + integrand_inv[j - 1]) * dtheta
    Cd_cumulative_inv *= 0.5

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.plot(theta_sorted, Cd_cumulative_exp, 'o-', markersize=4, linewidth=1, label='Experimental (trapezoidal)')
    ax2.plot(theta_theory, Cd_cumulative_inv, '-', linewidth=1.5, label="Inviscid theory (d'Alembert: $C_d = 0$)")
    ax2.set_xlabel(r'$\theta$ [deg]')
    ax2.set_ylabel(r'$C_d(\theta)$  (cumulative)')
    ax2.set_title(r'Cumulative Drag Coefficient $C_d$ vs Angular Position $\theta$')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(FIG_DIR, "Cd_vs_theta.png"), dpi=300)
    print(f"Saved {os.path.join(FIG_DIR, 'Cd_vs_theta.png')}")

    # Print final Cd value (full integral over 0 to max angle)
    print(f"\nExperimental C_d (integrated 0 to {theta_sorted[-1]:.0f} deg): {Cd_cumulative_exp[-1]:.4f}")
    print(f"Inviscid theory C_d (integrated 0 to 360 deg):       {Cd_cumulative_inv[-1]:.6f}  (≈ 0, d'Alembert's paradox)")

    # ─── Summary statistics ─────────────────────────────────────────────────────
    q_avg   = sum(q_inf_list) / N
    U_avg   = sum(U_inf_list) / N

    with open(OUTPUT_FILE, "w", newline="") as f:
        f.write(f"Ambient conditions:")
        f.write(f"  P_amb   = {P_AMB} Pa")
        f.write(f"  T_amb   = {T_AMB} K")
        f.write(f"  rho_air = {RHO_AIR} kg/m^3")
        f.write(f"\nDerived quantities (averages over all rows):")
        f.write(f"  q_inf   = {q_avg:.2f} Pa")
        f.write(f"  U_inf   = {U_avg:.2f} m/s")

    print(f"\nSummary written to '{OUTPUT_FILE}'.")

    # ─── Write output table ─────────────────────────────────────────────────────
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["theta_deg", "P-P_inf_Pa", "Cp", "P_Pa"])

        for i in range(N):
            # P_actual = P_amb + (P_surface - P_inf) i.e. ambient + differential
            P_actual = P_AMB + delta_P_list[i]
            writer.writerow([
                f"{theta_deg_list[i]:.1f}",
                f"{delta_P_list[i]:.3f}",
                f"{Cp_list[i]:.4f}",
                f"{P_actual:.3f}",
            ])

    print(f"\nResults written to '{OUTPUT_CSV}'.")


if __name__ == "__main__":
    main()
