import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
kon = 1.0   # M^-1 s^-1
koff = 0.1  # s^-1
D_total = 1.0  # Initial drug concentration (uM)
T_total = 1.0  # Initial target concentration (uM)
DT0 = 0.0      # Initial complex concentration

t_span = (0, 20)
t_eval = np.linspace(*t_span, 200)

# --- 1. Full ODE system ---
def reversible_binding_odes(t, y):
    D, T, DT = y
    dD = koff*DT - kon*D*T
    dT = koff*DT - kon*D*T
    dDT = kon*D*T - koff*DT
    return [dD, dT, dDT]

y0 = [D_total, T_total, DT0]
sol_full = solve_ivp(reversible_binding_odes, t_span, y0, t_eval=t_eval)

# --- 2. Only DT ODE, mass conservation for D and T ---
def dt_only_ode(t, y):
    DT = y[0]
    D = D_total - DT
    T = T_total - DT
    dDT = kon*D*T - koff*DT
    return [dDT]

sol_dt = solve_ivp(dt_only_ode, t_span, [DT0], t_eval=t_eval)
DT = sol_dt.y[0]
D = D_total - DT
T = T_total - DT

# --- Plotting ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# Full ODE system
axes[0].plot(sol_full.t, sol_full.y[0], label='[D] (full ODE)')
axes[0].plot(sol_full.t, sol_full.y[1], label='[T] (full ODE)')
axes[0].plot(sol_full.t, sol_full.y[2], label='[DT] (full ODE)')
axes[0].set_title('Explicit ODEs')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Concentration')
axes[0].legend()

# DT-only ODE
axes[1].plot(sol_dt.t, D, label='[D] (DT ODE)')
axes[1].plot(sol_dt.t, T, label='[T] (DT ODE)')
axes[1].plot(sol_dt.t, DT, label='[DT] (DT ODE)')
axes[1].set_title('DT ODE + Mass Conservation')
axes[1].set_xlabel('Time')
axes[1].legend()

plt.tight_layout()
plt.show()