import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

def drug_target_binding(state, t, kon, koff, drug_conc, target_conc):
    """
    ODE system for drug-target binding
    state[0] = [DT] (drug-target complex concentration)
    """
    DT = state[0]
    D = drug_conc - DT  # Free drug concentration
    T = target_conc - DT  # Free target concentration
    
    # d[DT]/dt = kon*[D]*[T] - koff*[DT]
    dDT = kon * D * T - koff * DT
    
    return [dDT]

def simulate_binding(kon, koff, drug_conc, target_conc, t_span):
    """
    Simulate drug-target binding over time
    """
    # Initial conditions: [DT] = 0
    DT0 = 0
    
    # Solve ODE
    solution = odeint(drug_target_binding, [DT0], t_span, 
                     args=(kon, koff, drug_conc, target_conc))
    
    return solution[:, 0]

def plot_binding_analysis(kon, koff, target_conc, ic50, half_ic50):
    """
    Create comprehensive plot of binding analysis for MEK inhibitor
    """
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Time course plot for both concentrations
    ax1 = fig.add_subplot(gs[0, 0])
    t_span = np.linspace(0, 5000, 5000)
    
    # Simulate binding for IC50
    binding_ic50 = simulate_binding(kon, koff, ic50, target_conc, t_span)
    ax1.plot(t_span, binding_ic50, label=f'IC50 ({ic50:.2e} M)')
    
    # Simulate binding for half IC50
    binding_half_ic50 = simulate_binding(kon, koff, half_ic50, target_conc, t_span)
    ax1.plot(t_span, binding_half_ic50, label=f'Half IC50 ({half_ic50:.2e} M)')
    
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('[DT] (M)')
    ax1.set_title('Time Course of MEK Inhibitor Binding')
    ax1.grid(True)
    ax1.legend()
    
    # 2. Dose-response curve
    ax2 = fig.add_subplot(gs[0, 1])
    drug_concs = np.logspace(-8, -4, 100)  # Adjusted range for MEK inhibitor
    max_binding = []
    
    for conc in drug_concs:
        binding = simulate_binding(kon, koff, conc, target_conc, t_span)
        max_binding.append(binding[-1])
    
    ax2.semilogx(drug_concs, max_binding)
    ax2.axvline(x=ic50, color='r', linestyle='--', 
                label=f'IC50: {ic50:.2e} M')
    ax2.axvline(x=half_ic50, color='g', linestyle='--', 
                label=f'Half IC50: {half_ic50:.2e} M')
    ax2.set_xlabel('Drug Concentration (M)')
    ax2.set_ylabel('Equilibrium [DT] (M)')
    ax2.set_title('Dose-Response Curve')
    ax2.grid(True)
    ax2.legend()
    
    # 3. Free drug and target concentrations
    ax3 = fig.add_subplot(gs[1, 0])
    DT_ic50 = binding_ic50[-1]
    DT_half_ic50 = binding_half_ic50[-1]
    
    # Calculate free concentrations at equilibrium
    free_drug_ic50 = ic50 - DT_ic50
    free_drug_half_ic50 = half_ic50 - DT_half_ic50
    free_target_ic50 = target_conc - DT_ic50
    free_target_half_ic50 = target_conc - DT_half_ic50
    
    # Plot free concentrations
    concentrations = [free_drug_ic50, free_drug_half_ic50, 
                     free_target_ic50, free_target_half_ic50]
    labels = ['Free Drug (IC50)', 'Free Drug (Half IC50)',
              'Free Target (IC50)', 'Free Target (Half IC50)']
    
    ax3.bar(range(len(concentrations)), concentrations)
    ax3.set_xticks(range(len(concentrations)))
    ax3.set_xticklabels(labels, rotation=45)
    ax3.set_ylabel('Concentration (M)')
    ax3.set_title('Free Concentrations at Equilibrium')
    ax3.grid(True)
    
    # 4. Binding efficiency comparison
    ax4 = fig.add_subplot(gs[1, 1])
    binding_efficiency_ic50 = DT_ic50 / ic50
    binding_efficiency_half_ic50 = DT_half_ic50 / half_ic50
    
    efficiencies = [binding_efficiency_ic50, binding_efficiency_half_ic50]
    labels = ['IC50', 'Half IC50']
    
    ax4.bar(range(len(efficiencies)), efficiencies)
    ax4.set_xticks(range(len(efficiencies)))
    ax4.set_xticklabels(labels)
    ax4.set_ylabel('Binding Efficiency ([DT]/[D])')
    ax4.set_title('Binding Efficiency Comparison')
    ax4.grid(True)
    
    plt.tight_layout()
    return fig

def main():
    # MEK inhibitor parameters
    ic50 = 31e-06  # IC50 in mM
    half_ic50 = ic50 / 2
    
    # Binding parameters (can be adjusted)
    kon = 1    # Association rate (M⁻¹s⁻¹)
    koff = 1   # Dissociation rate (s⁻¹)
    target_conc = 1  # Target concentration (M)
    
    # Create analysis plots
    fig = plot_binding_analysis(kon, koff, target_conc, ic50, half_ic50)
    plt.savefig('mek_inhibitor_binding_analysis.png')
    print(f"plot saved to file: {os.getcwd()}/mek_inhibitor_binding_analysis.png")
    
    # Calculate and print key metrics
    Kd = koff/kon
    print(f"Kd = {Kd:.2e} M")
    print(f"IC50 = {ic50:.2e} M")
    print(f"Half IC50 = {half_ic50:.2e} M")
    
    # Simulate equilibrium binding for both concentrations
    t_span = np.linspace(0, 10000, 10000)
    binding_ic50 = simulate_binding(kon, koff, ic50, target_conc, t_span)
    binding_half_ic50 = simulate_binding(kon, koff, half_ic50, target_conc, t_span)
    
    print(f"\nEquilibrium binding at IC50: {binding_ic50[-1]:.2e} M")
    print(f"Equilibrium binding at half IC50: {binding_half_ic50[-1]:.2e} M")
    print(f"Ratio of binding (IC50/half IC50): {binding_ic50[-1]/binding_half_ic50[-1]:.2f}")

if __name__ == "__main__":
    main() 