# This script plots the growth and apoptosis Hill curves for the growth and apoptosis response curves
# As arguments: takes min, max, half-max, Hill coefficient

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def plot_Hill_curve(min, max, half_max, Hill_coefficient):
    x = np.linspace(min, max, 100)
    y = 1 / (1 + (x / half_max)**Hill_coefficient)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=f'Hill coefficient = {Hill_coefficient}')
    plt.xlabel('Drug concentration')
    plt.ylabel('Growth/Apoptosis')
    plt.title(f'Hill curve for {half_max} half-max, {Hill_coefficient} Hill coefficient')
    plt.legend()
    plt.show()

    # save to results folder
    results_path = "./results/"
    plt.savefig(results_path + f'Hill_curve_halfmax_{half_max}_Hill_coefficient_{Hill_coefficient}.png', dpi=500)
    print(f'Hill curve saved to {results_path + f'Hill_curve_halfmax_{half_max}_Hill_coefficient_{Hill_coefficient}.png'}')

# For the growth TF:
min = 0
max = 1
half_max = 0.01
Hill_coefficient = 9.0

plot_Hill_curve(min, max, half_max, Hill_coefficient)