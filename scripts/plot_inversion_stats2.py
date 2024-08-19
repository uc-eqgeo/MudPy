# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 13:55:43 2024

@author: jmc753
"""

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

n_ruptures = 137

b, N = 1.1, 21.5
a = np.log10(N) + (b * 5)
min_Mw, max_Mw = 6.0, 9.5

bin_Mw = np.arange(min_Mw, max_Mw, 0.1)
Mw = np.array([min_Mw, max_Mw])

plot_results = True

inputRuptures = pd.read_csv(f"Z:\\McGrath\\HikurangiFakeQuakes\\hikkerk3D\\output\\ruptures\\input_n{n_ruptures}.txt", sep='\t', index_col=0)
try:
    results_file = f"Z:\\McGrath\\HikurangiFakeQuakes\\hikkerk3D\\output\\ruptures\\preferred_rate_n{n_ruptures}.txt"
    results = pd.read_csv(results_file, sep='\t', index_col=0)
except FileNotFoundError:
    plot_results = False

sort_ix = np.argsort(inputRuptures['Mw'])

initial_rate = []
for mw in inputRuptures['Mw']:
    cumfreq = np.sum(inputRuptures['initial_rate'][np.where(inputRuptures['Mw'] >= mw)[0]])
    if cumfreq > 0:
        initial_rate.append(np.log10(cumfreq))  # Calculate log(N) for each magnitude
    else:
        initial_rate.append(np.nan)
initial_rate = np.array(initial_rate)

noisy_rate = []
for mw in inputRuptures['Mw']:
    cumfreq = np.sum(inputRuptures['noisy_rate'][np.where(inputRuptures['Mw'] >= mw)[0]])
    if cumfreq > 0:
        noisy_rate.append(np.log10(cumfreq))  # Calculate log(N) for each magnitude
    else:
        noisy_rate.append(np.nan)
noisy_rate = np.array(noisy_rate)

if plot_results:
    inv_rate = []
    for mw in results['Mw']:
        cumfreq = np.sum(results['rate'][np.where(results['Mw'] >= mw)[0]])
        if cumfreq > 0:
            inv_rate.append(np.log10(cumfreq))  # Calculate log(N) for each magnitude
        else:
            inv_rate.append(np.nan)
    inv_rate = np.array(inv_rate)
# %%
#plt.plot(inputRuptures['Mw'][sort_ix], (inputRuptures['target'][sort_ix]), color='red', label='Target GR Relation')
plt.plot(inputRuptures['Mw'][sort_ix], (inputRuptures['lower'][sort_ix] + 1e-9), color='green', linestyle='-.', label='Upper Bound')
plt.plot(inputRuptures['Mw'][sort_ix], (inputRuptures['upper'][sort_ix]), color='green', linestyle='-.')
plt.ylabel('log10(N)')
plt.xlim([min_Mw, max_Mw])
plt.ylim([-1, 35])
#sns.scatterplot(x=inputRuptures['Mw'], y=initial_rate, sizes=10, label='Actual Inital')
sns.scatterplot(x=inputRuptures['Mw'], y=inputRuptures['target'], sizes=10, label='Target')
sns.scatterplot(x=inputRuptures['Mw'], y=inputRuptures['noisy_rate'], sizes=10, label='Actual Noisy Inital')
if plot_results:
    sns.scatterplot(x=results['Mw'], y=results['rate'], sizes=5, label='Actual Inverted')
plt.legend(loc='upper right')
plt.show()

print('Initial GR RMS', np.sqrt(np.mean((np.log10(inputRuptures['target'][sort_ix] - noisy_rate[sort_ix])) ** 2)))
print('Inverted GR RMS', np.sqrt(np.mean((np.log10(inputRuptures['target'][sort_ix] - inv_rate[sort_ix])) ** 2)))
# %%
if plot_results:
    sns.scatterplot(x=inputRuptures['noisy_rate'], y=results['rate'], hue=results['Mw'])
    plt.plot([1e-6, 30], [1e-6, 30])
    plt.xlabel('Initial Noisy Rate')
    plt.ylabel('Inverted Rate')
    #plt.xscale('log')
    #plt.yscale('log')
    plt.show()

    # plt.scatter(results['Mw'], inputRuptures['noisy_rate'] / results['rate'])
    # plt.xlabel('Mw')
    # plt.ylabel('Change Ratio')
    # plt.yscale('log')
    # plt.show()