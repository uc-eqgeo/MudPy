# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 13:55:43 2024

@author: jmc753
"""

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

n_ruptures = 36000
min_Mw, max_Mw = 6.0, 9.5

outdir = "Z:\\McGrath\\HikurangiFakeQuakes\\hikkerk3D\\output"

plot_results = True

try:
    ruptures = pd.read_csv(f"{outdir}\\n{n_ruptures}_inverted_ruptures.txt", sep='\t', index_col=0).sort_values('Mw')
    bins = pd.read_csv(f"{outdir}\\n{n_ruptures}_inverted_bins.txt", sep='\t', index_col=0)
except FileNotFoundError:
    plot_results = False
    ruptures = pd.read_csv(f"{outdir}\\n{n_ruptures}_input_ruptures.txt", sep='\t', index_col=0).sort_values('Mw')
    bins = pd.read_csv(f"{outdir}\\n{n_ruptures}_input_bins.txt", sep='\t', index_col=0)

gr_matrix = np.zeros((bins.shape[0], ruptures.shape[0])).astype('bool')
for ix, mag in enumerate(bins['Mw_bin']):
    gr_matrix[ix, :] = (np.round(ruptures['Mw'], 1) >= mag)
gr_matrix = gr_matrix.astype('int')

rupture_matrix = np.zeros((ruptures.shape[0], ruptures.shape[0])).astype('bool')
for ix, mag in enumerate(ruptures['Mw']):
    rupture_matrix[ix, :] = (ruptures['Mw'] >= mag)
rupture_matrix = rupture_matrix.astype('int')

initial_rate = np.matmul(rupture_matrix, ruptures['initial_rate'])
initial_bins = np.matmul(gr_matrix, ruptures['initial_rate'])
if plot_results:
    inverted_rate = np.matmul(rupture_matrix, ruptures['inverted_rate'])
    inverted_bins = np.matmul(gr_matrix, ruptures['inverted_rate'])
# %%
plt.plot(ruptures['Mw'], ruptures['target_rate'].apply(lambda x: np.log10(x)), color='red', label='Target GR Relation')
plt.plot(ruptures['Mw'], np.log10(ruptures['lower'] + 1e-12), color='green', linestyle=':')
plt.plot(ruptures['Mw'], np.log10(ruptures['upper']), color='green', linestyle=':')
plt.plot(bins['Mw_bin'], np.log10(bins['upper']), color='green', linestyle='-.', label='Upper Limit')  # Binned upper bound
# sns.scatterplot(x=ruptures['Mw'], y=np.log10(initial_rate + 1e-10), s=20, label='Initial', edgecolors=None)
# if plot_results:
#     sns.scatterplot(x=ruptures['Mw'], y=np.log10(inverted_rate + 1e-10), s=10, label='Inverted', edgecolors=None)
sns.scatterplot(x=bins['Mw_bin'], y=np.log10(initial_bins + 1e-10), s=20, label='Initial Bins', edgecolors=None)
if plot_results:
    sns.scatterplot(x=bins['Mw_bin'], y=np.log10(inverted_bins + 1e-10), s=10, label='Inverted Bins', edgecolors=None)
plt.ylabel('log10(N)')
plt.xlim([min_Mw, max_Mw])
plt.ylim([-6, 3])
plt.legend(loc='upper right')
plt.title(f"# Ruptures: {n_ruptures}")
plt.show()

if plot_results:
    plt.scatter(ruptures['initial_rate'], ruptures['inverted_rate'], c=ruptures['Mw'])
    plt.plot([1e-6, 1], [1e-6, 1])
    plt.plot([1e-6, 1e-1], [1e-5, 1])
    plt.plot([1e-5, 1], [1e-6, 1e-1])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Input/Ideal Rates')
    plt.ylabel('Inverted Rates')
    plt.title(f"# Ruptures: {n_ruptures}")
    plt.colorbar()
    plt.show()
# %%
plot_extras = False
if plot_extras:
    plt.scatter(ruptures['Mw'], ruptures['inverted_rate'] / ruptures['initial_rate'])
    plt.xlabel('Magnitude')
    plt.ylabel('Change ratio (inv / inp)')
    plt.yscale('log')
    plt.show()
