# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 13:55:43 2024

@author: jmc753
"""

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

inversion_name = 'start_rand'
n_ruptures = 15000
slip_weight = 1
GR_weight = 1
max_iter = 44500


min_Mw, max_Mw = 4.5, 9.5

outdir = f"Z:\\McGrath\\HikurangiFakeQuakes\\hikkerk3D\\output\\{inversion_name}"

plot_results = True
if max_iter == 0:
    max_iter = '*'

rupture_list = glob(f'{outdir}\\n{n_ruptures}_S{slip_weight}_GR{GR_weight}_nIt{max_iter}_inverted_ruptures.txt')
n_iter = [int(file.split('nIt')[1].split('_')[0]) for file in rupture_list]
order = np.array(n_iter).argsort()
if len(order) != 0:
    ruptures_list = []
    bins_list = []
    for ix in order:
        file = rupture_list[ix]
        ruptures_list.append(pd.read_csv(file, sep='\t', index_col=0).sort_values('Mw'))
        bins_df = pd.read_csv(file.replace('ruptures', 'bins'), sep='\t', index_col=0)
        bins_df['upper'] = pd.read_csv(f"{outdir}\\n{n_ruptures}_S{slip_weight}_GR{GR_weight}_input_bins.txt", sep='\t', index_col=0)['upper']
        bins_list.append(bins_df)
else:
    plot_results = False
    ruptures_list = [pd.read_csv(f"{outdir}\\n{n_ruptures}_S{slip_weight}_GR{GR_weight}_input_ruptures.txt", sep='\t', index_col=0).sort_values('Mw')]
    bins_list = [pd.read_csv(f"{outdir}\\n{n_ruptures}_S{slip_weight}_GR{GR_weight}_input_bins.txt", sep='\t', index_col=0)]

n_runs = len(ruptures_list)
if n_runs > 1:
    min_Mw = 4.0

gr_matrix = np.zeros((bins_list[0].shape[0], ruptures_list[0].shape[0])).astype('bool')
rupture_matrix = np.zeros((ruptures_list[0].shape[0], ruptures_list[0].shape[0])).astype('bool')

for ix, mag in enumerate(bins_list[0]['Mw_bin']):
    gr_matrix[ix, :] = (np.round(ruptures_list[0]['Mw'], 1) >= mag)
gr_matrix = gr_matrix.astype('int')

for ix, mag in enumerate(ruptures_list[0]['Mw']):
    rupture_matrix[ix, :] = (ruptures_list[0]['Mw'] >= mag)
rupture_matrix = rupture_matrix.astype('int')

for run in range(n_runs):
    bins = bins_list[run]
    ruptures = ruptures_list[run]

    initial_rate = np.matmul(rupture_matrix, ruptures['initial_rate'])
    initial_bins = np.matmul(gr_matrix, ruptures['initial_rate'])
    lim_ix = np.where(bins['upper'] != 0)[0]
    if plot_results:
        inverted_rate = np.matmul(rupture_matrix, ruptures['inverted_rate'])
        inverted_bins = np.matmul(gr_matrix, ruptures['inverted_rate'])
        inverted_rate[inverted_rate == 0] = 1e-10
        inverted_bins[inverted_bins == 0] = 1e-10
        ruptures[ruptures['inverted_rate'] == 0] = 1e-10

    ruptures[ruptures['initial_rate'] == 0] = 1e-10
    ruptures[ruptures['upper'] == 0] = 1e-10

# %%
    if run == 0:
        # %%
        plt.plot(ruptures['Mw'], ruptures['target_rate'].apply(lambda x: np.log10(x)), color='black' , label='Target GR Relation', zorder=6)
        # plt.plot(ruptures['Mw'], np.log10(ruptures['lower'] + 1e-12), color='green', linestyle=':')
        # plt.plot(ruptures['Mw'], np.log10(ruptures['upper']), color='green', linestyle=':')
        # plt.plot(bins['Mw_bin'][lim_ix], np.log10(bins['upper'][lim_ix]), color='green', linestyle='-.', label='Upper Limit')  # Binned upper bound

        # sns.histplot(x=ruptures['Mw'], y=np.log10(ruptures['initial_rate']), binwidth=(0.1,0.1), binrange=((5.95,9.55),(-12.05,3.05)))

        sns.scatterplot(x=ruptures['Mw'], y=np.log10(initial_rate), s=20, color='blue', label='Initial GR', edgecolors=None, zorder=4)
        sns.scatterplot(x=ruptures['Mw'], y=np.log10(ruptures['initial_rate']), s=2, label='Initial rate', color='blue', edgecolors=None, zorder=1)
        # sns.histplot(x=ruptures['Mw'], y=np.log10(ruptures['initial_rate'] + 1e-10), binwidth=(0.05, 0.1))
        try:
            sns.scatterplot(x=ruptures['Mw'], y=np.log10(ruptures['upper']), s=5, color='green', label='Individual limit', edgecolors=None, zorder=0)
        except:
            sns.scatterplot(x=ruptures['Mw'], y=ruptures['upper'], s=5, color='green', label='Individual limit', edgecolors=None, zorder=0)
        if plot_results:
            sns.scatterplot(x=ruptures['Mw'], y=np.log10(ruptures['inverted_rate']), s=2, label='Inverted rate', color='orange', edgecolors=None, zorder=2)
            sns.scatterplot(x=ruptures['Mw'], y=np.log10(inverted_rate), s=10, color='red', label='Inverted GR', edgecolors=None, zorder=5)
        # sns.scatterplot(x=bins['Mw_bin'], y=np.log10(initial_bins + 1e-12), s=20, label='Initial Bins', edgecolors=None)
        plt.ylabel('log10(N)')
        plt.xlim([min_Mw, max_Mw])
        plt.ylim([-10, 3])
        plt.legend(loc='lower left')
        plt.title(f"# Ruptures: {n_ruptures}")
        # %%
    # if plot_results:
    #     sns.scatterplot(x=bins['Mw_bin'], y=np.log10(inverted_bins + 1e-12), s=15, label=f'Inverted Bins {n_iter[order[run]]}', edgecolors=None)
plt.show()
# %%
if plot_results:
    binwidth = 0.5
    sns.histplot(x=np.log10(ruptures['initial_rate']), y=np.log10(ruptures['inverted_rate']), binwidth=binwidth, binrange=(-12 - binwidth / 2, 1 + binwidth / 2), zorder=0)
    plt.scatter(np.log10(ruptures['initial_rate']), np.log10(ruptures['inverted_rate']), c=ruptures['Mw'], s=3, zorder=2)
    plt.plot([-10, 1], [-10, 1], color='red', zorder=1)
    plt.xlim([-10, 1])
    plt.ylim([-10, 1])
    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel('Log(Input Rate)')
    plt.ylabel('Log(Inverted Rate)')
    plt.title(f"# Ruptures: {n_ruptures}")
    plt.colorbar()
    plt.show()

    binwidth = 0.1
    sns.histplot(x=np.log10(ruptures['initial_rate']), y=np.log10(ruptures['inverted_rate']), binwidth=binwidth, binrange=(-12 - binwidth / 2, 1 + binwidth / 2), zorder=0)
    plt.plot([-10, 1], [-10, 1], color='red', zorder=1)
    plt.xlim([-10, 1])
    plt.ylim([-10, 1])
    plt.xlabel('Log(Input Rate)')
    plt.ylabel('Log(Inverted Rate)')
    plt.title(f"# Ruptures: {n_ruptures}")
    plt.show()
# %%
plot_extras = False
if plot_extras:
    plt.scatter(ruptures['Mw'], ruptures['inverted_rate'] / ruptures['initial_rate'])
    plt.xlabel('Magnitude')
    plt.ylabel('Change ratio (inv / inp)')
    plt.yscale('log')
    plt.show()
