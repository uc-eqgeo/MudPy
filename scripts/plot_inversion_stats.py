# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 13:55:43 2024

@author: jmc753
"""

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

results_file = "C:\\Users\\jmc753\\Work\\MudPy\\examples\\fakequakes\\3D\\hikkerk3D_test\\output\\ruptures\\preferred_rate_n1184.txt"

results = pd.read_csv(results_file, sep='\t', index_col=0)

b, N = 1.1, 21.5
a = np.log10(N) + (b * 5)
min_a = np.log10(0.1) + (b * 5)
max_a = np.log10(100) + (b * 5)
min_Mw, max_Mw = 6.0, 9.5

Mw = np.array([min_Mw, max_Mw])

inv_rate = []
for mw in results['Mw']:
    cumfreq = np.sum(results['rate'][np.where(results['Mw'] >= mw)[0]])
    if cumfreq > 0:
        inv_rate.append(np.log10(cumfreq))  # Calculate log(N) for each magnitude
    else:
        inv_rate.append(np.nan)


sns.scatterplot(x=results['Mw'], y=inv_rate)
plt.plot(Mw, (a - (b * Mw)), color='red')
plt.plot(Mw, (min_a - (b * Mw)), color='red', linestyle=':')
plt.plot(Mw, (max_a - (b * Mw)), color='red', linestyle=':')

results['log_rate'] = results['rate'].apply(lambda x: np.log10(x) if x > 0 else x)
