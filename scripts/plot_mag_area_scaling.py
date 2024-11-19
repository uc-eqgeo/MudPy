import pandas as pd
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt

p_length, p_width = 5, 5

indicies_file = os.path.join('..', '..', 'occ-coseismic', 'data', 'sz_solutions', 'FakeQuakes_sz_n5000_S10_N1_GR500_b1-1_N21-5_nIt1000000_narchi10', 'ruptures', 'indices.csv')
ruptures_file = 'Z:/McGrath/HikurangiFakeQuakes/hikkerk3D_hires/output/ruptures_noDepthLimit/rupture_df_n50000.csv'


indices_df = pd.read_csv(indicies_file)
ruptures_df = pd.read_csv(ruptures_file)

mags = ruptures_df['mw']
rupture_ids = indices_df['Rupture Index']
area = indices_df['Num Sections'] * p_length * p_width

mags = np.array([mags[ix] for ix in rupture_ids])

sns.scatterplot(x=mags, y=area)
plt.plot(mags, 10 ** (mags - 4), color='black')
plt.yscale('log')
plt.ylabel('Area (km^2)')
plt.show()