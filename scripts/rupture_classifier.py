# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer


# data = load_breast_cancer()


# # construct a dataframe using pandas
# df1=pd.DataFrame(data['data'],columns=data['feature_names'])
 
# # Scale data before applying PCA
# scaling=StandardScaler()
 
# # Use fit and transform method 
# scaling.fit(df1)
# Scaled_data=scaling.transform(df1)
 
# # Set the n_components=3
# principal=PCA(n_components=3)
# principal.fit(Scaled_data)
# x=principal.transform(Scaled_data)
 
# # Check the dimensions of data after PCA
# print(x.shape)

# principal.components_



# plt.figure(figsize=(10,10))
# plt.scatter(x[:,0],x[:,1],c=data['target'],cmap='plasma')
# plt.xlabel('pc1')
# plt.ylabel('pc2')



# # import relevant libraries for 3d graph
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure(figsize=(10,10))
 
# # choose projection 3d for creating a 3d graph
# axis = fig.add_subplot(111, projection='3d')
 
# # x[:,0]is pc1,x[:,1] is pc2 while x[:,2] is pc3
# axis.scatter(x[:,0],x[:,1],x[:,2], c=data['target'],cmap='plasma')
# axis.set_xlabel("PC1", fontsize=10)
# axis.set_ylabel("PC2", fontsize=10)
# axis.set_zlabel("PC3", fontsize=10)




# # check how much variance is explained by each principal component
# print(principal.explained_variance_ratio_)


# df1 = pd.read_csv('Z:\\McGrath\\HikurangiFakeQuakes\\hikkerk3D\\output\\rupture_PCA_n5000.csv')
# # %%
# # Scale data before applying PCA
# scaling=StandardScaler()
 
# # Use fit and transform method 
# scaling.fit(df1)
# Scaled_data=scaling.transform(df1)
 
# # Set the n_components=3
# principal=PCA(n_components=10)
# principal.fit(Scaled_data)
# x=principal.transform(Scaled_data)
 
# # Check the dimensions of data after PCA
# print(x.shape)

# principal.components_



# # check how much variance is explained by each principal component
# print(principal.explained_variance_ratio_)
# print(sum(principal.explained_variance_ratio_))


# # 
# plt.figure(figsize=(10,10))
# plt.scatter(x[:,0],x[:,1],c=df1['mw'],cmap='plasma')
# plt.xlabel('pc1')
# plt.ylabel('pc2')
# plt.colorbar()


# # %%
# # import relevant libraries for 3d graph
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure(figsize=(10,10))
 
# # choose projection 3d for creating a 3d graph
# axis = fig.add_subplot(111, projection='3d')
 
# # x[:,0]is pc1,x[:,1] is pc2 while x[:,2] is pc3
# axis.scatter(x[:,0],x[:,1],x[:,2], c=df1['mw'],cmap='plasma')
# axis.set_xlabel("PC1", fontsize=10)
# axis.set_ylabel("PC2", fontsize=10)
# axis.set_zlabel("PC3", fontsize=10)

# %%
import os
# Define directories
inversion_dir = 'Z:\\McGrath\\HikurangiFakeQuakes\\hikkerk3D\\output\\archi'
rupture_dir = os.path.abspath(os.path.join(inversion_dir, '..', 'ruptures'))
rupture_file_prefix = 'hikkerk3D_locking_NZNSHMscaling.Mw'

# Define flags for results csv
n_ruptures = 5000
slip_weight = 1
gr_weight = 10
n_its = 100000

# %% No user inputs below here
# Create filepaths
rupture_csv = os.path.join(inversion_dir, '..', 'rupture_df_n15000.csv')  # CSV containing rupture slips
inv_file = f"n{n_ruptures}_S{slip_weight}_GR{gr_weight}_nIt{n_its}_inverted_ruptures.csv"
inv_file = os.path.join(inversion_dir, inv_file)
patch_file = os.path.join(inversion_dir, '..', '..', 'data', 'model_info', 'hk.fault')

# %% Load data
rupture_df = pd.read_csv(rupture_csv, nrows=n_ruptures)
rupture_ids = rupture_df['rupt_id']

del rupture_df


# %%
def read_log(log_file):
    parameters = {'Corr. length used Lstrike': None,
                  'Corr. length used Ldip': None,
                  'Slip std. dev.': None,
                  'Maximum length Lmax': None,
                  'Maximum width Wmax': None,
                  'Effective length Leff': None,
                  'Effective width Weff': None,
                  'Target magnitude': None,
                  'Actual magnitude': None,
                  'Hypocenter (lon,lat,z[km])': None,
                  'Centroid (lon,lat,z[km])': None,
                  'Average Risetime (s)': None,
                  'Average Rupture Velocity (km/s)': None,
                  'Avg. length': None,
                  'Avg. width': None}

    with open(log_file, 'r') as f:
        line = f.readline()
        while '\n' in line:
            if len(line.split(':')) == 2:
                parameter, value = line.split(':')
                if parameter in parameters:
                    if '(' in value:
                        tuple_string = value.strip().strip('()')
                        value = ()
                        for element in tuple_string.split(','):
                            value += (float(element),)
                        parameters[parameter] = [value]
                    else:
                        value = value.strip().strip('\n').strip(' km').strip('Mw ')
                        parameters[parameter] = float(value)
                        parameters[parameter] = float(value)
                    print(parameter + ':', value)
                elif 'Run number' in line:
                    id = value.strip().strip('\n')
                    print(id)
            line = f.readline()
    log_df = pd.DataFrame(parameters, index=[id])
    return log_df

log_file = os.path.join(rupture_dir, rupture_file_prefix + rupture_ids[0] + '.log')
# %%
pca_df = read_log(log_file)

for ix, rupture_id in enumerate(rupture_ids[1:]):
    print(ix, end='\r')
    log_file = os.path.join(rupture_dir, rupture_file_prefix + rupture_id + '.log')
    log_df = read_log(log_file)
    pca_df = pd.concat([pca_df, log_df])
# %%
inv_df = pd.read_csv(inv_file, sep='\t', index_col=0)
trim_df = pca_df.drop(columns=['Hypocenter (lon,lat,z[km])', 'Centroid (lon,lat,z[km])'])
trim_df['rate'] = np.log10(np.array(inv_df['inverted_rate_0']))
trim_df = trim_df.sort_values(by='rate', ascending=True)

# Scale data before applying PCA
scaling=StandardScaler()
 
# Use fit and transform method 
scaling.fit(trim_df)
Scaled_data=scaling.transform(trim_df)
 
# Set the n_components=3
principal=PCA(n_components=3)
principal.fit(Scaled_data)
x=principal.transform(Scaled_data)
 
# Check the dimensions of data after PCA
print(x.shape)

principal.components_


for col in trim_df.columns:
    plt.figure(figsize=(10,10))
    plt.scatter(x[:,0],x[:,1],c=trim_df[col],cmap='plasma')
    plt.xlabel('pc1')
    plt.ylabel('pc2')
    plt.colorbar()
    plt.title(col)
    plt.show()

# %%
plt.figure(figsize=(10,10))
plt.scatter(x[:,0],x[:,1], c=trim_df['rate'] > -6, cmap='plasma')
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.colorbar()
plt.title('Inverted Rate')
plt.show()

# %%
# import relevant libraries for 3d graph
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10,10))
 
# choose projection 3d for creating a 3d graph
axis = fig.add_subplot(111, projection='3d')
 
# x[:,0]is pc1,x[:,1] is pc2 while x[:,2] is pc3
axis.scatter(x[:,0],x[:,1],x[:,2], c=np.log10(inv_df['inverted_rate_0']),cmap='plasma')
axis.set_xlabel("PC1", fontsize=10)
axis.set_ylabel("PC2", fontsize=10)
axis.set_zlabel("PC3", fontsize=10)


# check how much variance is explained by each principal component
print(principal.explained_variance_ratio_)
# %%
