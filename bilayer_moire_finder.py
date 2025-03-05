import moire_lattice_vector_finder as mlf
import numpy as np
import pandas as pd

file_1 = "mos2_mono_layer.vasp"
file_2 = "mos2_mono_layer.vasp"

lattice_vectors1, atom_type_arr1, dat1 = mlf.read_vasp(file_1)

lattice_vectors2, atom_type_arr2, dat2 = mlf.read_vasp(file_2)

test = mlf.find_index(a1=lattice_vectors1['a'],a2=lattice_vectors1['b'],a3=lattice_vectors1['c'],
                    b1=lattice_vectors2['a'],b2=lattice_vectors2['b'],b3=lattice_vectors2['c'],
                    theta_min=20,theta_max=22,theta_step=0.1,
                    n_min = 1, n_max=30,
                    dtol = 0.005, ang_lat = 60.0, num_cores=16)

# Define the column names
columns = ['degree', 'n1', 'n2', 'n1p', 'n2p','vec_del', 'delta_angle', 'norm','A1','A2']
df = pd.DataFrame(test, columns=columns)


# Sort the DataFrame by 'degree' in ascending order and 'norm' in descending order, followed by 'vec_del' and 'delta_angle'
df_sorted = df.sort_values(by=['degree', 'norm', 'vec_del', 'delta_angle','n1','n2'], ascending=[True, True, True, True, False, False])


# Group by 'degree' and take the first entry in each group
final_df = df_sorted.groupby('degree', as_index=False).first()

final_df.to_csv('mos2_bilayer_candidates.csv')

final_df.to_pickle('mos2_bilayer_candidates.pkl')