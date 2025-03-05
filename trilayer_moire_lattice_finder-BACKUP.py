import moire_lattice_vector_finder as mlf
import numpy as np
import pandas as pd

file_1 = "graphene_monolayer.vasp"
file_2 = "graphene_monolayer.vasp"
file_3 = "graphene_monolayer.vasp"


lattice_vectors1, atom_type_arr1, dat1 = mlf.read_vasp(file_1)
lattice_vectors2, atom_type_arr2, dat2 = mlf.read_vasp(file_2)
lattice_vectors3, atom_type_arr3, dat3 = mlf.read_vasp(file_3)

mlf.find_index(a1=lattice_vectors1['a'],a2=lattice_vectors1['b'],a3=lattice_vectors1['c'],
                    b1=lattice_vectors2['a'],b2=lattice_vectors2['b'],b3=lattice_vectors2['c'],
                    theta_min=1,theta_max=66,theta_step=1,
                    n_min = 1, n_max=4,
                    dtol = 0.009, ang_lat = 60.0, num_cores=64)

print("Initialisation done")

test = mlf.find_index(a1=lattice_vectors1['a'],a2=lattice_vectors1['b'],a3=lattice_vectors1['c'],
                    b1=lattice_vectors2['a'],b2=lattice_vectors2['b'],b3=lattice_vectors2['c'],
                    theta_min=0.5,theta_max=30.0,theta_step=0.05,
                    n_min = 1, n_max=40,
                    dtol = 0.009, ang_lat = 60.0, num_cores=64)

# Define the column names
columns = ['degree', 'n1', 'n2', 'n1p', 'n2p','vec_del', 'delta_angle', 'norm','A1','A2']
df = pd.DataFrame(test, columns=columns)

# Sort the DataFrame by 'degree' in ascending order and 'norm' in descending order, followed by 'vec_del' and 'delta_angle'
df_sorted = df.sort_values(by=['degree', 'norm', 'vec_del', 'delta_angle','n1','n2'], ascending=[True, True, True, True, False, False])

# Group by 'degree' and take the first entry in each group
final_df = df_sorted.groupby('degree', as_index=False).first()

final_df.to_pickle(f'bilayer_graphene.pkl')

print("Done with Bilayer")
print("#####################")
print("Starting with Trilayer")

for i,row in final_df.iterrows():
    print(i)
    print(f"Degree: {row['degree']}")
    moire_vector_a = np.array(row['A1'])
    moire_vector_b = np.array(row['A2'])
    moire_vector_c = np.array([0.0,0.0,1.0])
    print(f"Moire Vector A: {moire_vector_a}")
    print(f"Moire Vector B: {moire_vector_b}")
    print(f"Moire Vector C: {moire_vector_c}")
    print(f"Lattice Vector A: {lattice_vectors3['a']}")
    print(f"Lattice Vector B: {lattice_vectors3['b']}")
    print(f"Lattice Vector C: {lattice_vectors3['c']}")
    test2 = mlf.find_index(a1=moire_vector_a,a2=moire_vector_b,a3=moire_vector_c,
                    b1=lattice_vectors3['a'],b2=lattice_vectors3['b'],b3=lattice_vectors3['c'],
                    theta_min=0.5,theta_max=30.0,theta_step=0.05,
                    n_min = 1, n_max=200,
                    dtol = 0.009, ang_lat = 60.0, num_cores=64)
    columns = ['degree', 'n1', 'n2', 'n1p', 'n2p','vec_del', 'delta_angle', 'norm','A1','A2']
    df = pd.DataFrame(test2, columns=columns)

    df_sorted = df.sort_values(by=['degree', 'norm','vec_del', 'delta_angle'])

    # Group by 'degree' and take the first entry in each group
    final2_df = df_sorted.groupby('degree', as_index=False).first()
    final2_df['theta'] = row['degree']
    theta = row['degree']
    final2_df.to_pickle(f'trilayer_graphene_theta_{theta}.pkl')
