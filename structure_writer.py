import numpy as np 
import pandas as pd 
from finalised_scripts import angle_between as angle

def read_vasp(file_path):
    lattice_vectors = {}
    lattice_vectors['a'] = np.loadtxt(file_path, skiprows=2, max_rows=1)
    lattice_vectors['b'] = np.loadtxt(file_path, skiprows=3, max_rows=1)
    lattice_vectors['c'] = np.loadtxt(file_path, skiprows=4, max_rows=1)
    atom_type = np.loadtxt(file_path, skiprows=5, max_rows=1, dtype='str')
    atom_counts = np.loadtxt(file_path, skiprows=6,max_rows=1, dtype = 'int')
    atom_type_arr = np.repeat(atom_type, atom_counts)
    data = np.loadtxt(file_path, skiprows = 8)
    
    return lattice_vectors, atom_type_arr, data

def replicate_atoms(a: np.array,
                    b: np.array,
                    c: np.array,
                    atom_data: np.array,
                    atom_type_arr: np.array,
                    natoms: int,
                    na: int,
                    nb: int,
                    nc: int):
    """ replicates given unit cell in -na to na and -nb to nb and nc 
    for purpose of moire structure creation
    and new lattice vectors which are na*a, nb*b, nc*c
    
    """ 
    
    replicated_atom = np.tile(atom_data,(((2 * na ) * ( 2 * nb ) * nc),1) )
    
    
    # Create displacement vectors for each replication
    displacements = np.array([
        ia * a +
        ib * b +
        ic * c
        for ia in range(-na, na )
        for ib in range(-nb, nb )
        for ic in range(nc)
    ])
    
        # Apply displacements to atom positions
    replicated_atom += np.repeat(displacements, natoms, axis=0)
    replicated_type_arr = np.tile(atom_type_arr,((2 * na  ) * ( 2 * nb ) * nc) )
    # new_a = a * na
    # new_b = b * nb 
    # new_c = c * nc
    new_lattice_vectors = {}
    new_lattice_vectors['a'] = a * na
    new_lattice_vectors['b'] = b * nb
    new_lattice_vectors['c'] = c * nc
    new_total_atoms = natoms * (2 * na ) * (2 * nb ) * nc
    return new_lattice_vectors, new_total_atoms, replicated_atom, replicated_type_arr

def write_vasp_optimized(filename, A1, A2, A3, atom_data, num_Ba, num_O, num_Ti):
    # Buffer for all the lines to write
    lines = [
        "BTO\n",
        "1.0\n",
        f"{A1[0]} {A1[1]} {A1[2]}\n",
        f"{A2[0]} {A2[1]} {A2[2]}\n",
        f"{A3[0]} {A3[1]} {A3[2]}\n",
        "Ba O Ti\n",
        f"{num_Ba} {num_O} {num_Ti}\n",
        "Cartesian\n"
    ]
    
    # Write the buffer to the file
    with open(filename, 'w') as file:
        file.writelines(lines)
        # Use to_csv for efficient DataFrame writing
        atom_data.to_csv(file, sep=' ', index=False, header=False, mode='a')
        
def write_lammps_data(filename, centered_x_bounds, centered_y_bounds, centered_z_bounds, atom_data):
    with open(filename, 'w') as file:
        # Write box bounds
        file.write("LAMMPS data file\n\n")
        file.write(f"{len(atom_data)} atoms\n")
        file.write(f"{int(atom_data['Atom'].nunique())} atom types\n\n")
        file.write(f"{centered_x_bounds[0]} {centered_x_bounds[1]} xlo xhi\n")
        file.write(f"{centered_y_bounds[0]} {centered_y_bounds[1]} ylo yhi\n")
        file.write(f"{centered_z_bounds[0]} {centered_z_bounds[1]} zlo zhi\n\n")
        file.write("Masses\n\n")
        file.write("1 137.327  # Ba\n")
        file.write("2 47.867  # Ti\n")
        file.write("3 15.9994  # O\n\n")
        file.write("Atoms # atomic\n\n")
        file.write(atom_data.to_string(index=False, header=False))

