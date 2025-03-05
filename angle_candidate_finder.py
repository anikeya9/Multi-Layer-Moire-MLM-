import numpy as np
import pandas as pd
from numba import njit, prange

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

@njit
def rotation_matrix(theta: float) -> np.ndarray:
    rad = np.deg2rad(theta)
    return np.array([[np.cos(rad), -np.sin(rad), 0.0],
                     [np.sin(rad), np.cos(rad), 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)
    
@njit
def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    norm_v1 = np.sqrt(np.dot(v1, v1))
    norm_v2 = np.sqrt(np.dot(v2, v2))
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0  # Handle zero-length vectors gracefully
    
    cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
    
    # Clamp the cosine value to avoid numerical errors
    if cosine_angle < -1.0:
        cosine_angle = -1.0
    elif cosine_angle > 1.0:
        cosine_angle = 1.0
    
    angle_rad = np.arccos(cosine_angle)
    return np.rad2deg(angle_rad)  # Convert radians to degrees

@njit(parallel=True, nogil=True)
def find_index(a1, a2, a3, b1, b2, b3, theta_min, theta_max, theta_step, n_min, n_max, d_tol, ang_lat):
    theta_array = np.arange(theta_min, theta_max, theta_step)
    final_selected_index = []
    
    range_1 = list(range(-n_max,-n_min+1))
    range_2 = list(range(n_min,n_max+1))
    combined_list = range_1 + range_2
    
    #combined_list = np.array([i for i in range(-n_max, n_max + 1) if i != 0], dtype=np.int32)
    
    for deg in prange(len(theta_array)):
        qualified_index = []
        rot_matrix = rotation_matrix(theta_array[deg])
        b_1r = rot_matrix @ b1.T
        b_2r = rot_matrix @ b2.T
        b_3r = rot_matrix @ b3.T

        for n1 in combined_list:
            for n2 in combined_list:
                for m1 in combined_list:
                    for m2 in combined_list:
                        v1 = n1 * a1 + n2 * a2
                        v2 = m1 * b_1r + m2 * b_2r
                        vec_del = np.linalg.norm(v1 - v2)
                        if vec_del < d_tol:
                            qualified_index.append((deg, n1, n2, vec_del))
                            
        for i in range(len(qualified_index)):
            n_1, n_2, v_d = qualified_index[i][1], qualified_index[i][2], qualified_index[i][3]
            for j in range(len(qualified_index)):
                n_1p, n_2p = qualified_index[j][1], qualified_index[j][2]
                A1 = n_1 * a1 + n_2 * a2
                A2 = n_1p * a1 + n_2p * a2
                #norm_A1 = np.linalg.norm(A1)
                #norm_A2 = np.linalg.norm(A2)
                angle_A1_A2 = angle_between(A1,A2)
                delta_angle = np.abs(angle_A1_A2 - ang_lat)
                if delta_angle < 0.1:
                    final_selected_index.append((theta_array[deg], n_1, n_2, n_1p, n_2p, v_d, delta_angle,
                                                  np.linalg.norm(A1+A2), A1, A2))
    
    return final_selected_index
