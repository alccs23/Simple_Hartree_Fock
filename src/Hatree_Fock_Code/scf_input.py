# input.py

from .basis import *
from .scf_run import run_scf, prepare_scf_inputs

def run_custom_scf(orbital_list, nat_pos, num_elec, max_scf=50, conv_thr=1e-6):
    """
    Runs SCF calculation with provided orbitals, nuclear positions, and number of electrons.
    
    Parameters:
        orbital_list (list): List of orbital objects created via `create_orbital`.
        nat_pos (list): Nuclear positions and charges, e.g., [[[0, 0, 0], 1], [[0.7, 0, 0], 1]].
        num_elec (int): Number of electrons in the system.
        max_scf (int): Maximum SCF iterations. Default = 50.
        conv_thr (float): Convergence threshold for SCF. Default = 1e-6.

    Returns:
        Tuple: (energy, final_density, C_matrix)
    """
    H_core, repulsion_matrix, S_inv_sqrt = prepare_scf_inputs(orbital_list, nat_pos)
    return run_scf(H_core, repulsion_matrix, S_inv_sqrt, num_elec, orbital_list, nat_pos, max_scf, conv_thr)


if __name__ == "__main__":
    Be_1s = create_orbital(STO_3G_1s_alphas, zetas['Be'], STO_3G_1s_coeff, [0,0,0])
    Be_2s = create_orbital(STO_3G_2s_alphas, zetas['Be'], STO_3G_2s_coeff, [0,0,0])
    H1_1s = create_orbital(STO_3G_1s_alphas, zetas['H'], STO_3G_1s_coeff, [1.2906,0,0])
    H2_1s = create_orbital(STO_3G_1s_alphas, zetas['H'], STO_3G_1s_coeff, [-1.2906,0,0])

    orbital_list = [Be_1s, Be_2s, H1_1s, H2_1s]
    nat_pos = [ [[0,0,0], 4], [[1.2906,0,0], 1], [[-1.2906,0,0], 1] ]
    num_elec = 6

    energy, final_density, C_matrix = run_custom_scf(orbital_list, nat_pos, num_elec)
    print(f"SCF converged energy: {energy:.6f} Hartree")

