import numpy as np
from .integrals import calculate_orbital_integral, S_overlap, generate_core_hamiltonian

from .scf_utils import create_repulsion_matrix, create_G_matrix, create_P_matrix, compute_energy

def prepare_scf_inputs(orbital_list, nat_pos):
    """
    Generates H_core, repulsion matrix, and S^{-1/2} from a list of orbitals and nuclear positions.

    Args:
        orbital_list (list): List of basis orbitals
        nat_pos (list): Nuclear coordinates and atomic numbers [[[x,y,z], Z], ...]

    Returns:
        H_core (np.ndarray), repulsion_matrix (np.ndarray), S_inv_sqrt (np.ndarray)
    """
    num_orbitals = len(orbital_list)
    S_matrix = np.empty((num_orbitals, num_orbitals))
    for i in range(num_orbitals):
        for j in range(num_orbitals):
            S_matrix[i][j] = calculate_orbital_integral(orbital_list[i], orbital_list[j], S_overlap)

    H_core = generate_core_hamiltonian(orbital_list, nat_pos)
    repulsion_matrix = create_repulsion_matrix(orbital_list)

    # S^{-1/2} construction via SVD with regularization
    U, S_svd, Vh = np.linalg.svd(S_matrix, full_matrices=True)

    threshold = 1e-8  # regularization threshold
    inv_sqrt_singulars = []
    for val in S_svd:
        if val < threshold:
            print(f"Warning: Singular value too small ({val:.2e}), applying regularization.")
            inv_sqrt_singulars.append(0.0)  # discard near-zero modes
        else:
            inv_sqrt_singulars.append(1.0 / np.sqrt(val))

    Sigma_inv_sqrt = np.diag(inv_sqrt_singulars)
    S_inv_sqrt = U @ Sigma_inv_sqrt @ U.T

    return H_core, repulsion_matrix, S_inv_sqrt

def run_scf(H_core, repulsion_matrix, S_inv_sqrt, num_elec, orbital_list, nat_pos,
            max_scf=100, conv_thr=1e-6, initial_density=None):
    """
    Performs SCF iterations for Hartree-Fock energy convergence.
    Outputs SCF progress to 'output' file.
    """
    with open('output', 'w') as log_file:
        density_matrix = initial_density if initial_density is not None else np.zeros_like(H_core)
        cur_energy = 0.0

        for n in range(max_scf):
            G_matrix = create_G_matrix(density_matrix, repulsion_matrix)
            F_core = H_core + G_matrix

            # Orthogonalize and diagonalize Fock matrix
            F_ortho = S_inv_sqrt @ F_core @ S_inv_sqrt
            epsilon, C_prime = np.linalg.eigh(F_ortho)
            C_matrix = S_inv_sqrt @ C_prime

            new_density_matrix = create_P_matrix(C_matrix, num_elec, len(orbital_list))
            new_energy = compute_energy(new_density_matrix, H_core, F_core, nat_pos)

            energy_diff = np.abs(new_energy - cur_energy)
            log_file.write(f"Iter {n+1}: E = {new_energy:.8f}, ΔE = {energy_diff:.2e}\n")

            if energy_diff < conv_thr:
                log_file.write(f"SCF converged in {n+1} iterations with ΔE = {energy_diff:.2e}\n")
                log_file.write(f"Final Energy : {new_energy:.12f}\n Hartrees")
                return new_energy, new_density_matrix, C_matrix

            cur_energy = new_energy
            density_matrix = new_density_matrix

        log_file.write(f"SCF failed to converge in {max_scf} iterations\n")
        raise RuntimeError("SCF failed to converge")

