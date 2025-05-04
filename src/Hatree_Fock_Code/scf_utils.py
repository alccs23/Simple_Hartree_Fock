import numpy as np
from scipy.special import erf
import math
from .integrals import evaluate_repulsion_integral

def create_repulsion_matrix(orbital_list):
    """
    Constructs the full 4-dimensional electron repulsion integral tensor for a set of orbitals.

    The resulting matrix contains all two-electron repulsion integrals in the atomic orbital basis

    Args:
        orbital_list (list[dict]): List of orbital dictionaries, where each dictionary contains:
            - 'coeffs' (list[float]): Contraction coefficients for primitive Gaussians
            - 'alphas' (list[float]): Exponential coefficients for primitive Gaussians
            - 'R' (np.ndarray): Cartesian coordinates [x, y, z] of orbital center

    Returns:
        np.ndarray: 4D array of shape (nbf, nbf, nbf, nbf) where:
            - nbf = number of basis functions (len(orbital_list))
    """
    nbf = len(orbital_list)  # Number of basis functions
    rep_matrix = np.zeros((nbf, nbf, nbf, nbf))  # Initialize 4D ERI tensor

    # Compute all unique two-electron integrals
    for i in range(nbf):
        for j in range(nbf):
            for k in range(nbf):
                for l in range(nbf):
                    rep_matrix[i,j,k,l] = evaluate_repulsion_integral(
                        orbital_list[i],
                        orbital_list[j],
                        orbital_list[k],
                        orbital_list[l]
                    )
    
    return rep_matrix

def create_P_matrix(coeff_matrix, num_elec, number_basis):
    """
    Constructs the density matrix (P) from molecular orbital coefficients.

    Args:
        coeff_matrix (np.ndarray):
            Coefficient matrix of shape (number_basis, number_basis) containing the molecular orbital
            coefficients, where coeff_matrix[u,a] is the coefficient of basis function 'u' in molecular orbital 'a'.

        num_elec (int):
            Total number of electrons in the system. Must be even (closed-shell system assumed).

        number_basis (int):
            Number of basis functions used in the calculation.

    Returns:
        np.ndarray:
            The density matrix P of shape (number_basis, number_basis), where each element P[u,v]
            represents the probability of finding an electron in the overlap between basis functions u and v.
    """
    P_matrix = np.empty((number_basis,number_basis))

    for u in range(number_basis):
        for v in range(number_basis):
            total_sum = 0
            for a in range(int(num_elec/2)):
                total_sum += coeff_matrix[u][a]*np.conjugate(coeff_matrix[v][a])
            P_matrix[u][v] = 2 * total_sum

    return P_matrix

def create_G_matrix(density_matrix, ERI_matrix):
    """
    Constructs the G matrix (two-electron repulsion contribution) for Hartree-Fock calculations.

    The G matrix represents the electron-electron repulsion terms in the Fock matrix and is computed
    as a contraction of the density matrix with the two-electron repulsion integrals (ERIs).

    Parameters:
        density_matrix (np.ndarray):
            The density matrix P (nbf x nbf) where nbf is the number of basis functions.
            Must be symmetric and real-valued.

        ERI_matrix (np.ndarray):
            The four-index electron repulsion integral tensor (nbf x nbf x nbf x nbf)

    Returns:
        np.ndarray:
            The G matrix (nbf x nbf) containing the two-electron contributions to the Fock matrix.

    """
    num_orbitals = len(density_matrix)

    G_matrix = np.zeros(density_matrix.shape)

    for u in range(num_orbitals):
        for v in range(num_orbitals):
            total_sum = 0
            for lam in range(num_orbitals):
                for sig in range(num_orbitals):
                    total_sum += density_matrix[lam][sig] * (ERI_matrix[u][v][sig][lam] - 0.5*ERI_matrix[u][lam][sig][v])
            G_matrix[u][v] = total_sum

    return G_matrix

def compute_energy(P_matrix, H_core, F_matrix, nat_pos):
    """
    Computes the total Hartree-Fock energy (electronic + nuclear repulsion) for a molecular system.

    Args:
        P_matrix (np.ndarray):
            Density matrix of shape (nbf, nbf) where nbf is the number of basis functions.
            Represents the electron distribution in the basis set.

        H_core (np.ndarray):
            Core Hamiltonian matrix (nbf, nbf) containing kinetic energy and nuclear attraction terms.

        F_matrix (np.ndarray):
            Fock matrix (nbf, nbf) representing the effective one-electron operator.

        nat_pos (list):
            Nuclear positions and charges in the format:
            [ [ [x1,y1,z1], charge1 ], [ [x2,y2,z2], charge2 ], ... ]

    Returns:
        float: Total Hartree-Fock energy in atomic units (Hartrees), composed of:
               - Electronic energy (1-electron + 2-electron contributions)
               - Nuclear repulsion energy

    """
    # Compute electronic energy: 0.5 * Tr[P*(H + F)]
    electronic_energy = 0.5 * np.sum(P_matrix * (H_core + F_matrix))

    # Compute nuclear repulsion energy
    nuc_energy = 0.0
    for i in range(len(nat_pos)):
        for j in range(i+1, len(nat_pos)):
            pos_A, Z_A = nat_pos[i]
            pos_B, Z_B = nat_pos[j]
            R_AB = np.linalg.norm(np.array(pos_A) - np.array(pos_B))
            nuc_energy += (Z_A * Z_B) / R_AB

    return electronic_energy + nuc_energy
