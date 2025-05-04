import numpy as np
from scipy.special import erf
import math


def S_overlap(alpha_a, R_a, alpha_b, R_b):
    """
    Calculate the overlap integral between two Gaussian-type orbitals (GTOs).
    
    Parameters:
    alpha_a (float): Exponential coefficient for Gaussian A
    R_a (array-like): Center coordinates [x, y, z] of Gaussian A
    alpha_b (float): Exponential coefficient for Gaussian B
    R_b (array-like): Center coordinates [x, y, z] of Gaussian B
    
    Returns:
    float: The overlap integral value between the two GTOs

    Notes:
    ------
    The integral is derived from the product of two Gaussians and the kinetic energy operator.
    Reference: Szabo & Ostlund, "Modern Quantum Chemistry", Appendix A (A.9)
    """
    
    prefactor = (np.pi / (alpha_a + alpha_b)) ** (3/2)
    
    exp_alphas = (-alpha_a * alpha_b) / (alpha_a + alpha_b)
    
    distance_squared = np.linalg.norm(R_a - R_b) ** 2
    
    exp_factor = ((2 * alpha_a / np.pi) ** (3/4) * 
                  (2 * alpha_b / np.pi) ** (3/4) * 
                  np.exp(exp_alphas * distance_squared))
    
    return prefactor * exp_factor

def kinetic_overlap(alpha_a, center_a, alpha_b, center_b):
    """
    Computes the kinetic energy overlap integral between two Gaussian Type Orbitals (GTOs).

    This function evaluates the integral ⟨χ_a| -½∇² |χ_b⟩, where χ_a and χ_b are
    primitive Gaussian functions centered at positions center_a and center_b, with
    exponents alpha_a and alpha_b, respectively.

    Parameters:
    -----------
    alpha_a : float
        Gaussian exponent (orbital decay rate) for the first orbital.
    center_a : np.ndarray
        Cartesian coordinates [x, y, z] of the first orbital's center.
    alpha_b : float
        Gaussian exponent for the second orbital.
    center_b : np.ndarray
        Cartesian coordinates [x, y, z] of the second orbital's center.

    Returns:
    --------
    float
        The value of the kinetic energy overlap integral.

    Notes:
    ------
    The integral is derived from the product of two Gaussians and the kinetic energy operator.
    Reference: Szabo & Ostlund, "Modern Quantum Chemistry", Eq. (3.198)
    """
    normalization = ((2 * alpha_a / np.pi) ** (3/4)) * ((2 * alpha_b / np.pi) ** (3/4))
    combined_alpha = alpha_a + alpha_b
    reduced_alpha = (alpha_a * alpha_b) / combined_alpha
    distance_squared = np.sum((center_a - center_b) ** 2)

    S_ab = (np.pi / combined_alpha) ** (3/2) * np.exp(-reduced_alpha * distance_squared)

    term = reduced_alpha * (3 - 2 * reduced_alpha * distance_squared)
    kinetic_integral = term * S_ab

    return normalization * kinetic_integral

# Boys function F0 with special case for t=0
def F0(t):
    if t < 1e-12:  # Special case when t approaches 0
        return 1.0  # lim(t→0) F0(t) = 1
    return 0.5 * np.sqrt(np.pi/t) * erf(np.sqrt(t))

def nuclear_attraction_integral(alpha_a, center_a, alpha_b, center_b, nuclear_center, charge):
    """
    Computes the nuclear attraction integral between two Gaussian orbitals and a point charge nucleus.

    Parameters:
    -----------
    alpha_a : float
        Exponent of the first Gaussian orbital
    center_a : array-like
        [x,y,z] coordinates of the first orbital's center
    alpha_b : float
        Exponent of the second Gaussian orbital
    center_b : array-like
        [x,y,z] coordinates of the second orbital's center
    nuclear_center : array-like
        [x,y,z] coordinates of the nuclear center
    charge : float
        Nuclear charge (atomic number Z)

    Returns:
    --------
    float
        Value of the nuclear attraction integral
    """
    normalization = ((2 * alpha_a / np.pi) ** (3/4)) * ((2 * alpha_b / np.pi) ** (3/4))
    nuclear_center = np.asarray(nuclear_center)
    center_a = np.asarray(center_a)
    center_b = np.asarray(center_b)

    combined_alpha = alpha_a + alpha_b
    reduced_alpha = (alpha_a * alpha_b) / combined_alpha
    distance_squared = np.sum((center_a - center_b)**2)

    weighted_center = (alpha_a*center_a + alpha_b*center_b)/combined_alpha

    prefactor = (-2*np.pi * charge)/combined_alpha
    exp_term = np.exp(-reduced_alpha * distance_squared)

    t = combined_alpha * np.sum((weighted_center - nuclear_center)**2)
    erf_term = F0(t)

    return normalization * prefactor * exp_term * erf_term

def calculate_orbital_integral(orbital_1, orbital_2, integral_function, *extra_args):
    """
    Calculate the integral between two Gaussian Type Orbitals (GTOs) for any given operator.

    Parameters:
    - orbital_1 (dict): First orbital with keys:
        * 'coeffs' (list): Contraction coefficients
        * 'alphas' (list): Exponential coefficients
        * 'R' (list): Position vector [x, y, z]
    - orbital_2 (dict): Second orbital (same structure as orbital_1)
    - integral_function (callable): Function that computes the primitive integral between two GTOs.
        Must accept arguments: (alpha_a, R_a, alpha_b, R_b, *optional_args)
    - *extra_args: Additional arguments to pass to integral_function

    Returns:
    - float: Total integral value
    """
    integral_sum = 0.0
    for i in range(len(orbital_1['coeffs'])):
        for j in range(len(orbital_2['coeffs'])):
            integral_sum += (
                orbital_1['coeffs'][i] * orbital_2['coeffs'][j] *
                integral_function(
                    orbital_1['alphas'][i], orbital_1['R'],
                    orbital_2['alphas'][j], orbital_2['R'],
                    *extra_args
                )
            )
    return integral_sum

def generate_core_hamiltonian(orbital_list, nat_pos):
    """
    Computes the core Hamiltonian matrix (kinetic energy + nuclear attraction)
    for a given set of orbitals and nuclear positions.

    Args:
        orbital_list (list): List of orbital basis functions
        nat_pos (list): List of nuclear positions and charges in the form:
                       [[[x1,y1,z1], charge1], [[x2,y2,z2], charge2], ...]

    Returns:
        numpy.ndarray: Core Hamiltonian matrix (H_core = T + V) where:
                      - T is the kinetic energy matrix
                      - V is the total nuclear attraction matrix
    """
    num_orbitals = len(orbital_list)

    # Generate Kinetic Overlap
    kinetic_matrix = np.empty((num_orbitals,num_orbitals))
    for i in range(num_orbitals):
        for j in range(num_orbitals):
            kinetic_matrix[i][j] = calculate_orbital_integral(orbital_list[i], orbital_list[j], kinetic_overlap)

    total_nuclear_matrices = np.empty((num_orbitals,num_orbitals))
    for n in range(len(nat_pos)):
        nuclear_matrix = np.empty((num_orbitals,num_orbitals))
        for i in range(num_orbitals):
            for j in range(num_orbitals):
                nuclear_matrix[i][j] = calculate_orbital_integral(orbital_list[i], orbital_list[j], nuclear_attraction_integral,
                                                                 nat_pos[n][0], nat_pos[n][1])
        total_nuclear_matrices = total_nuclear_matrices + nuclear_matrix

    return kinetic_matrix + total_nuclear_matrices

def electron_repulsion(alpha_a, R_a, alpha_b, R_b, alpha_c, R_c, alpha_d, R_d):
    """
    Computes the two-electron repulsion integral (AB|CD) between four Gaussian orbitals.

    Implements equation (A.41) using pre-existing Boys function F0(t).

    Args:
        R_a, R_b, R_c, R_d (array-like): Cartesian coordinates (x,y,z) of centers A, B, C, D
        alpha_a, alpha_b, alpha_c, alpha_d (float): Gaussian exponents

    Returns:
        float: Value of the two-electron repulsion integral (AB|CD)
    """
    # Calculate composite exponents and centers
    alpha_p = alpha_a + alpha_b
    alpha_q = alpha_c + alpha_d
    R_p = (alpha_a*R_a + alpha_b*R_b)/alpha_p
    R_q = (alpha_c*R_c + alpha_d*R_d)/alpha_q

    # Calculate intermediate quantities
    prefactor = (2 * np.pi**(5/2)) / (alpha_p * alpha_q * np.sqrt(alpha_p + alpha_q))

    # Exponential terms
    AB_term = -alpha_a*alpha_b/alpha_p * np.sum((R_a - R_b)**2)
    CD_term = -alpha_c*alpha_d/alpha_q * np.sum((R_c - R_d)**2)

    # Boys function argument
    T = (alpha_p * alpha_q)/(alpha_p + alpha_q) * np.sum((R_p - R_q)**2)

    # Compute integral using existing F0 implementation
    integral = prefactor * np.exp(AB_term + CD_term) * F0(T)

    # Apply normalization constants (assuming primitive Gaussians)
    normalization = (2*alpha_a/np.pi)**(3/4) * (2*alpha_b/np.pi)**(3/4) * \
                    (2*alpha_c/np.pi)**(3/4) * (2*alpha_d/np.pi)**(3/4)

    return normalization * integral

def evaluate_repulsion_integral(orbital_1, orbital_2, orbital_3, orbital_4):
    """
    Computes the two-electron repulsion integral between four contracted Gaussian orbitals.
    Args:
        orbital_1 (dict): First orbital with keys:
            - 'coeffs' (list[float]): Contraction coefficients for primitive Gaussians
            - 'alphas' (list[float]): Exponential coefficients for primitive Gaussians
            - 'R' (np.ndarray): Cartesian coordinates [x, y, z] of orbital center
        orbital_2 (dict): Second orbital (same structure as orbital_1)
        orbital_3 (dict): Third orbital (same structure as orbital_1)
        orbital_4 (dict): Fourth orbital (same structure as orbital_1)

    Returns:
        float: The two-electron repulsion integral value in atomic units
    """
    integral_sum = 0.0

    # Quadruple loop over all primitive Gaussian combinations
    for i in range(len(orbital_1['coeffs'])):
        for j in range(len(orbital_2['coeffs'])):
            for k in range(len(orbital_3['coeffs'])):
                for l in range(len(orbital_4['coeffs'])):
                    # Compute contribution from this primitive combination
                    integral_sum += (
                        orbital_1['coeffs'][i] * orbital_2['coeffs'][j] *
                        orbital_3['coeffs'][k] * orbital_4['coeffs'][l] *
                        electron_repulsion(
                            orbital_1['alphas'][i], orbital_1['R'],
                            orbital_2['alphas'][j], orbital_2['R'],
                            orbital_3['alphas'][k], orbital_3['R'],
                            orbital_4['alphas'][l], orbital_4['R']
                        )
                    )

    return integral_sum
