import numpy as np

# STO-2G basis set parameters
STO_2G_1s_alphas = np.array([0.151623, 0.851819])
STO_2G_2s_alphas = np.array([0.0974545, 0.384244])
STO_2G_1s_coeff = np.array([0.678914, 0.430129])
STO_2G_2s_coeff = np.array([0.963782, 0.494718])

# STO-3G basis set parameters
STO_3G_1s_alphas = np.array([0.109818, 0.405771, 2.227660])
STO_3G_2s_alphas = np.array([0.0751386, 0.231031, 0.994203])
STO_3G_1s_coeff = np.array([0.444635, 0.535328, 0.154329])
STO_3G_2s_coeff = np.array([0.700115, 0.399513, -0.0999672])

# STO-4G basis set parameters
STO_4G_1s_alphas = np.array([0.0880187, 0.265204, 0.954620, 5.21686])
STO_4G_2s_alphas = np.array([0.0628104, 0.163451, 0.502989, 2.21250])
STO_4G_1s_coeff = np.array([0.291626, 0.532846, 0.260141, 0.0567523])
STO_4G_2s_coeff = np.array([0.497767, 0.558855, 0.0000297680, -0.0622071])

#Zeta Parameters for first 4 atoms
zetas = {
    'H' : 1.24,
    'He': 2.0925,
    'Li': 2.69,
    'Be':3.68
}

def create_orbital(alpha, zeta, coeff, position):
    """
    Creates an orbital dictionary with scaled exponents (alphas = alpha * zeta^2).

    Parameters:
    - alpha (np.array): Base exponents for the Gaussian primitives.
    - zeta (float): Scaling factor for the exponents (alphas = alpha * zeta^2).
    - coeff (np.array): Contraction coefficients for the primitives.
    - position (list or np.array): [x, y, z] position of the orbital center.

    Returns:
    - dict: Orbital dictionary with keys {'coeffs', 'alphas', 'R'}.
    """
    scaled_alphas = alpha * (zeta ** 2)  # Scale exponents by zeta^2
    orbital = {
        'coeffs': np.array(coeff),
        'alphas': scaled_alphas,
        'R': np.array(position)  # Ensure position is a numpy array
    }
    return orbital
