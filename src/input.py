from basis import *
from scf_run import run_scf, prepare_scf_inputs

# Here you will create all the orbitals necessary for your SCF
# If you choose to use an Na or Be atom, then make sure to create a 1s and 2s orbital centered on the atom
# Use 1s orbitals for H and He
# Change zeta to the corresponding atom for the orbital
# STO_2G, STO_3G, and STO-4G are all available basis sets

Be_1s = create_orbital(STO_3G_1s_alphas, zetas['Be'], STO_3G_1s_coeff, [0,0,0])
Be_2s = create_orbital(STO_3G_2s_alphas, zetas['Be'], STO_3G_2s_coeff, [0,0,0])
H1_1s = create_orbital(STO_3G_1s_alphas, zetas['H'], STO_3G_1s_coeff, [1.2906,0,0])
H2_1s = create_orbital(STO_3G_1s_alphas, zetas['H'], STO_3G_1s_coeff, [-1.2906,0,0])

# Define molecule parameters:

# For orbital_list, just create a list of all the orbitals you made
orbital_list = [Be_1s, Be_2s, H1_1s, H2_1s]

# For nat_pos, just make an array of arrays where each entry is of the form [[x, y, z], atomic_number]
# This array defines the nuclear positions and charges in the molecule
nat_pos = [ [[0,0,0], 4], [[1.2906,0,0], 1], [[-1.2906,0,0], 1] ]

# Number of electrons in your system
num_elec = 6

# Maximum number of iterations before stopping self-consistency check
max_scf = 50

# Convergence threshold of your SCF evaluations. Converges SCF when the energy difference between
# successive SCF iterations is less than conv_thr Hartrees
conv_thr = 1e-6

# Prepare matrices and run SCF. DO NOT TOUCH
H_core, repulsion_matrix, S_inv_sqrt = prepare_scf_inputs(orbital_list, nat_pos)
energy, final_density, C_matrix = run_scf(H_core, repulsion_matrix, S_inv_sqrt, num_elec, orbital_list, nat_pos, max_scf, conv_thr)

