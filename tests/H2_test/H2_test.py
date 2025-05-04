from Hatree_Fock_Code import scf_input
from Hatree_Fock_Code.basis import *

# Define a new molecule, e.g., H2
H1 = create_orbital(STO_3G_1s_alphas, zetas['H'], STO_3G_1s_coeff, [0.7, 0, 0])
H2 = create_orbital(STO_3G_1s_alphas, zetas['H'], STO_3G_1s_coeff, [-0.7, 0, 0])

orbital_list = [H1, H2]
nat_pos = [ [[0.7,0,0], 1], [[-0.7,0,0], 1] ]
num_elec = 2

energy, final_density, C_matrix = scf_input.run_custom_scf(orbital_list, nat_pos, num_elec)
print(f"Energy for H2: {energy:.6f} Hartree")

