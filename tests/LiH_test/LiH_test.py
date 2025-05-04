from Hatree_Fock_Code import scf_input
from Hatree_Fock_Code.basis import *

Li_1 = create_orbital(STO_3G_1s_alphas, zetas['Li'], STO_3G_1s_coeff, [0, 0, 0])
Li_2 = create_orbital(STO_3G_2s_alphas, zetas['Li'], STO_3G_2s_coeff, [0, 0, 0])
H_1 = create_orbital(STO_3G_1s_alphas, zetas['H'], STO_3G_1s_coeff, [0, 0, 3.0141129518])

orbital_list = [Li_1, Li_2, H_1]
nat_pos = [ [[0,0,0], 3], [[0,0,3.0141129518], 1] ]
num_elec = 4

energy, final_density, C_matrix = scf_input.run_custom_scf(orbital_list, nat_pos, num_elec)
print(f"Energy for LiH: {energy:.6f} Hartree")

