# Simple_Hartree_Fock

**Simple_Hartree_Fock** is a lightweight Python package for computing restricted closed-shell Hartree-Fock (HF) wavefunctions using minimal basis sets. It supports the STO-nG basis sets (n = 2, 3, or 4) and works with atoms H, He, Li, and Be — assuming only *s*-type orbitals.

## Features

- Restricted Hartree-Fock (RHF) implementation for closed-shell systems  
- Support for STO-2G, STO-3G, and STO-4G basis sets   
- Minimal atomic support: H, He, Li, Be  
- Modifiable Python interface for defining custom molecules  

---

## Installation

After cloning the repo, install the package as:

```bash
cd Simple_Hartree_Fock
pip install -e .
```

---

## How to Use

Navigate to the main input script located at:

```
src/Hartree_Fock_Code/scf_input.py
```

This script demonstrates how to define custom molecular systems and run SCF calculations using the `run_custom_scf()` function.

### Example Usage

Here's a minimal example included in `scf_input.py` for a BeH₂-like molecule:

```python
from .basis import create_orbital, STO_3G_1s_alphas, STO_3G_2s_alphas, STO_3G_1s_coeff, STO_3G_2s_coeff, zetas
from .scf_run import run_scf, prepare_scf_inputs
from .scf_input import run_custom_scf

# Define orbitals using basis parameters and atomic positions
Be_1s = create_orbital(STO_3G_1s_alphas, zetas['Be'], STO_3G_1s_coeff, [0, 0, 0])
Be_2s = create_orbital(STO_3G_2s_alphas, zetas['Be'], STO_3G_2s_coeff, [0, 0, 0])
H1_1s = create_orbital(STO_3G_1s_alphas, zetas['H'], STO_3G_1s_coeff, [1.29, 0, 0])
H2_1s = create_orbital(STO_3G_1s_alphas, zetas['H'], STO_3G_1s_coeff, [-1.29, 0, 0])

# Combine orbitals
orbital_list = [Be_1s, Be_2s, H1_1s, H2_1s]

# Define nuclear positions and charges: [position, nuclear_charge]
nat_pos = [ [[0, 0, 0], 4], [[1.29, 0, 0], 1], [[-1.29, 0, 0], 1] ]

# Define total number of electrons
num_elec = 6

# Run the SCF calculation
energy, final_density, C_matrix = run_custom_scf(orbital_list, nat_pos, num_elec)
print(f"SCF converged energy: {energy:.6f} Hartree")
```

---

## Input Format

To set up a custom molecule, you must define:

1. **Orbitals** using `create_orbital()` with:
   - Basis function exponents (alphas)
   - Contraction coefficients
   - Atomic positions
   - Atom-specific zeta values

2. **Nuclear configuration** as a list of `[position, charge]` pairs.

3. **Electron count** for the total number of electrons (must be even for RHF).

---

## Notes

- This code does **not** implement *d*- or *p*-type orbitals, open-shell systems, or post-Hartree-Fock methods.
- Make sure your system is charge-neutral and has an even number of electrons for proper RHF calculations.

---

## License

MIT License
