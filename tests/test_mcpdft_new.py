import sys
import os

# Insert workspace root to import local hilbert
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import psi4
import hilbert

# Set memory
psi4.set_memory('2 GB')

# Set up molecule
molecule = psi4.geometry("""
0 1
n
n 1 1.1
""")

# Set global options
psi4.set_options({
    'basis': 'cc-pvdz',
    'scf_type': 'disk_df',
    'd_convergence': 1e-10,
    'maxiter': 500,
    'restricted_docc': [2, 0, 0, 0, 0, 2, 0, 0],
    'active': [1, 0, 1, 1, 0, 1, 1, 1],
})

# Set local options for HILBERT
psi4.set_module_options('HILBERT', {
    'positivity': 'dqg',
    'opdm_write_full': True,
    'tpdm_write_full': True,
    'constrain_spin': True,
    'r_convergence': 1e-5,
    'e_convergence': 1e-5,
    'cg_convergence': 1e-6,
    'maxiter': 1000000,
    'optimize_orbitals': True,
    'mu_update_frequency': 500,
    'mcpdft_reference': 'v2rdm',
    'mcpdft_type': 'df',
    'mcpdft_functional': 'blyp',
    'mcpdft_method': 'mcpdft',
})

print("Running v2rdm-casscf baseline...")
# run v2rdm-casscf
ref_energy, wfn = psi4.energy('v2rdm-casscf', return_wfn=True)
print(f"CASSCF Energy: {ref_energy:.12f} Hartree\n")

# reference should be uks in mcpdft
psi4.set_options({'reference': 'uks'})

# 1. Run BLYP (standard)
print("1. Running standard BLYP MC-PDFT...")
psi4.core.set_local_option('HILBERT', 'MCPDFT_FUNCTIONAL', 'blyp')
blyp_energy = psi4.energy('mcpdft', ref_wfn=wfn)
print(f"BLYP MC-PDFT Energy: {blyp_energy:.12f} Hartree\n")

# 2. Run tM06L
print("2. Running tM06L MC-PDFT...")
psi4.core.set_local_option('HILBERT', 'MCPDFT_FUNCTIONAL', 'tm06l')
tm06l_energy = psi4.energy('mcpdft', ref_wfn=wfn)
print(f"tM06L MC-PDFT Energy: {tm06l_energy:.12f} Hartree\n")

# 3. Run tM06L with hybrid lambda (e.g. lambda = 0.25)
print("3. Running Hybrid tM06L MC-PDFT (lambda = 0.25 via options)...")
psi4.core.set_local_option('HILBERT', 'MCPDFT_LAMBDA', 0.25)
hybrid_energy_opt = psi4.energy('mcpdft', ref_wfn=wfn)
print(f"Hybrid tM06L Energy (options): {hybrid_energy_opt:.12f} Hartree")
# Expected manual mix: 0.25 * ref_energy + 0.75 * tm06l_energy
expected_hybrid = 0.25 * ref_energy + 0.75 * tm06l_energy
print(f"Expected Hybrid Energy:         {expected_hybrid:.12f} Hartree")
diff = abs(hybrid_energy_opt - expected_hybrid)
print(f"Difference:                    {diff:.12f} Hartree\n")

# Reset lambda to 0.0
psi4.core.set_local_option('HILBERT', 'MCPDFT_LAMBDA', 0.0)

# 4. Run tM06L with custom functional parameters for exchange and/or correlation
print("4. Running tM06L with custom functional parameters...")
# Let's perturb the first parameter of exchange and correlation
# Exchange default: _a0 = 0.3987756
custom_x = [0.5] + [0.2548219, 0.3923994, -2.103655, -6.302147, 10.97615, 30.97273, -23.18489, -56.7348, 21.60364, 34.21814, -9.049762, 0.6012244, 0.004748822, -0.008635108, -9.308062e-06, 4.482811e-05, 0.0]
# Correlation default: _gamma_ss = 0.06
custom_c = [0.1] + [0.0031, 0.00515088, 0.00304966, 0.5349466, 0.539662, -31.61217, 51.49592, -29.19613, 0.6042374, 177.6783, -251.3252, 76.35173, -12.55699, 0.4650534, 0.1617589, 0.1833657, 0.00046921, -0.004990573, 0.0, 0.3957626, -0.5614546, 0.01403963, 0.0009831442, -0.003577176, 0.0, 1e-10]

custom_energy = psi4.energy('mcpdft', ref_wfn=wfn, custom_params_x=custom_x, custom_params_c=custom_c)
print(f"Custom Parameters Energy: {custom_energy:.12f} Hartree")
print(f"Difference from standard tM06L: {abs(custom_energy - tm06l_energy):.12f} Hartree\n")

# 5. Run MC23 (hardcoded parameters & lambda)
print("5. Running MC23 MC-PDFT...")
psi4.core.set_local_option('HILBERT', 'MCPDFT_FUNCTIONAL', 'mc23')
mc23_energy = psi4.energy('mcpdft', ref_wfn=wfn)
print(f"MC23 MC-PDFT Energy: {mc23_energy:.12f} Hartree\n")

# 6. Run MC25 (hardcoded parameters & lambda)
print("6. Running MC25 MC-PDFT...")
psi4.core.set_local_option('HILBERT', 'MCPDFT_FUNCTIONAL', 'mc25')
mc25_energy = psi4.energy('mcpdft', ref_wfn=wfn)
print(f"MC25 MC-PDFT Energy: {mc25_energy:.12f} Hartree\n")

print("All tests completed!")
