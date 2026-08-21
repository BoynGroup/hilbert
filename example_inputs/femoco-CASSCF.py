import sys
import os
import time
import json
import numpy as np

# initialize Torch/CUDA
try:
    import torch
    if torch.cuda.is_available():
        _ = torch.zeros(1, device="cuda")
        torch.set_num_threads(1)
        print("  ==> [Debug] PyTorch/CUDA initialized successfully.")
except ImportError:
    pass

def run_femoco():
    import psi4
    import hilbert

    psi4.set_num_threads(32)
    psi4.set_memory('120000 MiB')

    psi4.core.clean()
    psi4.core.clean_options()
    psi4.set_output_file('psi4-output.log', False)

    # generate Femoco object from xyz
    with open('femoco-E0.xyz') as f:
        f.readline()
        f.readline()
        xyz = f.read()
    geom_str_2 = "-5 4\n\n" + xyz + "\n"
    geom_str_2 += "symmetry c1\n"
    psi4.geometry(geom_str_2)

    # Set Psi4 options
    psi4.set_options({
        'basis': 'def2-svp',
        'reference': 'rohf',
        'scf_type': 'disk_df',
        'maxiter': 1,
        'restricted_docc': [242],
        'active': [40],
        'fail_on_maxiter': False,
        'save_uhf_nos': True,
    })

    # set Hilbert v2RDM options
    hilbert_options = {
        'sdp_solver': 'gpu_admm',
        'positivity': 'dqg',
        'maxiter': 30500,
        'orbopt_frequency': 1500,
        'mu_update_frequency': 100,
        'GPU_ADMM_PROFILE': True,
        'optimize_orbitals': True,
        'r_convergence': 1e-4,
        'e_convergence': 1e-4,
        'orbopt_energy_convergence': 1e-6,
        "ORBOPT_GRADIENT_CONVERGENCE": 1.0e-3,
        "WRITE_CHECKPOINT_FILE": True,
        "CHECKPOINT_FILE": "4140.v2rdm.chk",
        "CHECKPOINT_WRITE_MODE": 'FINAL',
        "MOLDEN_WRITE": True,
        "MOLDEN_FILE": "4140.molden",
        'SCF_MAXITER': 25,
        'ORBOPT_MAXITER': 20,
        "ORBOPT_FOCAS_DF_C1_CUDA": True,

    }
    psi4.set_module_options('hilbert', hilbert_options)

    # Need to save DF integrals after initial ROHF
    psi4.core.set_local_option('SCF', 'DF_INTS_IO', 'SAVE')

    # Read in High-Spin UKS NOs as guess orbitals
    ref_wfn_18 = psi4.core.Wavefunction.from_file("./wfn_HS.npy")
    # Dummy ROHF 1-step SCF
    scf_energy, ref_wfn = psi4.energy('scf', return_wfn=True)
    # overwrite Ca and Cb coeffs with UKS NOs
    ref_wfn.Ca().copy(ref_wfn_18.Ca())
    ref_wfn.Cb().copy(ref_wfn_18.Cb())

    # Run v2RDM-CASSCF
    ref_energy, wfn = psi4.energy('v2rdm-casscf', ref_wfn=ref_wfn, return_wfn=True)
    print(f"CASSCF Energy: {ref_energy:.12f} Hartree\n")


if __name__ == "__main__":
    run_femoco()
