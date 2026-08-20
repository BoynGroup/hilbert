import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if os.path.basename(parent_dir) == 'hilbert-dev':
    sys.path.insert(0, os.path.dirname(parent_dir))
else:
    sys.path.insert(0, parent_dir)

import psi4
import hilbert

def run_gpu_admm_casscf():
    print(f"\n=======================================================")
    print(f"Testing v2RDM-CASSCF with SDP_SOLVER = GPU_ADMM")
    print(f"=======================================================")
    psi4.core.clean()
    psi4.core.clean_options()

    n2 = psi4.geometry("""
    0 1
    n
    n 1 r
    """)

    psi4.set_options({
      'basis':           'cc-pvdz',
      'scf_type':        'disk_df',
      'd_convergence':   1e-10,
      'maxiter':         500,
      'restricted_docc': [ 2, 0, 0, 0, 0, 2, 0, 0 ],
      'active':          [ 1, 0, 1, 1, 0, 1, 1, 1 ],
      'r_convergence':   1e-5,
      'e_convergence':   1e-4,
    })
    psi4.set_module_options('hilbert', {
      'sdp_solver':      'gpu_admm',
      'positivity':      'dqg',
      'maxiter':         20000,
    })

    n2.r = 1.1
    refscf = -108.95348837831371
    # Reference energy from BPSDP/CVXPY
    refv2rdm = -109.09440

    # save three-index integrals after scf
    psi4.core.set_local_option('SCF', 'DF_INTS_IO', 'SAVE')

    # get scf wfn
    scf_energy, ref_wfn = psi4.energy('scf', return_wfn=True)

    # grab options object
    options = psi4.core.get_options()
    options.set_current_module('HILBERT')

    # evaluate v2RDM CASSCF energy
    v2rdm = hilbert.v2RDMHelper(ref_wfn, options)
    current_energy = v2rdm.compute_energy()

    print(f"SCF energy: {scf_energy}")
    print(f"v2RDM CASSCF energy: {current_energy}")
    
    assert psi4.compare_values(refscf, scf_energy, 8, "SCF total energy")
    assert psi4.compare_values(refv2rdm, current_energy, 4, "v2RDM-CASSCF total energy")
    print("Test passed successfully!")

if __name__ == '__main__':
    run_gpu_admm_casscf()
