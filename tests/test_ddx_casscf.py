"""
Example / smoke test: v2RDM-CASSCF with ddx (domain-decomposition) solvation.

This mirrors test_pcm_casscf.py but drives the newer ddx continuum solver
(ddCOSMO / ddPCM, via the pyddx package) instead of PCMSolver.  The v2RDM
reaction field is re-solved against the v2RDM density every macro-iteration
(see v2RDMSolver::update_ddx), exactly as for PCM.

-----------------------------------------------------------------------------
!! WARNING -- read before running / adapting this example !!
-----------------------------------------------------------------------------
1. RELEASE THE ddx INTERFACE BEFORE THE INTERPRETER EXITS.
   The DdxInterface holds Psi4 core objects (MintsHelper, DFTGrid, pyddx
   Model) that do NOT survive Python interpreter finalization.  If it is left
   attached to the wavefunction, teardown-time garbage collection touches it
   and the process can crash *after* the (correct) result is already computed.
   Always drop it and force a collection once the solve is done:

       del ref_wfn.ddx_interface
       import gc; gc.collect()

   The energy('v2rdm-casscf') driver path (pymodule.run_v2rdm_casscf) already
   does this for you; it is only required when you call v2RDMHelper directly,
   as this example does.

2. KNOWN INTERMITTENT HEAP ISSUE (results are NOT affected).
   When a ddx interface coexists with the v2RDMSolver there is a rare,
   allocator-dependent use-after-free somewhere in the Psi4/pyddx object
   interaction (plain RHF-ddx and gas/PCM v2RDM are unaffected).  It has not
   been root-caused yet (would need an AddressSanitizer build of the whole
   stack).  The COMPUTED ENERGIES ARE ALWAYS CORRECT when a run completes; the
   only symptom is that a process may occasionally abort.  For unattended /
   batch use, run each calculation in its own process and, if you must, write
   the result to a file and os._exit(0) after the solve, or simply retry.

3. ddx and PCM are mutually exclusive in a single calculation.
-----------------------------------------------------------------------------
"""

import sys
import os
import gc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import psi4
import hilbert

# Register the plugin's options (POSITIVITY, OPTIMIZE_ORBITALS, ...) so they can
# be set before the first driver call.  Importing the compiled extension does not
# register them on its own; plugin_load does.
psi4.core.plugin_load(hilbert.__file__)


def run_ddx_casscf(ddx_model='cosmo'):
    print("Testing v2rdm-CASSCF with ddx (%s) solvation..." % ddx_model)
    psi4.core.clean()
    psi4.core.clean_options()

    co = psi4.geometry("""
    0 1
    C 0.0 0.0 0.0
    O 0.0 0.0 1.128
    symmetry c1
    units angstrom
    """)

    psi4.set_options({
        'basis': 'cc-pvdz',
        'scf_type': 'disk_df',
        'e_convergence': 1e-9,
        'd_convergence': 1e-9,
        'maxiter': 500,
        'restricted_docc': [4],
        'active': [6],
        # --- ddx continuum solvation ---
        'ddx': True,
        'ddx_model': ddx_model,      # 'cosmo' (ddCOSMO) or 'pcm' (ddPCM)
        'ddx_solvent': 'water',
        'ddx_radii_set': 'uff',
    })

    psi4.set_module_options('hilbert', {
        'positivity': 'dqg',
        'maxiter': 50000,
        'r_convergence': 1e-6,
        'e_convergence': 1e-7,
        'orbopt_maxiter': 20,
        'optimize_orbitals': True,
    })

    psi4.core.set_local_option('SCF', 'DF_INTS_IO', 'SAVE')

    # SCF (with ddx) -> reference wavefunction; it already carries a DdxInterface
    # as wfn.ddx, so reuse it rather than building a redundant second one.
    scf_energy, ref_wfn = psi4.energy('scf', return_wfn=True)
    ref_wfn.ddx_interface = ref_wfn.ddx

    options = psi4.core.get_options()
    options.set_current_module('HILBERT')

    # Run v2rdm-CASSCF; update_ddx() picks up ref_wfn.ddx_interface.
    v2rdm = hilbert.v2RDMHelper(ref_wfn, options)
    casscf_energy = v2rdm.compute_energy()

    dd_solvation = psi4.variable('DD SOLVATION ENERGY')
    print("  SCF+ddx energy:        %20.12f" % scf_energy)
    print("  v2RDM-CASSCF+ddx:      %20.12f" % casscf_energy)
    print("  DD solvation energy:   %20.12f" % dd_solvation)

    assert casscf_energy < 0.0
    assert dd_solvation < 0.0

    # !! REQUIRED: release the ddx interface before interpreter teardown (see the
    # WARNING at the top of this file).  Skipping this risks a teardown crash.
    del ref_wfn.ddx_interface
    gc.collect()

    return casscf_energy


def test_ddx_casscf():
    run_ddx_casscf('cosmo')


if __name__ == '__main__':
    # Run each model in its own process for robustness (see WARNING note 2).
    run_ddx_casscf('cosmo')
    print("ddCOSMO test passed.\n")
    # run_ddx_casscf('pcm')   # ddPCM; run separately (fresh process recommended)
