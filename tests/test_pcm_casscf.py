import sys
import os
import psi4

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import hilbert

def test_pcm_casscf():
    print("Testing v2rdm-CASSCF with PCM solvation...")
    psi4.core.clean()
    psi4.core.clean_options()

    co = psi4.geometry("""
    0 1
    C 0.0 0.0 0.0
    O 0.0 0.0 2.0
    symmetry c1
    """)

    pcm_input = """
    Units = Angstrom
    Medium {
        SolverType = IEFPCM
        Solvent = Water
    }
    Cavity {
        RadiiSet = UFF
        Type = GePol
        Scaling = False
        Area = 0.3
        Mode = Implicit
    }
    """

    psi4.set_options({
        'basis': 'sto-3g',
        'scf_type': 'disk_df',
        'e_convergence': 1e-8,
        'r_convergence': 1e-6,
        'maxiter': 500,
        'pcm': True,
        'pcm_scf_type': 'total',
        'PCM__INPUT': pcm_input,
        'restricted_docc': [4],
        'active': [6],
    })

    psi4.set_module_options('hilbert', {
        'positivity': 'dqg',
        'maxiter': 2000,
        'optimize_orbitals': True,
    })

    psi4.core.set_local_option('SCF', 'DF_INTS_IO', 'SAVE')

    # get SCF + PCM wavefunction
    scf_energy, ref_wfn = psi4.energy('scf', return_wfn=True)

    # grab options
    options = psi4.core.get_options()
    options.set_current_module('HILBERT')

    # Run v2rdm-CASSCF
    v2rdm = hilbert.v2RDMHelper(ref_wfn, options)
    casscf_energy = v2rdm.compute_energy()

    print(f"SCF+PCM Energy: {scf_energy}")
    print(f"CASSCF+PCM Energy: {casscf_energy}")
    pcm_polarization = psi4.variable('PCM POLARIZATION ENERGY')
    print(f"PCM Polarization Energy: {pcm_polarization}")

    assert casscf_energy < 0.0
    assert pcm_polarization < 0.0
    print("Test passed successfully!")

if __name__ == '__main__':
    test_pcm_casscf()
