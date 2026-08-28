#
# @BEGIN LICENSE
#
# Hilbert: a space for quantum chemistry plugins to Psi4
#
# Copyright (c) 2020 by its authors (LICENSE).
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see http://www.gnu.org/licenses/.
#
# @END LICENSE
#

import numpy as np
import os
import hashlib

import psi4
import psi4.driver.p4util as p4util
from psi4.driver.procrouting import proc_util
from psi4.driver.procrouting import proc
from psi4.driver.p4util.exceptions import ValidationError

_DQG_C1_CUDA_AU_MODULE = None
_DQG_C1_CUDA_NORMAL_MODULE = None
_DQG_SYM_CUDA_MODULE = None


def _load_dqg_c1_cuda_module(source_name, module_prefix, label, verbose=False):
    from torch.utils.cpp_extension import load

    module_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(module_dir, "src", "misc", source_name),
        os.path.join(os.getcwd(), "src", "misc", source_name),
        os.path.join(module_dir, source_name),
        os.path.join(os.getcwd(), source_name),
    ]
    source_path = next((path for path in candidates if os.path.exists(path)), None)
    if source_path is None:
        raise RuntimeError(
            f"{label} requested, but "
            f"{source_name} was not found. Searched: {candidates}"
        )

    with open(source_path, "rb") as handle:
        source_hash = hashlib.sha1(handle.read()).hexdigest()[:12]
    module_name = f"{module_prefix}_{source_hash}"

    print(
        f"  ==> GPU-ADMM: Loading {label} extension "
        f"{module_name} from {source_path}",
        flush=True,
    )
    module = load(
        name=module_name,
        sources=[source_path],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=bool(verbose),
    )
    version_fn = (
        getattr(module, "dqg_c1_au_version", None) or
        getattr(module, "dqg_sym_version", None) or
        (lambda: "unknown")
    )
    version = version_fn()
    print(
        f"  ==> GPU-ADMM: Loaded {label} extension "
        f"{module_name} ({version}) from {source_path}",
        flush=True,
    )
    return module


def _load_dqg_c1_cuda_au_module(verbose=False):
    global _DQG_C1_CUDA_AU_MODULE
    if _DQG_C1_CUDA_AU_MODULE is None:
        _DQG_C1_CUDA_AU_MODULE = _load_dqg_c1_cuda_module(
            "gpu_admm_dqg_c1_au_cuda.cu",
            "hilbert_dqg_c1_au_cuda",
            "CUDA Au/ATu",
            verbose=verbose,
        )
    return _DQG_C1_CUDA_AU_MODULE


def _load_dqg_c1_cuda_normal_module(verbose=False):
    global _DQG_C1_CUDA_NORMAL_MODULE
    if _DQG_C1_CUDA_NORMAL_MODULE is None:
        _DQG_C1_CUDA_NORMAL_MODULE = _load_dqg_c1_cuda_module(
            "gpu_admm_dqg_c1_normal_cuda.cu",
            "hilbert_dqg_c1_normal_cuda",
            "CUDA normal-operator",
            verbose=verbose,
        )
    return _DQG_C1_CUDA_NORMAL_MODULE


def _load_dqg_sym_cuda_module(verbose=False):
    global _DQG_SYM_CUDA_MODULE
    if _DQG_SYM_CUDA_MODULE is None:
        _DQG_SYM_CUDA_MODULE = _load_dqg_c1_cuda_module(
            "gpu_admm_dqg_sym_cuda.cu",
            "hilbert_dqg_sym_cuda",
            "CUDA symmetry Au/ATu",
            verbose=verbose,
        )
    return _DQG_SYM_CUDA_MODULE

def init_cc_cavity(name, **kwargs):

    # set method to CC_CAVITY
    psi4.core.set_local_option('HILBERT', 'HILBERT_METHOD', 'CC_CAVITY')

    # determine level of theory from name
    if 'ccsd-00' in name:
        psi4.core.set_local_option('HILBERT', 'QED_CC_TYPE', 'CCSD-00')
    elif 'ccsd-21' in name:
        psi4.core.set_local_option('HILBERT', 'QED_CC_TYPE', 'CCSD-21')
    elif 'ccsd-22' in name:
        psi4.core.set_local_option('HILBERT', 'QED_CC_TYPE', 'CCSD-22')

    # check if coupling strength is zero
    if np.allclose(psi4.core.get_option('HILBERT', 'CAVITY_COUPLING_STRENGTH'), 0.0):
        psi4.core.set_local_option('HILBERT', 'QED_CC_TYPE', 'CCSD-00')
        psi4.core.set_local_option('HILBERT', 'CAVITY_FREQUENCY', [0.0, 0.0, 1000.0])

    # determine if using eom-cc or ground-state CC
    if 'eom' in name:
        psi4.core.set_local_option('HILBERT', 'PERFORM_EOM', True)

        # check if 'eom-ea' is in the name and set EOM_TYPE to EA if so
        if 'eom-ea' in name:
            psi4.core.set_local_option('HILBERT', 'EOM_TYPE', 'EA')
    else:
        psi4.core.set_local_option('HILBERT', 'PERFORM_EOM', False)

    # set the number of threads for MADNESS with TiledArray
    try:
        mad_num_threads = str(psi4.core.get_local_option('HILBERT', 'MAD_NUM_THREADS'))
    except:
        # check if the number of threads is set in the environment; if not, set to 1
        mad_num_threads = os.environ.get('MAD_NUM_THREADS', 1)

    # set environment variable for MAD_NUM_THREADS
    os.environ['MAD_NUM_THREADS'] = mad_num_threads

    # get MPI communicator
    try:
        from mpi4py import MPI
        from hilbert import set_comm
        set_comm(MPI.COMM_WORLD)
    except ImportError:
        raise Exception('Hilbert is not compiled with TA support. Please recompile with `-D WITH_TA` cmake flag.')

    # upon exit, finalize MPI
    from atexit import register
    @register
    def cleanup():
        from hilbert import ta_finalize
        ta_finalize()


def run_qed_scf(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    qed-scf can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('qed-scf')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    optstash = p4util.OptionsState(
        ['SCF', 'DF_INTS_IO'])

    psi4.core.set_local_option('SCF', 'DF_INTS_IO', 'SAVE')

    reference = psi4.core.get_global_option('REFERENCE').lower()

    if ( lowername == 'qed-scf'):
        if ( reference == 'rhf'):
            psi4.core.set_local_option('HILBERT', 'HILBERT_METHOD', 'POLARITONIC_RHF')
        elif ( reference == 'rohf'):
            psi4.core.set_local_option('HILBERT', 'HILBERT_METHOD', 'POLARITONIC_ROHF')
        else:
            psi4.core.set_local_option('HILBERT', 'HILBERT_METHOD', 'POLARITONIC_UHF')
    elif ( lowername == 'qed-dft' ):
        if ( reference == 'rks' or reference == 'rhf'):
            psi4.core.set_local_option('HILBERT', 'HILBERT_METHOD', 'POLARITONIC_RKS')
        else:
            psi4.core.set_local_option('HILBERT', 'HILBERT_METHOD', 'POLARITONIC_UKS')
    elif ( lowername == 'qed-cis' ):
        psi4.core.set_local_option('HILBERT', 'HILBERT_METHOD', 'POLARITONIC_RCIS')
    elif ( lowername == 'qed-ccsd' ):
        psi4.core.set_local_option('HILBERT', 'HILBERT_METHOD', 'POLARITONIC_UCCSD')
    elif ( lowername == 'qed-tddft' ):
        if ( reference == 'rks' or reference == 'rhf'):
            psi4.core.set_local_option('HILBERT', 'HILBERT_METHOD', 'POLARITONIC_RTDDFT')
        else:
            psi4.core.set_local_option('HILBERT', 'HILBERT_METHOD', 'POLARITONIC_UTDDFT')
    elif ( 'qed-ccsd' in lowername): # cc_cavity
        init_cc_cavity(name, **kwargs)

    # Compute a SCF reference, a wavefunction is return which holds the molecule used, orbitals
    # Fock matrices, and more
    #print('Attention! This SCF may be density-fitted.')
    ref_wfn = kwargs.get('ref_wfn', None)
    #if ref_wfn is None:
    #    ref_wfn = psi4.driver.scf_helper(name, **kwargs)
    if ref_wfn is None:
        if ( lowername == 'qed-dft' or lowername == 'qed-tddft'):
            func = psi4.core.get_option('HILBERT','QED_DFT_FUNCTIONAL')
            en, ref_wfn = psi4.driver.energy(func, **kwargs, return_wfn=True)
        else :
            ref_wfn = psi4.driver.scf_helper(name, **kwargs)

    scf_aux_basis = psi4.core.BasisSet.build(ref_wfn.molecule(), "DF_BASIS_SCF",
                                        psi4.core.get_option("SCF", "DF_BASIS_SCF"),
                                        "JKFIT", psi4.core.get_global_option('BASIS'),
                                        puream=ref_wfn.basisset().has_puream())
    ref_wfn.set_basisset("DF_BASIS_SCF", scf_aux_basis)

    aux_basis = psi4.core.BasisSet.build(ref_wfn.molecule(), "DF_BASIS_CC",
                                        psi4.core.get_global_option("DF_BASIS_CC"),
                                        "RIFIT", psi4.core.get_global_option("BASIS"),
                                        puream=ref_wfn.basisset().has_puream())
    ref_wfn.set_basisset("DF_BASIS_CC", aux_basis)

    # Ensure IWL files have been written when not using DF/CD
    scf_type = psi4.core.get_option('SCF', 'SCF_TYPE')
    if ( scf_type == 'PK' or scf_type == 'DIRECT' ):
        proc_util.check_iwl_file_from_scf_type(psi4.core.get_option('SCF', 'SCF_TYPE'), ref_wfn)

    # Call the Psi4 plugin
    # Please note that setting the reference wavefunction in this way is ONLY for plugins
    rhf_wfn = psi4.core.plugin('hilbert.so', ref_wfn)

    optstash.restore()

    return rhf_wfn

def run_qed_scf_gradient(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    qed-scf can be called via :py:func:`~driver.gradient`. For post-scf plugins.

    >>> energy('qed-scf')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    optstash = p4util.OptionsState(
        ['SCF', 'DF_INTS_IO'])

    psi4.core.set_local_option('SCF', 'DF_INTS_IO', 'SAVE')

    reference = psi4.core.get_global_option('REFERENCE').lower()

    if ( lowername == 'qed-scf'):
        if ( reference == 'rhf'):
            psi4.core.set_local_option('HILBERT', 'HILBERT_METHOD', 'POLARITONIC_RHF')
        elif ( reference == 'rohf'):
            psi4.core.set_local_option('HILBERT', 'HILBERT_METHOD', 'POLARITONIC_ROHF')
        else:
            psi4.core.set_local_option('HILBERT', 'HILBERT_METHOD', 'POLARITONIC_UHF')
    elif ( lowername == 'qed-dft' ):
        if ( reference == 'rks' or reference == 'rhf'):
            psi4.core.set_local_option('HILBERT', 'HILBERT_METHOD', 'POLARITONIC_RKS')
        else:
            psi4.core.set_local_option('HILBERT', 'HILBERT_METHOD', 'POLARITONIC_UKS')
    elif ( lowername == 'qed-cis' ):
        psi4.core.set_local_option('HILBERT', 'HILBERT_METHOD', 'POLARITONIC_RCIS')
    elif ( lowername == 'qed-ccsd' ):
        psi4.core.set_local_option('HILBERT', 'HILBERT_METHOD', 'POLARITONIC_UCCSD')
    elif ( lowername == 'qed-tddft' ):
        if ( reference == 'rks' or reference == 'rhf'):
            psi4.core.set_local_option('HILBERT', 'HILBERT_METHOD', 'POLARITONIC_RTDDFT')
        else:
            psi4.core.set_local_option('HILBERT', 'HILBERT_METHOD', 'POLARITONIC_UTDDFT')
    elif ( 'qed-ccsd' in lowername): # cc_cavity
        init_cc_cavity(name, **kwargs)


    # Compute a SCF reference, a wavefunction is return which holds the molecule used, orbitals
    # Fock matrices, and more
    #print('Attention! This SCF may be density-fitted.')
    ref_wfn = kwargs.get('ref_wfn', None)
    if ref_wfn is None:
        if ( lowername == 'qed-dft' or lowername == 'qed-tddft' ):

            # get functional from options
            func = psi4.core.get_option('HILBERT','QED_DFT_FUNCTIONAL')

            # check if dertype is present in kwargs and handle accordingly
            try:
                dertype = kwargs.pop('dertype') # must remove dertype for energy call if present
            except:
                 dertype = "gradient" # set default dertype for analytic gradient if not present

            # call energy and grab wfn
            en, ref_wfn = psi4.driver.energy(func, **kwargs, return_wfn=True)
            kwargs['dertype'] = dertype # restore dertype for gradient call
        else :
            ref_wfn = psi4.driver.scf_helper(name, **kwargs)

    scf_aux_basis = psi4.core.BasisSet.build(ref_wfn.molecule(), "DF_BASIS_SCF",
                                        psi4.core.get_option("SCF", "DF_BASIS_SCF"),
                                        "JKFIT", psi4.core.get_global_option('BASIS'),
                                        puream=ref_wfn.basisset().has_puream())
    ref_wfn.set_basisset("DF_BASIS_SCF", scf_aux_basis)

    aux_basis = psi4.core.BasisSet.build(ref_wfn.molecule(), "DF_BASIS_CC",
                                         psi4.core.get_global_option("DF_BASIS_CC"),
                                         "RIFIT", psi4.core.get_global_option("BASIS"),
                                         puream=ref_wfn.basisset().has_puream())
    ref_wfn.set_basisset("DF_BASIS_CC", aux_basis)

    # Ensure IWL files have been written when not using DF/CD
    scf_type = psi4.core.get_option('SCF', 'SCF_TYPE')
    if ( scf_type == 'PK' or scf_type == 'DIRECT' ):
        proc_util.check_iwl_file_from_scf_type(psi4.core.get_option('SCF', 'SCF_TYPE'), ref_wfn)

    # Call the Psi4 plugin
    # Please note that setting the reference wavefunction in this way is ONLY for plugins
    rhf_wfn = psi4.core.plugin('hilbert.so', ref_wfn)

    # check if reference wave function is restricted
    if ("rks" in lowername or "rhf" in lowername or "rohf" in lowername):
        # copy alpha quantities to beta quantities in polaritonic wave function
        for irrep in range (0,ref_wfn.Cb().nirrep()):
            rhf_wfn.Cb().nph[irrep][:,:] = rhf_wfn.Ca().nph[irrep][:,:]
            rhf_wfn.Db().nph[irrep][:,:] = rhf_wfn.Da().nph[irrep][:,:]
            rhf_wfn.epsilon_b().nph[irrep][:] = rhf_wfn.epsilon_a().nph[irrep][:]

    # gradient of photon-free hamiltonian

    # some quantities aren't set correctly in hilbert's wave functions, so we can't call
    # scfgrad directly. to get the photon-free part of the gradient, just push 
    # (i)   polaritonic-scf orbitals 
    # (ii)  polaritonic-scf orbital energies
    # (iii) polaritonic-scf densities
    # onto reference wave function 

    # set alpha orbitals, densities, and energies
    for irrep in range (0,ref_wfn.Ca().nirrep()):
        ref_wfn.Ca().nph[irrep][:,:] = rhf_wfn.Ca().nph[irrep][:,:]
        ref_wfn.Cb().nph[irrep][:,:] = rhf_wfn.Cb().nph[irrep][:,:]
        ref_wfn.Da().nph[irrep][:,:] = rhf_wfn.Da().nph[irrep][:,:]
        ref_wfn.Db().nph[irrep][:,:] = rhf_wfn.Db().nph[irrep][:,:]
        ref_wfn.epsilon_a().nph[irrep][:] = rhf_wfn.epsilon_a().nph[irrep][:]
        ref_wfn.epsilon_b().nph[irrep][:] = rhf_wfn.epsilon_b().nph[irrep][:]

    #### call scfgrad for electron-only part of gradient ####
    gradient = psi4.core.scfgrad(ref_wfn)

    #### dipole self energy portion of gradient ####

    # OPDM
    Da = np.asarray(rhf_wfn.Da())
    Db = np.asarray(rhf_wfn.Db())

    # dipole integrals
    mints = psi4.core.MintsHelper(ref_wfn.basisset())
    dipole = mints.so_dipole()

    mu_z = np.asarray(dipole[2])
    if ( psi4.core.get_option("HILBERT","ROTATE_POLARIZATION_AXIS") == "YZX" ):
        mu_z = np.asarray(dipole[0])
    if ( psi4.core.get_option("HILBERT","ROTATE_POLARIZATION_AXIS") == "ZXY" ):
        mu_z = np.asarray(dipole[1])

    # exchange contribution to dipole self energy 

    #### D(p,q) = - mu(r,s) [ Da(p,r)Da(s,q) + Db(p,r) Da(s,q) ] ####

    tmpa = -np.einsum('rs,pr,sq->pq',mu_z, Da, Da) 
    tmpb = -np.einsum('rs,pr,sq->pq',mu_z, Db, Db)

    # test exchange energy from dressed RDM
    g = psi4.core.get_option("HILBERT","CAVITY_COUPLING_STRENGTH")
    w = psi4.core.get_option("HILBERT","CAVITY_FREQUENCY")
    lambda_z = g[2] * np.sqrt(2.0 * w[2])

    en  = 0.5 * lambda_z * lambda_z * np.einsum('pq,pq',tmpa,mu_z)
    en += 0.5 * lambda_z * lambda_z * np.einsum('pq,pq',tmpb,mu_z)

    D = tmpa + tmpb

    # symmetrize D because dipole_grad only uses 1/2 the elements
    D = 0.5 * ( D + np.einsum('rs->sr',D) )

    D = psi4.core.Matrix.from_array(D)

    # number of atoms
    mol = psi4.core.get_active_molecule()
    natom = mol.natom()

    tmp = mints.dipole_grad(D)
    dse_gradient = np.asarray(tmp)

    # unpack z-component 3N x 3 matrix (the third column)
    dse_gradient_z = np.zeros((natom,3))
    zdir = 2
    if ( psi4.core.get_option("HILBERT","ROTATE_POLARIZATION_AXIS") == "YZX" ):
        zdir = 0
    if ( psi4.core.get_option("HILBERT","ROTATE_POLARIZATION_AXIS") == "ZXY" ):
        zdir = 1
    for atom in range (0,natom):
        for cart in range (0,3):
            dse_gradient_z[atom,cart] = dse_gradient[atom*3+cart,zdir] 

    # scale by lambda^2
    dse_gradient_z_scaled = psi4.core.Matrix.from_array(dse_gradient_z)
    dse_gradient_z_scaled.scale(lambda_z*lambda_z)

    #### quadrupole integral gradient ####
      
    C = [0.0, 0.0, 0.0] # origin
    maxorder = 2 # quadrupole
    D = Da + Db # OPDM
    
    # symmetrize D because dipole_grad only uses 1/2 the elements
    D = 0.5 * ( D + np.einsum('rs->sr',D) )
    D = psi4.core.Matrix.from_array(D)
    
    # 3N x 9 matrix of quadrupole derivatives
    quad_grad = np.asarray(mints.multipole_grad(D, maxorder, C))
    
    # get requested component of quadrupole gradient
    zzdir = 8 # zz component
    if ( psi4.core.get_option("HILBERT","ROTATE_POLARIZATION_AXIS") == "YZX" ):
        zzdir = 3 # xx component
    if ( psi4.core.get_option("HILBERT","ROTATE_POLARIZATION_AXIS") == "ZXY" ):
        zzdir = 6 # yy component

    # unpack zz-component 3N x 3 matrix (the 9th column)
    dse_gradient_zz = np.zeros((natom,3))
    for atom in range (0,natom):
        for cart in range (0,3):
            dse_gradient_zz[atom,cart] = quad_grad[atom*3+cart,zzdir]
    
    dse_gradient_z_scaled_2 = psi4.core.Matrix.from_array(dse_gradient_zz)
    dse_gradient_z_scaled_2.scale(-0.5 * lambda_z*lambda_z)
    dse_gradient_z_scaled.add(dse_gradient_z_scaled_2)

    #### print out gradients ####

    # electronic gradient
    eg_norm = np.linalg.norm(gradient)
    eg_norm_xyz = np.linalg.norm(gradient, axis=0)
    psi4.core.print_out(f"\nElectronic Gradient: norm = {eg_norm:-20.12f}\n\n") # total norm
    gradient.print_out()
    psi4.core.Vector.from_array(eg_norm_xyz, name="Electronic Gradient |xyz|").print_out() # norm along each axis

    # total polaritonic gradient
    gradient.add(dse_gradient_z_scaled)
    pg_norm = np.linalg.norm(gradient)
    pg_norm_xyz = np.linalg.norm(gradient, axis=0)
    psi4.core.print_out(f"\nPolaritonic Gradient: norm = {pg_norm:-20.12f}\n\n") # total norm
    gradient.print_out() 
    psi4.core.Vector.from_array(pg_norm_xyz, name="Polaritonic Gradient |xyz|").print_out() # norm along each axis

    # difference between polaritonic and electronic gradients
    psi4.core.print_out(f"Gradient Difference: norm = {pg_norm-eg_norm:-20.12f}\n") # total norm
    pg_norm_xyz -= eg_norm_xyz
    psi4.core.Vector.from_array(pg_norm_xyz, name="Gradient Difference |xyz|").print_out() # norm along each axis

    optstash.restore()

    # set the gradient and return the wavefunction
    rhf_wfn.set_gradient(gradient)
    return rhf_wfn

def run_doci(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    doci can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('doci')

    """
    kwargs = p4util.kwargs_lower(kwargs)

    optstash = p4util.OptionsState(
        ['SCF', 'DF_INTS_IO'])

    psi4.core.set_local_option('SCF', 'DF_INTS_IO', 'SAVE')

    psi4.core.set_local_option('HILBERT', 'HILBERT_METHOD', 'DOCI')

    # Compute a SCF reference, a wavefunction is return which holds the molecule used, orbitals
    # Fock matrices, and more
    #print('Attention! This SCF may be density-fitted.')
    ref_wfn = kwargs.get('ref_wfn', None)
    if ref_wfn is None:
        ref_wfn = psi4.driver.scf_helper(name, **kwargs)

    # Ensure IWL files have been written when not using DF/CD
    scf_type = psi4.core.get_option('SCF', 'SCF_TYPE')
    if ( scf_type == 'PK' or scf_type == 'DIRECT' ):
        proc_util.check_iwl_file_from_scf_type(psi4.core.get_option('SCF', 'SCF_TYPE'), ref_wfn)

    # Call the Psi4 plugin
    # Please note that setting the reference wavefunction in this way is ONLY for plugins
    doci_wfn = psi4.core.plugin('hilbert.so', ref_wfn)

    optstash.restore()

    return doci_wfn

def run_pp2rdm(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    pp2rdm can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('pp2rdm')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    optstash = p4util.OptionsState(
        ['SCF', 'DF_INTS_IO'])

    psi4.core.set_local_option('SCF', 'DF_INTS_IO', 'SAVE')

    psi4.core.set_local_option('HILBERT', 'HILBERT_METHOD', 'PP2RDM')

    if lowername == 'pp2rdm':
        psi4.core.set_local_option('HILBERT', 'P2RDM_TYPE', 'K')
    elif lowername == 'pccd':
        psi4.core.set_local_option('HILBERT', 'P2RDM_TYPE', 'CCD')
    elif lowername == 'pcid':
        psi4.core.set_local_option('HILBERT', 'P2RDM_TYPE', 'CID')
    elif lowername == 'pcepa(1)':
        psi4.core.set_local_option('HILBERT', 'P2RDM_TYPE', 'CEPA(1)')
    elif lowername == 'pcepa(0)':
        psi4.core.set_local_option('HILBERT', 'P2RDM_TYPE', 'CEPA(0)')
    elif lowername == 'pacpf':
        psi4.core.set_local_option('HILBERT', 'P2RDM_TYPE', 'ACPF')
    elif lowername == 'paqcc':
        psi4.core.set_local_option('HILBERT', 'P2RDM_TYPE', 'AQCC')

    # Compute a SCF reference, a wavefunction is return which holds the molecule used, orbitals
    # Fock matrices, and more
    #print('Attention! This SCF may be density-fitted.')
    ref_wfn = kwargs.get('ref_wfn', None)
    if ref_wfn is None:
        ref_wfn = psi4.driver.scf_helper(name, **kwargs)

    # Ensure IWL files have been written when not using DF/CD
    scf_type = psi4.core.get_option('SCF', 'SCF_TYPE')
    if ( scf_type == 'PK' or scf_type == 'DIRECT' ):
        proc_util.check_iwl_file_from_scf_type(psi4.core.get_option('SCF', 'SCF_TYPE'), ref_wfn)

    # Call the Psi4 plugin
    # Please note that setting the reference wavefunction in this way is ONLY for plugins
    pp2rdm_wfn = psi4.core.plugin('hilbert.so', ref_wfn)

    optstash.restore()

    return pp2rdm_wfn

def run_v2rdm_doci(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    v2rdm_doci can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('v2rdm_doci')

    """

    kwargs = p4util.kwargs_lower(kwargs)

    optstash = p4util.OptionsState(
        ['SCF', 'DF_INTS_IO'])

    psi4.core.set_local_option('HILBERT', 'HILBERT_METHOD', 'V2RDM_DOCI')

    psi4.core.set_local_option('SCF', 'DF_INTS_IO', 'SAVE')

    # Your plugin's psi4 run sequence goes here
    ref_wfn = kwargs.get('ref_wfn', None)
    if ref_wfn is None:
        ref_wfn = psi4.driver.scf_helper(name, **kwargs)

    # Ensure IWL files have been written when not using DF/CD
    scf_type = psi4.core.get_option('SCF', 'SCF_TYPE')
    if ( scf_type == 'PK' or scf_type == 'DIRECT' ):
        proc_util.check_iwl_file_from_scf_type(psi4.core.get_option('SCF', 'SCF_TYPE'), ref_wfn)

    v2rdm_doci_wfn = psi4.core.plugin('hilbert.so', ref_wfn)

    optstash.restore()

    return v2rdm_doci_wfn

def run_v2rdm_casscf(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    v2rdm_casscf can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('v2rdm_casscf')

    """

    kwargs = p4util.kwargs_lower(kwargs)

    optstash = p4util.OptionsState(
        ['SCF', 'DF_INTS_IO'])

    psi4.core.set_local_option('SCF', 'DF_INTS_IO', 'SAVE')

    psi4.core.set_local_option('HILBERT', 'HILBERT_METHOD', 'V2RDM_CASSCF')

    # Your plugin's psi4 run sequence goes here
    ref_wfn = kwargs.get('ref_wfn', None)
    if ref_wfn is None:
        ref_wfn = psi4.driver.scf_helper(name, **kwargs)

    # Ensure IWL files have been written when not using DF/CD
    scf_type = psi4.core.get_option('SCF', 'SCF_TYPE')
    if ( scf_type == 'PK' or scf_type == 'DIRECT' ):
        proc_util.check_iwl_file_from_scf_type(psi4.core.get_option('SCF', 'SCF_TYPE'), ref_wfn)

    # reorder wavefuntions based on user input
    # apply a list of 2x2 rotation matrices to the orbitals in the form of [irrep, orbital1, orbital2, theta]
    # where an angle of 0 would do nothing and an angle of 90 would switch the two orbitals.
    # the indices of irreps and orbitals start from 0
    reorder_orbitals = psi4.core.get_option("HILBERT","MCSCF_ROTATE")
    for orbord in reorder_orbitals:
        if type(orbord) != list :
            raise psi4.p4util.PsiException("Each element of the orbtial rotate list requires 4 arguements (irrep, orb1, orb2, theta).")
        if len(orbord) != 4:
            raise psi4.p4util.PsiException("Each element of the orbtial rotate list requires 4 arguements (irrep, orb1, orb2, theta).")

        irrep, orb1, orb2, theta = orbord

        if irrep > ref_wfn.Ca().nirrep():
            raise psi4.p4util.PsiException("REORDER_ORBITALS: Expression %s irrep number is larger than the number of irreps" %
                                    (str(orbord)))

        if max(orb1, orb2) > ref_wfn.Ca().coldim()[irrep]:
            raise psi4.p4util.PsiException("REORDER_ORBITALS: Expression %s orbital number exceeds number of orbitals in irrep" %
                                    (str(orbord)))

        theta = np.deg2rad(theta)

        x_a = ref_wfn.Ca().nph[irrep][:, orb1].copy()
        y_a = ref_wfn.Ca().nph[irrep][:, orb2].copy()

        xp_a = np.cos(theta) * x_a - np.sin(theta) * y_a
        yp_a = np.sin(theta) * x_a + np.cos(theta) * y_a

        ref_wfn.Ca().nph[irrep][:, orb1] = xp_a
        ref_wfn.Ca().nph[irrep][:, orb2] = yp_a

        # beta orbitals are not changed because v2rdm does not use beta orbitals
        # if/when beta orbitals are used, you should change beta orbitals too,
        # but keep in mind that in RHF wavefunctions, beta orbital pointer may
        # point to the same location as alpha orbital pointer

    # --- ddx (domain-decomposition) implicit solvent -----------------------
    # PCM is flagged on the wavefunction by Psi4's SCF and picked up in C++ via
    # PCM_enabled().  ddx has no such C++ Wavefunction hook, so we build the
    # DdxInterface here and hand it to the plugin as a Python attribute; the
    # v2RDMSolver re-solves the reaction field against the v2RDM density each
    # macro-iteration (see v2RDMSolver::update_ddx).  DDX and PCM are mutually
    # exclusive (get_ddx_options enforces this).
    used_ddx = psi4.core.get_option('SCF', 'DDX')
    if used_ddx:
        from psi4.driver.procrouting.solvent.ddx import DdxInterface, get_ddx_options
        mol = ref_wfn.molecule()
        ref_wfn.ddx_interface = DdxInterface(mol, get_ddx_options(mol),
                                             ref_wfn.basisset())

    returnvalue = psi4.core.plugin('hilbert.so', ref_wfn)

    # Release the DdxInterface as soon as the solve is done.  It holds Psi4 core
    # objects (MintsHelper, DFTGrid, pyddx Model) that do not survive Python
    # interpreter finalization; leaving it attached to the wavefunction lets it
    # reach teardown-time GC and crash there.  Unlike PCM (a C++ object destroyed
    # with the wavefunction) this is a plain Python attribute, so we drop it here.
    # It also sits in a reference cycle (MintsHelper <-> basis set), so a forced
    # collection now -- while the interpreter is fully alive -- is what actually
    # frees it; dropping the reference alone is not enough.
    if used_ddx and hasattr(ref_wfn, 'ddx_interface'):
        del ref_wfn.ddx_interface
        import gc
        gc.collect()

    optstash.restore()

    return returnvalue

def run_v2rdm_casscf_gradient(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    v2rdm_casscf can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> gradient('v2rdm_casscf')

    """

    kwargs = p4util.kwargs_lower(kwargs)

    optstash = p4util.OptionsState(
        ['GLOBALS', 'DERTYPE'],
        ['HILBERT', 'OPTIMIZE_ORBITALS'],
        ['HILBERT', 'SEMICANONICALIZE_ORBITALS'],
        ['HILBERT', 'ORBOPT_ACTIVE_ACTIVE_ROTATIONS'],
        ['HILBERT', 'RESTART_FROM_CHECKPOINT_FILE'],
        ['HILBERT', 'WRITE_CHECKPOINT_FILE'])

    psi4.core.set_global_option('DERTYPE', 'FIRST')
    psi4.core.set_local_option("HILBERT","OPTIMIZE_ORBITALS",True)
    psi4.core.set_local_option("HILBERT","ORBOPT_ACTIVE_ACTIVE_ROTATIONS",True)
    psi4.core.set_local_option("HILBERT","SEMICANONICALIZE_ORBITALS",False)
    psi4.core.set_local_option("HILBERT","RESTART_FROM_CHECKPOINT_FILE","DUMMY")
    psi4.core.set_local_option("HILBERT","WRITE_CHECKPOINT_FILE",True)

    # analytic derivatives do not work with scf_type df/cd
    scf_type = psi4.core.get_option('SCF', 'SCF_TYPE')
    if ( scf_type == 'CD' or scf_type == 'DF' ):
        raise ValidationError("""Error: analytic v2RDM-CASSCF gradients not implemented for scf_type %s.""" % scf_type)

    v2rdm_wfn = run_v2rdm_casscf(name,**kwargs)
    derivobj = psi4.core.Deriv(v2rdm_wfn)
    derivobj.set_deriv_density_backtransformed(True)
    derivobj.set_ignore_reference(True)
    grad = derivobj.compute()

    v2rdm_wfn.set_gradient(grad)

    optstash.restore()

    return v2rdm_wfn

def run_p2rdm(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    p2rdm can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('p2rdm')

    """
    kwargs = p4util.kwargs_lower(kwargs)

    optstash = p4util.OptionsState(
        ['SCF', 'DF_INTS_IO'])

    psi4.core.set_local_option('SCF', 'DF_INTS_IO', 'SAVE')

    psi4.core.set_local_option('HILBERT', 'HILBERT_METHOD', 'P2RDM')

    #if lowername == 'p2rdm':
    #    psi4.core.set_local_option('HILBERT', 'P2RDM_TYPE', 'K')
    #elif lowername == 'cid':
    #    psi4.core.set_local_option('HILBERT', 'P2RDM_TYPE', 'CID')
    #elif lowername == 'cepa(0)':
    #    psi4.core.set_local_option('HILBERT', 'P2RDM_TYPE', 'CEPA(0)')

    # Compute a SCF reference, a wavefunction is return which holds the molecule used, orbitals
    # Fock matrices, and more
    #print('Attention! This SCF may be density-fitted.')
    ref_wfn = kwargs.get('ref_wfn', None)
    if ref_wfn is None:
        ref_wfn = psi4.driver.scf_helper(name, **kwargs)

    # Ensure IWL files have been written when not using DF/CD
    scf_type = psi4.core.get_option('SCF', 'SCF_TYPE')
    if ( scf_type == 'PK' or scf_type == 'DIRECT' ):
        proc_util.check_iwl_file_from_scf_type(psi4.core.get_option('SCF', 'SCF_TYPE'), ref_wfn)

    # Call the Psi4 plugin
    # Please note that setting the reference wavefunction in this way is ONLY for plugins
    p2rdm_wfn = psi4.core.plugin('hilbert.so', ref_wfn)

    optstash.restore()

    return p2rdm_wfn

def run_jellium_scf(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    jellium_scf can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('jellium-scf')

    """
    kwargs = p4util.kwargs_lower(kwargs)

    # build empty reference wavefunction to pass into plugin

    #ref_molecule = kwargs.get('molecule', psi4.core.get_active_molecule())
    mol = """H 0 0 0
    H 0 0 1"""
    ref_molecule = psi4.core.Molecule.from_string(mol)
    base_wfn = psi4.core.Wavefunction.build(ref_molecule, 'STO-3G')
    ref_wfn = proc.scf_wavefunction_factory('HF', base_wfn, psi4.core.get_global_option('REFERENCE'))

    psi4.core.set_local_option('HILBERT', 'HILBERT_METHOD', 'JELLIUM_SCF')

    jellium_scf_wfn = psi4.core.plugin('hilbert.so', ref_wfn)

    return jellium_scf_wfn

def density_analysis(**kwargs):
    r"""Function to evaluate real-space density"""

    kwargs = p4util.kwargs_lower(kwargs)

    psi4.core.set_local_option('SCF', 'DF_INTS_IO', 'SAVE')

    ref_wfn = kwargs.get('ref_wfn', None)
    if ref_wfn is None:
        raise ValidationError("""Error: density_analysis requires a reference wave function.""" )

    func = 'M06-2X'
    ref_molecule = kwargs.get('molecule', psi4.core.get_active_molecule())
    base_wfn = psi4.core.Wavefunction.build(ref_molecule, psi4.core.get_global_option('BASIS'))
    new_wfn = proc.scf_wavefunction_factory(func, base_wfn, 'UKS')

    # push reference orbitals onto new wave function 
    for irrep in range (0,ref_wfn.Ca().nirrep()):
        new_wfn.Ca().nph[irrep][:,:] = ref_wfn.Ca().nph[irrep][:,:]
        new_wfn.Cb().nph[irrep][:,:] = ref_wfn.Cb().nph[irrep][:,:]

    # push reference energies onto new wave function
    for irrep in range (0,ref_wfn.epsilon_a().nirrep()):
        new_wfn.epsilon_a().nph[irrep][:] = ref_wfn.epsilon_a().nph[irrep][:]
        new_wfn.epsilon_b().nph[irrep][:] = ref_wfn.epsilon_b().nph[irrep][:]

    # grab options object
    options = psi4.core.get_options()
    options.set_current_module('HILBERT')

    # build real-space density
    import hilbert
    real_space_density = hilbert.RealSpaceDensityHelper(new_wfn,options)
    real_space_density.read_opdm()
    real_space_density.build_rho()

    return real_space_density

def run_mcpdft(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    mcpdft can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('mcpdft')

    """
    kwargs = p4util.kwargs_lower(kwargs)

    # pylibxc
    try:
        import pylibxc
    except ImportError:
        print('')
        print('    error: mc-pdft requires the python interface to libxc. see https://gitlab.com/libxc/libxc/-/tree/devel#python-library')
        print('')
        exit()


    functional_name_dict = {
        'svwn' : ['lda_x', 'lda_c_vwn_rpa'],
        'lda' : ['lda_x', None],
        'blyp' : ['gga_x_b88', 'gga_c_lyp'],
        'bop' : ['gga_x_b88', 'gga_c_op_b88'],
        'pbe' : ['gga_x_pbe', 'gga_c_pbe'],
        'revpbe': ['gga_x_pbe_r', 'gga_c_pbe'],
        'tm06l': ['mgga_x_m06_l', 'mgga_c_m06_l'],
        'mc23': ['mgga_x_m06_l', 'mgga_c_m06_l'],
        'mc25': ['mgga_x_m06_l', 'mgga_c_m06_l'],
    }
    functional = psi4.core.get_option('HILBERT','MCPDFT_FUNCTIONAL').lower()

    if functional not in functional_name_dict.keys():
        raise ValidationError(f"Invalid functional choice {name} for MC-PDFT. try {functional_name_dict.keys()}")

    libxc_functional_name = functional_name_dict[functional]

    psi4.core.set_local_option('SCF', 'DF_INTS_IO', 'SAVE')

    ref_wfn = kwargs.get('ref_wfn', None)
    if ref_wfn is None:
        raise ValidationError("""Error: mcpdft requires a reference wave function.""" )

    # pick fake functional that requires second derivatives for new_wfn
    func = 'M06-2X'
    ref_molecule = kwargs.get('molecule', psi4.core.get_active_molecule())
    base_wfn = psi4.core.Wavefunction.build(ref_molecule, psi4.core.get_global_option('BASIS'))
    new_wfn = proc.scf_wavefunction_factory(func, base_wfn, 'UKS')

    # push reference orbitals onto new wave function
    for irrep in range (0,ref_wfn.Ca().nirrep()):
        new_wfn.Ca().nph[irrep][:,:] = ref_wfn.Ca().nph[irrep][:,:]
        new_wfn.Cb().nph[irrep][:,:] = ref_wfn.Cb().nph[irrep][:,:]

    # push reference energies onto new wave function
    for irrep in range (0,ref_wfn.epsilon_a().nirrep()):
        new_wfn.epsilon_a().nph[irrep][:] = ref_wfn.epsilon_a().nph[irrep][:]
        new_wfn.epsilon_b().nph[irrep][:] = ref_wfn.epsilon_b().nph[irrep][:]

    # grab options object
    options = psi4.core.get_options()
    options.set_current_module('HILBERT')

    psi4.core.flush_outfile()
    psi4.core.print_out('\n\n')
    psi4.core.print_out('        ********************************************************************\n')
    psi4.core.print_out('        *                                                                  *\n')
    psi4.core.print_out('        *    MC-PDFT:                                                      *\n')
    psi4.core.print_out('        *                                                                  *\n')
    psi4.core.print_out('        *    Multiconfigurational Pair Density Functional Theory           *\n')
    psi4.core.print_out('        *                                                                  *\n')
    psi4.core.print_out('        ********************************************************************\n')
    psi4.core.print_out('\n')

    import hilbert
    rho_helper = hilbert.RealSpaceDensityHelper(new_wfn, options)

    # get MO-basis opdm from disk or user input

    opdm_a = kwargs.get('opdm_a', None)
    opdm_b = kwargs.get('opdm_b', None)
    if opdm_a is None or opdm_b is None:
        rho_helper.read_opdm()
    else:
        rho_helper.set_opdm(opdm_a, opdm_b)

    # build real-space density from OPDM
    rho_helper.build_rho()

    # need Da and Db to build T+V+J. rho_helper has them in the MO basis
    Da = rho_helper.Da()
    Db = rho_helper.Db()

    # T + V
    mints = psi4.core.MintsHelper(new_wfn.basisset())

    # T
    Ta = mints.so_kinetic()
    Tb = Ta.clone()

    Ta.transform(new_wfn.Ca())
    Tb.transform(new_wfn.Cb())

    kinetic_energy = Da.vector_dot(Ta)
    kinetic_energy += Db.vector_dot(Tb)

    # V
    Va = mints.so_potential()
    Vb = Va.clone()

    Va.transform(new_wfn.Ca())
    Vb.transform(new_wfn.Cb())

    en_potential_energy = Da.vector_dot(Va)
    en_potential_energy += Db.vector_dot(Vb)

    # classical coulomb energy, J
    jk = psi4.core.JK.build(new_wfn.get_basisset("ORBITAL"),
                           aux=new_wfn.get_basisset("DF_BASIS_SCF"))

    jk.set_memory(int(5e8)) # 4GB of memory
    jk.set_do_K(False)
    jk.set_do_wK(False)
    jk.initialize()

    Cra = new_wfn.Ca().clone()
    Crb = new_wfn.Cb().clone()

    Cla = Cra.clone()
    Clb = Crb.clone()

    Cla.zero();
    Cla.gemm(False, True, 1.0, Cra, Da, 0.0);
    jk.C_left_add(Cla)
    jk.C_right_add(Cra)

    Clb.zero();
    Clb.gemm(False, True, 1.0, Crb, Db, 0.0);
    jk.C_left_add(Clb)
    jk.C_right_add(Crb)

    jk.compute()

    Ja = jk.J()[0]
    Jb = jk.J()[1]

    Ja.transform(new_wfn.Ca())
    Jb.transform(new_wfn.Cb())

    coulomb_energy = Da.vector_dot(Ja)
    coulomb_energy += Da.vector_dot(Jb)
    coulomb_energy += Db.vector_dot(Ja)
    coulomb_energy += Db.vector_dot(Jb)
    coulomb_energy *= 0.5

    # xc contribution to the energy

    # density in real space
    rho_a = np.asarray(rho_helper.rho_a())
    rho_b = np.asarray(rho_helper.rho_b())
    rho = rho_a + rho_b

    # gradient of density in real space
    rho_a_x = np.asarray(rho_helper.rho_a_x())
    rho_a_y = np.asarray(rho_helper.rho_a_y())
    rho_a_z = np.asarray(rho_helper.rho_a_z())

    rho_b_x = np.asarray(rho_helper.rho_b_x())
    rho_b_y = np.asarray(rho_helper.rho_b_y())
    rho_b_z = np.asarray(rho_helper.rho_b_z())

    rho_x = rho_a_x + rho_b_x
    rho_y = rho_a_y + rho_b_y
    rho_z = rho_a_z + rho_b_z

    # get MO-basis alpha-beta block of tpdm from disk or user input

    tpdm_ab = kwargs.get('tpdm_ab', None)
    if tpdm_ab is None:
        rho_helper.read_tpdm()
    else:
        rho_helper.set_tpdm(tpdm_ab)

    # on-top pair density in real space
    pi = np.asarray(rho_helper.pi())

    # on-top ratio
    R = 4.0 * np.divide(pi, rho * rho)

    # translated densities and gradients of densities

    # rhoa = [1 + zeta] * rho / 2
    # rhob = [1 - zeta] * rho / 2
    # zeta = sqrt(1-R), where 1-R > 0, 0 otherwise
    zeta = np.sqrt(1.0 - R, out = np.zeros_like(R), where = 1.0 - R > 0 )

    rho_a =  0.5 * rho * (1.0 + zeta)
    rho_b =  0.5 * rho * (1.0 - zeta)

    # translated gradients
    rho_a_x =  0.5 * rho_x * (1.0 + zeta)
    rho_a_y =  0.5 * rho_y * (1.0 + zeta)
    rho_a_z =  0.5 * rho_z * (1.0 + zeta)

    rho_b_x =  0.5 * rho_x * (1.0 - zeta)
    rho_b_y =  0.5 * rho_y * (1.0 - zeta)
    rho_b_z =  0.5 * rho_z * (1.0 - zeta)

    # get kinetic energy density
    tau_a = np.asarray(rho_helper.tau_a())
    tau_b = np.asarray(rho_helper.tau_b())
    tau = tau_a + tau_b

    # translated kinetic energy density
    tau_a = 0.5 * tau * (1.0 + zeta)
    tau_b = 0.5 * tau * (1.0 - zeta)

    combined_tau = np.zeros([2 * len(rho)])
    combined_tau[::2] = tau_a
    combined_tau[1::2] = tau_b

    # with translated rho_a, rho_b, etc. evaluate xc contribution to the energy

    # we need grids for ex/ec
    grid_w = np.asarray(rho_helper.grid_w())

    # combined rho as rho_a[0], rho_b[0], rho_a[1], rho_b[1], etc.
    combined_rho = np.zeros([2 * len(rho)])
    combined_rho[::2] = rho_a
    combined_rho[1::2] = rho_b

    # contracted gradient as drho.drho / aa[0], ab[0], bb[0], aa[1], ab[1], bb[1], etc.
    sigma = np.zeros([3 * len(rho)])

    sigma_aa = rho_a_x * rho_a_x +  rho_a_y * rho_a_y +  rho_a_z * rho_a_z
    sigma_ab = rho_a_x * rho_b_x +  rho_a_y * rho_b_y +  rho_a_z * rho_b_z
    sigma_bb = rho_b_x * rho_b_x +  rho_b_y * rho_b_y +  rho_b_z * rho_b_z

    sigma[::3] = sigma_aa
    sigma[1::3] = sigma_ab
    sigma[2::3] = sigma_bb

    inp = {
        "rho" : combined_rho,
        "sigma" : sigma,
        "lapl" : None,
        "tau" : combined_tau
    }

    ex = 0.0
    ec = 0.0

    # custom functional parameters & hybrid lambda
    mcpdft_lambda = kwargs.get('mcpdft_lambda', None)
    if mcpdft_lambda is None:
        opt_lambda = psi4.core.get_option('HILBERT', 'MCPDFT_LAMBDA')
        if opt_lambda != 0.0:
            mcpdft_lambda = opt_lambda
        elif functional == 'mc23':
            mcpdft_lambda = 0.2952
        elif functional == 'mc25':
            mcpdft_lambda = 0.28
        else:
            mcpdft_lambda = 0.0

    custom_params_x = kwargs.get('custom_params_x', None)
    if custom_params_x is None:
        opt_params_x = psi4.core.get_option('HILBERT', 'MCPDFT_CUSTOM_PARAMS_X')
        if opt_params_x:
            custom_params_x = [float(val) for val in opt_params_x.split()]
        elif functional == 'mc23':
            custom_params_x = [3.352197e+00, 6.332929e-01, -9.469553e-01, 2.030835e-01, 2.503819e+00, 8.085354e-01, -3.619144e+00, -5.572321e-01, -4.506606e+00, 9.614774e-01, 6.977048e+00, -1.309337e+00, -2.426371e+00, -7.896540e-03, 1.364510e-02, -1.714252e-06, -4.698672e-05, 0.0]
        elif functional == 'mc25':
            custom_params_x = [3.465030e+00, 5.375447e-01, -7.191629e-01, -9.915646e-01, 2.229138e+00, 5.404209e+00, -4.004898e+00, -5.983860e+00, -2.086931e+00, -8.878196e-01, 4.888665e+00, 2.868958e+00, -2.499939e+00, -9.852771e-03, 8.596984e-03, -1.220706e-05, -1.336275e-05, 0.0]

    custom_params_c = kwargs.get('custom_params_c', None)
    if custom_params_c is None:
        opt_params_c = psi4.core.get_option('HILBERT', 'MCPDFT_CUSTOM_PARAMS_C')
        if opt_params_c:
            custom_params_c = [float(val) for val in opt_params_c.split()]
        elif functional == 'mc23':
            custom_params_c = [0.06, 0.0031, 0.00515088, 0.00304966, 2.427648e+00, 3.707473e+00, -7.943377e+00, -2.521466e+00, 2.658691e+00, 2.932276e+00, -8.832841e-01, -1.895247e+00, -2.899644e+00, -5.068570e-01, -2.712838e+00, 9.416102e-02, -3.485860e-03, -5.811240e-04, 6.668814e-04, 0.0, 2.669169e-01, -7.563289e-02, 7.036292e-02, 3.493904e-04, 6.360837e-04, 0.0, 1e-10]
        elif functional == 'mc25':
            custom_params_c = [0.06, 0.0031, 0.00515088, 0.00304966, 1.489435e+00, 2.942442e+00, -6.297330e+00, -2.062505e+00, 1.634904e+00, 1.608843e+00, -1.233955e+00, -1.964674e+00, -2.471985e+00, -5.392796e-01, -1.509794e+00, 2.794569e-02, 1.061909e-01, 5.095118e-04, -2.927055e-03, 0.0, 9.690385e-01, -4.546714e-02, 4.151718e-02, 1.789189e-04, 1.024388e-03, 0.0, 1e-10]

    if libxc_functional_name[0] is not None:

        functional = pylibxc.LibXCFunctional(libxc_functional_name[0], "polarized")
        if custom_params_x is not None:
            functional.set_ext_params(custom_params_x)
        ret = functional.compute( inp, do_vxc = False )
        zk = ret['zk'].flatten()
        ex = np.sum( zk * rho * grid_w )

    if libxc_functional_name[1] is not None:

        functional = pylibxc.LibXCFunctional(libxc_functional_name[1], "polarized")
        if custom_params_c is not None:
            functional.set_ext_params(custom_params_c)
        ret = functional.compute( inp, do_vxc = False )
        zk = ret['zk'].flatten()
        ec = np.sum( zk * rho * grid_w )

    nuclear_repulsion_energy = new_wfn.molecule().nuclear_repulsion_energy()

    total_energy = kinetic_energy + en_potential_energy + coulomb_energy + ex + ec + nuclear_repulsion_energy

    psi4.core.flush_outfile()
    psi4.core.print_out('    ==> MC-PDFT energy by component <==\n')
    psi4.core.print_out('\n')

    psi4.core.print_out('        nuclear repulsion energy =          %20.12f\n' % (nuclear_repulsion_energy) )
    psi4.core.print_out('        electron-nucleus potential energy = %20.12f\n' % (en_potential_energy) )
    psi4.core.print_out('        electron kinetic energy =           %20.12f\n' % (kinetic_energy) )
    psi4.core.print_out('        classical coulomb energy  =         %20.12f\n' % (coulomb_energy) )
    psi4.core.print_out('        exchange energy =                   %20.12f\n' % (ex) )
    psi4.core.print_out('        correlation energy =                %20.12f\n' % (ec) )
    psi4.core.print_out('\n')
    psi4.core.print_out('    * MC-PDFT total energy   =        %20.12f\n\n' % (total_energy));

    if mcpdft_lambda != 0.0:
        ref_energy = ref_wfn.energy()
        hybrid_energy = mcpdft_lambda * ref_energy + (1.0 - mcpdft_lambda) * total_energy
        psi4.core.print_out('    * Reference CASSCF energy =       %20.12f\n' % (ref_energy))
        psi4.core.print_out('    * Hybrid MC-PDFT energy  =        %20.12f (lambda = %6.4f)\n\n' % (hybrid_energy, mcpdft_lambda))
        final_energy = hybrid_energy
    else:
        final_energy = total_energy

    psi4.core.set_variable('CURRENT ENERGY', final_energy)

    return final_energy


def cvxpy_solve(c, A, b, block_dims, solver_name, verbose=False):
    import cvxpy as cp

    # 1. Create semidefinite variables for each block
    vars = []
    for d in block_dims:
        if d > 0:
            vars.append(cp.Variable((d, d), PSD=True))

    # 2. Flatten and concatenate variables to form primal variable vector x_vec
    x_vec = cp.hstack([v.flatten(order='C') for v in vars])

    # 3. Define constraints
    constraints = [A @ x_vec == b]

    # 4. Formulate optimization problem
    prob = cp.Problem(cp.Minimize(c @ x_vec), constraints)

    # 5. Solve the problem
    if solver_name:
        if solver_name not in cp.installed_solvers():
            raise ValueError(f"Requested CVXPY solver '{solver_name}' is not installed. Installed: {cp.installed_solvers()}")
        prob.solve(solver=solver_name, verbose=verbose)
    else:
        prob.solve(verbose=verbose)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"CVXPY solve failed with status: {prob.status}")

    # 6. Retrieve optimal primal and dual values
    x_opt = x_vec.value
    # Negate dual values due to sign convention difference
    y_opt = -constraints[0].dual_value

    return x_opt, y_opt


class _DQGC1MatrixFreeOperator:
    """Matrix-free DQG/C1 operator based on src/v2rdm_casscf/sparse_a.cc."""

    def __init__(self, meta, block_dims, n_primal, n_dual, device, dtype,
                 cuda_au=False, cuda_atu=False, cuda_normal=False,
                 cuda_verbose=False):
        import torch
        import math

        self.torch = torch
        self.n = int(meta["amo"])
        self.amopi = int(meta["amopi"])
        self.gab = int(meta["gab"])
        self.gaa = int(meta["gaa"])
        self.na = float(meta["nalpha_active"])
        self.nb = float(meta["nbeta_active"])
        self.spin_singlet = abs(self.na - self.nb) < 1.0e-12
        self.constrain_sz = bool(meta["constrain_sz"])
        self.constrain_spin = bool(meta["constrain_spin"])
        self.n_primal = int(n_primal)
        self.n_dual = int(n_dual)
        self.device = device
        self.dtype = dtype
        self.cuda_au = bool(cuda_au)
        self.cuda_atu = bool(cuda_atu)
        self.cuda_normal = bool(cuda_normal)
        self.cuda_au_module = None
        self.cuda_atu_module = None
        self.cuda_normal_module = None

        if self.amopi != self.n:
            raise ValueError("DQG/C1 matrix-free path currently requires amopi == amo.")
        if self.gab != self.n * self.n:
            raise ValueError("DQG/C1 matrix-free metadata has inconsistent gab.")
        if self.gaa != self.n * (self.n - 1) // 2:
            raise ValueError("DQG/C1 matrix-free metadata has inconsistent gaa.")

        names = ["d2ab", "d2aa", "d2bb"]
        if self.constrain_spin:
            names.append("d200")
        names += [
            "d1a", "d1b", "q1a", "q1b",
            "q2ab", "q2aa", "q2bb", "g2ab", "g2ba", "g2aa",
        ]
        d200_dim = self.gab if self.spin_singlet else 2 * self.gab
        expected_dims = {
            "d2ab": self.gab,
            "d2aa": self.gaa,
            "d2bb": self.gaa,
            "d200": d200_dim,
            "d1a": self.n,
            "d1b": self.n,
            "q1a": self.n,
            "q1b": self.n,
            "q2ab": self.gab,
            "q2aa": self.gaa,
            "q2bb": self.gaa,
            "g2ab": self.gab,
            "g2ba": self.gab,
            "g2aa": 2 * self.gab,
        }
        if len(block_dims) != len(names):
            raise ValueError(
                f"DQG/C1 matrix-free expected {len(names)} primal blocks, got {len(block_dims)}."
            )

        self.blocks = {}
        self.block_offsets = {}
        offset = 0
        for name, dim in zip(names, block_dims):
            dim = int(dim)
            if dim != expected_dims[name]:
                raise ValueError(
                    f"DQG/C1 matrix-free block {name} expected dim "
                    f"{expected_dims[name]}, got {dim}."
                )
            self.blocks[name] = (offset, dim)
            self.block_offsets[name] = offset
            offset += dim * dim
        if offset != self.n_primal:
            raise ValueError("DQG/C1 matrix-free primal dimension mismatch.")

        if self.cuda_au or self.cuda_atu or self.cuda_normal:
            if device.type != "cuda":
                raise ValueError("GPU_ADMM matrix-free CUDA kernels require a CUDA device.")
            if dtype != torch.double:
                raise ValueError("GPU_ADMM matrix-free CUDA kernels currently support float64 solves only.")
            if not self.constrain_sz or not self.constrain_spin:
                raise ValueError(
                    "GPU_ADMM matrix-free CUDA kernels currently support the "
                    "constrain_sz=True, constrain_spin=True DQG/C1 path."
                )
            module = (
                _load_dqg_c1_cuda_normal_module(cuda_verbose)
                if self.cuda_normal else
                _load_dqg_c1_cuda_au_module(cuda_verbose)
            )
            if self.cuda_au:
                self.cuda_au_module = module
            if self.cuda_atu:
                self.cuda_atu_module = module
            if self.cuda_normal:
                self.cuda_normal_module = module

        self.dual_block_spans = []
        pos = 0

        def add_span(name, count):
            nonlocal pos
            count = int(count)
            self.dual_block_spans.append((name, pos, pos + count))
            pos += count

        add_span("trace_d2ab", 1 if self.constrain_sz else 0)
        add_span("trace_d2aa", 1 if self.constrain_sz else 0)
        add_span("trace_d2bb", 1 if self.constrain_sz else 0)
        if not self.constrain_sz:
            add_span("trace_d2_total", 1)
        add_span("herm_d2aa", self.gaa * self.gaa)
        add_span("herm_d2bb", self.gaa * self.gaa)
        add_span("herm_d2ab", self.gab * self.gab)
        add_span("d1a_q1a", self.n * self.n)
        add_span("d1b_q1b", self.n * self.n)
        if self.constrain_sz:
            add_span("contract_d2ab_d1a", self.n * self.n)
            add_span("contract_d2ab_d1b", self.n * self.n)
            add_span("contract_d2aa_d1a", self.n * self.n)
            add_span("contract_d2bb_d1b", self.n * self.n)
        else:
            add_span("contract_d2aa_d2ab_d1a", self.n * self.n)
            add_span("contract_d2bb_d2ab_d1b", self.n * self.n)
        if self.constrain_spin:
            add_span("spin_trace", 1)
            if self.spin_singlet:
                add_span("spin_d1a_eq_d1b", self.n * self.n)
                add_span("spin_d2aa_eq_d2bb", self.gaa * self.gaa)
                add_span("spin_d2aa_from_d2ab", self.gaa * self.gaa)
                add_span("spin_d2bb_from_d2ab", self.gaa * self.gaa)
                add_span("spin_d200", self.gab * self.gab)
                add_span("spin_d2ab_sym", self.gab * self.gab)
            else:
                add_span("spin_d200_nonsinglet", 4 * self.gab * self.gab)
            add_span("spin_g2ba_col_trace", self.gab)
            add_span("spin_g2ba_row_trace", self.gab)
        add_span("q2ab", self.gab * self.gab)
        add_span("q2aa", self.gaa * self.gaa)
        add_span("q2bb", self.gaa * self.gaa)
        add_span("g2ab", self.gab * self.gab)
        add_span("g2ba", self.gab * self.gab)
        add_span("g2aa", 4 * self.gab * self.gab)
        if pos != self.n_dual:
            raise ValueError(
                f"DQG/C1 matrix-free dual block spans cover {pos} rows, "
                f"expected {self.n_dual}."
            )

        ar = torch.arange(self.n, device=device, dtype=torch.long)
        self.ab_index = ar.view(self.n, 1) + self.n * ar.view(1, self.n)
        ab = torch.arange(self.gab, device=device, dtype=torch.long)
        self.ab_first = ab % self.n
        self.ab_second = torch.div(ab, self.n, rounding_mode="floor")
        self.ab_perm = self.ab_first * self.n + self.ab_second
        self.ab_diag = ar * (self.n + 1)
        self.ab_diag_scale = torch.where(
            self.ab_first == self.ab_second,
            torch.full((self.gab,), math.sqrt(2.0), device=device, dtype=dtype),
            torch.ones(self.gab, device=device, dtype=dtype),
        )
        self.ab_range = torch.arange(self.gab, device=device, dtype=torch.long)

        aa_i = []
        aa_j = []
        for j in range(self.n):
            for i in range(j):
                aa_i.append(i)
                aa_j.append(j)
        self.aa_i = torch.tensor(aa_i, device=device, dtype=torch.long)
        self.aa_j = torch.tensor(aa_j, device=device, dtype=torch.long)
        aa_index = torch.zeros((self.n, self.n), device=device, dtype=torch.long)
        aa_sign = torch.zeros((self.n, self.n), device=device, dtype=dtype)
        for idx, (i, j) in enumerate(zip(aa_i, aa_j)):
            aa_index[i, j] = idx
            aa_index[j, i] = idx
            aa_sign[i, j] = 1.0
            aa_sign[j, i] = -1.0
        self.aa_index = aa_index
        self.aa_sign = aa_sign
        self.aa_ab_fwd = self.aa_j * self.n + self.aa_i
        self.aa_ab_rev = self.aa_i * self.n + self.aa_j

    def _cuda_kernel_args(self, output_size):
        offsets = self.block_offsets
        return (
            self.n, float(self.na), float(self.nb),
            offsets["d2ab"], offsets["d2aa"], offsets["d2bb"],
            offsets.get("d200", -1), offsets["d1a"], offsets["d1b"],
            offsets["q1a"], offsets["q1b"], offsets["q2ab"],
            offsets["q2aa"], offsets["q2bb"], offsets["g2ab"],
            offsets["g2ba"], offsets["g2aa"], int(output_size),
        )

    def _views(self, vec):
        return {
            name: vec[offset:offset + dim * dim].view(dim, dim)
            for name, (offset, dim) in self.blocks.items()
        }

    def _put(self, out, pos, value):
        flat = value.reshape(-1)
        out[pos:pos + flat.numel()].copy_(flat)
        return pos + flat.numel()

    def _ab4(self, mat):
        n = self.n
        return mat.view(n, n, n, n).permute(1, 0, 3, 2)

    def _from_ab4(self, tensor4):
        n = self.n
        return tensor4.permute(1, 0, 3, 2).contiguous().view(n * n, n * n)

    def _outer_add(self, target, row_idx, col_idx, values):
        rr = row_idx.view(-1, 1).expand_as(values)
        cc = col_idx.view(1, -1).expand_as(values)
        target.index_put_(
            (rr.reshape(-1), cc.reshape(-1)),
            values.reshape(-1),
            accumulate=True,
        )

    def _masked_add(self, target, row_idx, col_idx, values, mask):
        target.index_put_(
            (row_idx[mask], col_idx[mask]),
            values[mask],
            accumulate=True,
        )

    def _ab_same_second_contract(self, mat):
        torch = self.torch
        out = torch.zeros((self.n, self.n), device=self.device, dtype=self.dtype)
        for k in range(self.n):
            idx = self.ab_index[:, k]
            out += mat.index_select(0, idx).index_select(1, idx)
        return out

    def _ab_same_first_contract(self, mat):
        torch = self.torch
        out = torch.zeros((self.n, self.n), device=self.device, dtype=self.dtype)
        for k in range(self.n):
            idx = self.ab_index[k, :]
            out += mat.index_select(0, idx).index_select(1, idx)
        return out

    def _add_ab_same_second_adj(self, target, y_mat, scale):
        for k in range(self.n):
            idx = self.ab_index[:, k]
            self._outer_add(target, idx, idx, scale * y_mat)

    def _add_ab_same_first_adj(self, target, y_mat, scale):
        for k in range(self.n):
            idx = self.ab_index[k, :]
            self._outer_add(target, idx, idx, scale * y_mat)

    def _aa_contract(self, mat):
        torch = self.torch
        out = torch.zeros((self.n, self.n), device=self.device, dtype=self.dtype)
        for k in range(self.n):
            idx = self.aa_index[:, k]
            sign = self.aa_sign[:, k]
            out += sign.view(-1, 1) * sign.view(1, -1) * mat.index_select(0, idx).index_select(1, idx)
        return out

    def _add_aa_contract_adj(self, target, y_mat, scale):
        for k in range(self.n):
            idx = self.aa_index[:, k]
            sign = self.aa_sign[:, k]
            weighted = scale * sign.view(-1, 1) * sign.view(1, -1) * y_mat
            self._outer_add(target, idx, idx, weighted)

    def _aa_cross_ab_contract(self, mat):
        idx_il = self.aa_index[:, None, None, :]
        idx_kj = self.aa_index.t()[None, :, :, None]
        coeff = self.aa_sign[:, None, None, :] * self.aa_sign.t()[None, :, :, None]
        return coeff * mat[idx_il, idx_kj]

    def _add_aa_cross_ab_adj(self, target, y4, scale):
        idx_il = self.aa_index[:, None, None, :].expand(self.n, self.n, self.n, self.n)
        idx_kj = self.aa_index.t()[None, :, :, None].expand(self.n, self.n, self.n, self.n)
        coeff = self.aa_sign[:, None, None, :] * self.aa_sign.t()[None, :, :, None]
        values = scale * coeff * y4
        target.index_put_(
            (idx_il.reshape(-1), idx_kj.reshape(-1)),
            values.reshape(-1),
            accumulate=True,
        )

    def _q2aa_Au(self, d2, q2, d1):
        torch = self.torch
        i = self.aa_i.view(-1, 1)
        j = self.aa_j.view(-1, 1)
        k = self.aa_i.view(1, -1)
        l = self.aa_j.view(1, -1)
        zero = torch.zeros((), device=self.device, dtype=self.dtype)
        out = d2 - q2
        out = out + torch.where(j == l, -d1[k, i], zero)
        out = out + torch.where(j == k, d1[l, i], zero)
        out = out + torch.where(i == l, d1[k, j], zero)
        out = out + torch.where(i == k, -d1[l, j], zero)
        return out

    def _q2aa_ATu(self, d2, q2, d1, y_mat):
        d2 += y_mat
        q2 -= y_mat
        i = self.aa_i.view(-1, 1).expand(self.gaa, self.gaa)
        j = self.aa_j.view(-1, 1).expand(self.gaa, self.gaa)
        k = self.aa_i.view(1, -1).expand(self.gaa, self.gaa)
        l = self.aa_j.view(1, -1).expand(self.gaa, self.gaa)
        self._masked_add(d1, k, i, -y_mat, j == l)
        self._masked_add(d1, l, i, y_mat, j == k)
        self._masked_add(d1, k, j, y_mat, i == l)
        self._masked_add(d1, l, j, -y_mat, i == k)

    def Au(self, x):
        torch = self.torch
        if self.cuda_au_module is not None:
            return self.cuda_au_module.dqg_c1_au(
                x.contiguous(), *self._cuda_kernel_args(self.n_dual),
            )

        b = self._views(x)
        out = torch.empty(self.n_dual, device=self.device, dtype=self.dtype)
        pos = 0
        d2ab = b["d2ab"]
        d2aa = b["d2aa"]
        d2bb = b["d2bb"]
        d1a = b["d1a"]
        d1b = b["d1b"]
        q1a = b["q1a"]
        q1b = b["q1b"]
        q2ab = b["q2ab"]
        q2aa = b["q2aa"]
        q2bb = b["q2bb"]
        g2ab = b["g2ab"]
        g2ba = b["g2ba"]
        g2aa = b["g2aa"]

        if self.constrain_sz:
            out[pos] = torch.trace(d2ab)
            pos += 1
            out[pos] = 2.0 * torch.trace(d2aa)
            pos += 1
            out[pos] = 2.0 * torch.trace(d2bb)
            pos += 1
        else:
            out[pos] = 2.0 * (torch.trace(d2ab) + torch.trace(d2aa) + torch.trace(d2bb))
            pos += 1

        pos = self._put(out, pos, d2aa - d2aa.t())
        pos = self._put(out, pos, d2bb - d2bb.t())
        pos = self._put(out, pos, d2ab - d2ab.t())
        pos = self._put(out, pos, d1a.t() + q1a)
        pos = self._put(out, pos, d1b.t() + q1b)

        if self.constrain_sz:
            pos = self._put(out, pos, self.nb * d1a - self._ab_same_second_contract(d2ab))
            pos = self._put(out, pos, self.na * d1b - self._ab_same_first_contract(d2ab))
            pos = self._put(out, pos, (self.na - 1.0) * d1a - self._aa_contract(d2aa))
            pos = self._put(out, pos, (self.nb - 1.0) * d1b - self._aa_contract(d2bb))
        else:
            nele = self.na + self.nb
            pos = self._put(
                out, pos,
                (nele - 1.0) * d1a - self._ab_same_second_contract(d2ab) - self._aa_contract(d2aa),
            )
            pos = self._put(
                out, pos,
                (nele - 1.0) * d1b - self._ab_same_first_contract(d2ab) - self._aa_contract(d2bb),
            )

        if self.constrain_spin:
            out[pos] = d2ab[self.ab_range, self.ab_perm].sum()
            pos += 1
            if self.spin_singlet:
                pos = self._put(out, pos, d1a - d1b)
                pos = self._put(out, pos, d2aa - d2bb)
                f = self.aa_ab_fwd
                r = self.aa_ab_rev
                pos = self._put(
                    out, pos,
                    d2aa - 0.5 * d2ab[f[:, None], f[None, :]]
                    + 0.5 * d2ab[r[:, None], f[None, :]]
                    + 0.5 * d2ab[f[:, None], r[None, :]]
                    - 0.5 * d2ab[r[:, None], r[None, :]],
                )
                pos = self._put(
                    out, pos,
                    d2bb - 0.5 * d2ab[f[:, None], f[None, :]]
                    + 0.5 * d2ab[r[:, None], f[None, :]]
                    + 0.5 * d2ab[f[:, None], r[None, :]]
                    - 0.5 * d2ab[r[:, None], r[None, :]],
                )
                scale = -0.5 / (self.ab_diag_scale.view(-1, 1) * self.ab_diag_scale.view(1, -1))
                pos = self._put(
                    out, pos,
                    b["d200"] + scale * (
                        d2ab + d2ab[self.ab_perm, :] +
                        d2ab[:, self.ab_perm] + d2ab[self.ab_perm, :][:, self.ab_perm]
                    ),
                )
                pos = self._put(out, pos, d2ab - d2ab[self.ab_perm, :][:, self.ab_perm])
            else:
                d200 = b["d200"]
                d_ij = self.ab_diag_scale.view(-1, 1)
                d_kl = self.ab_diag_scale.view(1, -1)
                d2_ji_kl = d2ab[self.ab_perm, :]
                d2_ij_lk = d2ab[:, self.ab_perm]
                d2_ji_lk = d2_ji_kl[:, self.ab_perm]
                d200_rows = torch.empty((2 * self.gab, 2 * self.gab), device=self.device, dtype=self.dtype)
                d200_rows[:self.gab, :self.gab] = (
                    d200[:self.gab, :self.gab]
                    - 0.5 / (d_ij * d_kl) * (d2ab + d2_ji_kl + d2_ij_lk + d2_ji_lk)
                )
                d200_rows[:self.gab, self.gab:] = (
                    d200[:self.gab, self.gab:]
                    - 0.5 / d_ij * (d2ab + d2_ji_kl)
                    + 0.5 / d_ij * (d2_ij_lk + d2_ji_lk)
                )
                d200_rows[self.gab:, :self.gab] = (
                    d200[self.gab:, :self.gab]
                    - 0.5 / d_kl * (d2ab + d2_ij_lk)
                    + 0.5 / d_kl * (d2_ji_kl + d2_ji_lk)
                )
                d200_rows[self.gab:, self.gab:] = (
                    d200[self.gab:, self.gab:]
                    - 0.5 * d2ab
                    + 0.5 * d2_ji_kl
                    + 0.5 * d2_ij_lk
                    - 0.5 * d2_ji_lk
                )
                pos = self._put(out, pos, d200_rows)
            pos = self._put(out, pos, g2ba[:, self.ab_diag].sum(dim=1))
            pos = self._put(out, pos, g2ba[self.ab_diag, :].sum(dim=0))

        d2ab4 = self._ab4(d2ab)
        q2ab4 = self._ab4(q2ab)
        r4 = d2ab4 - q2ab4
        r4 = r4.clone()
        for t in range(self.n):
            r4[:, t, :, t] -= d1a.t()
            r4[t, :, t, :] -= d1b.t()
        pos = self._put(out, pos, self._from_ab4(r4))
        pos = self._put(out, pos, self._q2aa_Au(d2aa, q2aa, d1a))
        pos = self._put(out, pos, self._q2aa_Au(d2bb, q2bb, d1b))

        r4 = -self._ab4(g2ab)
        r4 = r4.clone()
        for t in range(self.n):
            r4[:, t, :, t] += d1a
        r4 -= d2ab4.permute(0, 3, 2, 1)
        pos = self._put(out, pos, self._from_ab4(r4))

        r4 = -self._ab4(g2ba)
        r4 = r4.clone()
        for t in range(self.n):
            r4[:, t, :, t] += d1b
        r4 -= d2ab4.permute(1, 2, 3, 0)
        pos = self._put(out, pos, self._from_ab4(r4))

        g2aa_tl = g2aa[:self.gab, :self.gab]
        g2aa_tr = g2aa[:self.gab, self.gab:]
        g2aa_bl = g2aa[self.gab:, :self.gab]
        g2aa_br = g2aa[self.gab:, self.gab:]

        r_tl = -self._ab4(g2aa_tl)
        r_tl = r_tl.clone()
        for t in range(self.n):
            r_tl[:, t, :, t] += d1a
        r_tl -= self._aa_cross_ab_contract(d2aa)
        r_br = -self._ab4(g2aa_br)
        r_br = r_br.clone()
        for t in range(self.n):
            r_br[:, t, :, t] += d1b
        r_br -= self._aa_cross_ab_contract(d2bb)
        r_tr = -self._ab4(g2aa_tr) + d2ab4.permute(0, 2, 3, 1)
        r_bl = -self._ab4(g2aa_bl) + d2ab4.permute(1, 3, 2, 0)
        g2aa_rows = torch.empty((2 * self.gab, 2 * self.gab), device=self.device, dtype=self.dtype)
        g2aa_rows[:self.gab, :self.gab] = self._from_ab4(r_tl)
        g2aa_rows[:self.gab, self.gab:] = self._from_ab4(r_tr)
        g2aa_rows[self.gab:, :self.gab] = self._from_ab4(r_bl)
        g2aa_rows[self.gab:, self.gab:] = self._from_ab4(r_br)
        pos = self._put(out, pos, g2aa_rows)

        if pos != self.n_dual:
            raise ValueError(f"DQG/C1 matrix-free Au wrote {pos} rows, expected {self.n_dual}.")
        return out

    def Au_out(self, x, out):
        if self.cuda_au_module is not None:
            self.cuda_au_module.dqg_c1_au_out(
                x.contiguous(), out, *self._cuda_kernel_args(self.n_dual),
            )
            return out

        out.copy_(self.Au(x))
        return out

    def ATu(self, y):
        torch = self.torch
        if self.cuda_atu_module is not None:
            return self.cuda_atu_module.dqg_c1_atu(
                y.contiguous(), *self._cuda_kernel_args(self.n_primal),
            )

        out = torch.zeros(self.n_primal, device=self.device, dtype=self.dtype)
        b = self._views(out)
        pos = 0

        def take(count, shape=None):
            nonlocal pos
            item = y[pos:pos + count]
            pos += count
            return item.view(shape) if shape is not None else item

        d2ab = b["d2ab"]
        d2aa = b["d2aa"]
        d2bb = b["d2bb"]
        d1a = b["d1a"]
        d1b = b["d1b"]
        q1a = b["q1a"]
        q1b = b["q1b"]
        q2ab = b["q2ab"]
        q2aa = b["q2aa"]
        q2bb = b["q2bb"]
        g2ab = b["g2ab"]
        g2ba = b["g2ba"]
        g2aa = b["g2aa"]

        if self.constrain_sz:
            d2ab.diagonal().add_(take(1)[0])
            d2aa.diagonal().add_(2.0 * take(1)[0])
            d2bb.diagonal().add_(2.0 * take(1)[0])
        else:
            yy = take(1)[0]
            d2ab.diagonal().add_(2.0 * yy)
            d2aa.diagonal().add_(2.0 * yy)
            d2bb.diagonal().add_(2.0 * yy)

        yy = take(self.gaa * self.gaa, (self.gaa, self.gaa))
        d2aa += yy - yy.t()
        yy = take(self.gaa * self.gaa, (self.gaa, self.gaa))
        d2bb += yy - yy.t()
        yy = take(self.gab * self.gab, (self.gab, self.gab))
        d2ab += yy - yy.t()

        yy = take(self.n * self.n, (self.n, self.n))
        d1a += yy.t()
        q1a += yy
        yy = take(self.n * self.n, (self.n, self.n))
        d1b += yy.t()
        q1b += yy

        if self.constrain_sz:
            yy = take(self.n * self.n, (self.n, self.n))
            d1a += self.nb * yy
            self._add_ab_same_second_adj(d2ab, yy, -1.0)
            yy = take(self.n * self.n, (self.n, self.n))
            d1b += self.na * yy
            self._add_ab_same_first_adj(d2ab, yy, -1.0)
            yy = take(self.n * self.n, (self.n, self.n))
            d1a += (self.na - 1.0) * yy
            self._add_aa_contract_adj(d2aa, yy, -1.0)
            yy = take(self.n * self.n, (self.n, self.n))
            d1b += (self.nb - 1.0) * yy
            self._add_aa_contract_adj(d2bb, yy, -1.0)
        else:
            nele = self.na + self.nb
            yy = take(self.n * self.n, (self.n, self.n))
            d1a += (nele - 1.0) * yy
            self._add_ab_same_second_adj(d2ab, yy, -1.0)
            self._add_aa_contract_adj(d2aa, yy, -1.0)
            yy = take(self.n * self.n, (self.n, self.n))
            d1b += (nele - 1.0) * yy
            self._add_ab_same_first_adj(d2ab, yy, -1.0)
            self._add_aa_contract_adj(d2bb, yy, -1.0)

        if self.constrain_spin:
            yy = take(1)[0]
            d2ab.index_put_((self.ab_range, self.ab_perm), yy.expand(self.gab), accumulate=True)
            if self.spin_singlet:
                yy = take(self.n * self.n, (self.n, self.n))
                d1a += yy
                d1b -= yy
                yy = take(self.gaa * self.gaa, (self.gaa, self.gaa))
                d2aa += yy
                d2bb -= yy
                f = self.aa_ab_fwd
                r = self.aa_ab_rev
                yy = take(self.gaa * self.gaa, (self.gaa, self.gaa))
                d2aa += yy
                self._outer_add(d2ab, f, f, -0.5 * yy)
                self._outer_add(d2ab, r, f, 0.5 * yy)
                self._outer_add(d2ab, f, r, 0.5 * yy)
                self._outer_add(d2ab, r, r, -0.5 * yy)
                yy = take(self.gaa * self.gaa, (self.gaa, self.gaa))
                d2bb += yy
                self._outer_add(d2ab, f, f, -0.5 * yy)
                self._outer_add(d2ab, r, f, 0.5 * yy)
                self._outer_add(d2ab, f, r, 0.5 * yy)
                self._outer_add(d2ab, r, r, -0.5 * yy)

                yy = take(self.gab * self.gab, (self.gab, self.gab))
                b["d200"] += yy
                scale = -0.5 / (self.ab_diag_scale.view(-1, 1) * self.ab_diag_scale.view(1, -1))
                weighted = scale * yy
                d2ab += weighted
                self._outer_add(d2ab, self.ab_perm, self.ab_range, weighted)
                self._outer_add(d2ab, self.ab_range, self.ab_perm, weighted)
                self._outer_add(d2ab, self.ab_perm, self.ab_perm, weighted)

                yy = take(self.gab * self.gab, (self.gab, self.gab))
                d2ab += yy
                self._outer_add(d2ab, self.ab_perm, self.ab_perm, -yy)
            else:
                yy = take(4 * self.gab * self.gab, (2 * self.gab, 2 * self.gab))
                b["d200"] += yy
                d_ij = self.ab_diag_scale.view(-1, 1)
                d_kl = self.ab_diag_scale.view(1, -1)
                y00 = yy[:self.gab, :self.gab]
                y01 = yy[:self.gab, self.gab:]
                y10 = yy[self.gab:, :self.gab]
                y11 = yy[self.gab:, self.gab:]

                weighted = -0.5 / (d_ij * d_kl) * y00
                d2ab += weighted
                self._outer_add(d2ab, self.ab_perm, self.ab_range, weighted)
                self._outer_add(d2ab, self.ab_range, self.ab_perm, weighted)
                self._outer_add(d2ab, self.ab_perm, self.ab_perm, weighted)

                weighted = -0.5 / d_ij * y01
                d2ab += weighted
                self._outer_add(d2ab, self.ab_perm, self.ab_range, weighted)
                weighted = 0.5 / d_ij * y01
                self._outer_add(d2ab, self.ab_range, self.ab_perm, weighted)
                self._outer_add(d2ab, self.ab_perm, self.ab_perm, weighted)

                weighted = -0.5 / d_kl * y10
                d2ab += weighted
                self._outer_add(d2ab, self.ab_range, self.ab_perm, weighted)
                weighted = 0.5 / d_kl * y10
                self._outer_add(d2ab, self.ab_perm, self.ab_range, weighted)
                self._outer_add(d2ab, self.ab_perm, self.ab_perm, weighted)

                d2ab += -0.5 * y11
                self._outer_add(d2ab, self.ab_perm, self.ab_range, 0.5 * y11)
                self._outer_add(d2ab, self.ab_range, self.ab_perm, 0.5 * y11)
                self._outer_add(d2ab, self.ab_perm, self.ab_perm, -0.5 * y11)

            yy = take(self.gab)
            rows = self.ab_range.view(-1, 1).expand(self.gab, self.n)
            cols = self.ab_diag.view(1, -1).expand(self.gab, self.n)
            g2ba.index_put_((rows.reshape(-1), cols.reshape(-1)), yy.view(-1, 1).expand(self.gab, self.n).reshape(-1), accumulate=True)
            yy = take(self.gab)
            rows = self.ab_diag.view(-1, 1).expand(self.n, self.gab)
            cols = self.ab_range.view(1, -1).expand(self.n, self.gab)
            g2ba.index_put_((rows.reshape(-1), cols.reshape(-1)), yy.view(1, -1).expand(self.n, self.gab).reshape(-1), accumulate=True)

        yy = take(self.gab * self.gab, (self.gab, self.gab))
        y4 = self._ab4(yy)
        d2ab += yy
        q2ab -= yy
        for t in range(self.n):
            d1a -= y4[:, t, :, t].t()
            d1b -= y4[t, :, t, :].t()

        yy = take(self.gaa * self.gaa, (self.gaa, self.gaa))
        self._q2aa_ATu(d2aa, q2aa, d1a, yy)
        yy = take(self.gaa * self.gaa, (self.gaa, self.gaa))
        self._q2aa_ATu(d2bb, q2bb, d1b, yy)

        yy = take(self.gab * self.gab, (self.gab, self.gab))
        y4 = self._ab4(yy)
        g2ab -= yy
        for t in range(self.n):
            d1a += y4[:, t, :, t]
        self._ab4(d2ab).add_(-y4.permute(0, 3, 2, 1))

        yy = take(self.gab * self.gab, (self.gab, self.gab))
        y4 = self._ab4(yy)
        g2ba -= yy
        for t in range(self.n):
            d1b += y4[:, t, :, t]
        self._ab4(d2ab).add_(-y4.permute(3, 0, 1, 2))

        y_g2aa = take(4 * self.gab * self.gab, (2 * self.gab, 2 * self.gab))
        y_tl = y_g2aa[:self.gab, :self.gab]
        y_tr = y_g2aa[:self.gab, self.gab:]
        y_bl = y_g2aa[self.gab:, :self.gab]
        y_br = y_g2aa[self.gab:, self.gab:]

        y4 = self._ab4(y_tl)
        g2aa[:self.gab, :self.gab] -= y_tl
        for t in range(self.n):
            d1a += y4[:, t, :, t]
        self._add_aa_cross_ab_adj(d2aa, y4, -1.0)

        y4 = self._ab4(y_br)
        g2aa[self.gab:, self.gab:] -= y_br
        for t in range(self.n):
            d1b += y4[:, t, :, t]
        self._add_aa_cross_ab_adj(d2bb, y4, -1.0)

        y4 = self._ab4(y_tr)
        g2aa[:self.gab, self.gab:] -= y_tr
        self._ab4(d2ab).add_(y4.permute(0, 3, 1, 2))

        y4 = self._ab4(y_bl)
        g2aa[self.gab:, :self.gab] -= y_bl
        self._ab4(d2ab).add_(y4.permute(3, 0, 2, 1))

        if pos != self.n_dual:
            raise ValueError(f"DQG/C1 matrix-free ATu consumed {pos} rows, expected {self.n_dual}.")
        return out

    def ATu_out(self, y, out):
        if self.cuda_atu_module is not None:
            self.cuda_atu_module.dqg_c1_atu_out(
                y.contiguous(), out, *self._cuda_kernel_args(self.n_primal),
            )
            return out

        out.copy_(self.ATu(y))
        return out

    def normal_out(self, y, scratch, out):
        module = self.cuda_normal_module or self.cuda_atu_module or self.cuda_au_module
        if (
            module is not None and
            hasattr(module, "dqg_c1_normal_direct_out") and
            not self.spin_singlet
        ):
            module.dqg_c1_normal_direct_out(
                y.contiguous(), out,
                *self._cuda_kernel_args(self.n_primal), self.n_dual,
            )
            return out
        if module is not None and hasattr(module, "dqg_c1_normal_out"):
            module.dqg_c1_normal_out(
                y.contiguous(), scratch, out,
                *self._cuda_kernel_args(self.n_primal), self.n_dual,
            )
            return out

        self.ATu_out(y, scratch)
        self.Au_out(scratch, out)
        return out

    def normal_direct_cache_size(self):
        module = self.cuda_normal_module or self.cuda_atu_module or self.cuda_au_module
        if (
            module is None or
            not hasattr(module, "dqg_c1_normal_direct_cache_size") or
            self.spin_singlet
        ):
            raise RuntimeError("DQG/C1 direct normal cache is unavailable.")
        return int(module.dqg_c1_normal_direct_cache_size(self.n))

    def normal_cached_out(self, y, cache, out):
        module = self.cuda_normal_module or self.cuda_atu_module or self.cuda_au_module
        if (
            module is None or
            not hasattr(module, "dqg_c1_normal_direct_cached_out") or
            self.spin_singlet
        ):
            raise RuntimeError("DQG/C1 cached direct normal kernel is unavailable.")
        module.dqg_c1_normal_direct_cached_out(
            y.contiguous(), cache, out,
            *self._cuda_kernel_args(self.n_primal), self.n_dual,
        )
        return out

    def normal_build_cache_out(self, y, cache):
        module = self.cuda_normal_module or self.cuda_atu_module or self.cuda_au_module
        if (
            module is None or
            not hasattr(module, "dqg_c1_normal_direct_build_cache_out") or
            self.spin_singlet
        ):
            raise RuntimeError("DQG/C1 cached direct normal kernel is unavailable.")
        module.dqg_c1_normal_direct_build_cache_out(
            y.contiguous(), cache, self.n, self.na, self.nb, self.n_dual,
        )
        return cache

    def normal_cached_range_out(self, y, cache, out, row_start, row_stop):
        module = self.cuda_normal_module or self.cuda_atu_module or self.cuda_au_module
        if (
            module is None or
            not hasattr(module, "dqg_c1_normal_direct_cached_range_out") or
            self.spin_singlet
        ):
            raise RuntimeError("DQG/C1 cached direct normal row-range kernel is unavailable.")
        module.dqg_c1_normal_direct_cached_range_out(
            y.contiguous(), cache, out, int(row_start), int(row_stop),
            *self._cuda_kernel_args(self.n_primal), self.n_dual,
        )
        return out

    def normal_direct_range_out(self, y, out, row_start, row_stop):
        module = self.cuda_normal_module or self.cuda_atu_module or self.cuda_au_module
        if (
            module is None or
            not hasattr(module, "dqg_c1_normal_direct_range_out") or
            self.spin_singlet
        ):
            raise RuntimeError("DQG/C1 direct normal row-range kernel is unavailable.")
        module.dqg_c1_normal_direct_range_out(
            y.contiguous(), out, int(row_start), int(row_stop),
            *self._cuda_kernel_args(self.n_primal), self.n_dual,
        )
        return out

    def validate_au_range(self, n_chunks=4, atol=1.0e-9, rtol=1.0e-7):
        """For Debugging and Testing.
        Validates that the dqg_c1_au_range_out kernel and the atu_range_add -> au_range
        recomposition reproduce the full single-GPU operators, using standard
        (global) offsets, full buffers, a single device, and an arbitrary
        row partition. Per-row independence of A and additivity of A^T over
        rows mean an arbitrary partition must reconstruct the full result;
        Returns True/False and is gated by GPU_ADMM_VALIDATE_AURANGE."""
        torch = self.torch
        module = self.cuda_normal_module
        if module is None or not hasattr(module, "dqg_c1_au_range_out"):
            print("  ==> GPU-ADMM P1 au_range validation skipped: "
                  "normal module / dqg_c1_au_range_out unavailable.", flush=True)
            return None
        dev, dt = self.device, self.dtype
        gen = torch.Generator(device=dev)
        gen.manual_seed(20260629)
        x = torch.randn(self.n_primal, device=dev, dtype=dt, generator=gen)
        y = torch.randn(self.n_dual, device=dev, dtype=dt, generator=gen)
        au_args = self._cuda_kernel_args(self.n_dual)
        atu_args = self._cuda_kernel_args(self.n_primal)
        bounds = [(i * self.n_dual) // n_chunks for i in range(n_chunks + 1)]

        # (1) au_range over the full range == full Au.
        au_ref = module.dqg_c1_au(x.contiguous(), *au_args)
        au_full = torch.empty(self.n_dual, device=dev, dtype=dt)
        module.dqg_c1_au_range_out(x.contiguous(), au_full, 0, self.n_dual, *au_args)
        e_full = (au_full - au_ref).abs().max().item()

        # (2) au_range over an arbitrary partition == full Au.
        au_part = torch.empty(self.n_dual, device=dev, dtype=dt)
        for a, b in zip(bounds, bounds[1:]):
            module.dqg_c1_au_range_out(x.contiguous(), au_part[a:b], a, b, *au_args)
        e_part = (au_part - au_ref).abs().max().item()

        # (3) atu_range_add accumulated over a partition == full ATu.
        atu_ref = module.dqg_c1_atu(y.contiguous(), *atu_args)
        aty = torch.zeros(self.n_primal, device=dev, dtype=dt)
        for a, b in zip(bounds, bounds[1:]):
            module.dqg_c1_atu_range_add_out(
                y[a:b].contiguous(), aty, a, b, *atu_args)
        e_atu = (aty - atu_ref).abs().max().item()

        # (4) PRIMARY: the distributed operator computes Au(A^T y); its
        # single-GPU reference is the composed Au(ATu(y)). The range-compose must match this.
        ref = self.Au(self.ATu(y))
        comp = torch.empty(self.n_dual, device=dev, dtype=dt)
        for a, b in zip(bounds, bounds[1:]):
            module.dqg_c1_au_range_out(aty.contiguous(), comp[a:b], a, b, *au_args)
        e_norm = (comp - ref).abs().max().item()

        # (5) diagnostic: fused normal_out equal composed Au(ATu(y))
        # If large, the CG fused operator differs from A.A^T and the
        # distributed path must replicate that difference.
        fused = torch.empty(self.n_dual, device=dev, dtype=dt)
        scratch = torch.empty(self.n_primal, device=dev, dtype=dt)
        try:
            self.normal_out(y, scratch, fused)
            e_fused = (fused - ref).abs().max().item()
        except Exception as exc:  # noqa: BLE001
            e_fused = float("nan")
            print(f"  ==> GPU-ADMM P1 au_range validation: normal_out "
                  f"unavailable for this layout ({exc}); skipping fused check.",
                  flush=True)

        scale = ref.abs().max().item() + 1.0e-300
        ok = (max(e_full, e_part, e_atu) <= atol and
              e_norm <= atol + rtol * scale)
        print("  ==> GPU-ADMM P1 au_range validation: "
              f"au_full={e_full:.2e}, au_partition={e_part:.2e}, "
              f"atu_partition={e_atu:.2e}, range_compose_vs_AuATu={e_norm:.2e}, "
              f"fused_vs_AuATu={e_fused:.2e} (ref|max|={scale:.2e}) -> "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
        return ok

    def validate_au_range_compact(self, atol=1.0e-9, rtol=1.0e-7):
        """For debugging and Testing. Validation of single device, no NCCL.
        Confirm the compact offset-remapping of the sharded operator. Models a device
        that owns the g2aa block and shares D2/D1, builds its compact cache with
        remapped offsets via atu_range_add over g2aa's definition rows, and
        applies au_range from that compact cache. Compares both to the full
        single-GPU operators. Unowned blocks map to a zeroed guard region that
        must stay untouched, verifies the owned rows only read owned+shared
        blocks. Gated by GPU_ADMM_VALIDATE_AURANGE."""
        torch = self.torch
        module = self.cuda_normal_module
        if module is None or not hasattr(module, "dqg_c1_au_range_out"):
            print("  ==> GPU-ADMM P1 compact validation skipped: "
                  "normal module / dqg_c1_au_range_out unavailable.", flush=True)
            return None
        if "g2aa" not in self.blocks:
            print("  ==> GPU-ADMM P1 compact validation skipped: no g2aa block.",
                  flush=True)
            return None
        dev, dt = self.device, self.dtype
        gen = torch.Generator(device=dev)
        gen.manual_seed(20260630)
        y = torch.randn(self.n_dual, device=dev, dtype=dt, generator=gen)
        aty_full = self.ATu(y)
        au_ref = self.Au(aty_full)

        # Discover g2aa's definition-row range via an indicator probe.
        off_g, dim_g = self.blocks["g2aa"]
        szg = dim_g * dim_g
        probe = torch.zeros(self.n_primal, device=dev, dtype=dt)
        probe[off_g:off_g + szg] = torch.randn(szg, device=dev, dtype=dt,
                                               generator=gen)
        au_probe = self.Au(probe)
        thr = 1.0e-12 * (au_probe.abs().max().item() + 1.0e-300)
        rows = (au_probe.abs() > thr).nonzero(as_tuple=False).flatten()
        if rows.numel() == 0:
            print("  ==> GPU-ADMM P1 compact validation: g2aa probe found no "
                  "rows; skipping.", flush=True)
            return None
        r0 = int(rows.min().item())
        r1 = int(rows.max().item()) + 1
        contiguous = (rows.numel() == r1 - r0)

        # Compact layout: [g2aa | shared D2/D1 | guard]; unowned blocks -> guard.
        owned = ["g2aa"]
        shared = [b for b in ("d2ab", "d2aa", "d2bb", "d1a", "d1b")
                  if b in self.blocks]
        layout = owned + shared
        remap = {}
        pos = 0
        for name in layout:
            _, dim = self.blocks[name]
            remap[name] = pos
            pos += dim * dim
        guard = pos
        max_sz = max(dim * dim for (_, dim) in self.blocks.values())
        total = pos + max_sz
        for name in self.block_offsets:
            remap.setdefault(name, guard)

        def remapped_args(tail):
            g = lambda nm: int(remap.get(nm, guard))
            return (self.n, float(self.na), float(self.nb),
                    g("d2ab"), g("d2aa"), g("d2bb"), g("d200"), g("d1a"),
                    g("d1b"), g("q1a"), g("q1b"), g("q2ab"), g("q2aa"),
                    g("q2bb"), g("g2ab"), g("g2ba"), g("g2aa"), int(tail))

        # Build side: compact cache via remapped atu_range_add over g2aa rows.
        compact_b = torch.zeros(total, device=dev, dtype=dt)
        module.dqg_c1_atu_range_add_out(
            y[r0:r1].contiguous(), compact_b, r0, r1, *remapped_args(total))
        full_ref = torch.zeros(self.n_primal, device=dev, dtype=dt)
        module.dqg_c1_atu_range_add_out(
            y[r0:r1].contiguous(), full_ref, r0, r1,
            *self._cuda_kernel_args(self.n_primal))
        e_build = 0.0
        for name in layout:
            o, dim = self.blocks[name]
            sz = dim * dim
            e_build = max(e_build, (compact_b[remap[name]:remap[name] + sz] -
                                    full_ref[o:o + sz]).abs().max().item())
        e_guard = compact_b[guard:guard + max_sz].abs().max().item()

        # Apply side: au_range from a compact cache with remapped offsets.
        compact_x = torch.zeros(total, device=dev, dtype=dt)
        for name in layout:
            o, dim = self.blocks[name]
            sz = dim * dim
            compact_x[remap[name]:remap[name] + sz] = aty_full[o:o + sz]
        out_range = torch.empty(r1 - r0, device=dev, dtype=dt)
        module.dqg_c1_au_range_out(
            compact_x.contiguous(), out_range, r0, r1,
            *remapped_args(self.n_dual))
        e_apply = (out_range - au_ref[r0:r1]).abs().max().item()

        scale = au_ref.abs().max().item() + 1.0e-300
        ok = (contiguous and e_build <= atol and e_guard <= atol and
              e_apply <= atol + rtol * scale)
        print("  ==> GPU-ADMM P1 compact-offset validation (g2aa shard): "
              f"rows=[{r0},{r1}) contiguous={contiguous}, "
              f"build_err={e_build:.2e}, guard={e_guard:.2e}, "
              f"apply_err={e_apply:.2e} (ref|max|={scale:.2e}) -> "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
        return ok

    def validate_distributed_normal(self, device_ids, block_owner,
                                    atol=1.0e-9, rtol=1.0e-7):
        """Debugging/Testing. Validate the full multi-device distributed normal
        operator must equal single-GPU Au(ATu(y)). Each device builds a compact
        cache of its owned blocks & shared D2/D1 via remapped atu_range_add over
        its owned dual-row runs. The shared D2/D1 region is summed across devices
        with NCCL all_reduce, then each device applies au_range over its owned
        rows. Persistent vectors are not sharded to isolate the
        distributed-operator and NCCL correctness. Needs >=2 GPUs."""
        torch = self.torch
        module = self.cuda_normal_module
        if module is None or not hasattr(module, "dqg_c1_au_range_out"):
            print("  ==> P1 distributed-normal validation skipped: kernel n/a.",
                  flush=True)
            return None
        try:
            import torch.cuda.nccl as nccl
        except Exception as exc:  # noqa: BLE001
            print(f"  ==> P1 distributed-normal validation skipped: nccl n/a "
                  f"({exc}).", flush=True)
            return None
        device_ids = [int(d) for d in device_ids]
        if len(device_ids) < 2:
            print("  ==> P1 distributed-normal validation skipped: need >=2 GPUs.",
                  flush=True)
            return None
        base = int(torch.cuda.current_device())
        dt = self.dtype

        def runs_from_sorted(idx):
            # idx: ascending 1D LongTensor -> list of contiguous [start, stop).
            if idx.numel() == 0:
                return []
            v = idx.tolist()
            runs = []
            s = p = v[0]
            for x in v[1:]:
                if x == p + 1:
                    p = x
                else:
                    runs.append((s, p + 1))
                    s = p = x
            runs.append((s, p + 1))
            return runs

        with torch.cuda.device(base):
            gen = torch.Generator(device=f"cuda:{base}")
            gen.manual_seed(20260701)
            y = torch.randn(self.n_dual, device=f"cuda:{base}", dtype=dt,
                            generator=gen)
            ref = self.Au(self.ATu(y))

        shared_names = [b for b in ("d2ab", "d2aa", "d2bb", "d1a", "d1b")
                        if b in self.blocks]
        shared_set = set(shared_names)
        ownable = [nm for nm in self.blocks if nm not in shared_set]

        # Discover ownable families (lists of contiguous row runs) via probes.
        fam = {}
        with torch.cuda.device(base):
            for nm in ownable:
                o, dim = self.blocks[nm]
                sz = dim * dim
                probe = torch.zeros(self.n_primal, device=f"cuda:{base}", dtype=dt)
                probe[o:o + sz] = torch.randn(sz, device=f"cuda:{base}", dtype=dt,
                                              generator=gen)
                ap = self.Au(probe)
                thr = 1.0e-12 * (ap.abs().max().item() + 1.0e-300)
                rws = (ap.abs() > thr).nonzero(as_tuple=False).flatten()
                fam[nm] = runs_from_sorted(rws)

        # Assign each ownable family to its block owner, verify clean tiling.
        owned_rows = {d: [] for d in device_ids}
        all_runs = []
        for nm in ownable:
            d = block_owner.get(nm, base)
            if d not in owned_rows:
                d = base
            for run in fam[nm]:
                owned_rows[d].append(run)
                all_runs.append(run)
        all_runs.sort()
        cur = 0
        leftover = []
        for a, b in all_runs:
            if a < cur:
                print(f"  ==> P1 distributed-normal validation FAIL: overlapping "
                      f"owned rows near {a} (cur={cur}).", flush=True)
                return False
            if a > cur:
                leftover.append((cur, a))
            cur = b
        if cur < self.n_dual:
            leftover.append((cur, self.n_dual))
        owned_rows[base].extend(leftover)

        # Per-device compact layout: [shared D2/D1 | owned ownable | guard].
        shared_size = sum(self.blocks[nm][1] ** 2 for nm in shared_names)
        owned_blocks = {
            d: [nm for nm in ownable
                if block_owner.get(nm, base) == d and fam[nm]]
            for d in device_ids
        }
        layouts = {}
        for d in device_ids:
            remap = {}
            pos = 0
            for nm in shared_names:
                remap[nm] = pos
                pos += self.blocks[nm][1] ** 2
            for nm in owned_blocks[d]:
                remap[nm] = pos
                pos += self.blocks[nm][1] ** 2
            # A device's owned rows never index an unowned block (verified by
            # compact-offset validation guard==0). Unowned blocks map to offset 0, never accessed.
            guard = 0
            total = pos
            for nm in self.block_offsets:
                remap.setdefault(nm, 0)
            layouts[d] = (remap, total, guard)

        def args_for(remap, guard, tail):
            g = lambda nm: int(remap.get(nm, guard))
            return (self.n, float(self.na), float(self.nb),
                    g("d2ab"), g("d2aa"), g("d2bb"), g("d200"), g("d1a"),
                    g("d1b"), g("q1a"), g("q1b"), g("q2ab"), g("q2aa"),
                    g("q2bb"), g("g2ab"), g("g2ba"), g("g2aa"), int(tail))

        # Build compact caches (Aᵀ y restricted to each device's owned rows).
        caches = {}
        for d in device_ids:
            remap, total, guard = layouts[d]
            with torch.cuda.device(d):
                yd = y if d == base else y.to(f"cuda:{d}")
                cache = torch.zeros(total, device=f"cuda:{d}", dtype=dt)
                for a, b in owned_rows[d]:
                    module.dqg_c1_atu_range_add_out(
                        yd[a:b].contiguous(), cache, a, b,
                        *args_for(remap, guard, total))
                caches[d] = cache
        for d in device_ids:
            torch.cuda.synchronize(d)

        # NCCL all-reduce the shared D2/D1 region (same layout on every device).
        nccl.all_reduce([caches[d][:shared_size] for d in device_ids])
        for d in device_ids:
            torch.cuda.synchronize(d)

        # Apply au_range over each device's owned rows, gather to base.
        out = torch.empty(self.n_dual, device=f"cuda:{base}", dtype=dt)
        for d in device_ids:
            remap, total, guard = layouts[d]
            with torch.cuda.device(d):
                for a, b in owned_rows[d]:
                    obuf = torch.empty(b - a, device=f"cuda:{d}", dtype=dt)
                    module.dqg_c1_au_range_out(
                        caches[d].contiguous(), obuf, a, b,
                        *args_for(remap, guard, self.n_dual))
                    out[a:b].copy_(obuf if d == base else obuf.to(f"cuda:{base}"))
        for d in device_ids:
            torch.cuda.synchronize(d)

        e = (out - ref).abs().max().item()
        scale = ref.abs().max().item() + 1.0e-300
        ok = e <= atol + rtol * scale
        n_runs = sum(len(v) for v in owned_rows.values())
        print("  ==> GPU-ADMM P1 distributed-normal validation: "
              f"devices={device_ids}, owned-row-runs={n_runs}, "
              f"shared={shared_size} elems, err={e:.2e} "
              f"(ref|max|={scale:.2e}) -> {'PASS' if ok else 'FAIL'}", flush=True)

        # Cross-check the production methods (build_shard_plan & distributed_normal_out) against the same reference,
        # also validates the exact code path the solve will use.
        plan = self.build_shard_plan(device_ids, block_owner)
        if plan is not None:
            out2 = torch.empty(self.n_dual, device=f"cuda:{base}", dtype=dt)
            self.distributed_normal_out(plan, y, out2)
            e2 = (out2 - ref).abs().max().item()
            ok2 = e2 <= atol + rtol * scale
            print("  ==> GPU-ADMM P1 distributed-normal (production path): "
                  f"err={e2:.2e} -> {'PASS' if ok2 else 'FAIL'}", flush=True)
            ok = bool(ok) and ok2
        return ok

    def _shard_args(self, remap, guard, tail):
        """Build the (n, na, nb, ...block offsets..., tail) argument tuple for
        the matrix-free kernels, with block offsets remapped into a device's
        compact buffer (unowned blocks point at the zeroed guard region)."""
        g = lambda nm: int(remap.get(nm, guard))
        return (self.n, float(self.na), float(self.nb),
                g("d2ab"), g("d2aa"), g("d2bb"), g("d200"), g("d1a"), g("d1b"),
                g("q1a"), g("q1b"), g("q2ab"), g("q2aa"), g("q2bb"),
                g("g2ab"), g("g2ba"), g("g2aa"), int(tail))

    def build_shard_plan(self, device_ids, block_owner):
        """Build once the locality shard plan for the distributed normal
        operator. per-device owned dual-row runs, compact [shared | owned |
        guard] primal layouts with remapped offsets, the shared D2/D1 region
        size, and persistent per-device compact cache buffers. Returns a dict
        or None if unavailable or the row partition is inconsistent."""
        torch = self.torch
        module = self.cuda_normal_module
        if module is None or not hasattr(module, "dqg_c1_au_range_out"):
            return None
        device_ids = [int(d) for d in device_ids]
        base = int(torch.cuda.current_device())
        dt = self.dtype

        shared_names = [b for b in ("d2ab", "d2aa", "d2bb", "d1a", "d1b")
                        if b in self.blocks]
        ownable = [nm for nm in self.blocks if nm not in set(shared_names)]

        # Partition dual rows by op.dual_block_spans, exact constraint-family layout.
        # Each span constrains one ownable block (q1*, q2*, g2*, d200) plus the shared D2/D1 blocks;
        # spans that touch only shared blocks (traces, hermiticity, contractions, spin-symmetry) go
        # on base device. A span's rows go to the PSD owner of its ownable block,
        # which holds that block in its compact cache.
        span_ownable = {
            "d1a_q1a": "q1a", "d1b_q1b": "q1b",
            "spin_d200": "d200", "spin_d2ab_sym": "d200",
            "spin_d200_nonsinglet": "d200",
            "spin_g2ba_col_trace": "g2ba", "spin_g2ba_row_trace": "g2ba",
            "q2ab": "q2ab", "q2aa": "q2aa", "q2bb": "q2bb",
            "g2ab": "g2ab", "g2ba": "g2ba", "g2aa": "g2aa",
        }
        owned_rows = {d: [] for d in device_ids}
        owned_block_set = {d: set() for d in device_ids}
        for name, start, stop in self.dual_block_spans:
            if stop <= start:
                continue
            blk = span_ownable.get(name)
            if blk is not None and blk in self.blocks:
                d = block_owner.get(blk, base)
                if d not in owned_rows:
                    d = base
                owned_block_set[d].add(blk)
            else:
                d = base
            owned_rows[d].append((int(start), int(stop)))
        # Merge contiguous runs per device.
        for d in device_ids:
            merged = []
            for a, b in sorted(owned_rows[d]):
                if merged and merged[-1][1] == a:
                    merged[-1] = (merged[-1][0], b)
                else:
                    merged.append((a, b))
            owned_rows[d] = merged

        shared_size = sum(self.blocks[nm][1] ** 2 for nm in shared_names)
        owned_blocks = {
            d: [nm for nm in ownable if nm in owned_block_set[d]]
            for d in device_ids
        }
        layouts = {}
        caches = {}
        for d in device_ids:
            remap = {}
            pos = 0
            for nm in shared_names:
                remap[nm] = pos
                pos += self.blocks[nm][1] ** 2
            for nm in owned_blocks[d]:
                remap[nm] = pos
                pos += self.blocks[nm][1] ** 2
            # A device's owned rows never index an unowned block. Unowned blocks map to offset 0 (never accessed).
            guard = 0
            total = pos
            for nm in self.block_offsets:
                remap.setdefault(nm, 0)
            layouts[d] = (remap, total, guard)
            with torch.cuda.device(d):
                caches[d] = torch.zeros(total, device=f"cuda:{d}", dtype=dt)
        # Per-device dual-shard layout: runs annotated with their local offset in
        # the concatenated shard tensor, and the total shard length.
        run_offsets = {}
        shard_len = {}
        for d in device_ids:
            ro = []
            off = 0
            for a, b in owned_rows[d]:
                ro.append((a, b, off))
                off += b - a
            run_offsets[d] = ro
            shard_len[d] = off
        # Primal storage layout (disjoint, aligned with PSD block ownership).
        # each primal block including the shared D2/D1 lives on one device.
        # Used to store c/x/z/U sharded so no full primal is on base GPU.
        # Distinct from the operator compact layout (which replicates D2/D1),
        # conversion occurs in operator scratch build.
        store_blocks = {d: [] for d in device_ids}
        for nm in self.blocks:
            d = block_owner.get(nm, base)
            if d not in store_blocks:
                d = base
            store_blocks[d].append(nm)
        store_layout = {}
        store_total = {}
        for d in device_ids:
            lay = {}
            pos = 0
            for nm in store_blocks[d]:
                lay[nm] = pos
                pos += self.blocks[nm][1] ** 2
            store_layout[d] = lay
            store_total[d] = pos
        return {
            "device_ids": device_ids, "base": base, "owned_rows": owned_rows,
            "layouts": layouts, "caches": caches, "shared_size": shared_size,
            "shared_names": shared_names, "owned_blocks": owned_blocks,
            "run_offsets": run_offsets, "shard_len": shard_len,
            "block_owner": dict(block_owner), "store_blocks": store_blocks,
            "store_layout": store_layout, "store_total": store_total,
        }

    def distributed_normal_out(self, plan, u, out):
        """Matvec out = A (A^T u) across devices using a shard plan. Each device
        builds its compact cache (remapped atu_range_add over its owned rows),
        the shared D2/D1 region is summed with NCCL all_reduce, then each device
        applies au_range over its owned rows. `u` and `out` are full dual
        vectors on the base device and the per-device caches are compact."""
        torch = self.torch
        import torch.cuda.nccl as nccl
        module = self.cuda_normal_module
        device_ids = plan["device_ids"]
        base = plan["base"]
        owned_rows = plan["owned_rows"]
        layouts = plan["layouts"]
        caches = plan["caches"]
        shared_size = plan["shared_size"]
        for d in device_ids:
            remap, total, guard = layouts[d]
            with torch.cuda.device(d):
                cache = caches[d]
                cache.zero_()
                for a, b in owned_rows[d]:
                    chunk = u[a:b] if d == base else u[a:b].to(f"cuda:{d}")
                    module.dqg_c1_atu_range_add_out(
                        chunk.contiguous(), cache, a, b,
                        *self._shard_args(remap, guard, total))
        for d in device_ids:
            torch.cuda.synchronize(d)
        nccl.all_reduce([caches[d][:shared_size] for d in device_ids])
        for d in device_ids:
            torch.cuda.synchronize(d)
        for d in device_ids:
            remap, total, guard = layouts[d]
            with torch.cuda.device(d):
                for a, b in owned_rows[d]:
                    obuf = torch.empty(b - a, device=f"cuda:{d}", dtype=self.dtype)
                    module.dqg_c1_au_range_out(
                        caches[d].contiguous(), obuf, a, b,
                        *self._shard_args(remap, guard, self.n_dual))
                    out[a:b].copy_(obuf if d == base else obuf.to(f"cuda:{base}"))
        for d in device_ids:
            torch.cuda.synchronize(d)
        return out

    # ---- Sharded-dual storage ops, keep dual vectors row-sharded
    # across devices, base GPU never holds a full dual vector -------------

    def alloc_dual_shards(self, plan):
        """Allocate a zeroed per-device dual shard set {dev: tensor(shard_len)}."""
        torch = self.torch
        out = {}
        for d in plan["device_ids"]:
            with torch.cuda.device(d):
                out[d] = torch.zeros(plan["shard_len"][d], device=f"cuda:{d}",
                                     dtype=self.dtype)
        return out

    def scatter_dual_to_shards(self, plan, full):
        """Full dual vector on base -> {dev: shard} (device d gets its owned rows,
        runs concatenated in order)."""
        torch = self.torch
        shards = {}
        for d in plan["device_ids"]:
            parts = [full[a:b] for a, b in plan["owned_rows"][d]]
            s = (torch.cat(parts) if parts
                 else torch.empty(0, device=full.device, dtype=self.dtype))
            # Always land on the owner GPU. When `full` already lives on the base
            # GPU (standard path) this is a no-op for d==base, when `full` is a CPU
            # master (P1 host-scatter) this performs the host->device copy so the
            # base shard is correctly resident on cuda:base, not left on the host.
            shards[d] = s.to(f"cuda:{d}")
        return shards

    def gather_shards_to_dual(self, plan, shards, out_full):
        """{dev: shard} -> full dual vector on base (inverse of scatter)."""
        base = plan["base"]
        for d in plan["device_ids"]:
            src = shards[d] if d == base else shards[d].to(f"cuda:{base}")
            for a, b, off in plan["run_offsets"][d]:
                out_full[a:b].copy_(src[off:off + (b - a)])
        return out_full

    def sharded_dot(self, plan, xs, ys):
        """Sum_d (xs[d] . ys[d]) as a python float, computing each device's dot
        on-GPU without per-device host sync, summing the 1-element results on base,
        and syncing to host only once."""
        torch = self.torch
        base = plan["base"]
        acc = None
        for d in plan["device_ids"]:
            if xs[d].numel() == 0:
                continue
            with torch.cuda.device(d):
                dd = torch.dot(xs[d], ys[d])
            if d != base:
                dd = dd.to(f"cuda:{base}")
            acc = dd if acc is None else acc + dd
        return float(acc) if acc is not None else 0.0

    def distributed_normal_sharded(self, plan, p_shards, ap_shards):
        """Sharded matvec: ap = A(A^T p), with p and ap row-sharded (no full dual).
        Compact caches & NCCL all_reduce of the shared D2/D1 region."""
        torch = self.torch
        import torch.cuda.nccl as nccl
        module = self.cuda_normal_module
        device_ids = plan["device_ids"]
        caches = plan["caches"]
        layouts = plan["layouts"]
        shared_size = plan["shared_size"]
        for d in device_ids:
            remap, total, guard = layouts[d]
            with torch.cuda.device(d):
                cache = caches[d]
                cache.zero_()
                for a, b, off in plan["run_offsets"][d]:
                    module.dqg_c1_atu_range_add_out(
                        p_shards[d][off:off + (b - a)].contiguous(), cache, a, b,
                        *self._shard_args(remap, guard, total))
        for d in device_ids:
            torch.cuda.synchronize(d)
        nccl.all_reduce([caches[d][:shared_size] for d in device_ids])
        for d in device_ids:
            torch.cuda.synchronize(d)
        for d in device_ids:
            remap, total, guard = layouts[d]
            with torch.cuda.device(d):
                for a, b, off in plan["run_offsets"][d]:
                    module.dqg_c1_au_range_out(
                        caches[d].contiguous(),
                        ap_shards[d][off:off + (b - a)], a, b,
                        *self._shard_args(remap, guard, self.n_dual))
        for d in device_ids:
            torch.cuda.synchronize(d)
        return ap_shards

    def _fill_compact_from_full(self, plan, d, x_full):
        """Fill device d persistent cache which is reused as compact primal scratch
        with its blocks. shared D2/D1 & its owned blocks, copied from
        full primal vector on base. Reuses plan['caches'][d]."""
        remap = plan["layouts"][d][0]
        xc = plan["caches"][d]
        xc.zero_()
        for nm in plan["shared_names"] + plan["owned_blocks"][d]:
            o, dim = self.blocks[nm]
            sz = dim * dim
            src = x_full[o:o + sz]
            xc[remap[nm]:remap[nm] + sz] = src if d == plan["base"] else src.to(
                f"cuda:{d}")
        return xc

    def au_to_shards(self, plan, x_full, ap_shards=None):
        """A x with x a full primal vector on base -> row-sharded dual output.
        Reuses the persistent per-device caches as the compact primal scratch
        called sequentially with the matvec, never concurrently."""
        torch = self.torch
        module = self.cuda_normal_module
        if ap_shards is None:
            ap_shards = self.alloc_dual_shards(plan)
        for d in plan["device_ids"]:
            remap, total, guard = plan["layouts"][d]
            with torch.cuda.device(d):
                xc = self._fill_compact_from_full(plan, d, x_full)
                for a, b, off in plan["run_offsets"][d]:
                    module.dqg_c1_au_range_out(
                        xc.contiguous(), ap_shards[d][off:off + (b - a)], a, b,
                        *self._shard_args(remap, guard, self.n_dual))
        for d in plan["device_ids"]:
            torch.cuda.synchronize(d)
        return ap_shards

    def atu_to_full(self, plan, y_shards, out_full):
        """A^T y with y row-sharded -> full primal on base, owned blocks gathered,
        shared D2/D1 summed with NCCL then scattered to their global offsets."""
        torch = self.torch
        import torch.cuda.nccl as nccl
        module = self.cuda_normal_module
        base = plan["base"]
        caches = plan["caches"]
        layouts = plan["layouts"]
        shared_size = plan["shared_size"]
        for d in plan["device_ids"]:
            remap, total, guard = layouts[d]
            with torch.cuda.device(d):
                cache = caches[d]
                cache.zero_()
                for a, b, off in plan["run_offsets"][d]:
                    module.dqg_c1_atu_range_add_out(
                        y_shards[d][off:off + (b - a)].contiguous(), cache, a, b,
                        *self._shard_args(remap, guard, total))
        for d in plan["device_ids"]:
            torch.cuda.synchronize(d)
        nccl.all_reduce([caches[d][:shared_size] for d in plan["device_ids"]])
        for d in plan["device_ids"]:
            torch.cuda.synchronize(d)
        out_full.zero_()
        base_remap = layouts[base][0]
        for nm in plan["shared_names"]:
            o, dim = self.blocks[nm]
            sz = dim * dim
            out_full[o:o + sz] = caches[base][base_remap[nm]:base_remap[nm] + sz]
        for d in plan["device_ids"]:
            remap = layouts[d][0]
            for nm in plan["owned_blocks"][d]:
                o, dim = self.blocks[nm]
                sz = dim * dim
                src = caches[d][remap[nm]:remap[nm] + sz]
                out_full[o:o + sz] = src if d == base else src.to(f"cuda:{base}")
        return out_full

    # ---- Sharded-primal storage ops per-device block shards without full primal on the base GPU -----------------

    def alloc_primal_storage(self, plan):
        """Allocate a zeroed disjoint per-device primal store {dev: tensor}."""
        torch = self.torch
        out = {}
        for d in plan["device_ids"]:
            with torch.cuda.device(d):
                out[d] = torch.zeros(plan["store_total"][d], device=f"cuda:{d}",
                                     dtype=self.dtype)
        return out

    def scatter_primal_to_storage(self, plan, full, store=None):
        """Full primal on base -> disjoint per-device block store with each block on
        its PSD-owner device."""
        torch = self.torch
        if store is None:
            store = self.alloc_primal_storage(plan)
        for d in plan["device_ids"]:
            lay = plan["store_layout"][d]
            with torch.cuda.device(d):
                buf = store[d]
                for nm, loff in lay.items():
                    o, dim = self.blocks[nm]
                    sz = dim * dim
                    src = full[o:o + sz]
                    # `.to(cuda:d)` is a no-op when `full` is already on cuda:d (base,
                    # standard path) and performs the host->device copy when `full`
                    # is a CPU master (P1 host-scatter, without full primal on any GPU).
                    buf[loff:loff + sz] = src.to(f"cuda:{d}")
        return store

    def gather_storage_to_primal(self, plan, store, out_full):
        """Disjoint per-device block store -> full primal on base."""
        base = plan["base"]
        for d in plan["device_ids"]:
            lay = plan["store_layout"][d]
            for nm, loff in lay.items():
                o, dim = self.blocks[nm]
                sz = dim * dim
                src = store[d][loff:loff + sz]
                out_full[o:o + sz] = src if d == base else src.to(f"cuda:{base}")
        return out_full

    def atu_to_store(self, plan, y_shards, store):
        """A^T y with y row-sharded -> disjoint primal store. Same compact-cache
        build & NCCL D2/D1 reduce as atu_to_full, but each device then copies the
        blocks it owns from its cache into its store slot for all local copies so a
        device cache holds its owned_ownable blocks the reduced D2/D1 in the shared region,
        which covers the blocks it stores."""
        torch = self.torch
        import torch.cuda.nccl as nccl
        module = self.cuda_normal_module
        caches = plan["caches"]
        layouts = plan["layouts"]
        shared_size = plan["shared_size"]
        for d in plan["device_ids"]:
            remap, total, guard = layouts[d]
            with torch.cuda.device(d):
                cache = caches[d]
                cache.zero_()
                for a, b, off in plan["run_offsets"][d]:
                    module.dqg_c1_atu_range_add_out(
                        y_shards[d][off:off + (b - a)].contiguous(), cache, a, b,
                        *self._shard_args(remap, guard, total))
        for d in plan["device_ids"]:
            torch.cuda.synchronize(d)
        nccl.all_reduce([caches[d][:shared_size] for d in plan["device_ids"]])
        for d in plan["device_ids"]:
            torch.cuda.synchronize(d)
        for d in plan["device_ids"]:
            remap = layouts[d][0]
            slay = plan["store_layout"][d]
            with torch.cuda.device(d):
                for nm in plan["store_blocks"][d]:
                    o, dim = self.blocks[nm]
                    sz = dim * dim
                    store[d][slay[nm]:slay[nm] + sz] = \
                        caches[d][remap[nm]:remap[nm] + sz]
        return store

    def build_U_store(self, plan, y_shards, x_store, c_store, U_store,
                      mu, admm_relaxation=1.0, z_store=None):
        """U = mu*x + A^T y - c, built entirely in the disjoint store per-device
        elementwise, with optional over relaxation of the
        coupling term. No full primal, full U on base GPU."""
        torch = self.torch
        self.atu_to_store(plan, y_shards, U_store)  # U <- A^T y
        for d in plan["device_ids"]:
            with torch.cuda.device(d):
                if admm_relaxation != 1.0 and z_store is not None:
                    one_minus = 1.0 - admm_relaxation
                    U_store[d].mul_(admm_relaxation)
                    U_store[d].add_(c_store[d], alpha=one_minus)
                    U_store[d].sub_(z_store[d], alpha=one_minus)
                U_store[d].add_(x_store[d], alpha=mu)
                U_store[d].sub_(c_store[d])
        return U_store

    def validate_u_store(self, plan, atol=1.0e-9, rtol=1.0e-7):
        """Debugging/Testing. Check U = mu*x + A^T y - c built in disjoint store must
        equal the full-vector U (gathered)."""
        torch = self.torch
        base = plan["base"]
        dt = self.dtype
        mu = 0.371
        with torch.cuda.device(base):
            gen = torch.Generator(device=f"cuda:{base}")
            gen.manual_seed(20260705)
            x_full = torch.randn(self.n_primal, device=f"cuda:{base}", dtype=dt,
                                 generator=gen)
            c_full = torch.randn(self.n_primal, device=f"cuda:{base}", dtype=dt,
                                 generator=gen)
            y_full = torch.randn(self.n_dual, device=f"cuda:{base}", dtype=dt,
                                 generator=gen)
            u_ref = self.ATu(y_full)
            u_ref = u_ref + mu * x_full - c_full
        x_store = self.scatter_primal_to_storage(plan, x_full)
        c_store = self.scatter_primal_to_storage(plan, c_full)
        y_shards = self.scatter_dual_to_shards(plan, y_full)
        u_store = self.alloc_primal_storage(plan)
        self.build_U_store(plan, y_shards, x_store, c_store, u_store, mu)
        u_back = torch.zeros(self.n_primal, device=f"cuda:{base}", dtype=dt)
        self.gather_storage_to_primal(plan, u_store, u_back)
        e = (u_back - u_ref).abs().max().item()
        scale = u_ref.abs().max().item() + 1.0e-300
        ok = e <= atol + rtol * scale
        print("  ==> GPU-ADMM P1 U-store validation: err="
              f"{e:.2e} (|U|max={scale:.2e}) -> {'PASS' if ok else 'FAIL'}",
              flush=True)
        return ok

    def au_from_store(self, plan, x_store, ap_shards=None):
        """A x with x held in the disjoint store -> row-sharded dual output. Each
        device fills its compact cache from its locally-stored owned blocks, then
        the shared D2/D1 blocks are broadcast from their owners into every cache and
        then au_range produces each device's owned dual rows."""
        torch = self.torch
        module = self.cuda_normal_module
        if ap_shards is None:
            ap_shards = self.alloc_dual_shards(plan)
        base = plan["base"]
        for d in plan["device_ids"]:
            remap = plan["layouts"][d][0]
            slay = plan["store_layout"][d]
            with torch.cuda.device(d):
                cache = plan["caches"][d]
                cache.zero_()
                for nm in plan["owned_blocks"][d]:
                    o, dim = self.blocks[nm]
                    sz = dim * dim
                    cache[remap[nm]:remap[nm] + sz] = \
                        x_store[d][slay[nm]:slay[nm] + sz]
        for nm in plan["shared_names"]:
            owner = plan["block_owner"].get(nm, base)
            if owner not in plan["device_ids"]:
                owner = base
            o, dim = self.blocks[nm]
            sz = dim * dim
            oslay = plan["store_layout"][owner]
            src = x_store[owner][oslay[nm]:oslay[nm] + sz]
            for d in plan["device_ids"]:
                remap = plan["layouts"][d][0]
                with torch.cuda.device(d):
                    plan["caches"][d][remap[nm]:remap[nm] + sz] = \
                        src if d == owner else src.to(f"cuda:{d}")
        for d in plan["device_ids"]:
            remap, total, guard = plan["layouts"][d]
            with torch.cuda.device(d):
                for a, b, off in plan["run_offsets"][d]:
                    module.dqg_c1_au_range_out(
                        plan["caches"][d].contiguous(),
                        ap_shards[d][off:off + (b - a)], a, b,
                        *self._shard_args(remap, guard, self.n_dual))
        for d in plan["device_ids"]:
            torch.cuda.synchronize(d)
        return ap_shards

    def validate_au_from_store(self, plan, atol=1.0e-9, rtol=1.0e-7):
        """Debugging/Testing. check that A x with x in the store (gathered) == full Au(x)."""
        torch = self.torch
        base = plan["base"]
        dt = self.dtype
        with torch.cuda.device(base):
            gen = torch.Generator(device=f"cuda:{base}")
            gen.manual_seed(20260707)
            x_full = torch.randn(self.n_primal, device=f"cuda:{base}", dtype=dt,
                                 generator=gen)
            ref = self.Au(x_full)
        x_store = self.scatter_primal_to_storage(plan, x_full)
        aus = self.au_from_store(plan, x_store)
        back = torch.zeros(self.n_dual, device=f"cuda:{base}", dtype=dt)
        self.gather_shards_to_dual(plan, aus, back)
        e = (back - ref).abs().max().item()
        scale = ref.abs().max().item() + 1.0e-300
        ok = e <= atol + rtol * scale
        print("  ==> GPU-ADMM P1 au-from-store validation: err="
              f"{e:.2e} (|Ax|max={scale:.2e}) -> {'PASS' if ok else 'FAIL'}",
              flush=True)
        return ok

    def project_psd_store(self, plan, U_store, x_store, z_store, mu,
                          serial_dim_threshold=16000):
        """PSD projection of U, read from and written to the disjoint store.
        each device eighs the blocks it owns locally without cross-device block
        transfers writing x = pos_part/mu and z = pos_part - sym(U) back into
        its store. Devices run in parallel threads and torch.linalg.eigh (cuSOLVER)
        synchronizes internally. A serial device loop would run the per-device
        eighs back-to-back and the thread pool overlaps them (eigh releases the GIL).
        Distributes the eigh workspace and keeps no full U/x/z on base."""
        torch = self.torch
        from concurrent.futures import ThreadPoolExecutor

        def _work(d):
            slay = plan["store_layout"][d]
            by_dim = {}
            for nm in plan["store_blocks"][d]:
                by_dim.setdefault(self.blocks[nm][1], []).append(nm)
            with torch.cuda.device(d):
                for dim, names in by_dim.items():
                    bs = dim * dim

                    def _project_one(nm):
                        # Serial path for the large monolithic block (g2aa). fp64
                        # eigh of n=2*gab matrix dominates base-GPU memory, so keeping
                        # the live temp set as small as possible. free the eigenvector
                        # buffers before the z/x writes and reuse pp in place for x.
                        Ub = U_store[d][slay[nm]:slay[nm] + bs].view(dim, dim)
                        sym = Ub.add(Ub.t()).mul_(0.5)  # symmetrized copy (Ub untouched)
                        lam, V = torch.linalg.eigh(sym)
                        lam.clamp_(min=0.0)             # positive part, in place
                        Vp = V * lam                    # scaled eigenvectors
                        pp = Vp @ V.t()                 # V (+pos) V^T
                        del Vp, V
                        # z = pp - sym(U); then reuse pp in place for x = pp/mu.
                        z_store[d][slay[nm]:slay[nm] + bs] = (pp - sym).reshape(-1)
                        del sym
                        pp.div_(mu)
                        x_store[d][slay[nm]:slay[nm] + bs] = pp.reshape(-1)

                    if dim >= serial_dim_threshold:
                        for nm in names:
                            torch.cuda.empty_cache()
                            _project_one(nm)
                        continue
                    Ub = torch.stack([
                        U_store[d][slay[nm]:slay[nm] + bs].view(dim, dim)
                        for nm in names])
                    Ub = 0.5 * (Ub + Ub.transpose(1, 2))
                    lam, V = torch.linalg.eigh(Ub)
                    pos = torch.clamp(lam, min=0.0)
                    pp = torch.bmm(V * pos.unsqueeze(1), V.transpose(1, 2))
                    for i, nm in enumerate(names):
                        x_store[d][slay[nm]:slay[nm] + bs] = pp[i].reshape(-1) / mu
                        z_store[d][slay[nm]:slay[nm] + bs] = \
                            (pp[i] - Ub[i]).reshape(-1)
                torch.cuda.synchronize(d)

        devs = plan["device_ids"]
        # first call runs serially to warm PyTorch's lazily-initialized CUDA
        # op wrappers. Subsequent calls run the per-device eighs in parallel.
        if len(devs) > 1 and getattr(self, "_psd_store_warmed", False):
            with ThreadPoolExecutor(max_workers=len(devs)) as ex:
                for _ in ex.map(_work, devs):
                    pass
        else:
            for d in devs:
                _work(d)
            self._psd_store_warmed = True
        return x_store, z_store

    def validate_psd_store(self, plan, mu=0.371, atol=1.0e-9, rtol=1.0e-7):
        """Testing/Debuggin. Check store-based PSD projection == the full-vector PSD math
        (symmetrize; eigh; x = pos_part/mu; z = pos_part - sym(U))."""
        torch = self.torch
        base = plan["base"]
        dt = self.dtype
        with torch.cuda.device(base):
            gen = torch.Generator(device=f"cuda:{base}")
            gen.manual_seed(20260706)
            u_full = torch.randn(self.n_primal, device=f"cuda:{base}", dtype=dt,
                                 generator=gen)
            x_ref = torch.zeros(self.n_primal, device=f"cuda:{base}", dtype=dt)
            z_ref = torch.zeros(self.n_primal, device=f"cuda:{base}", dtype=dt)
            for nm, (o, dim) in self.blocks.items():
                bs = dim * dim
                Ub = u_full[o:o + bs].view(dim, dim)
                Ub = 0.5 * (Ub + Ub.t())
                lam, V = torch.linalg.eigh(Ub)
                pos = torch.clamp(lam, min=0.0)
                pp = (V * pos) @ V.t()
                x_ref[o:o + bs] = pp.reshape(-1) / mu
                z_ref[o:o + bs] = (pp - Ub).reshape(-1)
        u_store = self.scatter_primal_to_storage(plan, u_full)
        x_store = self.alloc_primal_storage(plan)
        z_store = self.alloc_primal_storage(plan)
        self.project_psd_store(plan, u_store, x_store, z_store, mu)
        x_back = torch.zeros(self.n_primal, device=f"cuda:{base}", dtype=dt)
        z_back = torch.zeros(self.n_primal, device=f"cuda:{base}", dtype=dt)
        self.gather_storage_to_primal(plan, x_store, x_back)
        self.gather_storage_to_primal(plan, z_store, z_back)
        ex = (x_back - x_ref).abs().max().item()
        ez = (z_back - z_ref).abs().max().item()
        sx = x_ref.abs().max().item() + 1.0e-300
        sz = z_ref.abs().max().item() + 1.0e-300
        ok = ex <= atol + rtol * sx and ez <= atol + rtol * sz
        print("  ==> GPU-ADMM P1 PSD-store validation: x_err="
              f"{ex:.2e}, z_err={ez:.2e} (|x|max={sx:.2e}, |z|max={sz:.2e}) -> "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
        return ok

    def validate_primal_storage(self, plan, atol=0.0):
        """Testing/Debugging. Scatter a full primal to the disjoint store
        and gather it back, must be bit-identical."""
        torch = self.torch
        base = plan["base"]
        dt = self.dtype
        with torch.cuda.device(base):
            gen = torch.Generator(device=f"cuda:{base}")
            gen.manual_seed(20260704)
            full = torch.randn(self.n_primal, device=f"cuda:{base}", dtype=dt,
                               generator=gen)
        store = self.scatter_primal_to_storage(plan, full)
        back = torch.zeros(self.n_primal, device=f"cuda:{base}", dtype=dt)
        self.gather_storage_to_primal(plan, store, back)
        e = (back - full).abs().max().item()
        covered = sum(plan["store_total"][d] for d in plan["device_ids"])
        per_dev = ", ".join(
            f"cuda:{d}={plan['store_total'][d] * 8 / (1024 ** 3):.2f}GiB"
            for d in plan["device_ids"])
        ok = (e <= atol) and (covered == self.n_primal)
        print("  ==> GPU-ADMM P1 primal-storage validation: roundtrip_err="
              f"{e:.2e}, covered={covered}/{self.n_primal}, {per_dev} -> "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
        return ok

    def sharded_cg_solve(self, plan, y_shards, rhs_shards, cg_conv, cg_max_it):
        """Conjugate gradient for A(A^T)y = rhs entirely in sharded-dual space.
        y_shards is updated in place and returned with the iteration count."""
        torch = self.torch
        # ap and p are allocated once per solve (not per CG iteration) and freed
        # on return so they dont persist in the memory-peak PSD phase.
        ap = self.alloc_dual_shards(plan)
        self.distributed_normal_sharded(plan, y_shards, ap)
        # Reuse rhs_shards in place as the residual r = rhs - A(A^T)y0 (caller
        # discards rhs afterwards).
        r = rhs_shards
        for d in plan["device_ids"]:
            with torch.cuda.device(d):
                r[d].sub_(ap[d])
        p = {d: r[d].clone() for d in plan["device_ids"]}
        rr = self.sharded_dot(plan, r, r)
        cg_conv_sq = float(cg_conv) * float(cg_conv)
        if rr < cg_conv_sq:
            return y_shards, 0
        cg_it = 0
        for cg_it in range(int(cg_max_it)):
            self.distributed_normal_sharded(plan, p, ap)
            pap = self.sharded_dot(plan, p, ap)
            alpha = rr / pap
            for d in plan["device_ids"]:
                with torch.cuda.device(d):
                    y_shards[d].add_(p[d], alpha=alpha)
                    r[d].add_(ap[d], alpha=-alpha)
            rr_new = self.sharded_dot(plan, r, r)
            if rr_new < cg_conv_sq:
                cg_it += 1
                break
            beta = rr_new / rr
            for d in plan["device_ids"]:
                with torch.cuda.device(d):
                    p[d].mul_(beta).add_(r[d])
            rr = rr_new
        return y_shards, cg_it

    def validate_sharded_ops(self, plan, atol=1.0e-9, rtol=1.0e-7):
        """Testing/Debugging. Validate sharded-dual ops against the full single-GPU operators.
        au_to_shards (gathered) == Au; atu_to_full == ATu; and sharded_cg_solve
        drives A(A^T)y - rhs to zero. Gated by GPU_ADMM_VALIDATE_AURANGE."""
        torch = self.torch
        base = plan["base"]
        dt = self.dtype
        with torch.cuda.device(base):
            gen = torch.Generator(device=f"cuda:{base}")
            gen.manual_seed(20260703)
            x_full = torch.randn(self.n_primal, device=f"cuda:{base}", dtype=dt,
                                 generator=gen)
            y_full = torch.randn(self.n_dual, device=f"cuda:{base}", dtype=dt,
                                 generator=gen)

        au_shards = self.au_to_shards(plan, x_full)
        au_gathered = torch.zeros(self.n_dual, device=f"cuda:{base}", dtype=dt)
        self.gather_shards_to_dual(plan, au_shards, au_gathered)
        e_au = (au_gathered - self.Au(x_full)).abs().max().item()

        y_shards = self.scatter_dual_to_shards(plan, y_full)
        atu_full = torch.zeros(self.n_primal, device=f"cuda:{base}", dtype=dt)
        self.atu_to_full(plan, y_shards, atu_full)
        e_atu = (atu_full - self.ATu(y_full)).abs().max().item()

        # Sharded CG: solve A(A^T) y = rhs from y0=0, check residual.
        rhs_full = self.Au(self.ATu(y_full))  # a consistent right-hand side
        rhs_shards = self.scatter_dual_to_shards(plan, rhs_full)
        y0 = self.alloc_dual_shards(plan)
        # sharded_cg_solve consumes its rhs in place (reuses it as the residual r),
        # so pass a clone and keep rhs_shards intact for the residual check below.
        rhs_for_cg = {d: rhs_shards[d].clone() for d in plan["device_ids"]}
        y0, iters = self.sharded_cg_solve(plan, y0, rhs_for_cg, 1.0e-10, 2000)
        ap = self.alloc_dual_shards(plan)
        self.distributed_normal_sharded(plan, y0, ap)
        res = 0.0
        for d in plan["device_ids"]:
            res += float(((ap[d] - rhs_shards[d]) ** 2).sum())
        res = res ** 0.5
        rhs_norm = (rhs_full * rhs_full).sum().item() ** 0.5 + 1.0e-300

        ok = (e_au <= atol and e_atu <= atol and
              res <= 1.0e-6 * rhs_norm)
        print("  ==> GPU-ADMM P1 sharded-ops validation: "
              f"au_to_shards={e_au:.2e}, atu_to_full={e_atu:.2e}, "
              f"cg_residual={res:.2e} (rhs|2|={rhs_norm:.2e}, iters={iters}) -> "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
        return ok


_DQG_SYM_META_MAGIC = -20260623


_DQG_SYM_SPAN = {
    "trace_d2ab": 0,
    "trace_d2aa": 1,
    "trace_d2bb": 2,
    "trace_d2_total": 3,
    "herm_d2aa": 4,
    "herm_d2bb": 5,
    "herm_d2ab": 6,
    "d1a_q1a": 7,
    "d1b_q1b": 8,
    "contract_ab_d1a": 9,
    "contract_ab_d1b": 10,
    "contract_aa_d1a": 11,
    "contract_bb_d1b": 12,
    "contract_mix_a": 13,
    "contract_mix_b": 14,
    "spin_trace": 15,
    "spin_d1_eq": 16,
    "spin_d2aa_eq": 17,
    "spin_d2aa_from_ab": 18,
    "spin_d2bb_from_ab": 19,
    "spin_d200_singlet": 20,
    "spin_d2ab_sym": 21,
    "spin_d200_nonsinglet": 22,
    "spin_g2ba_col_trace": 23,
    "spin_g2ba_row_trace": 24,
    "q2ab": 25,
    "q2aa": 26,
    "q2bb": 27,
    "g2ab": 28,
    "g2ba": 29,
    "g2aa": 30,
}


def _parse_dqg_sym_meta(meta):
    ints = [int(v) for v in meta["int_meta"]]
    if len(ints) < 9 or ints[0] != _DQG_SYM_META_MAGIC:
        raise ValueError("DQG/sym matrix-free metadata has an invalid header.")
    version = ints[1]
    if version != 1:
        raise ValueError(f"Unsupported DQG/sym matrix-free metadata version {version}.")
    nirrep = ints[2]
    amo = ints[3]
    pos = 9

    def take(count):
        nonlocal pos
        values = ints[pos:pos + count]
        if len(values) != count:
            raise ValueError("DQG/sym matrix-free metadata is truncated.")
        pos += count
        return values

    amopi = take(nirrep)
    pitzer_offset = take(nirrep)
    symmetry = take(amo)
    sym_pair = take(64)
    gems_ab = take(nirrep)
    gems_aa = take(nirrep)
    offsets = take(14 * nirrep)
    ab_pair_offsets = take(nirrep + 1)
    aa_pair_offsets = take(nirrep + 1)
    total_ab = ab_pair_offsets[-1]
    total_aa = aa_pair_offsets[-1]
    ab_pairs = take(2 * total_ab)
    aa_pairs = take(2 * total_aa)
    ibas_ab = take(nirrep * amo * amo)
    ibas_aa = take(nirrep * amo * amo)
    if pos != len(ints):
        raise ValueError("DQG/sym matrix-free metadata has trailing fields.")
    return {
        "version": version,
        "nirrep": nirrep,
        "amo": amo,
        "constrain_sz": bool(ints[4]),
        "constrain_spin": bool(ints[5]),
        "constrain_q2": bool(ints[6]),
        "constrain_g2": bool(ints[7]),
        "spin_singlet": bool(ints[8]),
        "amopi": amopi,
        "pitzer_offset": pitzer_offset,
        "symmetry": symmetry,
        "sym_pair": sym_pair,
        "gems_ab": gems_ab,
        "gems_aa": gems_aa,
        "offsets": offsets,
        "ab_pair_offsets": ab_pair_offsets,
        "aa_pair_offsets": aa_pair_offsets,
        "ab_i": ab_pairs[0::2],
        "ab_j": ab_pairs[1::2],
        "aa_i": aa_pairs[0::2],
        "aa_j": aa_pairs[1::2],
        "ibas_ab": ibas_ab,
        "ibas_aa": ibas_aa,
    }


def _dqg_sym_block_names_and_dims(meta):
    parsed = _parse_dqg_sym_meta(meta)
    names = []
    dims = []

    def add_family(prefix, values, scale=1):
        for h, dim in enumerate(values):
            names.append(f"{prefix}[{h}]")
            dims.append(scale * int(dim))

    add_family("d2ab", parsed["gems_ab"])
    add_family("d2aa", parsed["gems_aa"])
    add_family("d2bb", parsed["gems_aa"])
    if parsed["constrain_spin"]:
        add_family("d200", parsed["gems_ab"], 1 if parsed["spin_singlet"] else 2)
    add_family("d1a", parsed["amopi"])
    add_family("d1b", parsed["amopi"])
    add_family("q1a", parsed["amopi"])
    add_family("q1b", parsed["amopi"])
    if parsed["constrain_q2"]:
        add_family("q2ab", parsed["gems_ab"])
        add_family("q2aa", parsed["gems_aa"])
        add_family("q2bb", parsed["gems_aa"])
    if parsed["constrain_g2"]:
        add_family("g2ab", parsed["gems_ab"])
        add_family("g2ba", parsed["gems_ab"])
        add_family("g2aa", parsed["gems_ab"], 2)
    return names, dims


class _DQGSymMatrixFreeOperator:
    """Matrix-free DQG operator using the CPU irrep/pair maps on CUDA."""

    def __init__(self, meta, block_dims, n_primal, n_dual, device, dtype,
                 cuda_au=False, cuda_atu=False, cuda_normal=False,
                 cuda_verbose=False):
        import torch

        parsed = _parse_dqg_sym_meta(meta)
        self.torch = torch
        self.parsed = parsed
        self.nirrep = parsed["nirrep"]
        self.amo = parsed["amo"]
        self.na = float(meta["nalpha_active"])
        self.nb = float(meta["nbeta_active"])
        self.spin_singlet = parsed["spin_singlet"]
        self.constrain_sz = parsed["constrain_sz"]
        self.constrain_spin = parsed["constrain_spin"]
        self.n_primal = int(n_primal)
        self.n_dual = int(n_dual)
        self.device = device
        self.dtype = dtype
        self.cuda_au = bool(cuda_au)
        self.cuda_atu = bool(cuda_atu)
        self.cuda_normal = False
        self.cuda_normal_module = None

        if not self.constrain_sz or not self.constrain_spin:
            raise ValueError(
                "DQG/sym CUDA matrix-free currently requires "
                "constrain_sz=True and constrain_spin=True."
            )
        if not parsed["constrain_q2"] or not parsed["constrain_g2"]:
            raise ValueError("DQG/sym matrix-free requires DQG with Q2 and G2 blocks.")
        if device.type != "cuda":
            raise ValueError("DQG/sym matrix-free CUDA kernels require a CUDA device.")
        if dtype != torch.double:
            raise ValueError("DQG/sym matrix-free CUDA kernels currently support float64 solves only.")
        if not (self.cuda_au and self.cuda_atu):
            raise ValueError("DQG/sym matrix-free requires both CUDA Au and CUDA ATu kernels.")
        if cuda_normal:
            # The generic symmetry kernel intentionally uses the existing
            # scratch-backed normal path: ATu(y) followed by Au(ATu(y)).
            cuda_normal = False

        expected_names, expected_dims = _dqg_sym_block_names_and_dims(meta)
        if len(block_dims) != len(expected_dims):
            raise ValueError(
                f"DQG/sym matrix-free expected {len(expected_dims)} primal blocks, "
                f"got {len(block_dims)}."
            )
        for name, got, expected in zip(expected_names, block_dims, expected_dims):
            if int(got) != int(expected):
                raise ValueError(
                    f"DQG/sym matrix-free block {name} expected dim {expected}, got {got}."
                )
        if sum(int(dim) * int(dim) for dim in expected_dims) != self.n_primal:
            raise ValueError("DQG/sym matrix-free primal dimension mismatch.")

        self.span_type = []
        self.span_h = []
        self.span_size = []
        pos = 0

        def add_span(name, h, count):
            nonlocal pos
            count = int(count)
            if count <= 0:
                return
            self.span_type.append(_DQG_SYM_SPAN[name])
            self.span_h.append(int(h))
            self.span_size.append(count)
            pos += count

        if self.constrain_sz:
            add_span("trace_d2ab", -1, 1)
            add_span("trace_d2aa", -1, 1)
            add_span("trace_d2bb", -1, 1)
        else:
            add_span("trace_d2_total", -1, 1)
        for h in range(self.nirrep):
            add_span("herm_d2aa", h, parsed["gems_aa"][h] ** 2)
        for h in range(self.nirrep):
            add_span("herm_d2bb", h, parsed["gems_aa"][h] ** 2)
        for h in range(self.nirrep):
            add_span("herm_d2ab", h, parsed["gems_ab"][h] ** 2)
        for h in range(self.nirrep):
            add_span("d1a_q1a", h, parsed["amopi"][h] ** 2)
        for h in range(self.nirrep):
            add_span("d1b_q1b", h, parsed["amopi"][h] ** 2)
        if self.constrain_sz:
            for h in range(self.nirrep):
                add_span("contract_ab_d1a", h, parsed["amopi"][h] ** 2)
            for h in range(self.nirrep):
                add_span("contract_ab_d1b", h, parsed["amopi"][h] ** 2)
            for h in range(self.nirrep):
                add_span("contract_aa_d1a", h, parsed["amopi"][h] ** 2)
            for h in range(self.nirrep):
                add_span("contract_bb_d1b", h, parsed["amopi"][h] ** 2)
        else:
            for h in range(self.nirrep):
                add_span("contract_mix_a", h, parsed["amopi"][h] ** 2)
            for h in range(self.nirrep):
                add_span("contract_mix_b", h, parsed["amopi"][h] ** 2)
        add_span("spin_trace", -1, 1)
        if self.spin_singlet:
            for h in range(self.nirrep):
                add_span("spin_d1_eq", h, parsed["amopi"][h] ** 2)
            for h in range(self.nirrep):
                add_span("spin_d2aa_eq", h, parsed["gems_aa"][h] ** 2)
            for h in range(self.nirrep):
                add_span("spin_d2aa_from_ab", h, parsed["gems_aa"][h] ** 2)
            for h in range(self.nirrep):
                add_span("spin_d2bb_from_ab", h, parsed["gems_aa"][h] ** 2)
            for h in range(self.nirrep):
                add_span("spin_d200_singlet", h, parsed["gems_ab"][h] ** 2)
            for h in range(self.nirrep):
                add_span("spin_d2ab_sym", h, parsed["gems_ab"][h] ** 2)
        else:
            for h in range(self.nirrep):
                add_span("spin_d200_nonsinglet", h, 4 * parsed["gems_ab"][h] ** 2)
        add_span("spin_g2ba_col_trace", 0, parsed["gems_ab"][0])
        add_span("spin_g2ba_row_trace", 0, parsed["gems_ab"][0])
        for h in range(self.nirrep):
            add_span("q2ab", h, parsed["gems_ab"][h] ** 2)
        for h in range(self.nirrep):
            add_span("q2aa", h, parsed["gems_aa"][h] ** 2)
        for h in range(self.nirrep):
            add_span("q2bb", h, parsed["gems_aa"][h] ** 2)
        for h in range(self.nirrep):
            add_span("g2ab", h, parsed["gems_ab"][h] ** 2)
        for h in range(self.nirrep):
            add_span("g2ba", h, parsed["gems_ab"][h] ** 2)
        for h in range(self.nirrep):
            add_span("g2aa", h, 4 * parsed["gems_ab"][h] ** 2)
        if pos != self.n_dual:
            raise ValueError(
                f"DQG/sym matrix-free dual block spans cover {pos} rows, "
                f"expected {self.n_dual}."
            )

        def int_tensor(values):
            return torch.tensor(values, device=device, dtype=torch.int32).contiguous()

        def long_tensor(values):
            return torch.tensor(values, device=device, dtype=torch.long).contiguous()

        self.amopi_t = int_tensor(parsed["amopi"])
        self.pitzer_t = int_tensor(parsed["pitzer_offset"])
        self.symmetry_t = int_tensor(parsed["symmetry"])
        self.sym_pair_t = int_tensor(parsed["sym_pair"])
        self.gems_ab_t = int_tensor(parsed["gems_ab"])
        self.gems_aa_t = int_tensor(parsed["gems_aa"])
        self.offsets_t = long_tensor(parsed["offsets"])
        self.ab_offsets_t = long_tensor(parsed["ab_pair_offsets"])
        self.aa_offsets_t = long_tensor(parsed["aa_pair_offsets"])
        self.ab_i_t = int_tensor(parsed["ab_i"])
        self.ab_j_t = int_tensor(parsed["ab_j"])
        self.aa_i_t = int_tensor(parsed["aa_i"])
        self.aa_j_t = int_tensor(parsed["aa_j"])
        self.ibas_ab_t = int_tensor(parsed["ibas_ab"])
        self.ibas_aa_t = int_tensor(parsed["ibas_aa"])
        self.span_type_t = int_tensor(self.span_type)
        self.span_h_t = int_tensor(self.span_h)
        self.span_size_t = long_tensor(self.span_size)
        self.row_type_t = torch.repeat_interleave(
            self.span_type_t, self.span_size_t, output_size=self.n_dual,
        ).contiguous()
        self.row_h_t = torch.repeat_interleave(
            self.span_h_t, self.span_size_t, output_size=self.n_dual,
        ).contiguous()
        self.row_local_t = torch.cat([
            torch.arange(int(count), device=device, dtype=torch.int32)
            for count in self.span_size
        ]).contiguous()
        if self.row_type_t.numel() != self.n_dual:
            raise ValueError("DQG/sym matrix-free row metadata size mismatch.")

        module = _load_dqg_sym_cuda_module(cuda_verbose)
        self.cuda_au_module = module
        self.cuda_atu_module = module

    def _cuda_kernel_args(self, output_size):
        return (
            self.nirrep, self.amo, float(self.na), float(self.nb),
            self.amopi_t, self.pitzer_t, self.symmetry_t, self.sym_pair_t,
            self.gems_ab_t, self.gems_aa_t, self.offsets_t,
            self.ab_offsets_t, self.aa_offsets_t, self.ab_i_t, self.ab_j_t,
            self.aa_i_t, self.aa_j_t, self.ibas_ab_t, self.ibas_aa_t,
            self.row_type_t, self.row_h_t, self.row_local_t, int(output_size),
        )

    def Au(self, x):
        return self.cuda_au_module.dqg_sym_au(
            x.contiguous(), *self._cuda_kernel_args(self.n_dual),
        )

    def Au_out(self, x, out):
        self.cuda_au_module.dqg_sym_au_out(
            x.contiguous(), out, *self._cuda_kernel_args(self.n_dual),
        )
        return out

    def ATu(self, y):
        return self.cuda_atu_module.dqg_sym_atu(
            y.contiguous(), *self._cuda_kernel_args(self.n_primal),
        )

    def ATu_out(self, y, out):
        self.cuda_atu_module.dqg_sym_atu_out(
            y.contiguous(), out, *self._cuda_kernel_args(self.n_primal),
        )
        return out

    def normal_out(self, y, scratch, out):
        self.ATu_out(y, scratch)
        self.Au_out(scratch, out)
        return out

    def normal_direct_cache_size(self):
        raise RuntimeError("DQG/sym cached direct normal kernel is unavailable.")

    def normal_cached_out(self, y, cache, out):
        raise RuntimeError("DQG/sym cached direct normal kernel is unavailable.")

    def normal_build_cache_out(self, y, cache):
        raise RuntimeError("DQG/sym cached direct normal kernel is unavailable.")

    def normal_cached_range_out(self, y, cache, out, row_start, row_stop):
        raise RuntimeError("DQG/sym cached direct normal row-range kernel is unavailable.")

    def normal_direct_range_out(self, y, out, row_start, row_stop):
        raise RuntimeError("DQG/sym direct normal row-range kernel is unavailable.")


def _make_dqg_matrix_free_operator(meta, block_dims, n_primal, n_dual, device,
                                   dtype, cuda_au=False, cuda_atu=False,
                                   cuda_normal=False, cuda_verbose=False):
    operator = meta.get("operator")
    if operator == "dqg_c1":
        return _DQGC1MatrixFreeOperator(
            meta, block_dims, n_primal, n_dual, device, dtype,
            cuda_au=cuda_au, cuda_atu=cuda_atu, cuda_normal=cuda_normal,
            cuda_verbose=cuda_verbose,
        )
    if operator == "dqg_sym":
        return _DQGSymMatrixFreeOperator(
            meta, block_dims, n_primal, n_dual, device, dtype,
            cuda_au=cuda_au, cuda_atu=cuda_atu, cuda_normal=cuda_normal,
            cuda_verbose=cuda_verbose,
        )
    raise ValueError(f"Unsupported matrix-free DQG operator '{operator}'.")


def gpu_admm_solve(c, b, block_dims, rows, cols, vals, progress_monitor_py,
                   x_init, y_init, z_init, mu_init,
                   maxiter, sdp_error_convergence, sdp_objective_convergence,
                   cg_maxiter, cg_convergence, dynamic_cg_convergence,
                   mu_update_frequency, print_level, oiter_start, iiter_start,
                   profile_timing=False,
                   a_crow=None, a_col=None, a_vals=None,
                   at_crow=None, at_col=None, at_vals=None,
                   matrix_free_meta=None, matrix_free=False,
                   matrix_free_validate=False, matrix_free_at_csr=False,
                   matrix_free_cuda_au=False,
                   matrix_free_cuda_atu=False,
                   cuda_verbose=False,
                   validate_au_range=False,
                   cg_dynamic_factor=0.01,
                   cg_fused_normal=False,
                   shard_primal=False,
                   shard_primal_store=False,
                   psd_projection_multi_gpu=False,
                   psd_projection_devices="",
                   psd_projection_base_memory_fraction=0.92,
                   psd_projection_workspace_scale=5.0,
                   psd_projection_avoid_base_large_blocks=True,
                   accel_options=None):
    """Solve an SDP with PyTorch based GPU-ADMM (BPSDP).

    This is the internal C++/Python bridge. Runtime controls are explicit
    arguments so a solve is fully determined by the corresponding Psi4
    `GPU_ADMM_*` module options.
    """
    if print_level > 1:
        print("  ==> [Debug Python] Entered gpu_admm_solve", flush=True)
    import torch
    torch.set_num_threads(1)
    import numpy as np
    import time
    solve_wall_start = time.perf_counter()
    if print_level > 1:
        print("  ==> [Debug Python] torch/numpy imported successfully", flush=True)
        print("  ==> [Debug Python] block_dims:", list(block_dims), flush=True)



    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    else:
        device = torch.device("cpu")

    profile_timing = bool(profile_timing)
    matrix_free = bool(matrix_free)
    matrix_free_validate = bool(matrix_free_validate)
    matrix_free_at_csr = bool(matrix_free_at_csr)
    matrix_free_cuda_au = bool(matrix_free_cuda_au)
    matrix_free_cuda_atu = bool(matrix_free_cuda_atu)
    cuda_verbose = bool(cuda_verbose)
    validate_au_range = bool(validate_au_range)
    cg_dynamic_factor = max(0.0, float(cg_dynamic_factor))
    if device.type != "cuda" and (matrix_free_cuda_au or matrix_free_cuda_atu):
        if print_level > 0:
            print(
                "  ==> GPU-ADMM: custom matrix-free CUDA kernels were "
                "requested, but no CUDA device is available; using the "
                "portable matrix-free operator.",
                flush=True,
            )
        matrix_free_cuda_au = False
        matrix_free_cuda_atu = False
    cg_fused_normal = bool(cg_fused_normal)
    shard_primal = bool(shard_primal)
    shard_primal_store = bool(shard_primal_store)
    psd_projection_multi_gpu = bool(psd_projection_multi_gpu)
    psd_projection_devices = "" if psd_projection_devices is None else str(psd_projection_devices)
    psd_projection_base_memory_fraction = float(psd_projection_base_memory_fraction)
    psd_projection_workspace_scale = float(psd_projection_workspace_scale)
    psd_projection_avoid_base_large_blocks = bool(psd_projection_avoid_base_large_blocks)

    def parse_cuda_device_ids(device_string, require_multi=True):
        if device.type != "cuda":
            return []
        visible = torch.cuda.device_count()
        if require_multi and visible <= 1:
            return []
        if device_string.strip():
            ids = []
            for part in device_string.split(","):
                part = part.strip()
                if not part:
                    continue
                idx = int(part)
                if idx < 0 or idx >= visible:
                    raise ValueError(
                        f"GPU-ADMM requested CUDA device {idx}, "
                        f"but only {visible} visible device(s) are available."
                    )
                if idx not in ids:
                    ids.append(idx)
        else:
            ids = list(range(visible))
        if 0 not in ids:
            ids.insert(0, 0)
        if require_multi and len(ids) <= 1:
            return []
        return ids

    def parse_psd_device_ids():
        if not psd_projection_multi_gpu:
            return []
        return parse_cuda_device_ids(psd_projection_devices, require_multi=True)

    psd_device_ids = parse_psd_device_ids()

    has_matrix_free_meta = (
        isinstance(matrix_free_meta, dict) and
        matrix_free_meta.get("operator") in ("dqg_c1", "dqg_sym")
    )
    use_matrix_free_operator = matrix_free and has_matrix_free_meta
    if matrix_free and not has_matrix_free_meta:
        raise ValueError("GPU_ADMM_MATRIX_FREE was requested, but DQG matrix-free metadata was not supplied.")
    matrix_free_operator_label = "DQG/C1"
    if has_matrix_free_meta and matrix_free_meta.get("operator") == "dqg_sym":
        matrix_free_operator_label = "DQG/symmetry"
    shard_primal_operator_eligible = (
        use_matrix_free_operator and
        matrix_free_meta.get("operator") == "dqg_c1" and
        not matrix_free_at_csr and
        matrix_free_cuda_au and
        matrix_free_cuda_atu and
        cg_fused_normal and
        device.type == "cuda"
    )
    has_a_csr = (
        a_crow is not None and a_col is not None and a_vals is not None and
        getattr(a_vals, "shape", (0,))[0] > 0
    )
    has_at_csr = (
        at_crow is not None and at_col is not None and at_vals is not None and
        getattr(at_vals, "shape", (0,))[0] > 0
    )
    has_direct_csr = has_a_csr and has_at_csr

    report_device_ids = []
    if device.type == "cuda":
        for dev_idx in psd_device_ids:
            if dev_idx not in report_device_ids:
                report_device_ids.append(dev_idx)
        if not report_device_ids:
            report_device_ids = [torch.cuda.current_device()]

    if device.type == "cuda" and report_device_ids:
        for dev_idx in report_device_ids:
            with torch.cuda.device(dev_idx):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(dev_idx)

    timings = {
        # Setup time for the matvec operator explicit CSR(A,A^T) build on the
        # legacy path, or the matrix-free operator construction which includes the one-time
        # CUDA-extension JIT compile on the matrix-free path.
        "operator_setup": 0.0,
        "initial_residual": 0.0,
        "cg_rhs": 0.0,
        "cg_solve": 0.0,
        "cg_atu": 0.0,
        "cg_au": 0.0,
        "cg_normal": 0.0,
        "cg_normal_cache": 0.0,
        "cg_normal_rows": 0.0,
        "xz_aty": 0.0,
        "psd_projection": 0.0,
        "psd_projection_float64": 0.0,
        "residual_objective": 0.0,
        "stage_copy": 0.0,
        "host_copy": 0.0,
    }
    psd_dim_timings = {}
    psd_dim_counts = {}
    psd_device_timings = {}
    psd_device_counts = {}

    def sync_for_timing():
        if profile_timing and device.type == "cuda":
            for dev_idx in report_device_ids:
                torch.cuda.synchronize(dev_idx)

    def timer_start():
        sync_for_timing()
        return time.perf_counter()

    def timer_stop(name, start):
        sync_for_timing()
        elapsed = time.perf_counter() - start
        if profile_timing:
            timings[name] = timings.get(name, 0.0) + elapsed
        return elapsed

    def timer_elapsed(start):
        sync_for_timing()
        return time.perf_counter() - start

    def cuda_memory_report(label):
        if print_level > 0 and device.type == "cuda":
            parts = []
            for dev_idx in report_device_ids:
                torch.cuda.synchronize(dev_idx)
                allocated = torch.cuda.memory_allocated(dev_idx) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(dev_idx) / (1024 ** 3)
                peak = torch.cuda.max_memory_allocated(dev_idx) / (1024 ** 3)
                parts.append(
                    f"cuda:{dev_idx} allocated={allocated:.3f} GiB, "
                    f"reserved={reserved:.3f} GiB, peak_allocated={peak:.3f} GiB"
                )
            print(f"  ==> GPU-ADMM CUDA memory {label}: " + "; ".join(parts), flush=True)

    if print_level > 0:
        print(f"  ==> GPU-ADMM: Solving SDP on device: {device} <==")
        if use_matrix_free_operator:
            if matrix_free_at_csr:
                print(
                    "  ==> GPU-ADMM: Hybrid matrix-free mode enabled; "
                    "Au is matrix-free and ATu uses explicit CSR(A^T).",
                    flush=True,
                )
            else:
                print(
                    f"  ==> GPU-ADMM: Matrix-free {matrix_free_operator_label} operator enabled; "
                    "explicit CSR(A,A^T) will not be used for matvecs.",
                    flush=True,
                )
            if matrix_free_cuda_au:
                print(
                    "  ==> GPU-ADMM: Matrix-free Au will use the custom CUDA kernel.",
                    flush=True,
                )
                if not matrix_free_validate:
                    print(
                        "  ==> GPU-ADMM WARNING: custom CUDA Au is running without "
                        "GPU_ADMM_MATRIX_FREE_VALIDATE; use validation after kernel changes.",
                        flush=True,
                    )
            if matrix_free_cuda_atu:
                print(
                    "  ==> GPU-ADMM: Matrix-free ATu will use the custom CUDA kernel.",
                    flush=True,
                )
                if matrix_free_at_csr:
                    print(
                        "  ==> GPU-ADMM: GPU_ADMM_MATRIX_FREE_AT_CSR is enabled, "
                        "so production ATu matvecs will use CSR(A^T) after validation.",
                        flush=True,
                    )
                if not matrix_free_validate:
                    print(
                        "  ==> GPU-ADMM WARNING: custom CUDA ATu is running without "
                        "GPU_ADMM_MATRIX_FREE_VALIDATE; use validation after kernel changes.",
                        flush=True,
                    )
            if matrix_free_validate:
                print(
                    "  ==> GPU-ADMM: Matrix-free validation requested; "
                    "explicit CSR(A,A^T) must also be present for random matvec checks.",
                    flush=True,
                )
        if psd_projection_multi_gpu:
            if psd_device_ids:
                print(
                    "  ==> GPU-ADMM: PSD projection will use multiple CUDA GPUs: "
                    + ",".join(f"cuda:{idx}" for idx in psd_device_ids),
                    flush=True,
                )
            else:
                print(
                    "  ==> GPU-ADMM: PSD multi-GPU requested, but fewer than two "
                    "valid CUDA devices are visible; using the single-GPU PSD path.",
                    flush=True,
                )
        if dynamic_cg_convergence and abs(cg_dynamic_factor - 0.01) > 1.0e-15:
            print(
                "  ==> GPU-ADMM: Dynamic CG convergence factor set to "
                f"{cg_dynamic_factor:.3e} (legacy default 1.000e-02).",
                flush=True,
            )

    # Verify input shapes
    sum_block_size = sum(d * d for d in block_dims if d > 0)
    if sum_block_size != c.shape[0]:
        raise ValueError(f"Block dimensions sum {sum_block_size} does not match primal dimension {c.shape[0]}")
    if x_init.shape[0] != c.shape[0] or z_init.shape[0] != c.shape[0]:
        raise ValueError(f"Primal vectors shape mismatch: x {x_init.shape}, z {z_init.shape}, c {c.shape}")
    if y_init.shape[0] != b.shape[0]:
        raise ValueError(f"Dual vectors shape mismatch: y {y_init.shape}, b {b.shape}")

    def infer_primal_block_names():
        names = [f"block{idx}" for idx in range(len(block_dims))]
        if not has_matrix_free_meta:
            return names
        if matrix_free_meta.get("operator") == "dqg_sym":
            try:
                expected, expected_dims = _dqg_sym_block_names_and_dims(matrix_free_meta)
                if len(expected) != len(block_dims):
                    return names
                for name, dim, expected_dim in zip(expected, block_dims, expected_dims):
                    if int(dim) != int(expected_dim):
                        return names
                return expected
            except (KeyError, TypeError, ValueError):
                return names
        try:
            n = int(matrix_free_meta["amo"])
            gab = int(matrix_free_meta["gab"])
            gaa = int(matrix_free_meta["gaa"])
            constrain_spin = bool(matrix_free_meta["constrain_spin"])
            spin_singlet = (
                abs(float(matrix_free_meta["nalpha_active"]) -
                    float(matrix_free_meta["nbeta_active"])) < 1.0e-12
            )
            expected = ["d2ab", "d2aa", "d2bb"]
            if constrain_spin:
                expected.append("d200")
            expected += [
                "d1a", "d1b", "q1a", "q1b",
                "q2ab", "q2aa", "q2bb", "g2ab", "g2ba", "g2aa",
            ]
            d200_dim = gab if spin_singlet else 2 * gab
            expected_dims = {
                "d2ab": gab,
                "d2aa": gaa,
                "d2bb": gaa,
                "d200": d200_dim,
                "d1a": n,
                "d1b": n,
                "q1a": n,
                "q1b": n,
                "q2ab": gab,
                "q2aa": gaa,
                "q2bb": gaa,
                "g2ab": gab,
                "g2ba": gab,
                "g2aa": 2 * gab,
            }
            if len(expected) != len(block_dims):
                return names
            for name, dim in zip(expected, block_dims):
                if int(dim) != int(expected_dims[name]):
                    return names
            return expected
        except (KeyError, TypeError, ValueError):
            return names

    block_names = infer_primal_block_names()

    # Build block groups for batched eigensolves
    block_groups = {}
    offset = 0
    for block_idx, dim in enumerate(block_dims):
        if dim <= 0:
            continue
        if dim not in block_groups:
            block_groups[dim] = []
        block_groups[dim].append((block_idx, offset))
        offset += dim * dim

    psd_work_by_device = {}
    if psd_device_ids:
        psd_work_by_device = {idx: [] for idx in psd_device_ids}
        psd_device_loads = {idx: 0 for idx in psd_device_ids}
        psd_device_dim_counts = {idx: {} for idx in psd_device_ids}
        psd_items = []
        for dim, groups in block_groups.items():
            for group in groups:
                block_idx, _ = group
                block_name = block_names[block_idx] if block_idx < len(block_names) else f"block{block_idx}"
                psd_items.append((int(dim) ** 3, int(dim), block_name, group))

        # Host-scatter changes base-GPU memory, full masters live
        # on the host, so cuda:0 holds only its balanced shard of the store+dual like
        # every other device, not 3*primal+2*dual. Without this, the scheduler thinks
        # cuda:0 is full (budget=0) and idles it, putting every block including the
        # largest g2aa onto the non-base GPUs. Detect host-scatter from the same
        # settings and hardware conditions used for the master build below.
        _host_scatter_sched = (
            shard_primal_operator_eligible and shard_primal_store and
            bool(psd_device_ids) and len(psd_device_ids) > 1 and
            device.type == "cuda"
        )
        base_idx = torch.cuda.current_device()
        base_psd_budget = 0
        base_psd_fraction = psd_projection_base_memory_fraction
        base_psd_workspace_scale = psd_projection_workspace_scale
        base_resident_vector_bytes = 0
        if psd_items and base_idx in psd_device_ids:
            if base_psd_fraction > 0.0:
                base_psd_fraction = min(max(base_psd_fraction, 0.50), 0.99)
                base_psd_workspace_scale = max(base_psd_workspace_scale, 1.0)
                base_total_bytes = torch.cuda.get_device_properties(base_idx).total_memory
                primal_bytes = int(c.shape[0]) * 8
                dual_bytes = int(b.shape[0]) * 8
                if _host_scatter_sched:
                    # cuda:0 holds ~one balanced shard of 5 store vectors (c,x,z,U,cz)
                    # + 2 dual shards (y,b), divided across the PSD devices.
                    base_resident_vector_bytes = (
                        5 * primal_bytes + 2 * dual_bytes) // len(psd_device_ids)
                else:
                    base_resident_vector_bytes = 3 * primal_bytes + 2 * dual_bytes
                base_psd_budget = max(
                    0,
                    int(base_psd_fraction * base_total_bytes) -
                    base_resident_vector_bytes,
                )
            else:
                base_psd_fraction = 0.0
                base_psd_workspace_scale = max(base_psd_workspace_scale, 1.0)

        max_psd_dim = max((dim for _, dim, _, _ in psd_items), default=0)
        max_dim_count = sum(1 for _, dim, _, _ in psd_items if dim == max_psd_dim)
        non_base_device_count = sum(1 for idx in psd_device_ids if idx != base_idx)
        avoid_base_for_largest = (
            psd_projection_avoid_base_large_blocks and
            not _host_scatter_sched and  # with host-scatter cuda:0 is the BEST home
            max_psd_dim > 0 and          # for the largest block (it is otherwise empty)
            non_base_device_count >= max_dim_count
        )

        def psd_workspace_estimate(dim, count):
            block_bytes = int(dim) * int(dim) * 8
            return int(base_psd_workspace_scale * int(count) * block_bytes)

        def base_can_accept(dim):
            if base_idx not in psd_device_ids or base_psd_budget <= 0:
                return False
            counts = dict(psd_device_dim_counts[base_idx])
            counts[dim] = counts.get(dim, 0) + 1
            estimated_peak = max(
                psd_workspace_estimate(existing_dim, count)
                for existing_dim, count in counts.items()
            )
            return estimated_peak <= base_psd_budget

        psd_nonbase_transfer_penalty_dim = 256 if len(psd_device_ids) > 1 else 0

        def psd_assignment_cost(dim, dev_idx):
            dim = int(dim)
            cost = dim ** 3
            if psd_nonbase_transfer_penalty_dim > 0 and dev_idx != base_idx:
                cost += psd_nonbase_transfer_penalty_dim * dim * dim
            return cost

        for cost, dim, block_name, group in sorted(psd_items, key=lambda item: item[0], reverse=True):
            candidate_devices = [
                dev_idx for dev_idx in psd_device_ids
                if (
                    dev_idx != base_idx or
                    (
                        base_can_accept(dim) and
                        not (avoid_base_for_largest and dim == max_psd_dim)
                    )
                )
            ]
            if not candidate_devices:
                candidate_devices = [dev_idx for dev_idx in psd_device_ids if dev_idx != base_idx]
            if not candidate_devices:
                candidate_devices = list(psd_device_ids)
            dev_idx = min(
                candidate_devices,
                key=lambda idx: psd_device_loads[idx] + psd_assignment_cost(dim, idx),
            )
            psd_work_by_device[dev_idx].append((dim, block_name, group))
            psd_device_loads[dev_idx] += psd_assignment_cost(dim, dev_idx)
            psd_device_dim_counts[dev_idx][dim] = psd_device_dim_counts[dev_idx].get(dim, 0) + 1

        if print_level > 0:
            if base_idx in psd_device_ids and psd_items:
                print(
                    "  ==> GPU-ADMM PSD base-GPU scheduler: "
                    f"cuda:{base_idx} workspace budget={base_psd_budget / (1024 ** 3):.3f} GiB, "
                    f"resident vectors={base_resident_vector_bytes / (1024 ** 3):.3f} GiB, "
                    f"workspace scale={base_psd_workspace_scale:.2f}, "
                    f"usable memory fraction={base_psd_fraction:.2f}, "
                    f"non-base transfer penalty dim={psd_nonbase_transfer_penalty_dim}, "
                    f"avoid largest on base={'yes' if avoid_base_for_largest else 'no'}",
                    flush=True,
                )
            assignment_parts = []
            largest_parts = []
            for dev_idx in psd_device_ids:
                dim_counts = {}
                for dim, block_name, _ in psd_work_by_device[dev_idx]:
                    dim_counts[dim] = dim_counts.get(dim, 0) + 1
                    if dim == max_psd_dim:
                        largest_parts.append(f"{block_name}({dim})->cuda:{dev_idx}")
                if dim_counts:
                    dims = ",".join(
                        f"{count}x{dim}" for dim, count in sorted(dim_counts.items(), reverse=True)
                    )
                else:
                    dims = "idle"
                assignment_parts.append(f"cuda:{dev_idx}=[{dims}]")
            print(
                "  ==> GPU-ADMM PSD multi-GPU block assignment: "
                + "; ".join(assignment_parts),
                flush=True,
            )
            if largest_parts:
                print(
                    "  ==> GPU-ADMM PSD largest block placement: "
                    + "; ".join(largest_parts),
                    flush=True,
                )

    # ====== Validate and report the primal-vector shard layout ===========
    # Persistent primal-sized vectors (c, x, z, U, CG scratch) were initially
    # replicated on the base GPU, giving a huge single-GPU footprint defining the memory ceiling.
    # Here precompute, from the PSD block->device assignment, a per-device "ownership"
    # layout where each device owns the contiguous primal segments of the PSD blocks
    # assigned to it. This stage validates coverage and reports the projected
    # per-device footprint, the CUDA operator builds the executable plan later.
    # Gated by GPU_ADMM_SHARD_PRIMAL.
    shard_primal_enabled = (
        shard_primal and shard_primal_operator_eligible and
        bool(psd_device_ids) and len(psd_device_ids) > 1
    )
    if shard_primal and not shard_primal_enabled and print_level > 0:
        print(
            "  ==> GPU-ADMM: GPU_ADMM_SHARD_PRIMAL requested, but its "
            "matrix-free DQG/C1 fused-normal CUDA prerequisites or multi-GPU "
            "PSD device assignment are unavailable; using full vectors.",
            flush=True,
        )
    if shard_primal_enabled:
        # global primal segment (offset, size) for every PSD block, grouped by owner
        seg_by_device = {idx: [] for idx in psd_device_ids}
        for dev_idx, work_items in psd_work_by_device.items():
            for dim, _block_name, (_block_idx, offset) in work_items:
                seg_by_device[dev_idx].append((int(offset), int(dim) * int(dim)))
        # compact local offset within each device's shard (kept in global order)
        local_nprimal = {}
        covered = 0
        for dev_idx in psd_device_ids:
            loff = 0
            for _goff, bs in sorted(seg_by_device[dev_idx]):
                loff += bs
            local_nprimal[dev_idx] = loff
            covered += loff
        n_primal_total = int(c.shape[0])
        # sanity check, owned PSD-block segments must tile the full primal exactly once
        if covered != n_primal_total:
            if print_level > 0:
                print(
                    "  ==> GPU-ADMM P1 WARNING: owned primal segments cover "
                    f"{covered} of {n_primal_total} entries; disabling primal "
                    "sharding (each PSD block must be owned by exactly one device).",
                    flush=True,
                )
            shard_primal_enabled = False
        else:
            if print_level > 0:
                bytes_per = 8  # float64
                full_gib = n_primal_total * bytes_per / (1024 ** 3)
                parts = [
                    f"cuda:{d}={local_nprimal[d] * bytes_per / (1024 ** 3):.2f}GiB"
                    for d in psd_device_ids
                ]
                print(
                    "  ==> GPU-ADMM P1 primal-shard plan (per replicated primal "
                    f"vector): full={full_gib:.2f}GiB -> " + ", ".join(parts),
                    flush=True,
                )

    # ==============================================================================
    # host-scatter when the primal store is active (SHARD_PRIMAL + PRIMAL_STORE,
    # multi-GPU, single double stage). Keep the five full masters on the HOST so no
    # full vector is ever materialized on the base GPU. get_precision_tensors then
    # leaves the solve tensors on the host and the scatters copy host->device per
    # block. Removes the ~n^4 residents floor that may OOM before the loop.
    # Any non-store path pulls the solve tensors back onto the GPU inside get_precision_tensors.
    host_scatter = (
        shard_primal_enabled and shard_primal_store and
        device.type == "cuda"
    )
    _master_device = "cpu" if host_scatter else device
    if host_scatter and print_level > 0:
        print("  ==> GPU-ADMM: P1 host-scatter active. Full masters kept on the "
              "host; no full primal/dual vector is built on the base GPU.",
              flush=True)

    # Setup initial double precision tensors
    c_t = torch.from_numpy(c).to(device=_master_device, dtype=torch.double)
    b_t = torch.from_numpy(b).to(device=_master_device, dtype=torch.double)
    x_t = torch.from_numpy(x_init).to(device=_master_device, dtype=torch.double)
    y_t = torch.from_numpy(y_init).to(device=_master_device, dtype=torch.double)
    z_t = torch.from_numpy(z_init).to(device=_master_device, dtype=torch.double)
    cuda_memory_report("after vector transfer")

    # Initial variables
    mu = mu_init
    iiter_total = iiter_start
    bpsdp_iter = oiter_start
    mu_update_frequency = int(mu_update_frequency)
    prev_max_residual = None
    stagnation_consecutive = 0

    # ------------------------------------------------------------------
    # Convergence-acceleration options passed as a flat dict of numbers
    # from the C++ side with keywords except the always-on
    # energy-stagnation stop.
    # ------------------------------------------------------------------
    accel = dict(accel_options) if accel_options else {}

    def _accel_get(key, default):
        val = accel.get(key, default)
        try:
            return type(default)(val)
        except (TypeError, ValueError):
            return default

    # Over-relaxation: U <- alpha*A^T y + (1-alpha)(c - z); alpha=1 is off.
    admm_relaxation = _accel_get("relaxation", 1.0)
    if not (0.0 < admm_relaxation < 2.0):
        admm_relaxation = 1.0

    # Relative duality-gap target: effective tol = max(abs, rel*|E|).
    gap_relative = bool(_accel_get("gap_relative", 0))
    gap_relative_tol = _accel_get("gap_relative_tol", 0.0)

    # Always-on energy-stagnation stop. When feasibility is already met and the
    # primal (variational) energy has been flat to tol over a window, the energy
    # is converged even if the gap criterion is still moving on slowly.
    stagnation_window = _accel_get("stagnation_window", 100)
    stagnation_energy_tol = _accel_get("stagnation_energy_tol", 0.0)
    if stagnation_energy_tol <= 0.0:
        stagnation_energy_tol = float(sdp_objective_convergence)
    if stagnation_window < 2:
        stagnation_window = 2

    def effective_gap_tol(obj_primal, obj_dual):
        tol = float(sdp_objective_convergence)
        if gap_relative and gap_relative_tol > 0.0:
            scale = max(abs(obj_primal), abs(obj_dual), 1.0)
            tol = max(tol, gap_relative_tol * scale)
        return tol

    if print_level > 0:
        accel_msgs = []
        if admm_relaxation != 1.0:
            accel_msgs.append(f"over-relaxation alpha={admm_relaxation:.3f}")
        if gap_relative and gap_relative_tol > 0.0:
            accel_msgs.append(f"relative gap tol={gap_relative_tol:.2e}")
        accel_msgs.append(
            f"energy-stagnation stop(window={stagnation_window}, "
            f"tol={stagnation_energy_tol:.2e}, always on)"
        )
        print(
            "  ==> GPU-ADMM acceleration: " + "; ".join(accel_msgs) + ".",
            flush=True,
        )

    mu_update_frequency_scale = min(1.0, max(0.5, mu_update_frequency / 250.0))
    mu_update_coarse_threshold = 100.0
    mu_update_active_residual_threshold = 3.0
    mu_update_active_objective_threshold = 10.0
    mu_update_coarse_deadband = 2.0
    mu_update_active_deadband = 1.5
    mu_update_polish_deadband = 1.25
    mu_update_coarse_min_factor = 1.0e-3
    mu_update_coarse_max_factor = 500.0
    mu_update_active_min_factor = 0.05
    mu_update_active_max_factor = 20.0
    mu_update_polish_min_factor = 0.8
    mu_update_polish_max_factor = 1.25
    mu_update_polish_damping = 0.25

    if print_level > 0:
        if mu_update_frequency > 0:
            print(
                "  ==> GPU-ADMM: Hybrid adaptive mu update enabled "
                f"every {mu_update_frequency} iteration(s): "
                f"coarse damping={mu_update_frequency_scale:.2f}, "
                f"coarse clamp=[{mu_update_coarse_min_factor:.3g},"
                f"{mu_update_coarse_max_factor:.3g}], "
                f"polish clamp=[{mu_update_polish_min_factor:.2f},"
                f"{mu_update_polish_max_factor:.2f}].",
                flush=True,
            )
        else:
            print(
                "  ==> GPU-ADMM: Adaptive mu update disabled "
                "(MU_UPDATE_FREQUENCY <= 0).",
                flush=True,
            )

    def adaptive_mu_update(mu_value, primal_residual, dual_residual, objective_gap):
        nonlocal prev_max_residual, stagnation_consecutive
        if mu_update_frequency <= 0:
            return mu_value
        if not np.isfinite(mu_value) or mu_value <= 0.0:
            return mu_value
        if not np.isfinite(objective_gap):
            objective_gap = 0.0
        if (
            not np.isfinite(primal_residual) or
            not np.isfinite(dual_residual) or
            primal_residual < 0.0 or
            dual_residual < 0.0
        ):
            return mu_value

        ratio = primal_residual / max(dual_residual, 1.0e-15)
        if not np.isfinite(ratio) or ratio <= 0.0:
            return mu_value

        residual_level = (
            max(primal_residual, dual_residual) /
            max(sdp_error_convergence, 1.0e-30)
        )
        objective_level = (
            abs(objective_gap) /
            max(sdp_objective_convergence, 1.0e-30)
        )
        progress_level = max(residual_level, objective_level)

        # Stagnation detection
        curr_max_residual = max(primal_residual, dual_residual)
        is_stagnating = False
        if prev_max_residual is not None and prev_max_residual > 0.0:
            # If residuals decreased by less than 5% over the update cycle
            if curr_max_residual / prev_max_residual > 0.95:
                is_stagnating = True
        prev_max_residual = curr_max_residual

        if is_stagnating:
            stagnation_consecutive += 1
        else:
            stagnation_consecutive = 0

        if progress_level > mu_update_coarse_threshold:
            deadband = mu_update_coarse_deadband
            damping = mu_update_frequency_scale
            min_factor = mu_update_coarse_min_factor
            max_factor = mu_update_coarse_max_factor
        elif (
            residual_level > mu_update_active_residual_threshold or
            objective_level > mu_update_active_objective_threshold
        ):
            deadband = mu_update_active_deadband
            damping = mu_update_frequency_scale
            min_factor = mu_update_active_min_factor
            max_factor = mu_update_active_max_factor
        else:
            deadband = mu_update_polish_deadband
            damping = mu_update_polish_damping
            min_factor = mu_update_polish_min_factor
            max_factor = mu_update_polish_max_factor

        # If stagnating, dynamically contract the deadband and boost damping
        if is_stagnating:
            deadband = max(1.01, deadband / (1.5 ** stagnation_consecutive))
            damping = min(1.0, damping * 1.5)

        if (1.0 / deadband) <= ratio <= deadband:
            if stagnation_consecutive >= 2:
                # Shake solver: alternate multiplying/dividing by 1.5
                perturb_factor = 1.5 if (stagnation_consecutive % 2 == 0) else (1.0 / 1.5)
                updated_mu = mu_value * perturb_factor
                if print_level > 0:
                    print(
                        f"  ==> GPU-ADMM: Stagnation detected (consecutive={stagnation_consecutive}). "
                        f"Forcing mu perturbation: {mu_value:.3e} -> {updated_mu:.3e}",
                        flush=True,
                    )
                return updated_mu
            return mu_value

        factor = ratio ** damping
        factor = min(max(factor, min_factor), max_factor)
        updated_mu = mu_value * factor
        if not np.isfinite(updated_mu) or updated_mu <= 0.0:
            return mu_value
        return updated_mu

    def sparse_storage_gib():
        if use_matrix_free_operator and not matrix_free_validate and not matrix_free_at_csr:
            return 0.0, 0.0
        if use_matrix_free_operator and matrix_free_at_csr and has_at_csr:
            final_bytes = at_crow.nbytes + at_col.nbytes + at_vals.shape[0] * 8
            return final_bytes / (1024 ** 3), 0.0
        if has_direct_csr:
            final_bytes = (
                a_crow.nbytes + a_col.nbytes + at_crow.nbytes + at_col.nbytes +
                a_vals.shape[0] * 8 + at_vals.shape[0] * 8
            )
            return final_bytes / (1024 ** 3), 0.0
        nnz = vals.shape[0]
        csr_bytes = 2 * nnz * (8 + 8) + (b.shape[0] + c.shape[0] + 2) * 8
        coo_peak_bytes = nnz * (8 + 8 + 8) + nnz * (8 + 8) + max(b.shape[0], c.shape[0]) * 8
        return csr_bytes / (1024 ** 3), coo_peak_bytes / (1024 ** 3)

    if print_level > 0:
        csr_gib, coo_stage_gib = sparse_storage_gib()
        if use_matrix_free_operator and matrix_free_at_csr and has_at_csr:
            sparse_nnz = at_vals.shape[0]
        elif has_direct_csr:
            sparse_nnz = a_vals.shape[0]
        else:
            sparse_nnz = vals.shape[0]
        if use_matrix_free_operator and not matrix_free_validate and not matrix_free_at_csr:
            print(
                f"  ==> GPU-ADMM: Matrix-free {matrix_free_operator_label} matvecs active; "
                "estimated explicit CSR(A+A^T)=0.000 GiB allocated by this path.",
                flush=True,
            )
        elif use_matrix_free_operator and matrix_free_at_csr:
            if has_at_csr:
                print(
                    f"  ==> GPU-ADMM: Hybrid matrix-free/CSR matvecs active; "
                    f"estimated GPU CSR(A^T)={csr_gib:.3f} GiB, "
                    f"index dtype={at_crow.dtype}",
                    flush=True,
                )
            else:
                print(
                    "  ==> GPU-ADMM: Hybrid matrix-free/CSR matvecs requested, "
                    "but CSR(A^T) was not supplied.",
                    flush=True,
                )
        elif has_direct_csr:
            print(
                f"  ==> GPU-ADMM: Direct CSR handoff nnz={sparse_nnz}, "
                f"estimated final CSR(A+A^T)={csr_gib:.3f} GiB, "
                f"index dtype={a_crow.dtype}",
                flush=True,
            )
        else:
            print(
                f"  ==> GPU-ADMM: Sparse A nnz={sparse_nnz}, "
                f"estimated final CSR(A+A^T)={csr_gib:.3f} GiB, "
                f"largest sequential COO->CSR stage={coo_stage_gib:.3f} GiB",
                flush=True,
            )
            print(
                "  ==> GPU-ADMM: Legacy COO->CSR path active; direct CSR "
                "handoff was not supplied by the C++ plugin.",
                flush=True,
            )

    # Helper function to switch precision
    def get_precision_tensors(stage_dtype):
        sparse_build_start = timer_start()
        CG_normal_out = None
        cg_cuda_module = None
        # Cast vector variables
        c_solve = c_t.to(dtype=stage_dtype)
        b_solve = b_t.to(dtype=stage_dtype)
        x_solve = x_t.to(dtype=stage_dtype)
        y_solve = y_t.to(dtype=stage_dtype)
        z_solve = z_t.to(dtype=stage_dtype)

        if use_matrix_free_operator:
            request_cg_normal_module = (
                cg_fused_normal and
                matrix_free_meta.get("operator") == "dqg_c1" and
                not matrix_free_at_csr and
                matrix_free_cuda_au and
                matrix_free_cuda_atu and
                device.type == "cuda" and
                stage_dtype == torch.double
            )
            op = _make_dqg_matrix_free_operator(
                matrix_free_meta, list(block_dims), c.shape[0], b.shape[0],
                device, stage_dtype, cuda_au=matrix_free_cuda_au,
                cuda_atu=matrix_free_cuda_atu,
                cuda_normal=request_cg_normal_module,
                cuda_verbose=cuda_verbose,
            )
            cg_cuda_module = op.cuda_atu_module or op.cuda_au_module

            _p1_validate = (
                stage_dtype == torch.double and
                hasattr(op, "validate_au_range") and
                validate_au_range)
            if _p1_validate and int(c.shape[0]) > 200_000_000:
                # The validation suite allocates ~10 FULL primal/dual reference
                # vectors on the base GPU (~n_primal*8 bytes each). It might OOM
                # at large sizes. Its a correctness check meant for small
                # cases. Skip it if too big!
                if print_level > 0:
                    print("  ==> GPU-ADMM P1 validation SKIPPED: n_primal="
                          f"{int(c.shape[0])} too large for the full-vector "
                          "validation suite; run it at a small active space "
                          "(amo<=~20).", flush=True)
                _p1_validate = False
            if _p1_validate:
                op.validate_au_range()
                op.validate_au_range_compact()
                if psd_work_by_device and len(psd_work_by_device) > 1:
                    # Map each PSD block (by primal offset) to its owning device.
                    _off_to_name = {
                        int(off): nm for nm, (off, _dim) in op.blocks.items()
                    }
                    _block_owner = {}
                    for _dev, _items in psd_work_by_device.items():
                        for _dim, _bname, (_bidx, _offset) in _items:
                            _nm = _off_to_name.get(int(_offset))
                            if _nm is not None:
                                _block_owner[_nm] = int(_dev)
                    op.validate_distributed_normal(
                        list(psd_work_by_device.keys()), _block_owner)
                    _plan_v = op.build_shard_plan(
                        list(psd_work_by_device.keys()), _block_owner)
                    if _plan_v is not None:
                        op.validate_sharded_ops(_plan_v)
                        op.validate_primal_storage(_plan_v)
                        op.validate_u_store(_plan_v)
                        op.validate_au_from_store(_plan_v)
                        op.validate_psd_store(_plan_v)

            def Au_torch(u_tensor):
                return op.Au(u_tensor)

            def ATu_torch(u_tensor):
                return op.ATu(u_tensor)

            def Au_torch_out(u_tensor, out_tensor):
                return op.Au_out(u_tensor, out_tensor)

            def ATu_torch_out(u_tensor, out_tensor):
                return op.ATu_out(u_tensor, out_tensor)

            normal_module = op.cuda_normal_module
            use_cg_fused_normal = (
                cg_fused_normal and
                not matrix_free_at_csr and
                matrix_free_cuda_au and
                matrix_free_cuda_atu and
                normal_module is not None and
                (
                    hasattr(normal_module, "dqg_c1_normal_direct_out") or
                    hasattr(normal_module, "dqg_c1_normal_out")
                ) and
                device.type == "cuda" and
                stage_dtype == torch.double
            )
            if use_cg_fused_normal:
                use_cached_direct_normal = (
                    hasattr(normal_module, "dqg_c1_normal_direct_cached_out") and
                    hasattr(normal_module, "dqg_c1_normal_direct_cached_range_out") and
                    hasattr(normal_module, "dqg_c1_normal_direct_build_cache_out") and
                    hasattr(normal_module, "dqg_c1_normal_direct_cache_size") and
                    not op.spin_singlet
                )
                normal_cache = None
                normal_cache_size = 0
                if use_cached_direct_normal:
                    normal_cache_size = op.normal_direct_cache_size()
                    normal_cache = torch.empty(
                        normal_cache_size, device=device, dtype=stage_dtype,
                    )

                    def CG_normal_out(u_tensor, out_tensor, scratch_tensor):
                        if profile_timing:
                            cache_timer = timer_start()
                            op.normal_build_cache_out(u_tensor, normal_cache)
                            timer_stop("cg_normal_cache", cache_timer)
                            rows_timer = timer_start()
                            op.normal_cached_range_out(
                                u_tensor, normal_cache, out_tensor, 0, b.shape[0],
                            )
                            timer_stop("cg_normal_rows", rows_timer)
                            return out_tensor
                        return op.normal_cached_out(u_tensor, normal_cache, out_tensor)
                    CG_normal_out.needs_scratch = False
                else:
                    def CG_normal_out(u_tensor, out_tensor, scratch_tensor):
                        return op.normal_out(u_tensor, scratch_tensor, out_tensor)
                    CG_normal_out.needs_scratch = not (
                        hasattr(normal_module, "dqg_c1_normal_direct_out") and
                        not op.spin_singlet
                    )

                if print_level > 0:
                    if use_cached_direct_normal:
                        cache_gib = normal_cache_size * 8 / (1024 ** 3)
                        normal_mode = (
                            "compact-cache row-partitioned open-shell v2, "
                            f"cache={cache_gib:.3f} GiB"
                        )
                    elif (
                        hasattr(normal_module, "dqg_c1_normal_direct_out") and
                        not op.spin_singlet
                    ):
                        normal_mode = "direct row-partitioned open-shell v1"
                    else:
                        normal_mode = "scratch-backed q1-active v1"
                    print(
                        "  ==> GPU-ADMM: CG normal operator will use CUDA "
                        f"fused-normal entry point ({normal_mode}).",
                        flush=True,
                    )


            elif cg_fused_normal and print_level > 0:
                reasons = []
                if matrix_free_at_csr:
                    reasons.append("GPU_ADMM_MATRIX_FREE_AT_CSR is enabled")
                if matrix_free_meta.get("operator") != "dqg_c1":
                    reasons.append("the selected matrix-free operator does not provide a fused-normal kernel")
                if not matrix_free_cuda_au:
                    reasons.append("GPU_ADMM_MATRIX_FREE_CUDA_AU is disabled")
                if not matrix_free_cuda_atu:
                    reasons.append("GPU_ADMM_MATRIX_FREE_CUDA_ATU is disabled")
                if normal_module is None:
                    reasons.append("CUDA matrix-free module is unavailable")
                elif not (
                    hasattr(normal_module, "dqg_c1_normal_direct_out") or
                    hasattr(normal_module, "dqg_c1_normal_out")
                ):
                    reasons.append("CUDA module lacks a normal operator entry point")
                if device.type != "cuda":
                    reasons.append("solve device is not CUDA")
                if stage_dtype != torch.double:
                    reasons.append("stage dtype is not float64")
                if not reasons:
                    reasons.append("requirements were not met")
                print(
                    "  ==> GPU-ADMM: GPU_ADMM_CG_FUSED_NORMAL requested, "
                    "but disabled: " + "; ".join(reasons) + ".",
                    flush=True,
                )

            hybrid_at_csr = None
            if matrix_free_validate:
                if not has_direct_csr:
                    raise ValueError(
                        "GPU_ADMM_MATRIX_FREE_VALIDATE requires direct CSR(A,A^T) "
                        "handoff from C++."
                    )
                a_crow_t = torch.from_numpy(a_crow).to(device=device)
                a_col_t = torch.from_numpy(a_col).to(device=device)
                a_values = torch.from_numpy(a_vals).to(device=device, dtype=stage_dtype)
                A_csr = torch.sparse_csr_tensor(
                    a_crow_t, a_col_t, a_values, size=(b.shape[0], c.shape[0])
                )
                at_crow_t = torch.from_numpy(at_crow).to(device=device)
                at_col_t = torch.from_numpy(at_col).to(device=device)
                at_values = torch.from_numpy(at_vals).to(device=device, dtype=stage_dtype)
                A_t_csr = torch.sparse_csr_tensor(
                    at_crow_t, at_col_t, at_values, size=(c.shape[0], b.shape[0])
                )

                gen = torch.Generator(device=device)
                gen.manual_seed(20260602)
                x_probe = torch.randn(c.shape[0], device=device, dtype=stage_dtype, generator=gen)
                y_probe = torch.randn(b.shape[0], device=device, dtype=stage_dtype, generator=gen)
                ax_ref = torch.mv(A_csr, x_probe)
                ax_mf = Au_torch(x_probe)
                aty_ref = torch.mv(A_t_csr, y_probe)
                aty_mf = ATu_torch(y_probe)
                ax_py = None
                aty_py = None
                can_build_py_ref = matrix_free_meta.get("operator") == "dqg_c1"
                if matrix_free_cuda_au and can_build_py_ref:
                    py_ref_op = _make_dqg_matrix_free_operator(
                        matrix_free_meta, list(block_dims), c.shape[0], b.shape[0],
                        device, stage_dtype, cuda_au=False, cuda_atu=False,
                    )
                    ax_py = py_ref_op.Au(x_probe)
                if matrix_free_cuda_atu and can_build_py_ref:
                    if ax_py is None:
                        py_ref_op = _make_dqg_matrix_free_operator(
                            matrix_free_meta, list(block_dims), c.shape[0], b.shape[0],
                            device, stage_dtype, cuda_au=False, cuda_atu=False,
                        )
                    aty_py = py_ref_op.ATu(y_probe)
                ax_abs = float(torch.max(torch.abs(ax_ref - ax_mf)))
                aty_abs = float(torch.max(torch.abs(aty_ref - aty_mf)))
                ax_rel = float(torch.linalg.norm(ax_ref - ax_mf) / torch.clamp(torch.linalg.norm(ax_ref), min=1.0))
                aty_rel = float(torch.linalg.norm(aty_ref - aty_mf) / torch.clamp(torch.linalg.norm(aty_ref), min=1.0))
                dot1 = float(torch.dot(ax_mf, y_probe))
                dot2 = float(torch.dot(x_probe, aty_mf))
                dot_diff = abs(dot1 - dot2)
                dot_rel = dot_diff / max(abs(dot1), abs(dot2), 1.0)
                if print_level > 0:
                    print(
                        "  ==> GPU-ADMM matrix-free validation: "
                        f"max|Ax|diff={ax_abs:.3e}, rel={ax_rel:.3e}; "
                        f"max|ATy|diff={aty_abs:.3e}, rel={aty_rel:.3e}; "
                        f"adjoint_diff={dot_diff:.3e}, adjoint_rel={dot_rel:.3e}",
                        flush=True,
                    )
                    if ax_py is not None:
                        cuda_py_abs = float(torch.max(torch.abs(ax_py - ax_mf)))
                        cuda_py_rel = float(
                            torch.linalg.norm(ax_py - ax_mf) /
                            torch.clamp(torch.linalg.norm(ax_py), min=1.0)
                        )
                        print(
                            "  ==> GPU-ADMM CUDA Au vs Python Au: "
                            f"max|diff|={cuda_py_abs:.3e}, rel={cuda_py_rel:.3e}",
                            flush=True,
                        )
                    if aty_py is not None:
                        cuda_py_abs = float(torch.max(torch.abs(aty_py - aty_mf)))
                        cuda_py_rel = float(
                            torch.linalg.norm(aty_py - aty_mf) /
                            torch.clamp(torch.linalg.norm(aty_py), min=1.0)
                        )
                        print(
                            "  ==> GPU-ADMM CUDA ATu vs Python ATu: "
                            f"max|diff|={cuda_py_abs:.3e}, rel={cuda_py_rel:.3e}",
                            flush=True,
                        )
                tolerance = 1e-8 if stage_dtype == torch.double else 1e-4
                if use_cg_fused_normal:
                    normal_scratch = torch.empty_like(c_solve)
                    normal_ref = torch.empty_like(b_solve)
                    normal_test = torch.empty_like(b_solve)
                    normal_module.dqg_c1_normal_out(
                        y_probe.contiguous(), normal_scratch, normal_ref,
                        *op._cuda_kernel_args(c.shape[0]), b.shape[0],
                    )
                    CG_normal_out(y_probe, normal_test, normal_scratch)
                    normal_diff = torch.abs(normal_ref - normal_test)
                    normal_abs = float(torch.max(normal_diff))
                    normal_rel = float(
                        torch.linalg.norm(normal_diff) /
                        torch.clamp(torch.linalg.norm(normal_ref), min=1.0)
                    )
                    if print_level > 0:
                        print(
                            "  ==> GPU-ADMM fused-normal validation: "
                            f"max|diff|={normal_abs:.3e}, rel={normal_rel:.3e}",
                            flush=True,
                        )
                    if max(normal_abs, normal_rel) > tolerance:
                        if print_level > 0:
                            global_row = int(torch.argmax(normal_diff).item())
                            print(
                                "  ==> GPU-ADMM fused-normal max-diff "
                                f"row={global_row}, "
                                f"ref={float(normal_ref[global_row]):.16e}, "
                                f"test={float(normal_test[global_row]):.16e}, "
                                f"diff={float(normal_diff[global_row]):.3e}",
                                flush=True,
                            )
                            block_parts = []
                            for name, start_row, end_row in op.dual_block_spans:
                                if end_row <= start_row:
                                    continue
                                block_diff = normal_diff[start_row:end_row]
                                block_ref = normal_ref[start_row:end_row]
                                block_max = float(torch.max(block_diff))
                                if block_max <= tolerance:
                                    continue
                                block_rel = float(
                                    torch.linalg.norm(block_diff) /
                                    torch.clamp(torch.linalg.norm(block_ref), min=1.0)
                                )
                                block_local = int(torch.argmax(block_diff).item())
                                block_parts.append(
                                    (
                                        block_max,
                                        f"{name}:max={block_max:.3e},rel={block_rel:.3e},"
                                        f"local={block_local},row={start_row + block_local}"
                                    )
                                )
                            block_parts.sort(key=lambda item: item[0], reverse=True)
                            if block_parts:
                                print(
                                    "  ==> GPU-ADMM fused-normal block diffs: " +
                                    "; ".join(part for _, part in block_parts[:8]),
                                    flush=True,
                                )
                        raise ValueError(
                            "GPU_ADMM_CG_FUSED_NORMAL validation failed "
                            "against scratch-backed CUDA normal."
                        )
                    del normal_scratch, normal_ref, normal_test, normal_diff
                if max(ax_abs, aty_abs, ax_rel, aty_rel, dot_rel) > tolerance:
                    if print_level > 0:
                        def report_ax_diffs(label, ref_vec, test_vec):
                            diff_vec = torch.abs(ref_vec - test_vec)
                            global_row = int(torch.argmax(diff_vec).item())
                            print(
                                f"  ==> GPU-ADMM {label} max-diff row={global_row}, "
                                f"ref={float(ref_vec[global_row]):.16e}, "
                                f"test={float(test_vec[global_row]):.16e}, "
                                f"diff={float(diff_vec[global_row]):.3e}",
                                flush=True,
                            )
                            block_parts = []
                            for name, start_row, end_row in op.dual_block_spans:
                                if end_row <= start_row:
                                    continue
                                block_diff = diff_vec[start_row:end_row]
                                block_ref = ref_vec[start_row:end_row]
                                block_max = float(torch.max(block_diff))
                                if block_max <= tolerance:
                                    continue
                                block_rel = float(
                                    torch.linalg.norm(block_diff) /
                                    torch.clamp(torch.linalg.norm(block_ref), min=1.0)
                                )
                                block_local = int(torch.argmax(block_diff).item())
                                block_parts.append(
                                    (
                                        block_max,
                                        f"{name}:max={block_max:.3e},rel={block_rel:.3e},"
                                        f"local={block_local},row={start_row + block_local}"
                                    )
                                )
                            block_parts.sort(key=lambda item: item[0], reverse=True)
                            if block_parts:
                                print(
                                    f"  ==> GPU-ADMM {label} block diffs: " +
                                    "; ".join(part for _, part in block_parts[:8]),
                                    flush=True,
                                )

                        report_ax_diffs("matrix-free Ax", ax_ref, ax_mf)
                        if ax_py is not None:
                            report_ax_diffs("CUDA Au vs Python Au", ax_py, ax_mf)
                        if aty_py is not None:
                            diff_vec = torch.abs(aty_py - aty_mf)
                            global_col = int(torch.argmax(diff_vec).item())
                            print(
                                f"  ==> GPU-ADMM CUDA ATu vs Python ATu max-diff col={global_col}, "
                                f"ref={float(aty_py[global_col]):.16e}, "
                                f"test={float(aty_mf[global_col]):.16e}, "
                                f"diff={float(diff_vec[global_col]):.3e}",
                                flush=True,
                            )
                    raise ValueError(
                        f"{matrix_free_operator_label} matrix-free validation failed against explicit CSR."
                    )
                if matrix_free_at_csr:
                    hybrid_at_csr = A_t_csr
                    del A_csr, a_crow_t, a_col_t, a_values
                else:
                    del A_csr, A_t_csr, a_crow_t, a_col_t, a_values, at_crow_t, at_col_t, at_values
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                cuda_memory_report("after matrix-free validation CSR release")

            if matrix_free_at_csr:
                if not has_at_csr:
                    raise ValueError(
                        "GPU_ADMM_MATRIX_FREE_AT_CSR requires direct CSR(A^T) "
                        "handoff from C++."
                    )
                if hybrid_at_csr is None:
                    at_crow_t = torch.from_numpy(at_crow).to(device=device)
                    at_col_t = torch.from_numpy(at_col).to(device=device)
                    at_values = torch.from_numpy(at_vals).to(device=device, dtype=stage_dtype)
                    hybrid_at_csr = torch.sparse_csr_tensor(
                        at_crow_t, at_col_t, at_values, size=(c.shape[0], b.shape[0])
                    )
                cuda_memory_report("after hybrid A^T CSR")

                def ATu_torch(u_tensor):
                    return torch.mv(hybrid_at_csr, u_tensor)

                def hybrid_ATu_torch_out(u_tensor, out_tensor):
                    out_tensor.copy_(ATu_torch(u_tensor))
                    return out_tensor

                ATu_torch_out = hybrid_ATu_torch_out


            if not matrix_free_validate and not matrix_free_at_csr:
                cuda_memory_report("after matrix-free operator setup")

        elif has_direct_csr:
            a_crow_t = torch.from_numpy(a_crow).to(device=device)
            a_col_t = torch.from_numpy(a_col).to(device=device)
            a_values = torch.from_numpy(a_vals).to(device=device, dtype=stage_dtype)
            A_csr = torch.sparse_csr_tensor(
                a_crow_t, a_col_t, a_values, size=(b.shape[0], c.shape[0])
            )
            cuda_memory_report("after A CSR")

            at_crow_t = torch.from_numpy(at_crow).to(device=device)
            at_col_t = torch.from_numpy(at_col).to(device=device)
            at_values = torch.from_numpy(at_vals).to(device=device, dtype=stage_dtype)
            A_t_csr = torch.sparse_csr_tensor(
                at_crow_t, at_col_t, at_values, size=(c.shape[0], b.shape[0])
            )
            cuda_memory_report("after A^T CSR")
        else:
            # Legacy fallback: build A and A^T one at a time through COO.
            row_t = torch.from_numpy(rows).to(device=device, dtype=torch.long)
            col_t = torch.from_numpy(cols).to(device=device, dtype=torch.long)
            values = torch.from_numpy(vals).to(device=device, dtype=stage_dtype)

            indices = torch.stack([row_t, col_t])
            A_sparse = torch.sparse_coo_tensor(indices, values, size=(b.shape[0], c.shape[0])).coalesce()
            A_csr = A_sparse.to_sparse_csr()
            del A_sparse, indices
            if device.type == "cuda":
                torch.cuda.empty_cache()
            cuda_memory_report("after legacy A CSR")

            indices_t = torch.stack([col_t, row_t])
            A_sparse_t = torch.sparse_coo_tensor(indices_t, values, size=(c.shape[0], b.shape[0])).coalesce()
            A_t_csr = A_sparse_t.to_sparse_csr()
            del A_sparse_t, indices_t, row_t, col_t
            if device.type == "cuda":
                torch.cuda.empty_cache()
            cuda_memory_report("after legacy A^T CSR")

        if not use_matrix_free_operator:
            def Au_torch(u_tensor):
                return torch.mv(A_csr, u_tensor)

            def ATu_torch(u_tensor):
                return torch.mv(A_t_csr, u_tensor)

            def csr_Au_torch_out(u_tensor, out_tensor):
                out_tensor.copy_(Au_torch(u_tensor))
                return out_tensor

            def csr_ATu_torch_out(u_tensor, out_tensor):
                out_tensor.copy_(ATu_torch(u_tensor))
                return out_tensor

            Au_torch_out = csr_Au_torch_out
            ATu_torch_out = csr_ATu_torch_out

        setup_elapsed = timer_stop("operator_setup", sparse_build_start)
        if print_level > 0:
            if use_matrix_free_operator:
                print(
                    f"  ==> GPU-ADMM: Matrix-free {matrix_free_operator_label} operator ready for dtype={stage_dtype} "
                    f"in {setup_elapsed:.2f} s",
                    flush=True,
                )
            else:
                print(
                    f"  ==> GPU-ADMM: CSR tensors ready for dtype={stage_dtype} "
                    f"in {setup_elapsed:.2f} s",
                    flush=True,
                )

        # Route the CG normal matvec through the distributed sharded
        # operator with compact per-device caches + NCCL all-reduce of the shared
        # D2/D1 region. This replaces whatever CG_normal_out was selected above. Gated by
        # GPU_ADMM_SHARD_PRIMAL; matrix-free C1, double precision, >=2 PSD devices only.
        if (use_matrix_free_operator and shard_primal_enabled and
                stage_dtype == torch.double and
                psd_work_by_device and len(psd_work_by_device) > 1 and
                getattr(op, "cuda_normal_module", None) is not None and
                hasattr(op, "build_shard_plan")):
            _off_to_name = {int(off): nm for nm, (off, _d) in op.blocks.items()}
            _block_owner = {}
            for _dev, _items in psd_work_by_device.items():
                for _dim, _bname, (_bidx, _offset) in _items:
                    _nm = _off_to_name.get(int(_offset))
                    if _nm is not None:
                        _block_owner[_nm] = int(_dev)
            _plan = op.build_shard_plan(list(psd_work_by_device.keys()),
                                        _block_owner)
            if _plan is not None:
                def CG_normal_out(u_tensor, out_tensor, scratch_tensor,
                                  _op=op, _plan=_plan):
                    return _op.distributed_normal_out(_plan, u_tensor, out_tensor)
                CG_normal_out.needs_scratch = False
                # Expose the plan + operator so the ADMM loop can store the dual
                # vectors row-sharded (P1 dual-vector sharding).
                CG_normal_out.shard_plan = _plan
                CG_normal_out.shard_op = op
                if print_level > 0:
                    _nr = sum(len(v) for v in _plan["owned_rows"].values())
                    print("  ==> GPU-ADMM: CG normal operator will use the P1 "
                          "sharded distributed operator (compact caches + NCCL "
                          f"D2/D1 all-reduce; owned-row-runs={_nr}).", flush=True)

        # Host-scatter, the solve tensors inherit the master device (host when
        # host_scatter is active). Keep them on the host ONLY when the primal store
        # will actually be used at this stage (sharded plan wired + PRIMAL_STORE option).
        # The store scatters copy host->device per block. In every other path
        # (plan didn't materialize, dual-only) pull them onto the GPU so the
        # standard full-vector code has its operands.
        _primal_store_stage = (
            getattr(CG_normal_out, "shard_plan", None) is not None and
            shard_primal_store and stage_dtype == torch.double
        )
        if not _primal_store_stage and c_solve.device.type != device.type:
            c_solve = c_solve.to(device=device)
            b_solve = b_solve.to(device=device)
            x_solve = x_solve.to(device=device)
            y_solve = y_solve.to(device=device)
            z_solve = z_solve.to(device=device)

        return (
            c_solve, b_solve, x_solve, y_solve, z_solve,
            Au_torch, ATu_torch, Au_torch_out, ATu_torch_out,
            CG_normal_out, cg_cuda_module,
        )

    # Determine execution stages. The solver runs entirely in float64.
    stages = []
    stages.append({
        'dtype': torch.double,
        'coarse_conv_mult': 1.0, # strict convergence
        'max_iters': maxiter
    })

    # Execute solve stages
    for stage_idx, stage in enumerate(stages):
        dtype_solve = stage['dtype']
        coarse_mult = stage['coarse_conv_mult']
        stage_max_iters = stage['max_iters']

        if print_level > 0:
            print(
                f"  ==> GPU-ADMM Stage {stage_idx+1}: Starting solve with dtype={dtype_solve} <==",
                flush=True,
            )

        (
            c_s, b_s, x_s, y_s, z_s,
            Au_torch, ATu_torch, Au_torch_out, ATu_torch_out,
            CG_normal_out, cg_cuda_module,
        ) = get_precision_tensors(dtype_solve)

        # Dual-vector sharding. store y_s and b_s row-sharded across devices so
        # the base GPU never holds a full dual vector. Primal vectors stay full
        # this step; A^T y is gathered to a full primal for U / residuals. Reuses
        # the shard plan + operator already built for the distributed matvec.
        shard_plan = getattr(CG_normal_out, "shard_plan", None)
        shard_op = getattr(CG_normal_out, "shard_op", None)
        dual_sharded = shard_plan is not None and shard_op is not None
        primal_sharded = dual_sharded and shard_primal_store
        if dual_sharded:
            # Scatter each full vector to shards then immediately evict it from the
            # base GPU, one at a time, so at most a single full vector is resident
            # during setup. y_s/b_s/c_s/... the alias persistent _t masters (`.to()` is a no-op in the double stage), so
            # freeing requires moving the _t tensor off-GPU and dropping the alias.
            y_shards = shard_op.scatter_dual_to_shards(shard_plan, y_s)
            y_t = y_t.to("cpu")
            del y_s
            if device.type == "cuda":
                torch.cuda.empty_cache()
            b_shards = shard_op.scatter_dual_to_shards(shard_plan, b_s)
            b_t = b_t.to("cpu")
            del b_s
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if primal_sharded:
                # Store c/x/z in the disjoint per-device primal store, evicting each
                # full primal from base before scattering the next. No full primal
                # vector or full U ever sits on the base GPU.
                c_store = shard_op.scatter_primal_to_storage(shard_plan, c_s)
                c_t = c_t.to("cpu")
                del c_s
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                x_store = shard_op.scatter_primal_to_storage(shard_plan, x_s)
                x_t = x_t.to("cpu")
                del x_s
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                z_store = shard_op.scatter_primal_to_storage(shard_plan, z_s)
                z_t = z_t.to("cpu")
                del z_s
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                # one scratch store shared by cg-rhs (cz = c - z) and step 2 (U).
                # they never overlap in an iteration (cz is consumed by au_from_store
                # inside _cg_rhs_store before step 2 overwrites the buffer with U), so
                # sharing saves a full primal-store vector on every device.
                U_store = shard_op.alloc_primal_storage(shard_plan)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if print_level > 0:
                _msg = ("dual (y, b) and primal (c, x, z, U)"
                        if primal_sharded else "dual (y, b)")
                print(f"  ==> GPU-ADMM: P1 {_msg} vectors row-sharded across "
                      "devices; base GPU holds no full vector.", flush=True)

            def _atu_y_to_full(out_full):
                return shard_op.atu_to_full(shard_plan, y_shards, out_full)

            def _dual_objective():
                return shard_op.sharded_dot(shard_plan, b_shards, y_shards)

            if primal_sharded:
                def _primal_residual():
                    axs = shard_op.au_from_store(shard_plan, x_store)
                    for _d in shard_plan["device_ids"]:
                        with torch.cuda.device(_d):
                            axs[_d].sub_(b_shards[_d])
                    return shard_op.sharded_dot(shard_plan, axs, axs) ** 0.5

                def _primal_objective():
                    return shard_op.sharded_dot(shard_plan, c_store, x_store)

                def _step2_and_project(mu_val):
                    shard_op.build_U_store(
                        shard_plan, y_shards, x_store, c_store, U_store, mu_val,
                        admm_relaxation, z_store)
                    shard_op.project_psd_store(
                        shard_plan, U_store, x_store, z_store, mu_val)

                def _dual_error_store():
                    shard_op.atu_to_store(shard_plan, y_shards, U_store)
                    for _d in shard_plan["device_ids"]:
                        with torch.cuda.device(_d):
                            U_store[_d].add_(z_store[_d])
                            U_store[_d].sub_(c_store[_d])
                    return shard_op.sharded_dot(shard_plan, U_store, U_store) ** 0.5

                def _cg_rhs_store(mu_val):
                    # Reuse the shared U scratch for cz = c - z (consumed immediately
                    # by au_from_store below; step 2 rebuilds U afterwards).
                    for _d in shard_plan["device_ids"]:
                        with torch.cuda.device(_d):
                            U_store[_d].copy_(c_store[_d])
                            U_store[_d].sub_(z_store[_d])
                    acz = shard_op.au_from_store(shard_plan, U_store)
                    ax = shard_op.au_from_store(shard_plan, x_store)
                    for _d in shard_plan["device_ids"]:
                        with torch.cuda.device(_d):
                            acz[_d].add_(b_shards[_d], alpha=mu_val)
                            acz[_d].add_(ax[_d], alpha=-mu_val)
                    return acz
            else:
                def _primal_residual():
                    axs = shard_op.au_to_shards(shard_plan, x_s)
                    for _d in shard_plan["device_ids"]:
                        with torch.cuda.device(_d):
                            axs[_d].sub_(b_shards[_d])
                    return shard_op.sharded_dot(shard_plan, axs, axs) ** 0.5
        else:
            def _atu_y_to_full(out_full):
                return ATu_torch_out(y_s, out_full)

            def _primal_residual():
                pv = torch.empty_like(b_s)
                Au_torch_out(x_s, pv)
                pv.sub_(b_s)
                return float(torch.linalg.norm(pv))

            def _dual_objective():
                return float(torch.dot(b_s, y_s))

        use_fused_cg_updates = (
            cg_cuda_module is not None and
            hasattr(cg_cuda_module, "cg_update_y_r") and
            hasattr(cg_cuda_module, "cg_update_p") and
            device.type == "cuda" and
            dtype_solve == torch.double
        )
        if print_level > 0 and use_fused_cg_updates:
            print(
                "  ==> GPU-ADMM: CG vector updates will use fused CUDA kernels.",
                flush=True,
            )

        psd_executor = None
        psd_warmed_dtypes = set()
        if psd_device_ids:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            active_psd_workers = sum(
                1 for work_items in psd_work_by_device.values() if work_items
            )
            if active_psd_workers > 0:
                psd_executor = ThreadPoolExecutor(max_workers=active_psd_workers)

        # PyTorch CG Solver (unpreconditioned, matches the CPU BPSDP)
        def cg_solve(y_cg, rhs_cg, cg_conv, cg_max_it):
            def cg_atu_out(u_tensor, out_tensor):
                if profile_timing:
                    matvec_timer = timer_start()
                    ATu_torch_out(u_tensor, out_tensor)
                    timer_stop("cg_atu", matvec_timer)
                    return out_tensor
                return ATu_torch_out(u_tensor, out_tensor)

            def cg_au_out(u_tensor, out_tensor):
                if profile_timing:
                    matvec_timer = timer_start()
                    Au_torch_out(u_tensor, out_tensor)
                    timer_stop("cg_au", matvec_timer)
                    return out_tensor
                return Au_torch_out(u_tensor, out_tensor)

            cg_normal_needs_scratch = (
                CG_normal_out is None or
                getattr(CG_normal_out, "needs_scratch", True)
            )
            primal_work = torch.empty_like(c_s) if cg_normal_needs_scratch else None

            def cg_normal_out(u_tensor, out_tensor):
                if CG_normal_out is not None:
                    if profile_timing:
                        matvec_timer = timer_start()
                        CG_normal_out(u_tensor, out_tensor, primal_work)
                        timer_stop("cg_normal", matvec_timer)
                        return out_tensor
                    return CG_normal_out(u_tensor, out_tensor, primal_work)

                cg_atu_out(u_tensor, primal_work)
                cg_au_out(primal_work, out_tensor)
                return out_tensor

            Ap = rhs_cg
            r = rhs_cg.clone()
            cg_normal_out(y_cg, Ap)
            r.sub_(Ap)
            p = r.clone()
            rr = torch.dot(r, r)

            cg_conv_sq = cg_conv * cg_conv
            if rr < cg_conv_sq:
                return y_cg, 0

            cg_it = 0
            for cg_it in range(cg_max_it):
                cg_normal_out(p, Ap)
                alpha = rr / torch.dot(p, Ap)
                if use_fused_cg_updates:
                    rr_new = cg_cuda_module.cg_update_y_r(
                        y_cg, r, p, Ap, alpha.contiguous()
                    )
                else:
                    y_cg.add_(p, alpha=alpha)
                    r.add_(Ap, alpha=-alpha)
                    rr_new = torch.dot(r, r)
                if rr_new < cg_conv_sq:
                    cg_it += 1
                    break

                beta = rr_new / rr
                if use_fused_cg_updates:
                    cg_cuda_module.cg_update_p(p, r, beta.contiguous())
                else:
                    p.mul_(beta)
                    p.add_(r)
                rr = rr_new
            return y_cg, cg_it

        # ADMM loop for current stage
        local_iter = 0
        psd_float64_iters = 0

        def update_psd_dim_timing(dim, dtype_label, elapsed, count):
            if profile_timing:
                dim_key = (int(dim), dtype_label)
                psd_dim_timings[dim_key] = psd_dim_timings.get(dim_key, 0.0) + elapsed
                psd_dim_counts[dim_key] = psd_dim_counts.get(dim_key, 0) + count

        def update_psd_device_timing(dev_idx, dtype_label, elapsed, count):
            if profile_timing:
                dev_key = (int(dev_idx), dtype_label)
                psd_device_timings[dev_key] = psd_device_timings.get(dev_key, 0.0) + elapsed
                psd_device_counts[dev_key] = psd_device_counts.get(dev_key, 0) + count

        def project_psd_single_gpu(U):
            serial_psd_dim_threshold = 16000
            for dim, groups in block_groups.items():
                dim_timer = timer_start()
                block_size = dim * dim
                if dim >= serial_psd_dim_threshold:
                    for _, offset in groups:
                        if U.is_cuda and dim >= serial_psd_dim_threshold:
                            torch.cuda.empty_cache()
                        U_block = U[offset:offset+block_size].view(dim, dim)
                        U_block = 0.5 * (U_block + U_block.t())

                        eigenvalues, eigenvectors = torch.linalg.eigh(U_block)
                        pos_eig = torch.clamp(eigenvalues, min=0.0)

                        pos_part = (eigenvectors * pos_eig) @ eigenvectors.t()
                        x_s[offset:offset+block_size] = pos_part.view(-1) / mu
                        z_s[offset:offset+block_size] = (pos_part - U_block).view(-1)
                        del U_block, eigenvalues, eigenvectors, pos_eig, pos_part
                else:
                    U_blocks = torch.stack([
                        U[offset:offset+block_size].view(dim, dim)
                        for _, offset in groups
                    ])
                    U_blocks = 0.5 * (U_blocks + U_blocks.transpose(1, 2))

                    eigenvalues, eigenvectors = torch.linalg.eigh(U_blocks)
                    pos_eig = torch.clamp(eigenvalues, min=0.0)
                    pos_parts = torch.bmm(
                        eigenvectors * pos_eig.unsqueeze(1),
                        eigenvectors.transpose(1, 2),
                    )
                    for idx, (_, offset) in enumerate(groups):
                        x_s[offset:offset+block_size] = pos_parts[idx].view(-1) / mu
                        z_s[offset:offset+block_size] = (pos_parts[idx] - U_blocks[idx]).view(-1)
                update_psd_dim_timing(
                    dim,
                    "f64",
                    timer_elapsed(dim_timer),
                    len(groups),
                )

        def project_psd_multi_gpu(U):
            base_idx = torch.cuda.current_device()
            base_device = torch.device(f"cuda:{base_idx}")
            dtype_label = "f64"
            nonlocal psd_warmed_dtypes

            if psd_executor is None:
                project_psd_single_gpu(U)
                return

            # PyTorch lazily initializes some CUDA op wrappers. Touch the PSD
            # worker ops serially before worker threads can race first use.
            if dtype_label not in psd_warmed_dtypes:
                for dev_idx, work_items in psd_work_by_device.items():
                    if not work_items:
                        continue
                    with torch.cuda.device(dev_idx):
                        warm_dtype = U.dtype
                        warm = torch.eye(2, device=f"cuda:{dev_idx}", dtype=warm_dtype)
                        warm_vals, warm_vecs = torch.linalg.eigh(warm)
                        warm_pos = torch.clamp(warm_vals, min=0.0)
                        (warm_vecs * warm_pos) @ warm_vecs.t()
                        warm_stack = torch.stack([warm, warm])
                        warm_stack = 0.5 * (warm_stack + warm_stack.transpose(1, 2))
                        warm_bvals, warm_bvecs = torch.linalg.eigh(warm_stack)
                        warm_bpos = torch.clamp(warm_bvals, min=0.0)
                        torch.bmm(
                            warm_bvecs * warm_bpos.unsqueeze(1),
                            warm_bvecs.transpose(1, 2),
                        )
                        del warm, warm_vals, warm_vecs, warm_pos
                        del warm_stack, warm_bvals, warm_bvecs, warm_bpos
                        torch.cuda.synchronize(dev_idx)
                psd_warmed_dtypes.add(dtype_label)

            def worker(dev_idx, work_items):
                target_device = torch.device(f"cuda:{dev_idx}")
                local_results = []
                groups_by_dim = {}
                for dim, _, group in work_items:
                    groups_by_dim.setdefault(dim, []).append(group)
                device_start = time.perf_counter()
                device_block_count = 0

                with torch.cuda.device(dev_idx):
                    for dim, groups in groups_by_dim.items():
                        dim_start = time.perf_counter()
                        device_block_count += len(groups)
                        block_size = dim * dim
                        if len(groups) == 1:
                            _, offset = groups[0]
                            U_block = U[offset:offset+block_size].view(dim, dim)
                            if target_device != base_device:
                                U_block = U_block.to(device=target_device, non_blocking=True)
                            U_block = 0.5 * (U_block + U_block.t())

                            eigenvalues, eigenvectors = torch.linalg.eigh(U_block)
                            pos_eig = torch.clamp(eigenvalues, min=0.0)
                            pos_part = (eigenvectors * pos_eig) @ eigenvectors.t()

                            x_blocks = (pos_part.reshape(1, -1) / mu).to(
                                device=base_device, non_blocking=True
                            )
                            z_blocks = (pos_part - U_block).reshape(1, -1).to(
                                device=base_device, non_blocking=True
                            )
                        else:
                            U_blocks = torch.stack([
                                U[offset:offset+block_size].view(dim, dim)
                                for _, offset in groups
                            ])
                            if target_device != base_device:
                                U_blocks = U_blocks.to(device=target_device, non_blocking=True)
                            U_blocks = 0.5 * (U_blocks + U_blocks.transpose(1, 2))

                            eigenvalues, eigenvectors = torch.linalg.eigh(U_blocks)
                            pos_eig = torch.clamp(eigenvalues, min=0.0)
                            pos_parts = torch.bmm(
                                eigenvectors * pos_eig.unsqueeze(1),
                                eigenvectors.transpose(1, 2),
                            )

                            x_blocks = (pos_parts.reshape(len(groups), -1) / mu).to(
                                device=base_device, non_blocking=True
                            )
                            z_blocks = (pos_parts - U_blocks).reshape(len(groups), -1).to(
                                device=base_device, non_blocking=True
                            )

                        if profile_timing:
                            torch.cuda.synchronize(dev_idx)
                            torch.cuda.synchronize(base_idx)
                            dim_elapsed = time.perf_counter() - dim_start
                        else:
                            dim_elapsed = 0.0
                        local_results.append((dim, groups, x_blocks, z_blocks, dim_elapsed))
                    torch.cuda.synchronize(dev_idx)
                torch.cuda.synchronize(base_idx)
                device_elapsed = time.perf_counter() - device_start if profile_timing else 0.0
                return dev_idx, local_results, device_elapsed, device_block_count

            futures = []
            for dev_idx, work_items in psd_work_by_device.items():
                if work_items:
                    futures.append(psd_executor.submit(worker, dev_idx, work_items))
            for future in as_completed(futures):
                dev_idx, local_results, device_elapsed, device_block_count = future.result()
                update_psd_device_timing(dev_idx, dtype_label, device_elapsed, device_block_count)
                for dim, groups, x_blocks, z_blocks, dim_elapsed in local_results:
                    block_size = dim * dim
                    for idx, (_, offset) in enumerate(groups):
                        x_s[offset:offset+block_size] = x_blocks[idx]
                        z_s[offset:offset+block_size] = z_blocks[idx]
                    update_psd_dim_timing(dim, dtype_label, dim_elapsed, len(groups))

        # Energy-stagnation-stop state (reset per stage).
        from collections import deque
        energy_history = deque(maxlen=stagnation_window)
        stagnation_triggered = False

        while local_iter < stage_max_iters:
            # step 1: CG y-update
            timer = timer_start()
            if primal_sharded:
                cg_rhs_shards = _cg_rhs_store(mu)
            elif dual_sharded:
                # rhs = A(c - z) + mu b - mu A x, built row-sharded and in place
                # (reusing the A(c-z) shards as the accumulator) to avoid extra
                # dual-shard temps. These are freed before the PSD phase.
                _acz = shard_op.au_to_shards(shard_plan, c_s - z_s)
                _ax = shard_op.au_to_shards(shard_plan, x_s)
                cg_rhs_shards = _acz
                for _d in shard_plan["device_ids"]:
                    with torch.cuda.device(_d):
                        cg_rhs_shards[_d].add_(b_shards[_d], alpha=mu)
                        cg_rhs_shards[_d].add_(_ax[_d], alpha=-mu)
                del _ax
            else:
                tmp_primal = c_s - z_s
                cg_rhs = torch.empty_like(b_s)
                Au_torch_out(tmp_primal, cg_rhs)
                del tmp_primal
                tmp_dual = torch.empty_like(b_s)
                Au_torch_out(x_s, tmp_dual)
                cg_rhs.add_(b_s, alpha=mu)
                cg_rhs.add_(tmp_dual, alpha=-mu)
                del tmp_dual
            timer_stop("cg_rhs", timer)

            if local_iter == 0:
                timer = timer_start()
                if primal_sharded:
                    dual_error = _dual_error_store()
                else:
                    dual_err_vec = torch.empty_like(c_s)
                    _atu_y_to_full(dual_err_vec)
                    dual_err_vec.sub_(c_s)
                    dual_err_vec.add_(z_s)
                    dual_error = float(torch.linalg.norm(dual_err_vec))
                    del dual_err_vec

                primal_error = _primal_residual()
                timer_stop("initial_residual", timer)

            if dynamic_cg_convergence:
                if local_iter == 0:
                    cg_conv_i = cg_dynamic_factor
                else:
                    cg_conv_i = cg_dynamic_factor * min(primal_error, dual_error)
                if cg_conv_i < cg_convergence:
                    cg_conv_i = cg_convergence
            else:
                cg_conv_i = cg_convergence

            cg_conv_i = float(cg_conv_i)

            timer = timer_start()
            if dual_sharded:
                y_shards, cg_iter = shard_op.sharded_cg_solve(
                    shard_plan, y_shards, cg_rhs_shards, cg_conv_i, cg_maxiter)
                del cg_rhs_shards
            else:
                y_s, cg_iter = cg_solve(y_s, cg_rhs, cg_conv_i, cg_maxiter)
                del cg_rhs
            timer_stop("cg_solve", timer)
            iiter_total += cg_iter

            # step 2: x and z updates
            if primal_sharded:
                # U = mu*x + A^T y - c and the PSD projection both happen entirely
                # in the disjoint store, no full U on the base GPU.
                timer = timer_start()
                _step2_and_project(mu)
                psd_elapsed = timer_stop("psd_projection_float64", timer)
                if profile_timing:
                    timings["psd_projection"] += psd_elapsed
                psd_float64_iters += 1

                # Step 3: residuals + objectives from the store.
                timer = timer_start()
                dual_error = _dual_error_store()
                primal_error = _primal_residual()
                objective_primal = _primal_objective()
                objective_dual = _dual_objective()
                primal_dual_objective_gap = abs(objective_primal - objective_dual)
            else:
                timer = timer_start()
                U = torch.empty_like(c_s)
                _atu_y_to_full(U)
                if admm_relaxation != 1.0:
                    # Eckstein-Bertsekas over-relaxation of the coupling term:
                    # A^T y  ->  alpha*A^T y + (1-alpha)(c - z).
                    one_minus = 1.0 - admm_relaxation
                    U.mul_(admm_relaxation)
                    U.add_(c_s, alpha=one_minus)
                    U.sub_(z_s, alpha=one_minus)
                U.add_(x_s, alpha=mu)
                U.sub_(c_s)
                timer_stop("xz_aty", timer)

                timer = timer_start()
                if psd_device_ids:
                    project_psd_multi_gpu(U)
                else:
                    project_psd_single_gpu(U)
                psd_elapsed = timer_stop("psd_projection_float64", timer)
                if profile_timing:
                    timings["psd_projection"] += psd_elapsed
                psd_float64_iters += 1

                # Step 3: check error and update mu
                timer = timer_start()
                _atu_y_to_full(U)
                U.sub_(c_s)
                U.add_(z_s)
                dual_error = float(torch.linalg.norm(U))
                del U

                primal_error = _primal_residual()

                objective_primal = float(torch.dot(c_s, x_s))
                objective_dual = _dual_objective()
                primal_dual_objective_gap = abs(objective_primal - objective_dual)

            # Notify C++ progress monitor with global iteration count
            if progress_monitor_py:
                progress_monitor_py(print_level, bpsdp_iter, cg_iter, objective_primal, objective_dual, mu, primal_error, dual_error)
            timer_stop("residual_objective", timer)

            if local_iter == 0:
                cuda_memory_report("after first ADMM iteration")

            bpsdp_iter += 1
            local_iter += 1

            mu_before = mu
            if mu_update_frequency > 0 and bpsdp_iter % mu_update_frequency == 0:
                mu = adaptive_mu_update(
                    mu, primal_error, dual_error, primal_dual_objective_gap
                )

            gap_tol_i = effective_gap_tol(objective_primal, objective_dual)
            if (primal_error < sdp_error_convergence * coarse_mult and
                dual_error < sdp_error_convergence * coarse_mult and
                primal_dual_objective_gap < gap_tol_i * coarse_mult):
                break

            # Always-on energy-stagnation stop: once feasibility is met and the
            # variational primal energy has been flat to tol across the window,
            # the energy is converged even if the duality-gap criterion is still
            # moving in its very slow linear tail.
            energy_history.append(objective_primal)
            if (coarse_mult <= 1.0 and
                    len(energy_history) >= stagnation_window and
                    primal_error < sdp_error_convergence and
                    dual_error < sdp_error_convergence):
                energy_span = max(energy_history) - min(energy_history)
                if energy_span < stagnation_energy_tol:
                    stagnation_triggered = True
                    if print_level > 0:
                        print(
                            f"  ==> GPU-ADMM: energy-stagnation stop at iter "
                            f"{bpsdp_iter}: feasibility met (eps_p={primal_error:.2e}, "
                            f"eps_d={dual_error:.2e}) and E(p) flat to "
                            f"{energy_span:.2e} < {stagnation_energy_tol:.2e} over "
                            f"{stagnation_window} iters (gap={primal_dual_objective_gap:.2e}).",
                            flush=True,
                        )
                    break

        # Reassemble the full solution from shards/store and propagate to the
        # double masters. In the sharded path do this one vector at a time
        # (gather -> copyback -> free) so at most a single full vector is resident
        # on the base GPU during teardown. The non-sharded path already holds full x/y/z.
        timer = timer_start()
        if primal_sharded:
            x_s = torch.zeros(int(c_t.shape[0]), device=device, dtype=dtype_solve)
            shard_op.gather_storage_to_primal(shard_plan, x_store, x_s)
            x_t.copy_(x_s.to(dtype=torch.double))
            del x_s, x_store
            if device.type == "cuda":
                torch.cuda.empty_cache()
            z_s = torch.zeros(int(c_t.shape[0]), device=device, dtype=dtype_solve)
            shard_op.gather_storage_to_primal(shard_plan, z_store, z_s)
            z_t.copy_(z_s.to(dtype=torch.double))
            del z_s, z_store, c_store, U_store
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            x_t.copy_(x_s.to(dtype=torch.double))
            z_t.copy_(z_s.to(dtype=torch.double))

        if dual_sharded:
            y_s = torch.zeros(int(b_t.shape[0]), device=device, dtype=dtype_solve)
            shard_op.gather_shards_to_dual(shard_plan, y_shards, y_s)
            y_t.copy_(y_s.to(dtype=torch.double))
            del y_s, y_shards, b_shards
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            y_t.copy_(y_s.to(dtype=torch.double))
        timer_stop("stage_copy", timer)
        if psd_executor is not None:
            psd_executor.shutdown(wait=True)
            psd_executor = None

    # Copy final results back into the NumPy arrays supplied by the C++ caller.
    # Returning fresh cpu().numpy() arrays would transiently duplicate x/y/z on host.
    timer = timer_start()
    def copy_back_to_host_array(device_tensor, host_array, label):
        host_np = np.asarray(host_array)
        if host_np.dtype != np.float64 or not host_np.flags.c_contiguous:
            if print_level > 0:
                print(
                    f"  ==> GPU-ADMM WARNING: final {label} array is not a "
                    "contiguous float64 buffer; falling back to an allocating "
                    "host copy.",
                    flush=True,
                )
            return device_tensor.detach().cpu().numpy().astype(np.float64, copy=False)
        host_tensor = torch.from_numpy(host_np)
        if host_tensor.numel() != device_tensor.numel():
            raise ValueError(
                f"GPU-ADMM final {label} copy size mismatch: "
                f"host={host_tensor.numel()} device={device_tensor.numel()}"
            )
        host_tensor.copy_(device_tensor.detach(), non_blocking=False)
        return host_np

    x_opt = copy_back_to_host_array(x_t, x_init, "x")
    y_opt = copy_back_to_host_array(y_t, y_init, "y")
    z_opt = copy_back_to_host_array(z_t, z_init, "z")
    timer_stop("host_copy", timer)

    # The always-on energy-stagnation stop certifies the variational energy has
    # converged to target even if the gap criterion was still moving, so treat
    # it as converged. Otherwise apply the gap tolerance.
    converged = (
        stagnation_triggered
        or (primal_error < sdp_error_convergence and
            dual_error < sdp_error_convergence and
            primal_dual_objective_gap <
            effective_gap_tol(objective_primal, objective_dual))
    )

    if profile_timing and print_level > 0:
        total_elapsed = time.perf_counter() - solve_wall_start
        print(
            "  ==> GPU-ADMM timing: "
            f"total={total_elapsed:.2f}s, operator_setup={timings['operator_setup']:.2f}s, "
            f"cg_rhs={timings['cg_rhs']:.2f}s, cg={timings['cg_solve']:.2f}s, "
            f"cg_normal={timings['cg_normal']:.2f}s, "
            f"cg_normal_cache={timings['cg_normal_cache']:.2f}s, "
            f"cg_normal_rows={timings['cg_normal_rows']:.2f}s, "
            f"cg_ATu={timings['cg_atu']:.2f}s, cg_Au={timings['cg_au']:.2f}s, "
            f"xz_ATy={timings['xz_aty']:.2f}s, psd={timings['psd_projection']:.2f}s, "
            f"psd64={timings['psd_projection_float64']:.2f}s, "
            f"residual={timings['residual_objective']:.2f}s, "
            f"copy={timings['stage_copy'] + timings['host_copy']:.2f}s, "
            f"admm_iters={bpsdp_iter - oiter_start}, cg_iters={iiter_total - iiter_start}, "
            f"psd64_iters={psd_float64_iters}",
            flush=True,
        )
        if psd_dim_timings:
            psd_dim_parts = []
            for (dim, dtype), elapsed in sorted(psd_dim_timings.items(), key=lambda item: (-item[0][0], item[0][1])):
                block_count = psd_dim_counts[(dim, dtype)]
                psd_dim_parts.append(f"{dim}{dtype}={elapsed:.2f}s/{block_count}blk")
            print("  ==> GPU-ADMM PSD blocks: " + ", ".join(psd_dim_parts), flush=True)
        if psd_device_timings:
            psd_device_parts = []
            for (dev_idx, dtype), elapsed in sorted(psd_device_timings.items(), key=lambda item: (item[0][0], item[0][1])):
                block_count = psd_device_counts[(dev_idx, dtype)]
                psd_device_parts.append(f"cuda:{dev_idx}{dtype}={elapsed:.2f}s/{block_count}blk")
            print("  ==> GPU-ADMM PSD devices: " + ", ".join(psd_device_parts), flush=True)

    return x_opt, y_opt, z_opt, mu, converged, bpsdp_iter, iiter_total


# Integration with driver routines

# jellium-scf
psi4.driver.procedures['energy']['jellium-scf'] = run_jellium_scf

# p2rdm
psi4.driver.procedures['energy']['p2rdm'] = run_p2rdm
#psi4.driver.procedures['energy']['cid']   = run_p2rdm

# pair methods:
psi4.driver.procedures['energy']['pp2rdm']   = run_pp2rdm
psi4.driver.procedures['energy']['pcid']     = run_pp2rdm
psi4.driver.procedures['energy']['pccd']     = run_pp2rdm
psi4.driver.procedures['energy']['pcepa(0)'] = run_pp2rdm
psi4.driver.procedures['energy']['pacpf']    = run_pp2rdm
psi4.driver.procedures['energy']['paqcc']    = run_pp2rdm

# doci
psi4.driver.procedures['energy']['doci'] = run_doci

# v2rdm-doci
psi4.driver.procedures['energy']['v2rdm-doci'] = run_v2rdm_doci

# v2rdm-casscf
psi4.driver.procedures['energy']['v2rdm-casscf'] = run_v2rdm_casscf
psi4.driver.procedures['gradient']['v2rdm-casscf'] = run_v2rdm_casscf_gradient

# qed-scf,dft,cc,tddft
psi4.driver.procedures['energy']['qed-scf']   = run_qed_scf
psi4.driver.procedures['energy']['qed-dft']   = run_qed_scf
psi4.driver.procedures['energy']['qed-tddft'] = run_qed_scf
psi4.driver.procedures['energy']['qed-ccsd']  = run_qed_scf

# qed-ccsd with tiled-array
psi4.driver.procedures['energy']['qed-ccsd-00']     = run_qed_scf
psi4.driver.procedures['energy']['qed-ccsd-21']     = run_qed_scf
psi4.driver.procedures['energy']['qed-ccsd-22']     = run_qed_scf

# eom qed methods
psi4.driver.procedures['energy']['eom-ee-qed-ccsd-00'] = run_qed_scf
psi4.driver.procedures['energy']['eom-ea-qed-ccsd-00'] = run_qed_scf
psi4.driver.procedures['energy']['eom-qed-ccsd-00']    = run_qed_scf

psi4.driver.procedures['energy']['eom-ee-qed-ccsd-21'] = run_qed_scf
psi4.driver.procedures['energy']['eom-ea-qed-ccsd-21'] = run_qed_scf
psi4.driver.procedures['energy']['eom-qed-ccsd-21']    = run_qed_scf 

# gradients for qed-scf,dft
psi4.driver.procedures['gradient']['qed-scf'] = run_qed_scf_gradient
psi4.driver.procedures['gradient']['qed-dft'] = run_qed_scf_gradient

# mcpdft
psi4.driver.procedures['energy']['mcpdft'] = run_mcpdft
