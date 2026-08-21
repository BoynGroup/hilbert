import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

import time
import json
import re
from contextlib import contextmanager

# ==============================================================================
# BENCHMARK CONFIGURATION
# ==============================================================================
K_VALUES = [2,3,4,5,6,8,9,10,11,12]  # chain length
BASIS = "cc-pvdz"
SDP_SOLVER = "gpu_admm"  # "gpu_admm" for GPU or "bpsdp" for CPU
E_CONVERGENCE = 1e-4
# ==============================================================================


@contextmanager
def redirect_stdout_fd(target_fd):
    sys.stdout.flush()
    saved_stdout_fd = os.dup(1)
    try:
        os.dup2(target_fd, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved_stdout_fd, 1)
        os.close(saved_stdout_fd)


@contextmanager
def capture_stdout_to_file(filepath):
    with open(filepath, "w") as f:
        with redirect_stdout_fd(f.fileno()):
            yield


def parse_admm_timing_line(line):
    parts = line.strip().split("==> GPU-ADMM timing:")
    if len(parts) < 2:
        return {}
    kv_str = parts[1].strip()
    results = {}
    for kv in kv_str.split(","):
        kv = kv.strip()
        if not kv or "=" not in kv:
            continue
        k, v = kv.split("=")
        k = k.strip()
        v = v.strip()
        if v.endswith("s"):
            results[k] = float(v[:-1])
        else:
            try:
                results[k] = int(v)
            except ValueError:
                try:
                    results[k] = float(v)
                except ValueError:
                    results[k] = v
    return results


def parse_admm_psd_blocks_line(line):
    parts = line.strip().split("==> GPU-ADMM PSD blocks:")
    if len(parts) < 2:
        return {}
    kv_str = parts[1].strip()
    results = {}
    pattern = re.compile(r'(\w+)=([\d\.]+)s/(\d+)blk')
    for item in kv_str.split(","):
        item = item.strip()
        match = pattern.match(item)
        if match:
            blk_name = match.group(1)
            time_val = float(match.group(2))
            count_val = int(match.group(3))
            results[blk_name] = {
                'time': time_val,
                'count': count_val
            }
    return results


def accumulate_admm_timings(all_stdout_lines):
    total_timing = {}
    total_psd_blocks = {}
    for line in all_stdout_lines:
        if "==> GPU-ADMM timing:" in line:
            t_data = parse_admm_timing_line(line)
            for k, v in t_data.items():
                if isinstance(v, (int, float)):
                    total_timing[k] = total_timing.get(k, 0.0) + v
                else:
                    total_timing[k] = v
        elif "==> GPU-ADMM PSD blocks:" in line:
            b_data = parse_admm_psd_blocks_line(line)
            for k, v in b_data.items():
                if k not in total_psd_blocks:
                    total_psd_blocks[k] = {'time': 0.0, 'count': 0}
                total_psd_blocks[k]['time'] += v['time']
                total_psd_blocks[k]['count'] += v['count']
    return {
        'gpu_admm_timings': total_timing,
        'gpu_admm_psd_blocks': total_psd_blocks
    }


def parse_natural_occupations(filepath):
    if not os.path.exists(filepath):
        return {}
    occupations = {'alpha': [], 'beta': []}
    current_spin = None
    with open(filepath, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if "# Natural Orbital Occupation Numbers (alpha) #" in line:
            current_spin = 'alpha'
            i += 1
            continue
        elif "# Natural Orbital Occupation Numbers (beta) #" in line:
            current_spin = 'beta'
            i += 1
            continue
        if current_spin is not None:
            stripped = line.strip()
            if not stripped or stripped.startswith("Irrep:") or stripped.startswith("#"):
                i += 1
                continue
            match = re.match(r'^\s*(\d+):\s*([\d\.\-\+eE]+)', line)
            if match:
                val = float(match.group(2))
                occupations[current_spin].append(val)
            else:
                if "Iteration" in line:
                    current_spin = None
        i += 1
    return occupations


def parse_gpu_memory(stdout_lines):
    max_peak_gb = 0.0
    pattern = re.compile(r'peak_allocated=([\d\.]+)\s*GiB')
    for line in stdout_lines:
        match = pattern.search(line)
        if match:
            val = float(match.group(1))
            if val > max_peak_gb:
                max_peak_gb = val
    return max_peak_gb


def parse_system_memory_requirements(filepath):
    if not os.path.exists(filepath):
        return 0.0
    pattern = re.compile(r'Total memory requirements:\s*([\d\.]+)\s*([a-zA-Z]+)')
    with open(filepath, "r") as f:
        for line in f:
            if "Total memory requirements:" in line:
                match = pattern.search(line)
                if match:
                    val = float(match.group(1))
                    unit = match.group(2).lower()
                    if unit.startswith('g'):
                        return val * 1024.0
                    elif unit.startswith('m'):
                        return val
                    elif unit.startswith('k'):
                        return val / 1024.0
                    else:
                        return val
    return 0.0


def run_acene_calc(k, basis, sdp_solver, e_convergence=1e-5):
    import psi4
    import hilbert

    _script_dir = os.path.dirname(os.path.abspath(__file__))

    psi4.set_num_threads(32)
    psi4.set_memory('100000 MiB')
    psi4.core.clean()
    psi4.core.clean_options()

    log_filename = f"acene_c1_k{k}_{sdp_solver}.log"
    psi4.set_output_file(log_filename, False)

    # Read geometry from XYZ file
    xyz_path = os.path.join(_script_dir, "geometries", f"acene_{k}_singlet.xyz")
    if not os.path.exists(xyz_path):
        raise FileNotFoundError(f"Geometry file not found: {xyz_path}")

    with open(xyz_path, "r") as f:
        xyz_lines = f.readlines()[2:]

    geom_str = "0 1\n" + "".join(xyz_lines)
    geom_str += "symmetry c1\n"
    # with symmetry d2h instead
    #geom_str += "symmetry d2h\n"
    #geom_str += "no_reorient\n"
    #geom_str += "no_com\n"

    psi4.geometry(geom_str)

    # C1 active space [4k+2, 4k+2]
    n_restricted = 11 * k + 7
    n_active = 4 * k + 2

    # D2h active space
    # [Ag, B1g, B2g, B3g, Au, B1u, B2u, B3u]
    # restricted = [3*k + 3, 5*k // 2 + 1, 0, 0, 0, 0, (5*k + 1) // 2 + 2, 3*k + 1]
    # active = [0, 0, k, k + 1, k, k + 1, 0, 0]


    print(f"  [Config] acene k={k}: C1 Symmetry | restricted_docc=[{n_restricted}] | active=[{n_active}]")

    psi4.set_options({
        'basis': basis,
        'scf_type': 'disk_df',
        'd_convergence': 1e-5,
        'maxiter': 0,
        'fail_on_maxiter': False,
        # change these to d2h arrays if running with symmetry
        'restricted_docc': [n_restricted],
        'active': [n_active],
        'r_convergence': 1e-5,
        'e_convergence': e_convergence,
        'reference': 'rhf'
    })

    hilbert_options = {
        'sdp_solver': sdp_solver,
        'positivity': 'dqg',
        'maxiter': 25000,
        'GPU_ADMM_PROFILE': True,
        'optimize_orbitals': True,
        'orbopt_frequency': 500,  # standardized orbital opt freq for benchmarking
        'e_convergence': e_convergence,
        'r_convergence': 1e-4,
        'mu_update_frequency': 100,
    }
    psi4.set_module_options('hilbert', hilbert_options)
    psi4.core.set_local_option('SCF', 'DF_INTS_IO', 'SAVE')

    scf_energy = float('nan')
    energy = float('nan')
    hf_time = 0.0
    v2rdm_time = 0.0
    status = "Success"
    admm_timings = {}
    admm_psd_blocks = {}
    natural_occupations = {}
    max_gpu_mem = 0.0
    sys_mem_req = 0.0

    t_hf_start = time.time()
    try:
        scf_energy, ref_wfn = psi4.energy('scf', return_wfn=True)
        hf_time = time.time() - t_hf_start
    except Exception as e:
        hf_time = time.time() - t_hf_start
        return {
            'k': k,
            'basis': basis,
            'sdp_solver': sdp_solver,
            'status': f"SCF Failed: {str(e)}",
            'scf_energy': float('nan'),
            'energy': float('nan'),
            'hf_time': hf_time,
            'v2rdm_time': 0.0,
            'gpu_admm_timings': {},
            'gpu_admm_psd_blocks': {},
            'natural_occupations': {},
            'max_gpu_memory_gib': 0.0,
            'system_memory_requirements_mb': 0.0
        }

    options = psi4.core.get_options()
    options.set_current_module('HILBERT')
    v2rdm = hilbert.v2RDMHelper(ref_wfn, options)

    temp_log_path = f"temp_stdout_acene_c1_k{k}_{sdp_solver}.log"

    t_v2rdm_start = time.time()
    try:
        with capture_stdout_to_file(temp_log_path):
            energy = v2rdm.compute_energy()
        v2rdm_time = time.time() - t_v2rdm_start
        status = "Success"
    except Exception as e:
        energy = float('nan')
        v2rdm_time = time.time() - t_v2rdm_start
        status = f"Solver Failed: {str(e)}"

    if os.path.exists(temp_log_path):
        try:
            with open(temp_log_path, "r") as f:
                stdout_content = f.read()
            sys.stdout.write(stdout_content)
            sys.stdout.flush()

            stdout_lines = stdout_content.splitlines()
            parsed_admm = accumulate_admm_timings(stdout_lines)
            admm_timings = parsed_admm.get('gpu_admm_timings', {})
            admm_psd_blocks = parsed_admm.get('gpu_admm_psd_blocks', {})
            max_gpu_mem = parse_gpu_memory(stdout_lines)
        except Exception as e:
            print(f"Error reading or parsing temp stdout log: {e}")
        finally:
            try:
                os.remove(temp_log_path)
            except Exception:
                pass

    if status == "Success":
        try:
            natural_occupations = parse_natural_occupations(log_filename)
        except Exception as e:
            print(f"Error parsing natural orbital occupations: {e}")

    try:
        sys_mem_req = parse_system_memory_requirements(log_filename)
    except Exception as e:
        print(f"Error parsing system memory requirements: {e}")

    return {
        'k': k,
        'basis': basis,
        'sdp_solver': sdp_solver,
        'status': status,
        'scf_energy': scf_energy,
        'energy': energy,
        'hf_time': hf_time,
        'v2rdm_time': v2rdm_time,
        'gpu_admm_timings': admm_timings,
        'gpu_admm_psd_blocks': admm_psd_blocks,
        'natural_occupations': natural_occupations,
        'max_gpu_memory_gib': max_gpu_mem,
        'system_memory_requirements_mb': sys_mem_req
    }


def main():
    print("==================================================================================")
    print(f"Acene Chain Benchmark (C1 Symmetry, BASIS: {BASIS}, SOLVER: {SDP_SOLVER.upper()})")
    print("==================================================================================")
    print(f"Series: k={K_VALUES}")
    print(f"V2RDM SDP Energy Convergence: {E_CONVERGENCE}")
    print("==================================================================================")

    output_filename = f"benchmark_acene_c1_{SDP_SOLVER}.json"

    # Check xyz files
    valid_k_values = []
    for k in K_VALUES:
        xyz_path = os.path.join(script_dir, "geometries", f"acene_{k}_singlet.xyz")
        if os.path.exists(xyz_path):
            valid_k_values.append(k)
        else:
            print(f"[Notice] Skipping k={k}: Geometry file not found at {xyz_path}")

    results = []
    completed_keys = set()
    if os.path.exists(output_filename):
        try:
            with open(output_filename, "r") as f:
                results = json.load(f)
            for r in results:
                if r.get('status') == 'Success':
                    completed_keys.add(r['k'])
            print(f"Loaded {len(results)} existing results from {output_filename}.")
        except Exception as e:
            print(f"Warning: Could not read existing results file {output_filename} ({e}).")
            results = []

    for k in valid_k_values:
        if k in completed_keys:
            print(f"[k={k}] -> Skipped (already completed).")
            continue

        print(f"[k={k}] -> Running CASSCF calculation...")
        try:
            res = run_acene_calc(k, BASIS, SDP_SOLVER, E_CONVERGENCE)
            status_str = res['status']
            if status_str == 'Success':
                energy_str = f"{res['energy']:.10f} Eh"
                time_str = f"HF: {res['hf_time']:.2f}s, V2RDM: {res['v2rdm_time']:.2f}s"
                mem_str = f"Max Mem: Sys {res['system_memory_requirements_mb']:.1f} MB, GPU {res['max_gpu_memory_gib']:.3f} GB"
                print(f"[k={k}] -> {status_str} (Energy: {energy_str}, {time_str}, {mem_str})")
            else:
                print(f"[k={k}] -> Failed: {status_str}")
            results.append(res)
        except Exception as e:
            print(f"[k={k}] -> Execution crashed: {e}")
            results.append({
                'k': k,
                'basis': BASIS,
                'sdp_solver': SDP_SOLVER,
                'status': f"Crashed: {str(e)}"
            })

        with open(output_filename, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nBenchmark completed. Results saved to {output_filename}")


if __name__ == "__main__":
    main()
