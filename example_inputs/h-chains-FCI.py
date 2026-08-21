import sys
import os
import time
import json
import re
import numpy as np
from contextlib import contextmanager

# ==============================================================================
# BENCHMARK SET-UP
LENGTHS = [10, 20, 40, 80, 100, 110, 120, 130]
DISTANCES = [1.25]
MULTIPLICITY = 1
BASIS = "sto-6g"
SDP_SOLVER = "gpu_admm"  # "gpu_admm" for GPU or "bpsdp" for CPU
E_CONVERGENCE = 1e-4
# ==============================================================================

# intialize torch/CUDA
try:
    import torch
    if torch.cuda.is_available():
        _ = torch.zeros(1, device="cuda")
        torch.set_num_threads(1)
        print("  ==> [Debug] PyTorch/CUDA initialized successfully.")
except ImportError:
    pass


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


# Parse timing info from std profiling output
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


# Parse PSD timing from profiling output
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

# accumulate timing across macroiterations
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

# parse NON data from psi4 output
def parse_natural_occupations(filepath):
    if not os.path.exists(filepath):
        return {}

    occupations = {
        'alpha': [],
        'beta': []
    }

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
            if not stripped:
                i += 1
                continue
            if stripped.startswith("Irrep:"):
                i += 1
                continue

            match = re.match(r'^\s*(\d+):\s*([\d\.\-\+eE]+)', line)
            if match:
                val = float(match.group(2))
                occupations[current_spin].append(val)
            else:
                if line.strip().startswith("#") or "Iteration" in line:
                    current_spin = None
        i += 1
    return occupations


# get memory info
def parse_gpu_memory(stdout_lines):
    max_peak_gb = 0.0
    import re
    pattern = re.compile(r'peak_allocated=([\d\.]+)\s*GiB')
    for line in stdout_lines:
        match = pattern.search(line)
        if match:
            val = float(match.group(1))
            if val > max_peak_gb:
                max_peak_gb = val
    return max_peak_gb


# get host/CPU RAM requirements
def parse_system_memory_requirements(filepath):
    if not os.path.exists(filepath):
        return 0.0

    import re
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

# run v2RDM-CASCI (FCI)
def run_hchain_calc(num_atoms, distance, multiplicity, basis, sdp_solver, e_convergence=1e-5):
    import psi4
    import hilbert

    psi4.set_num_threads(32)
    psi4.set_memory('230000 MiB')

    psi4.core.clean()
    psi4.core.clean_options()

    psi4_log_filename = f"hchain_l{num_atoms}_d{distance}_{sdp_solver}.log"
    psi4.set_output_file(psi4_log_filename, False)

    # linear H-chain
    geom_str = f"0 {multiplicity}\n"
    for i in range(num_atoms):
        geom_str += f"H 0.0 0.0 {i * distance}\n"
    geom_str += "symmetry c1\n"

    psi4.geometry(geom_str)

    # Set Psi4 Options
    psi4.set_options({
        'basis': basis,
        'scf_type': 'disk_df',
        'd_convergence': 1e-5,
        'maxiter': 100,
        'fail_on_maxiter': False,
        'restricted_docc': [0],
        'active': [num_atoms],
        'r_convergence': 1e-5,
        'e_convergence': e_convergence,
        'reference': 'rhf'
    })

    # v2RDM Options
    hilbert_options = {
        'sdp_solver': sdp_solver,
        'positivity': 'dqg',
        'maxiter': 100, # 100 for SDP profiling
        'GPU_ADMM_PROFILE': True,
        'optimize_orbitals': False,
        'e_convergence': e_convergence,
        'r_convergence': 1e-4,
        'mu_update_frequency': 100,
    }
    psi4.set_module_options('hilbert', hilbert_options)

    # Save DF integrals after HF
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

    # Run RHF / setup
    t_hf_start = time.time()
    try:
        scf_energy, ref_wfn = psi4.energy('scf', return_wfn=True)
        hf_time = time.time() - t_hf_start
    except Exception as e:
        hf_time = time.time() - t_hf_start
        return {
            'length': num_atoms,
            'distance': distance,
            'multiplicity': multiplicity,
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

    # Run v2RDM CASCI and redirect stdout
    options = psi4.core.get_options()
    options.set_current_module('HILBERT')
    v2rdm = hilbert.v2RDMHelper(ref_wfn, options)
    temp_log_path = f"temp_stdout_l{num_atoms}_d{distance}_{sdp_solver}.log"
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

    # Log
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

    # Parse Psi4 log
    if status == "Success":
        try:
            natural_occupations = parse_natural_occupations(psi4_log_filename)
        except Exception as e:
            print(f"Error parsing natural orbital occupations: {e}")

    try:
        sys_mem_req = parse_system_memory_requirements(psi4_log_filename)
    except Exception as e:
        print(f"Error parsing system memory requirements: {e}")

    return {
        'length': num_atoms,
        'distance': distance,
        'multiplicity': multiplicity,
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
    print(f"H-chain Benchmark (BASIS: {BASIS}, SOLVER: {SDP_SOLVER.upper()})")
    print("==================================================================================")
    print(f"Configs: Lengths={LENGTHS}, Distances={DISTANCES}")
    print(f"Multiplicity: {MULTIPLICITY}")
    print(f"V2RDM E Convergence: {E_CONVERGENCE}")
    print("==================================================================================")

    output_filename = f"benchmark_hchains_m{MULTIPLICITY}_{SDP_SOLVER}.json"

    # Restart if existing file found
    results = []
    completed_keys = set()
    if os.path.exists(output_filename):
        try:
            with open(output_filename, "r") as f:
                results = json.load(f)
            for r in results:
                if r.get('status') == 'Success':
                    completed_keys.add((r['length'], r['distance']))
            print(f"Loaded {len(results)} existing results from {output_filename}.")
            print(f"{len(completed_keys)} successful runs will be skipped.")
        except Exception as e:
            print(f"Warning: Could not read existing results file {output_filename} ({e}).")
            results = []

    for length in LENGTHS:
        for distance in DISTANCES:
            key = (length, distance)
            if key in completed_keys:
                print(f"[{length} atoms, {distance} A] -> Skipped (already completed).")
                continue

            print(f"[{length} atoms, {distance} A] -> Running...")
            res = run_hchain_calc(length, distance, MULTIPLICITY, BASIS, SDP_SOLVER, E_CONVERGENCE)

            # Print summary
            status_str = res['status']
            if status_str == 'Success':
                energy_str = f"{res['energy']:.10f} Eh"
                time_str = f"HF: {res['hf_time']:.2f}s, V2RDM: {res['v2rdm_time']:.2f}s"
                mem_str = f"Max Mem: Sys Req {res['system_memory_requirements_mb']:.1f} MB, GPU {res['max_gpu_memory_gib']:.3f} GB"
                print(f"[{length} atoms, {distance} A] -> {status_str} (Energy: {energy_str}, {time_str}, {mem_str})")
            else:
                print(f"[{length} atoms, {distance} A] -> Failed: {status_str}")

            results.append(res)

            # Save to file to allow restarts
            with open(output_filename, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\nBenchmark completed. Results saved to {output_filename}")


if __name__ == "__main__":
    main()
