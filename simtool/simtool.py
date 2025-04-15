# simtool.py

import os
import subprocess
import concurrent.futures
import pandas as pd
import re

# Mapping from preset names to extra info.
# Adjust these values to match what you set in your CMakePresets.json.

compiler_presets = {
    "gcc-O3":                {"compiler": "gcc",   "optimization_flags": "-O3"},
    "clang-O3":              {"compiler": "clang", "optimization_flags": "-O3"},
    "icx-O3":                {"compiler": "icx",   "optimization_flags": "-O3"},
    "gcc-O3-march-native":   {"compiler": "gcc",   "optimization_flags": "-O3 -march=native"},
    "clang-O3-march-native": {"compiler": "clang", "optimization_flags": "-O3 -march=native"},
    "icx-O3-march-native":   {"compiler": "icx",   "optimization_flags": "-O3 -march=native"},
    "gcc-O2":                {"compiler": "gcc",   "optimization_flags": "-O2"},
    "clang-O2":              {"compiler": "clang", "optimization_flags": "-O2"},
    "icx-O2":                {"compiler": "icx",   "optimization_flags": "-O2"},
    "gcc-O1":                {"compiler": "gcc",   "optimization_flags": "-O1"},
    "clang-O1":              {"compiler": "clang", "optimization_flags": "-O1"},
    "icx-O1":                {"compiler": "icx",   "optimization_flags": "-O1"},
    "gcc-O0":                {"compiler": "gcc",   "optimization_flags": "-O0"},
    "clang-O0":              {"compiler": "clang", "optimization_flags": "-O0"},
    "icx-O0":                {"compiler": "icx",   "optimization_flags": "-O0"}
}

profile_presets = {
    "gcc-O3_profile":                {"compiler": "gcc",   "optimization_flags": "-O3"},
  
}

optimization_presets = {
    "xoroshift":                {"compiler": "gcc",   "optimization_flags": "-O3"},
    "zero_padding":                {"compiler": "gcc",   "optimization_flags": "-O3"},
    "both":                {"compiler": "gcc",   "optimization_flags": "-O3"},
}


all_presets = {**compiler_presets, **profile_presets, **optimization_presets}
 


def parse_perf_output(filepath):
    """
    Parse the perf output file (stat_out) and return a dictionary of metrics.
    This version iterates over each line. For lines containing ":u", it extracts
    the first numeric token as the metric value and the token ending with ":u" as the metric name.
    It also handles the elapsed time line separately.
    """
    metrics = {}
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Process lines containing perf counters (they contain ":u")
            if ":u" in line:
                # Extract the first number in the line (the metric value)
                value_match = re.search(r"^([\d\.,]+)", line)
                # Extract the metric name (the token ending with ":u")
                metric_match = re.search(r"([\w\-_]+):u", line)
                if value_match and metric_match:
                    raw_value = value_match.group(1)
                    # Remove thousand separators (dots) and replace comma with dot for decimals.
                    # Example: "649.030.189.667" -> "649030189667" and "118.075,43" -> "118075,43" then "118075.43"
                    value_str = raw_value.replace('.', '').replace(',', '.')
                    try:
                        value = float(value_str)
                    except Exception:
                        value = None
                    metric_name = metric_match.group(1)
                    metrics[metric_name] = value

            # Process the elapsed time line separately.
            elif "seconds time elapsed" in line:
                time_match = re.search(r"([\d\.,]+)", line)
                if time_match:
                    raw_value = time_match.group(1)
                    value_str = raw_value.replace('.', '').replace(',', '.')
                    try:
                        value = float(value_str)
                    except Exception:
                        value = None
                    metrics["time_elapsed"] = value
    return metrics

def build(preset, L=1000, TRAN=5):
    """
    Build the binary for a given preset and problem size L.
    
    Returns a tuple: (binary_path, error)
    If error is None, binary_path is valid.
    """
    # For any preset starting with 'icx', check if the oneAPI environment script exists.
    env_prefix = ""
    if preset.lower().startswith("icx"):
        setvars_path = '/opt/intel/oneapi/setvars.sh'
        if not os.path.exists(setvars_path):
            return None, f"Skipping {preset}: {setvars_path} not found"
        env_prefix = f"source {setvars_path} && "
    
    # Define a unique build directory for this run.
    build_dir = os.path.abspath(os.path.join("out", f"{preset}_{L}_{TRAN}"))
    
    # Build command using the preset.
    build_cmd = (
        env_prefix +
        f"cmake -DL={L} -DTRAN={TRAN} --preset {preset} -Wno-dev -B {build_dir} && "
        f"cmake --build {build_dir} --preset {preset} -t tiny_ising"
    )
    print(f"Building preset '{preset}' for L={L} in directory {build_dir} ...")
    try:
        build_result = subprocess.run(
            build_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
            executable="/bin/bash"
        )
        if build_result.returncode != 0:
            error_msg = f"Build failed: {build_result.stderr}"
            print(f"Compilation failed for preset {preset} with L={L}: {error_msg}")
            return None, error_msg
    except Exception as e:
        return None, str(e)
    
    # The binary is assumed to be located at <build_dir>/tiny_ising.
    binary_path = os.path.abspath(os.path.join(build_dir, "tiny_ising"))
    if not os.path.exists(binary_path) or not os.access(binary_path, os.X_OK):
        return None, "Binary not found or not executable"
    
    return binary_path, None

def run(binary_path, preset, L=1000, TRAN=5):
    """
    Run the binary located at binary_path.
    
    Returns a tuple: (metric, error)
    """
    # Create the results directory for this run.
    result_dir = os.path.abspath(os.path.join("results", f"{preset}_{L}"))
    os.makedirs(result_dir, exist_ok=True)
    
    run_cmd = f"{binary_path}"
    try:
        run_result = subprocess.run(
            run_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
            cwd=result_dir
        )
        if run_result.returncode != 0:
            return None, f"Run failed: {run_result.stderr}"
        # Convert output to float as metric.
        metric = float(run_result.stdout.strip())
        return metric, None
    except Exception as e:
        return None, str(e)

def profile_run(binary_path, preset, L=1000, stat_n = 5):
    """
    Run the binary using perf.
    Executes:
       perf stat -o stat_out -ddd -r10 <binary_path> && perf record <binary_path>
    The working directory is set to a dedicated results folder.
    
    Returns a tuple: (profile_output, error)
    """
    result_dir = os.path.abspath(os.path.join("results", f"{preset}_{L}_profile"))
    os.makedirs(result_dir, exist_ok=True)
    
    # Construct the perf command.
    # Here we use the absolute path for the binary.
    perf_cmd = f"perf stat -o stat_out -ddd -r{stat_n} {binary_path} && perf record  --call-graph dwarf {binary_path}"
    print(f"Profiling binary using perf for preset '{preset}' with L={L} in {result_dir} ...")
    try:
        perf_result = subprocess.run(
            perf_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
            cwd=result_dir,
            executable="/bin/bash"
        )
        if perf_result.returncode != 0:
            return None, f"Perf run failed: {perf_result.stderr}"
        # Now parse the stat_out file.
        stat_file = os.path.join(result_dir, "stat_out")
        if not os.path.exists(stat_file):
            return None, "Perf stat output file not found"
        profile_metrics = parse_perf_output(stat_file)
        return profile_metrics, None
    except Exception as e:
        return None, str(e)

def build_and_run(preset, L=1000, TRAN=5):
    """
    Build and run the binary for a given preset and problem size L.
    
    Returns a dictionary with keys: compiler, optimization_flags, L, metric, error.
    """
    binary_path, build_error = build(preset, L, TRAN)
    if build_error is not None:
        return {
            "compiler": all_presets[preset]["compiler"],
            "optimization_flags": all_presets[preset]["optimization_flags"],
            "L": L,
            "TRAN": TRAN,
            "metric": None,
            "error": build_error
        }
    
    metric, run_error = run(binary_path, preset, L, TRAN)
    if run_error is not None:
        return {
            "compiler": all_presets[preset]["compiler"],
            "optimization_flags": all_presets[preset]["optimization_flags"],
            "L": L,
            "TRAN": TRAN,
            "metric": None,
            "error": run_error
        }
    
    return {
        "compiler": all_presets[preset]["compiler"],
        "optimization_flags": all_presets[preset]["optimization_flags"],
        "L": L,
        "metric": metric,
        "error": None
    }



def build_and_profile(preset='gcc-O3_profile', L=1000, stat_n = 5):
    """
    Build and run the binary for a given preset and problem size L.
    
    Returns a dictionary with keys: compiler, optimization_flags, L, metric, error.
    """
    binary_path, build_error = build(preset, L)
    if build_error is not None:
        return {
            "compiler": profile_presets[preset]["compiler"],
            "optimization_flags": profile_presets[preset]["optimization_flags"],
            "L": L,
            "profile": None,
            "error": build_error
        }
    
    profile_output, perf_error = profile_run(binary_path, preset, L, stat_n)
    if perf_error is not None:
        return {
            "compiler": profile_presets[preset]["compiler"],
            "optimization_flags": profile_presets[preset]["optimization_flags"],
            "L": L,
            "profile": None,
            "error": perf_error
        }
    
    return {
        "compiler": profile_presets[preset]["compiler"],
        "optimization_flags": profile_presets[preset]["optimization_flags"],
        "L": L,
        "profile": profile_output,
        "error": None
    }

def compiler_test(L=384):
    """
    Run builds for all presets in parallel for a fixed problem size L and return the results as a pandas DataFrame.
    
    Parameters:
        L (int): The problem size.
    
    Returns:
        pandas.DataFrame: Results with columns: compiler, optimization_flags, L, metric, error.
    """
    presets = list(compiler_presets.keys())
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(build_and_run, preset, L) for preset in presets]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(result)
            results.append(result)
    
    df = pd.DataFrame(results)
    return df

def problem_size_test(L_values, presets=None):
    """
    Run builds for all (or selected) presets for multiple problem sizes (L values).
    
    Parameters:
        L_values (iterable of int): Different problem sizes to test.
        presets (list of str, optional): Specific presets to run. Defaults to all presets in compiler_presets.
    
    Returns:
        pandas.DataFrame: Results with columns: compiler, optimization_flags, L, metric, error.
    """
    if presets is None:
        presets = ['gcc-O3']
    
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for preset in presets:
            for L in L_values:
                futures.append(executor.submit(build_and_run, preset, L))
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(result)
            results.append(result)
    
    return pd.DataFrame(results)

def montecarlo_step_test(TRAN_values,L=384, presets=None):
    """
    Run builds for all (or selected) presets for multiple problem sizes (L values).
    
    Parameters:
        TRAN_values (iterable of int): Different problem sizes to test.
        presets (list of str, optional): Specific presets to run. Defaults to all presets in compiler_presets.
    
    Returns:
        pandas.DataFrame: Results with columns: compiler, optimization_flags, L, metric, error.
    """
    if presets is None:
        presets = ['gcc-O3']
    
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for preset in presets:
            for TRAN in TRAN_values:
                futures.append(executor.submit(build_and_run, preset, L, TRAN))
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(result)
            results.append(result)
    
    return pd.DataFrame(results)

# def profile (L=1000, presets=None):
#     if presets is None:
#         presets = ['gcc-O3']

#     results = []
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = []
#         for preset in presets:
#             futures.append(executor.submit(build_and_run, preset, L))
#         for future in concurrent.futures.as_completed(futures):
#             result = future.result()
#             print(result)
#             results.append(result)

if __name__ == "__main__":
    # Allow running the module directly.
    # Example: Run all presets with L=1000.
    df = compiler_test(L=1000)
    print("\nFinal Results DataFrame (fixed L):")
    print(df)
    df.to_csv("build_run_results_fixed_L.csv", index=False)
    
    # Example: Run all presets for different L values.
    L_values = [500, 1000, 1500, 2000]
    df_varied = problem_size_test(L_values)
    print("\nFinal Results DataFrame (varying L):")
    print(df_varied)
    df_varied.to_csv("build_run_results_varying_L.csv", index=False)
