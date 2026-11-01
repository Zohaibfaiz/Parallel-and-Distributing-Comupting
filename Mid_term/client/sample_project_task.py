"""
================================================================================
CSC-334: Distributed Task Offloading - Complex Project Workload
Description: Heavy Multi-Stage Mathematical, Matrix & Monte Carlo Simulation
================================================================================
"""
import sys
import time
import math
import random
import json

def report_progress(percent, message):
    """Outputs standardized progress marker for remote executor to stream."""
    print(f"[PROGRESS] {percent:.1f}% - {message}", flush=True)

def heavy_monte_carlo_simulation(num_samples=2_000_000):
    report_progress(10.0, f"Starting Monte Carlo simulation ({num_samples:,} samples)...")
    inside_circle = 0
    batch_size = num_samples // 10
    
    for i in range(num_samples):
        x = random.random()
        y = random.random()
        if x * x + y * y <= 1.0:
            inside_circle += 1
        
        if (i + 1) % batch_size == 0:
            step = (i + 1) // batch_size
            pct = 10.0 + (step / 10.0) * 35.0  # 10% to 45%
            report_progress(pct, f"Monte Carlo batch {step}/10 processed ({i+1:,} samples)")
            
    pi_estimate = 4.0 * (inside_circle / num_samples)
    error = abs(pi_estimate - math.pi)
    report_progress(45.0, f"Monte Carlo completed: Pi ~ {pi_estimate:.6f} (Error: {error:.6f})")
    return {"pi_estimate": pi_estimate, "error": error, "samples": num_samples}

def heavy_matrix_decomposition(matrix_dim=750):
    report_progress(50.0, f"Starting intensive matrix tensor synthesis ({matrix_dim}x{matrix_dim})...")
    
    # Try importing numpy
    try:
        import numpy as np
        A = np.random.randn(matrix_dim, matrix_dim).astype(np.float64)
        B = np.random.randn(matrix_dim, matrix_dim).astype(np.float64)
        
        report_progress(65.0, f"Multiplying large matrices on Server compute cores...")
        C = np.dot(A, B)
        
        report_progress(80.0, f"Computing eigenvalues and matrix determinant...")
        norm_val = float(np.linalg.norm(C))
        trace_val = float(np.trace(C))
        eigenvalues = np.linalg.eigvals(C[:100, :100])
        max_eig = float(np.max(np.real(eigenvalues)))
        
        summary = {
            "engine": "NumPy (BLAS/LAPACK Multi-Threaded)",
            "matrix_dim": f"{matrix_dim}x{matrix_dim}",
            "frobenius_norm": norm_val,
            "matrix_trace": trace_val,
            "max_real_eigenvalue": max_eig
        }
    except Exception as e:
        report_progress(75.0, f"NumPy fallback: Running pure python math iteration...")
        # Pure Python fallback
        total = 0.0
        for i in range(100_000):
            total += math.sqrt(i) * math.sin(i)
        summary = {"engine": "Pure Python Mathematical Pipeline", "iterations": 100_000, "result": total}
        
    report_progress(90.0, "Synthesizing project output artifacts and summary metrics...")
    return summary

def main():
    print("=" * 65)
    print("  COMPLEX DISTRIBUTED PROJECT TASK EXECUTION (REMOTE SERVER WORKER)")
    print("=" * 65)
    t_start = time.time()
    
    report_progress(5.0, "Worker node initialized project environment...")
    time.sleep(0.3)
    
    mc_res = heavy_monte_carlo_simulation(num_samples=1_500_000)
    matrix_res = heavy_matrix_decomposition(matrix_dim=600)
    
    total_time = time.time() - t_start
    report_progress(100.0, f"All project stages completed successfully in {total_time:.2f}s!")
    
    result = {
        "task_name": "Complex Project Multi-Stage Simulation",
        "total_runtime_seconds": round(total_time, 3),
        "monte_carlo_results": mc_res,
        "matrix_computation_results": matrix_res,
        "status": "SUCCESS"
    }
    
    print("\n[OUTPUT SUMMARY JSON]")
    print(json.dumps(result, indent=2))
    
    # Save result to a file if output specified
    if len(sys.argv) > 1 and sys.argv[1]:
        with open(sys.argv[1], "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {sys.argv[1]}")

if __name__ == "__main__":
    main()
