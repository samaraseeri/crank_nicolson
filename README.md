🔧 What the code does
	•	Uses Crank–Nicolson time stepping → an implicit second-order scheme.
	•	Solves the nonlinear system each timestep with a Picard iteration (that’s the “Picard Δ” messages you saw).
	•	Parallelized with 2DECOMP&FFT → domain decomposition + FFT for Laplacian terms.
	•	Conserves mass (L² norm, printed as ≈ 7.87480) and monitors convergence.
This is a high-performance implementation of the 3D cubic NLS (focusing regime typically shows blow-up).
