mpif90 -O3 -cpp -march=native -ffast-math \
 -I/Users/aseeris/2decomp_fft/include nls3d_crank_nicolson.f90 \
 -L/Users/aseeris/2decomp_fft/lib -l2decomp_fft -L/opt/homebrew/Cellar/fftw/3.3.10_2/lib -lfftw3_mpi -lfftw3 -lm
