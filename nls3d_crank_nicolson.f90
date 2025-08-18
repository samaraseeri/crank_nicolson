!===============================================================================
! 3D Cubic Nonlinear Schrödinger (NLS) on [0,2π]^3 (no symmetry assumed)
! Pseudospectral space + Crank–Nicolson (CN) in time with fixed-point iteration.
! MPI + 2DECOMP&FFT (pencil decomposition) for massively parallel runs.
!
! Equation (focusing if sigma=+1, defocusing if sigma=-1):
!    i ∂t ψ + Δψ + sigma |ψ|^2 ψ = 0
!
! Discretization (CN midpoint with Picard iteration):
!   Let L = -iΔ,  N(ψ) = -i sigma |ψ|^2.
!   A = I - (dt/2) L,  B = I + (dt/2) L.  In Fourier: Â(k) = 1 - i (dt/2) |k|^2.
!   For m = 0,1,... until convergence within each time step:
!      ψ* = (ψ^{n+1,(m)} + ψ^n)/2
!      S  = N(ψ*)  (pointwise in x)
!      RHS = B ψ^n + (dt/2) S (ψ^{n+1,(m)} + ψ^n)
!      Solve A ψ^{n+1,(m+1)} = RHS  via FFT diagonal inversion in Fourier.
!   This minimizes storage and avoids a dense linear solve from the nonlinear term.
!
! Optional 2/3 de-aliasing for nonlinear term (define USE_DEALIAS at compile time).
!
! Build (example):
!   mpif90 -O3 -march=native -ffast-math \
!     -I/path/to/2decomp_fft/include nls3d_crank_nicolson.f90 \
!     -L/path/to/2decomp_fft/lib -ldecomp2d -lfftw3_mpi -lfftw3 -lm
!   mpirun -np 8 ./a.out
!
! Output: writes coordinates, time vector, and a few checkpoints of |ψ| at slices.
! You can extend I/O using decomp_2d_io for parallel file formats (HDF5/NetCDF).
!===============================================================================
program main
  use decomp_2d
  use decomp_2d_fft
  use mpi
  implicit none

  !---------------------- Parameters ------------------------------------------
  integer, parameter :: rk = selected_real_kind(15,307)
  integer, parameter :: ck = kind((0.0_rk,0.0_rk))
  integer, parameter :: Nx=128, Ny=128, Nz=128
  integer, parameter :: Lx=1,  Ly=1,  Lz=1     ! multiples of 2π for domain size
  integer, parameter :: Nt=200
!  integer, parameter :: rk = kind(1.0d0)
  real(rk), parameter :: pi = 3.1415926535897932384626433832795_rk
  real(rk), parameter :: Tfinal = 0.5_rk
  real(rk), parameter :: sigma  = +1.0_rk      ! +1 focusing, -1 defocusing
  real(rk), parameter :: tol_it = 1.0e-12_rk    ! Picard tolerance
  integer,  parameter :: it_max = 50            ! Picard cap per step

  !---------------------- Declarations ----------------------------------------
  type(DECOMP_INFO) :: dec
  integer :: i,j,k, n, it, ierr, myid, nprocs
  real(rk) :: dt, t, scalemodes
  integer :: clk0, clk1, rate

  ! grids and wavenumbers (stored as complex purely imaginary i*k for convenience)
  real(rk), allocatable :: x(:), y(:), z(:), time(:)
  complex(ck), allocatable :: kx(:), ky(:), kz(:)

  ! fields in physical space (complex): ψ and temporaries
  complex(ck), allocatable :: psi(:,:,:), psi_old(:,:,:), psi_new(:,:,:)
  complex(ck), allocatable :: rhs(:,:,:), psi_star(:,:,:), S(:,:,:)

  ! Fourier-space arrays (complex)
  complex(ck), allocatable :: psik(:,:,:), rhsk(:,:,:)

  ! scratch for Laplacian symbol |k|^2 and Â inverse
  real(rk), allocatable :: kk2(:,:,:)
  complex(ck), allocatable :: Ainv(:,:,:), Bfac(:,:,:)

  ! for convergence checks across MPI
  real(rk) :: mychg, allchg

  !---------------------- Init parallel libs ----------------------------------
  call MPI_INIT(ierr)
  call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
  call MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierr)

  call decomp_2d_init(Nx,Ny,Nz, 0,0)
  call decomp_info_init(Nx,Ny,Nz, dec)
  call decomp_2d_fft_init

  if (myid==0) then
     print *, 'NLS3D CN — grid=',Nx,'x',Ny,'x',Nz,' procs=',nprocs
  end if

  !---------------------- Allocate --------------------------------------------
  allocate(x(Nx),y(Ny),z(Nz), time(0:Nt))
  allocate(kx(Nx),ky(Ny),kz(Nz))

  allocate( &
    psi     (dec%xst(1):dec%xen(1), dec%xst(2):dec%xen(2), dec%xst(3):dec%xen(3)), &
    psi_old (dec%xst(1):dec%xen(1), dec%xst(2):dec%xen(2), dec%xst(3):dec%xen(3)), &
    psi_new (dec%xst(1):dec%xen(1), dec%xst(2):dec%xen(2), dec%xst(3):dec%xen(3)), &
    rhs     (dec%xst(1):dec%xen(1), dec%xst(2):dec%xen(2), dec%xst(3):dec%xen(3)), &
    psi_star(dec%xst(1):dec%xen(1), dec%xst(2):dec%xen(2), dec%xst(3):dec%xen(3)), &
    S       (dec%xst(1):dec%xen(1), dec%xst(2):dec%xen(2), dec%xst(3):dec%xen(3)) )

  allocate( &
    psik (dec%zst(1):dec%zen(1), dec%zst(2):dec%zen(2), dec%zst(3):dec%zen(3)), &
    rhsk (dec%zst(1):dec%zen(1), dec%zst(2):dec%zen(2), dec%zst(3):dec%zen(3)) )

  allocate( kk2  (dec%zst(1):dec%zen(1), dec%zst(2):dec%zen(2), dec%zst(3):dec%zen(3)) )
  allocate( Ainv (dec%zst(1):dec%zen(1), dec%zst(2):dec%zen(2), dec%zst(3):dec%zen(3)) )
  allocate( Bfac (dec%zst(1):dec%zen(1), dec%zst(2):dec%zen(2), dec%zst(3):dec%zen(3)) )

  if (myid==0) print *,'Allocated arrays.'

  !---------------------- Grid / Wavenumbers ----------------------------------
  call build_k_and_x
  scalemodes = 1.0_rk / real(Nx*Ny*Nz, rk)

  ! Precompute |k|^2 on z-pencil and factors Ainv=(1 - i dt/2 |k|^2)^{-1},
  ! Bfac=(1 + i dt/2 |k|^2)
  dt = Tfinal/real(Nt,rk)
  call precompute_symbols(dt)

  !---------------------- Initial condition -----------------------------------
  ! Example: modulated Gaussian that can lead to collapse in focusing case.
  call init_condition
  time = 0.0_rk

  if (myid==0) then
    print *,'dt=',dt,'  sigma=',sigma
  end if

  !---------------------- Time integration ------------------------------------
  call system_clock(clk0, rate)
  do n=1,Nt
     t = (n-1)*dt
     psi_old = psi

     ! initial Picard guess: take ψ^{n+1,(0)} = ψ^n (other options: linear-only step)
     psi_new = psi_old

     it = 0; allchg = 1.0_rk
     do while (allchg>tol_it .and. it<it_max)
        it = it + 1
        ! midpoint value
        psi_star = 0.5_rk * (psi_new + psi_old)
        ! S = -i * sigma * |psi_star|^2
        S = cmplx(0.0_rk, -sigma, kind=ck) * ( real(psi_star,kind=rk)**2 + aimag(psi_star)**2 )

        ! RHS = B ψ^n + (dt/2) S (ψ^{n+1,(m)} + ψ^n)
        rhs = psi_old
        call decomp_2d_fft_3d(rhs, rhsk, DECOMP_2D_FFT_FORWARD)
#ifdef USE_DEALIAS
        call apply_dealias(rhsk)
#endif
        rhsk = Bfac * rhsk
        call decomp_2d_fft_3d(rhsk, rhs, DECOMP_2D_FFT_BACKWARD)
        rhs = rhs * scalemodes

        rhs = rhs + 0.5_rk*dt * S * (psi_new + psi_old)

        ! Solve A ψ^{n+1} = RHS via FFT: ψ^{n+1} = Ainv * FFT(RHS)
        call decomp_2d_fft_3d(rhs, rhsk, DECOMP_2D_FFT_FORWARD)
#ifdef USE_DEALIAS
        call apply_dealias(rhsk)
#endif
        psik = Ainv * rhsk
        call decomp_2d_fft_3d(psik, psi_new, DECOMP_2D_FFT_BACKWARD)
        psi_new = psi_new * scalemodes

        mychg = maxval( abs(psi_new - psi) )
        call MPI_ALLREDUCE(mychg, allchg, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierr)
        psi = psi_new
        if (myid==0) print '(A,I4,A,1PE12.5)', 'step ',n,'  Picard Δ=',allchg
     end do

     time(n) = n*dt

     ! (Optional) basic diagnostics every 10 steps
     if (mod(n,10)==0) then
        if (myid==0) call write_diag(n, t+dt, psi)
     end if
  end do
  call system_clock(clk1, rate)

  if (myid==0) then
     print '(A,1PE12.5,A)','Wall time (s): ', real(clk1-clk0,kind=rk)/real(rate,kind=rk), ''
  end if

  call decomp_2d_fft_finalize
  call decomp_2d_finalize
  call MPI_FINALIZE(ierr)

contains

  subroutine build_k_and_x
    integer :: ii, jj, kk, ind
    real(rk), parameter :: two_pi = 6.28318530717958647692528676655900577_rk
    ! kx
    do ii=1,Nx/2+1; kx(ii) = (0.0_rk,1.0_rk) * real(ii-1,rk)/Lx; end do
    kx(1+Nx/2) = (0.0_rk,0.0_rk)
    do ii=1,Nx/2-1; kx(ii+1+Nx/2) = -kx(1-ii+Nx/2); end do
    ind=1; do ii=-Nx/2, Nx/2-1; x(ind) = two_pi*real(ii,rk)*Lx/real(Nx,rk); ind=ind+1; end do
    ! ky
    do jj=1,Ny/2+1; ky(jj) = (0.0_rk,1.0_rk) * real(jj-1,rk)/Ly; end do
    ky(1+Ny/2) = (0.0_rk,0.0_rk)
    do jj=1,Ny/2-1; ky(jj+1+Ny/2) = -ky(1-jj+Ny/2); end do
    ind=1; do jj=-Ny/2, Ny/2-1; y(ind) = two_pi*real(jj,rk)*Ly/real(Ny,rk); ind=ind+1; end do
    ! kz
    do kk=1,Nz/2+1; kz(kk) = (0.0_rk,1.0_rk) * real(kk-1,rk)/Lz; end do
    kz(1+Nz/2) = (0.0_rk,0.0_rk)
    do kk=1,Nz/2-1; kz(kk+1+Nz/2) = -kz(1-kk+Nz/2); end do
    ind=1; do kk=-Nz/2, Nz/2-1; z(ind) = two_pi*real(kk,rk)*Lz/real(Nz,rk); ind=ind+1; end do
  end subroutine build_k_and_x

  subroutine precompute_symbols(dt)
    real(rk), intent(in) :: dt
    integer :: i1,i2,i3
    real(rk) :: k2
    do i3=dec%zst(3), dec%zen(3)
    do i2=dec%zst(2), dec%zen(2)
    do i1=dec%zst(1), dec%zen(1)
       k2 = (aimag(kx(i1)))**2 + (aimag(ky(i2)))**2 + (aimag(kz(i3)))**2
       kk2(i1,i2,i3) = k2
       Ainv(i1,i2,i3) = 1.0_rk / ( 1.0_rk - (0.0_rk,1.0_rk) * 0.5_rk * dt * k2 )
       Bfac(i1,i2,i3) =       ( 1.0_rk + (0.0_rk,1.0_rk) * 0.5_rk * dt * k2 )
    end do; end do; end do
  end subroutine precompute_symbols

#ifdef USE_DEALIAS
  subroutine apply_dealias(fk)
    complex(ck), intent(inout) :: fk(dec%zst(1):dec%zen(1), dec%zst(2):dec%zen(2), dec%zst(3):dec%zen(3))
    integer :: i1,i2,i3
    integer, parameter :: cx=int(real(Nx,kind=rk)/3.0_rk), cy=int(real(Ny,kind=rk)/3.0_rk), cz=int(real(Nz,kind=rk)/3.0_rk)
    do i3=dec%zst(3), dec%zen(3)
    do i2=dec%zst(2), dec%zen(2)
    do i1=dec%zst(1), dec%zen(1)
       if (abs(i1-1) > cx .or. abs(i2-1) > cy .or. abs(i3-1) > cz) fk(i1,i2,i3) = (0.0_rk,0.0_rk)
    end do; end do; end do
  end subroutine apply_dealias
#endif

  subroutine init_condition
    ! Gaussian bump centered at (0,0,0): ψ(x,0) = A * exp(-r^2/w^2) * exp(i*kz z)
    real(rk), parameter :: A = 2.0_rk, w0 = 1.0_rk, kz0 = 0.0_rk
    integer :: i1,i2,i3
    real(rk) :: xx,yy,zz,r2
    do i3=dec%xst(3), dec%xen(3)
    do i2=dec%xst(2), dec%xen(2)
    do i1=dec%xst(1), dec%xen(1)
       xx = x(i1); yy = y(i2); zz = z(i3)
       r2 = xx*xx + yy*yy + zz*zz
       psi(i1,i2,i3) = A * exp(-r2/(w0*w0)) * exp( (0.0_rk,1.0_rk)*kz0*zz )
    end do; end do; end do
  end subroutine init_condition

  subroutine write_diag(step, tt, field)
    integer, intent(in) :: step
    real(rk), intent(in) :: tt
    complex(ck), intent(in) :: field(dec%xst(1):dec%xen(1), dec%xst(2):dec%xen(2), dec%xst(3):dec%xen(3))
    real(rk) :: mass_local, mass_global
    mass_local = sum( real(field,kind=rk)**2 + aimag(field)**2 ) * ( (2.0_rk*pi*Lx/Nx) * (2.0_rk*pi*Ly/Ny) * (2.0_rk*pi*Lz/Nz) )
    call MPI_ALLREDUCE(mass_local, mass_global, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
    print '(A,I6,A,1PE12.5,A,1PE12.5)', 'diag step=',step,' time=',tt,' mass≈',mass_global
  end subroutine write_diag

end program main
