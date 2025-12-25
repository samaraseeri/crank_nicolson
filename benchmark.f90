!===============================================================================
! 3D Cubic Nonlinear Schrödinger (NLS) on [0,2π]^3 (no symmetry assumed)
! Pseudospectral space + Crank–Nicolson (CN) in time with fixed-point iteration.
! MPI + 2DECOMP&FFT for parallel FFTs.
!
! Equation (focusing if sigma=+1, defocusing if sigma=-1):
!    i ∂t ψ + Δψ + sigma |ψ|^2 ψ = 0
!
! Benchmark mode:
!   - No video / no file output required
!   - Prints timing summary for Table: FFT, nonlinear, other, total (per step)
!   - Grid size chosen at runtime: Nx Ny Nz
!
! Run:
!   mpirun -np 8 ./nls_bench 128 128 128 100
!   mpirun -np 8 ./nls_bench 256 256 256 100
!   (512^3 likely NOT feasible on 16GB RAM with this many arrays)
!
! Notes on memory (rough estimate):
!   This code allocates multiple complex 3D arrays in physical space and k-space.
!   Complex(8-byte real + 8-byte imag) ~ 16 bytes/element (double complex).
!   At 512^3: 134,217,728 elements -> ~2.0 GB per complex field.
!   With several such arrays, total RAM can exceed 16GB quickly.
!===============================================================================
program main
  use decomp_2d
  use decomp_2d_fft
  use mpi
  implicit none

  !---------------------- Parameters ------------------------------------------
  integer, parameter :: rk = selected_real_kind(15,307)
  integer, parameter :: ck = kind((0.0_rk,0.0_rk))

  integer :: Nx, Ny, Nz
  integer, parameter :: Lx=1, Ly=1, Lz=1   ! domain size multipliers for 2π
  integer :: Nt_meas                         ! number of steps for averaging

  real(rk), parameter :: pi     = 3.1415926535897932384626433832795_rk
  real(rk), parameter :: two_pi = 2.0_rk*pi

  real(rk), parameter :: Tfinal = 0.5_rk
  real(rk), parameter :: sigma  = +1.0_rk      ! +1 focusing, -1 defocusing
  real(rk), parameter :: tol_it = 1.0e-12_rk   ! Picard tolerance
  integer,  parameter :: it_max = 50           ! Picard cap per step

  ! initial condition parameters (match your current code)
  real(rk), parameter :: A0  = 2.0_rk
  real(rk), parameter :: w0  = 1.0_rk
  real(rk), parameter :: kz0 = 0.0_rk

  logical, parameter :: verbose = .false.

  !---------------------- Declarations ----------------------------------------
  type(DECOMP_INFO) :: dec
  integer :: i1,i2,i3, n, it, ierr, myid, nprocs
  real(rk) :: dt, scalemodes

  ! grids and wavenumbers
  real(rk), allocatable :: x(:), y(:), z(:)
  real(rk), allocatable :: kx(:), ky(:), kz(:)

  ! fields in physical space (complex): ψ and temporaries (x-pencil)
  complex(ck), allocatable :: psi(:,:,:), psi_old(:,:,:), psi_new(:,:,:)
  complex(ck), allocatable :: rhs(:,:,:), psi_star(:,:,:), S(:,:,:)

  ! Fourier-space arrays (complex): z-pencil
  complex(ck), allocatable :: psik(:,:,:), rhsk(:,:,:)

  ! scratch: |k|^2 and factors
  real(rk),    allocatable :: kk2(:,:,:)
  complex(ck), allocatable :: Ainv(:,:,:), Bfac(:,:,:)

  ! convergence checks across MPI
  real(rk) :: mychg, allchg

  ! timing
  real(rk) :: t_fft_local, t_nl_local, t_total_local
  real(rk) :: t_fft_max,   t_nl_max,   t_total_max
  real(rk) :: t0, t1

  !---------------------- Init parallel libs ----------------------------------
  call MPI_INIT(ierr)
  call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
  call MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierr)

  !---------------------- Read runtime parameters -----------------------------
  call read_params_from_cli(Nx, Ny, Nz, Nt_meas)

  if (mod(Nx,2)/=0 .or. mod(Ny,2)/=0 .or. mod(Nz,2)/=0) then
     if (myid==0) print *, 'ERROR: Nx,Ny,Nz must be even for this FFT setup.'
     call MPI_ABORT(MPI_COMM_WORLD, 1, ierr)
  end if

  ! Initialize decomposition / FFT library
  call decomp_2d_init(Nx,Ny,Nz, 0,0)
  call decomp_info_init(Nx,Ny,Nz, dec)
  call decomp_2d_fft_init

  if (myid==0) then
     print *, 'NLS3D CN benchmark — grid=',Nx,'x',Ny,'x',Nz,' procs=',nprocs, ' steps=',Nt_meas
  end if

  !---------------------- Allocate --------------------------------------------
  allocate(x(Nx), y(Ny), z(Nz))
  allocate(kx(Nx), ky(Ny), kz(Nz))

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

  !---------------------- Grid / Wavenumbers ----------------------------------
  call build_grid_and_wavenumbers(Nx,Ny,Nz, Lx,Ly,Lz, x,y,z, kx,ky,kz)

  scalemodes = 1.0_rk / real(Nx*Ny*Nz, rk)

  dt = Tfinal/real(Nt_meas,rk)
  call precompute_symbols(dt, dec, kx,ky,kz, kk2, Ainv, Bfac)

  !---------------------- Initial condition -----------------------------------
  call init_condition(dec, x,y,z, psi)

  if (myid==0) then
    print '(A,1PE12.5,A,1PE12.5)', 'dt=',dt,'  sigma=',sigma
  end if

  !---------------------- Timing accumulators ---------------------------------
  t_fft_local   = 0.0_rk
  t_nl_local    = 0.0_rk
  t_total_local = 0.0_rk

  !---------------------- Time integration / Benchmark -------------------------
  do n=1,Nt_meas
     t0 = MPI_Wtime()

     psi_old = psi
     psi_new = psi_old

     it = 0
     allchg = huge(1.0_rk)

     do while (allchg>tol_it .and. it<it_max)
        it = it + 1

        psi_star = 0.5_rk * (psi_new + psi_old)

        ! Nonlinear coefficient S = -i*sigma*|psi_star|^2   (pointwise)
        t1 = MPI_Wtime()
        S = cmplx(0.0_rk, -sigma, kind=ck) * ( real(psi_star,kind=rk)**2 + aimag(psi_star)**2 )
        t_nl_local = t_nl_local + (MPI_Wtime() - t1)

        ! RHS = B psi^n + (dt/2) S (psi_new + psi_old)
        rhs = psi_old

        t1 = MPI_Wtime()
        call decomp_2d_fft_3d(rhs, rhsk, DECOMP_2D_FFT_FORWARD)
        t_fft_local = t_fft_local + (MPI_Wtime() - t1)

        rhsk = Bfac * rhsk

        t1 = MPI_Wtime()
        call decomp_2d_fft_3d(rhsk, rhs, DECOMP_2D_FFT_BACKWARD)
        t_fft_local = t_fft_local + (MPI_Wtime() - t1)

        rhs = rhs * scalemodes
        rhs = rhs + 0.5_rk*dt * S * (psi_new + psi_old)

        ! Solve A psi_new = RHS via FFT inversion
        t1 = MPI_Wtime()
        call decomp_2d_fft_3d(rhs, rhsk, DECOMP_2D_FFT_FORWARD)
        t_fft_local = t_fft_local + (MPI_Wtime() - t1)

        psik = Ainv * rhsk

        t1 = MPI_Wtime()
        call decomp_2d_fft_3d(psik, psi_new, DECOMP_2D_FFT_BACKWARD)
        t_fft_local = t_fft_local + (MPI_Wtime() - t1)

        psi_new = psi_new * scalemodes

        mychg = maxval( abs(psi_new - psi) )
        call MPI_ALLREDUCE(mychg, allchg, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierr)

        psi = psi_new

        if (verbose .and. myid==0) then
           print '(A,I5,A,I3,A,1PE12.5)', 'step=',n,' it=',it,' Picard Δ=',allchg
        end if
     end do

     t_total_local = t_total_local + (MPI_Wtime() - t0)
  end do

  !---------------------- Reduce timings and print summary ---------------------
  call MPI_ALLREDUCE(t_fft_local,   t_fft_max,   1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierr)
  call MPI_ALLREDUCE(t_nl_local,    t_nl_max,    1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierr)
  call MPI_ALLREDUCE(t_total_local, t_total_max, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierr)

  t_fft_max   = t_fft_max   / real(Nt_meas, rk)
  t_nl_max    = t_nl_max    / real(Nt_meas, rk)
  t_total_max = t_total_max / real(Nt_meas, rk)

  if (myid==0) then
     print '(A)', 'grid,procs,steps,dt,fft_s,nonlinear_s,other_s,total_s'
     print '(I0,A,I0,A,I0,A,I0,A,I0,A,1PE12.5,A,1PE12.5,A,1PE12.5,A,1PE12.5,A,1PE12.5)', &
          Nx,'x',Ny,'x',Nz, ',', nprocs, ',', Nt_meas, ',', dt, ',', &
          t_fft_max, ',', t_nl_max, ',', max(0.0_rk, t_total_max - t_fft_max - t_nl_max), ',', t_total_max
  end if

  !---------------------- Finalize --------------------------------------------
  call decomp_2d_fft_finalize
  call decomp_2d_finalize
  call MPI_FINALIZE(ierr)

contains

  subroutine read_params_from_cli(Nx, Ny, Nz, Nt)
    integer, intent(out) :: Nx, Ny, Nz, Nt
    character(len=64) :: arg
    integer :: ios

    Nx = 128; Ny = 128; Nz = 128
    Nt = 100

    call get_command_argument(1, arg)
    if (len_trim(arg)>0) then
       read(arg,*,iostat=ios) Nx
       if (ios/=0) Nx = 128
    end if

    call get_command_argument(2, arg)
    if (len_trim(arg)>0) then
       read(arg,*,iostat=ios) Ny
       if (ios/=0) Ny = Nx
    else
       Ny = Nx
    end if

    call get_command_argument(3, arg)
    if (len_trim(arg)>0) then
       read(arg,*,iostat=ios) Nz
       if (ios/=0) Nz = Nx
    else
       Nz = Nx
    end if

    call get_command_argument(4, arg)
    if (len_trim(arg)>0) then
       read(arg,*,iostat=ios) Nt
       if (ios/=0) Nt = 100
    end if
  end subroutine read_params_from_cli

  subroutine build_grid_and_wavenumbers(Nx,Ny,Nz, Lx,Ly,Lz, x,y,z, kx,ky,kz)
    integer, intent(in) :: Nx,Ny,Nz, Lx,Ly,Lz
    real(rk), intent(out) :: x(Nx), y(Ny), z(Nz)
    real(rk), intent(out) :: kx(Nx), ky(Ny), kz(Nz)
    integer :: i, m

    ! Grid in [0,2π*L)
    do i=1,Nx
       x(i) = two_pi*real(i-1,rk)*real(Lx,rk)/real(Nx,rk)
    end do
    do i=1,Ny
       y(i) = two_pi*real(i-1,rk)*real(Ly,rk)/real(Ny,rk)
    end do
    do i=1,Nz
       z(i) = two_pi*real(i-1,rk)*real(Lz,rk)/real(Nz,rk)
    end do

    ! Wavenumbers corresponding to exp(i*k*x) convention on [0,2π*L):
    ! k = 0,1,...,N/2, -(N/2-1),..., -1 scaled by 1/L
    do i=1,Nx
       m = i-1
       if (m <= Nx/2) then
          kx(i) = real(m,rk)/real(Lx,rk)
       else
          kx(i) = real(m-Nx,rk)/real(Lx,rk)
       end if
    end do
    do i=1,Ny
       m = i-1
       if (m <= Ny/2) then
          ky(i) = real(m,rk)/real(Ly,rk)
       else
          ky(i) = real(m-Ny,rk)/real(Ly,rk)
       end if
    end do
    do i=1,Nz
       m = i-1
       if (m <= Nz/2) then
          kz(i) = real(m,rk)/real(Lz,rk)
       else
          kz(i) = real(m-Nz,rk)/real(Lz,rk)
       end if
    end do
  end subroutine build_grid_and_wavenumbers

  subroutine precompute_symbols(dt, dec, kx,ky,kz, kk2, Ainv, Bfac)
    real(rk), intent(in) :: dt
    type(DECOMP_INFO), intent(in) :: dec
    real(rk), intent(in) :: kx(:), ky(:), kz(:)
    real(rk), intent(out) :: kk2(dec%zst(1):dec%zen(1), dec%zst(2):dec%zen(2), dec%zst(3):dec%zen(3))
    complex(ck), intent(out) :: Ainv(dec%zst(1):dec%zen(1), dec%zst(2):dec%zen(2), dec%zst(3):dec%zen(3))
    complex(ck), intent(out) :: Bfac(dec%zst(1):dec%zen(1), dec%zst(2):dec%zen(2), dec%zst(3):dec%zen(3))
    integer :: i1,i2,i3
    real(rk) :: k2

    do i3=dec%zst(3), dec%zen(3)
    do i2=dec%zst(2), dec%zen(2)
    do i1=dec%zst(1), dec%zen(1)
       k2 = kx(i1)*kx(i1) + ky(i2)*ky(i2) + kz(i3)*kz(i3)
       kk2(i1,i2,i3) = k2
       Ainv(i1,i2,i3) = 1.0_rk / ( 1.0_rk - cmplx(0.0_rk,1.0_rk,ck) * 0.5_rk*dt * k2 )
       Bfac(i1,i2,i3) =             ( 1.0_rk + cmplx(0.0_rk,1.0_rk,ck) * 0.5_rk*dt * k2 )
    end do
    end do
    end do
  end subroutine precompute_symbols

  subroutine init_condition(dec, x,y,z, psi)
    type(DECOMP_INFO), intent(in) :: dec
    real(rk), intent(in) :: x(:), y(:), z(:)
    complex(ck), intent(out) :: psi(dec%xst(1):dec%xen(1), dec%xst(2):dec%xen(2), dec%xst(3):dec%xen(3))
    integer :: i1,i2,i3
    real(rk) :: xx,yy,zz, r2

    ! Center at origin in [0,2π): treat coordinates as signed around 0 for a symmetric Gaussian:
    ! Map x in [0,2π) to x̃ in (-π,π]
    do i3=dec%xst(3), dec%xen(3)
    do i2=dec%xst(2), dec%xen(2)
    do i1=dec%xst(1), dec%xen(1)
       xx = x(i1); if (xx > pi) xx = xx - two_pi
       yy = y(i2); if (yy > pi) yy = yy - two_pi
       zz = z(i3); if (zz > pi) zz = zz - two_pi
       r2 = xx*xx + yy*yy + zz*zz
       psi(i1,i2,i3) = A0 * exp(-r2/(w0*w0)) * exp( cmplx(0.0_rk,1.0_rk,ck) * kz0 * zz )
    end do
    end do
    end do
  end subroutine init_condition

end program main
