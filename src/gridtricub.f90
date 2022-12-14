subroutine gridtricub( &
     x_newgrid,nx_new,y_newgrid,ny_new,z_newgrid,nz_new, &
     f_new,if1,if2, &
     nx,xpkg,ny,ypkg,nz,zpkg,fspl,inf4,inf5,iwarn,ier)
  use iso_c_binding, only: fp => c_double
  !
  !  regrid a spline function f defined vs. x,y,z as in xpkg, etc.
  !  to a new grid, given by x_newgrid, y_newgrid, z_newgrid
  !
  !  set warning flag if the range exceeds the range of the
  !  original x/y/zpkg's.
  !
  !  (xpkg/ypkg/zpkg -- axis data, see genxpkg subroutine)
  !
  !  input:
  !
  !============
  implicit none
  integer ny_new,nz_new,nx_new,iz,iy
  !============
  real(fp) :: x_newgrid(nx_new)            ! new x grid
  real(fp) :: y_newgrid(ny_new)            ! new y grid
  real(fp) :: z_newgrid(nz_new)            ! new z grid
  !
  !  output:
  !
  integer if1,if2                   ! 1st dimensions of f_new
  real(fp) :: f_new(if1,if2,nz_new)        ! f evaluated on this grid
  !
  !  input:
  !
  integer nx                        ! size of old grid
  real(fp) :: xpkg(nx,4)                   ! old grid "package"
  integer ny                        ! size of old grid
  real(fp) :: ypkg(ny,4)                   ! old grid "package"
  integer nz                        ! size of old grid
  real(fp) :: zpkg(nz,4)                   ! old grid "package"
  !
  integer inf4,inf5                 ! array dimensions
  real(fp) :: fspl(8,inf4,inf5,nz)     ! spline coefficients of f
  !
  !  output:
  !  condition codes, =0 for normal exit
  !
  integer iwarn                     ! =1 if new grid points out of range
  integer ier                       ! =1 if there is an argument error
  !
  !--------------------------------------------
  !  local
  !
  real(fp) :: ytmp(nx_new)
  real(fp) :: ztmp(nx_new)
  integer ict(10)
  !
  data ict/1,0,0,0,0,0,0,0,0,0/
  !
  !--------------------------------------------
  !
  do iz=1,nz_new
     ztmp=z_newgrid(iz)
     do iy=1,ny_new
        ytmp=y_newgrid(iy)
        call vectricub(ict,nx_new,x_newgrid,ytmp,ztmp, &
             nx_new,f_new(1,iy,iz), &
             nx,xpkg,ny,ypkg,nz,zpkg,fspl,inf4,inf5,iwarn,ier)
     end do
  end do
  !
  return
end subroutine gridtricub
