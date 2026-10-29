module global

use parameters

implicit none
real (8), dimension(ngpb) :: b
real (8), dimension (ngpb,ss) :: v, vo, sav, p,  def, i_s_star, dprob, p1,  cons, &
savnd, muc2, muc1, consnd, psav, psavnd, yy
real (8) :: c, assets, utils, price, izero
real (8), dimension (ss) :: shoc, rshoc, gams
real (8), dimension (simlen) :: simu, ww
real (8), dimension (ss,ss) :: ptran
integer :: i_ss

end module global