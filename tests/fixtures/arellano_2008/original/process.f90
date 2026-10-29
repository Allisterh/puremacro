module process

use global
use parameters

contains

subroutine utility(c, utils)
real (8), intent (in) :: c
real (8), intent (out) :: utils
utils=(c**(1-sigma))/(1-sigma)
end subroutine utility

subroutine grid
use global
integer :: i
b (1) = bmin
do i=2,bigg
b(i)=b(i-1)+(lbb-bmin)/bigg
end do
do i=bigg+1,fineg
b(i)=b(i-1)+(ubb-lbb)/fineg
end do 
do i=fineg+1,ngpb
b (i) = b (i-1) + (bmax-ubb)/((ngpb-fineg-bigg)-1)
if (b(i)> 0.0 .and. b(i-1) < 0.0) then
b(i)=0.0
izero=i
end if
end do
end subroutine grid

end module process