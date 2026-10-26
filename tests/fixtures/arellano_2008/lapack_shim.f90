subroutine dlinrg(n, a, lda, ainv, ldinv)
  implicit none
  integer, intent(in) :: n, lda, ldinv
  real(8), intent(in) :: a(lda,n)
  real(8), intent(out) :: ainv(ldinv,n)
  real(8) :: work(n,n)
  integer :: piv(n), info, i
  work = a(1:n,1:n)
  ainv = 0.0d0
  do i=1,n
    ainv(i,i) = 1.0d0
  end do
  call dgesv(n,n,work,n,piv,ainv,ldinv,info)
  if (info /= 0) stop 2
end subroutine dlinrg
