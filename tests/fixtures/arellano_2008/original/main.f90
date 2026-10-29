
program bond

use parameters
use process
use global

include 'link_f90_dll.h'

implicit none

!!  Define some vectors and scalars

real (8), dimension (ss) :: utilaut, vaut,  v00,  util00, temp2,  &
utilbhst, vbhst, condef, utildef, vdef,conaut, temp3
real (8), dimension (ss,ss) ::  ident, temp, tempinv, temp1, tempinv2

real (8) :: evo, evp, i_bp_star, gap, b00
real (8), dimension (ngpb, ss) :: unos
real (8), dimension (ss) :: shocks
integer ::  i_b, i_bp, i_r, i_rp, i_ssp, i, ite, kk, j , m

real (8), dimension (simlen+1) :: css, yss,  pss, pssnd, tbss, defss, countdef, income, dec
real (8), dimension (simlen+1) :: bss, inds, rand

unos=1.0

!! Upload the shock vector and transition matrix

open (unit=9, file="C:\Arellano\default\PIMAT.dat", position="rewind")
read (9, *) ptran
close (unit=9)

open (unit=9, file="C:\Arellano\default\YMAT.dat", position="rewind")
read (9, *) shocks
close (unit=9) 

shoc=exp(shocks(:))
rshoc=rs
ptran=transpose(ptran)
print *, "sum ptran", sum(ptran(1,:))

!! Calculate utility of permament autarky

do i_ss=1,ss
if (y*shoc(i_ss)<=y*gam) then
conaut(i_ss)=y*shoc(i_ss)
else
conaut(i_ss)=y*gam
end if
call utility (conaut(i_ss), utilaut(i_ss))
end do
ident = 0.0
do i=1,ss
ident(i,i) = 1.0
end do
temp = (ident-bita*ptran)
CALL DLINRG (ss, temp, ss, tempinv, ss)
vaut = (matmul(tempinv,utilaut))
print *, "vaut", vaut(1)

!! Construct bond grid
call grid
! save bond grid
open (unit=9, file="C:\Arellano\default\b.f90", position="rewind")
write (9, *) b
close (unit=9)


! If have good initial guesses for value function and bond price function, upload them

if (kkk==1) then

open (unit=9, file="C:\Arellano\default\v.f90", position="rewind")
read (9, *) v
close (unit=9) 

open (unit=9, file="C:\Arellano\default\p.f90", position="rewind")
read (9, *) p
close (unit=9) 

! otherwise start with a constant guess

else
do i_ss=1,ss
p(:,i_ss)=1/rs
v(:,i_ss)=vaut(i_ss)
end do
end if


!! PRICE INTERATION : OUTSIDE LOOP 

do kk = 1,vfitmax

!!  For a given bond price function , compute the optimal value function, by VALUE FUNCTION ITERATION

do ite = 1,vfitmax
vo = v
do i_ss=1,ss
do i_b=1,ngpb

!! For given future value , vo , compute optimal consumption and savings 
assets = b(i_b)
evo=-10000000000000000000.0
i_bp_star = 1
do i_bp=1,ngpb
	price = p(i_bp, i_ss)
	c = y*shoc(i_ss) + assets - price*b(i_bp)
	if (c<=0.0) then
		evp=-10000000000000000000.0
	else
	call utility(c, utils)
	evp = 0.0
	do i_ssp=1,ss
	evp = evp + ptran(i_ss,i_ssp)*bita*vo(i_bp,i_ssp)
	end do
	evp = evp + utils 
	end if
	if (evp>=evo) then
		evo = evp
		i_bp_star = i_bp
	end if
end do
i_s_star(i_b, i_ss) = i_bp_star
sav(i_b, i_ss) = b(i_bp_star)
psav(i_b, i_ss) = p(i_bp_star,i_ss)
v(i_b, i_ss) = evo
savnd=sav
psavnd=psav
end do
end do


!! Compute the value of default 
!! Value of default is a function of value at zero debt :  v_contract(0,s)
do i_ss=1,ss
v00(i_ss)=v(izero,i_ss)
end do

!! Value of default is defined recursively as a weighted average of value of default and v_contract(0,s)

!! Vdef=inv(eye(s,s)-beta*(1-theta)*P)*(U+beta*theta*P*v00); 

temp1 = (ident-bita*(1-theta)*ptran)
call dlinrg (ss, temp1, ss, tempinv2, ss)
temp2= matmul(ptran, v00)
utilbhst=utilaut+bita*theta* temp2
vbhst = matmul(tempinv2,utilbhst)
do i_ss=1,ss
call utility (conaut(i_ss), utildef(i_ss))
end do
temp3=matmul(ptran,vbhst)
vdef=utildef+bita*temp3


!! Now that we have v_contract and v_default (for a given guess of future value (vo) calculate v_option 

def=0
do i_ss=1,ss
do i_b=1,ngpb
if (b(i_b)<0.0) then
if (v(i_b, i_ss) < vdef(i_ss)) then
	v(i_b, i_ss) = vdef(i_ss)
	def(i_b, i_ss) = 1
	sav(i_b, i_ss)=0.0
	psav(i_b, i_ss)=0.0
end if
end if
end do
end do


!!  Check whether v_option new = v_option old 
!!  Iterate until convergence 

gap = maxval (abs (v - vo))
print *, "ite ", ite
if (gap<critvfit) then
 print *, "ite ", ite
 exit
end if
end do

!!  Now we have v_option computed for a given bond price function 


!! Check  whether initial guess on bond price function is consistent with default probabilities
do i_ss=1,ss
do i_b=1,ngpb
	evp=0.0
	do i_ssp=1,ss
	evp = evp + ptran(i_ss,i_ssp)*def(i_b,i_ssp)
	end do
   dprob(i_b, i_ss)=evp
   p1(i_b,i_ss)=(1.0-dprob(i_b,i_ss))/rshoc(i_ss)
end do
end do

!! Interate on the bond price function until convergence 

gap = maxval (abs (p1 - p))
if (gap<critprice) then
 print *, "kk ", kk
 exit
else
print *, "sum abs errors", sum (abs (p1-p))
p=.5*p+.5*p1;
end if
end do


!! Now we have all the solutions

! Calculate consumption decision rules
do i_ss=1,ss
do i_b=1,ngpb
if (def(i_b, i_ss) == 1) then
cons(i_b, i_ss)=conaut(i_ss)
else
cons(i_b, i_ss)=y*shoc(i_ss) + b(i_b) - psav(i_b, i_ss)*sav(i_b,i_ss)
end if
yy(i_b, i_ss)=y*shoc(i_ss)
consnd(i_b, i_ss)=y*shoc(i_ss) + b(i_b) - psavnd(i_b, i_ss)*savnd(i_b,i_ss)
end do
end do


!! Save all vectors 

open (unit=9, file="C:\Arellano\default\v.f90", position="rewind")
write (9, *) v
close (unit=9) 
open (unit=9, file="C:\Arellano\default\sav.f90", position="rewind")
write (9, *) sav
close (unit=9) 
open (unit=9, file="C:\Arellano\default\savnd.f90", position="rewind")
write (9, *) savnd
close (unit=9) 
open (unit=9, file="C:\Arellano\default\b.f90", position="rewind")
write (9, *) b
close (unit=9)
open (unit=9, file="C:\Arellano\default\i_s_star.f90", position="rewind")
write (9, *) i_s_star
close (unit=9)
open (unit=9, file="C:\Arellano\default\p.f90", position="rewind")
write (9, *) p
close (unit=9) 
open (unit=9, file="C:\Arellano\default\psav.f90", position="rewind")
write (9, *) psav
close (unit=9) 
open (unit=9, file="C:\Arellano\default\psavnd.f90", position="rewind")
write (9, *) psavnd
close (unit=9) 
open (unit=9, file="C:\Arellano\default\def.f90", position="rewind")
write (9, *) def
close (unit=9) 
open (unit=9, file="C:\Arellano\default\cons.f90", position="rewind")
write (9, *) cons
close (unit=9)
open (unit=9, file="C:\Arellano\default\consnd.f90", position="rewind")
write (9, *) consnd
close (unit=9)
open (unit=9, file="C:\Arellano\default\yy.f90", position="rewind")
write (9, *) yy
close (unit=9)
open (unit=9, file="C:\Arellano\default\dprob.f90", position="rewind")
write (9, *) dprob
close (unit=9)
open (unit=9, file="C:\Arellano\default\conaut.f90", position="rewind")
write (9, *) conaut
close (unit=9)
open (unit=9, file="C:\Arellano\default\izero.f90", position="rewind")
write (9, *) izero
close (unit=9)
open (unit=9, file="C:\Arellano\default\theta.f90", position="rewind")
write (9, *) theta
close (unit=9)


end program bond
