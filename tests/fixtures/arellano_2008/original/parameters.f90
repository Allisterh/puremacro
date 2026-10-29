module parameters


real (8), parameter ::  bita = 0.95282, sigma=2, y=10.0, theta=0.282, gam=0.969,  rs=1.017
integer, parameter ::  ngpb=200,  itmax = 50, maxitx = 2000, &
simlen = 10000,  vfitmax=500, peritmax=500, fineg=1, bigg=1, sy=21, sr=1, ss=sy*sr
real (8), parameter :: bmax = 1.500,  lbb=-3.300, ubb= -3.30, bmin=-3.30,  kkk=1

real (8), parameter :: errel = 0.000001, errels = 0.000001, &
critcon = 0.00001, critcon2 = 0.0001, critvfit = 0.00000000001, critprice=0.00000001, &
critpfit=0.000001

end module parameters
