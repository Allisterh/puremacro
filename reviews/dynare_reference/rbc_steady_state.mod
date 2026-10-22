var c k y a g;
varexo e;
parameters alpha beta delta rho gy;
alpha = 0.33;
beta = 0.99;
delta = 0.025;
rho = 0.8;
gy = 0.2;

model;
1/c = beta*(1/c(+1))*(alpha*exp(a(+1))*k^(alpha-1) + 1 - delta);
y = exp(a)*k(-1)^alpha;
k = y - c - g + (1-delta)*k(-1);
g = gy*STEADY_STATE(y);
a = rho*a(-1) + e;
end;

steady_state_model;
a = 0;
k = ((1/beta - 1 + delta)/alpha)^(1/(alpha-1));
y = k^alpha;
g = gy*y;
c = y - delta*k - g;
end;
steady;

shocks;
var e;
periods 1;
values 0.05;
end;

perfect_foresight_setup(periods=40);
perfect_foresight_solver(tolf=1e-12, tolx=1e-12);
