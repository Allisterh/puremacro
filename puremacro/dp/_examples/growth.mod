// Neoclassical growth: log utility and full capital depreciation.
parameters beta alpha;
beta = 0.95;
alpha = 0.36;
state k = geomspace(0.01, 2.0, 300);

model(name='Growth by value function iteration');
  c = k^alpha - k(+1);
  reward = log(c);
end;
constraints;
  c > 0;
end;
solve(method=vfi, tol=1e-9, distribution=false);
