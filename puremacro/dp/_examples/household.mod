// Stationary incomplete-markets household, at supplied prices.
parameters beta gamma r w;
beta = 0.96; gamma = 2.0; r = 0.02; w = 1.0;
state a = 40*linspace(0, 1, 120)^2;
exogenous z = rouwenhorst(rho=0.9, sigma=0.15, n=5);

model(name='Consumption and saving');
  c = (1+r)*a + w*exp(z) - a(+1);
  reward = crra(c, gamma);
end;
constraints; c > 0; end;
aggregates; A = a; C = c; end;
solve(method=egm, tol=1e-9);
