// Deterministic income profile, retirement, and survival.
parameters beta gamma r T;
beta = 0.96; gamma = 2.0; r = 0.02; T = 30;
state a = 30*linspace(0, 1, 120)^2;
horizon = T;
survival = full(T, 0.99);
model(name='Life cycle');
  income = (age < 20)*exp(0.04*age - 0.001*age^2) + (age >= 20)*0.6;
  c = (1+r)*a + income - a(+1);
  reward = crra(c, gamma);
end;
constraints; c > 0; end;
solve(method=egm);
