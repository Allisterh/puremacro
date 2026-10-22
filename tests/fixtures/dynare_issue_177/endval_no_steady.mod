var y x;
varexo e;
parameters rho;
rho = 0.5;

model;
y = 1 + rho*(y(-1)-1) + e;
x = STEADY_STATE(y);
end;

initval;
y = 1;
x = 1;
e = 0;
end;
steady;

endval;
e = 0.5;
y = 2;
x = 2;
end;

perfect_foresight_setup(periods=10);
perfect_foresight_solver(tolf=1e-12, tolx=1e-12);
