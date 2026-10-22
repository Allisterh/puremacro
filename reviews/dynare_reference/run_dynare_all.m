% Runner script to execute all reference models under Dynare in MATLAB R2026a
addpath('/Applications/Dynare/8-2026-05-26-1803/matlab');

disp('=== 1. RUNNING RBC MODEL ===');
try
    dynare rbc_steady_state.mod noclearall;
    endo_names = cellstr(M_.endo_names);
    sim_data = oo_.endo_simul';
    save('rbc_dynare_out.mat', 'endo_names', 'sim_data');
    writetable(array2table(sim_data, 'VariableNames', endo_names), 'rbc_dynare_out.csv');
    disp('DYNARE RBC SIMULATION COMPLETED SUCCESSFULLY');
catch ME
    disp(['ERROR in RBC: ', ME.message]);
end

disp('=== 2. RUNNING MINIMAL #177 MODEL ===');
try
    dynare minimal_177.mod noclearall;
    endo_names = cellstr(M_.endo_names);
    sim_data = oo_.endo_simul';
    save('minimal_177_dynare_out.mat', 'endo_names', 'sim_data');
    writetable(array2table(sim_data, 'VariableNames', endo_names), 'minimal_177_dynare_out.csv');
    disp('DYNARE MINIMAL 177 SIMULATION COMPLETED SUCCESSFULLY');
catch ME
    disp(['ERROR in MINIMAL 177: ', ME.message]);
end

disp('=== 3. RUNNING ENDVAL WITH STEADY ===');
try
    dynare endval_with_steady.mod noclearall;
    endo_names = cellstr(M_.endo_names);
    sim_data = oo_.endo_simul';
    save('endval_with_steady_dynare_out.mat', 'endo_names', 'sim_data');
    writetable(array2table(sim_data, 'VariableNames', endo_names), 'endval_with_steady_dynare_out.csv');
    disp('DYNARE ENDVAL WITH STEADY COMPLETED SUCCESSFULLY');
catch ME
    disp(['ERROR in ENDVAL WITH STEADY: ', ME.message]);
end

disp('=== 4. RUNNING ENDVAL NO STEADY ===');
try
    dynare endval_no_steady.mod noclearall;
    endo_names = cellstr(M_.endo_names);
    sim_data = oo_.endo_simul';
    save('endval_no_steady_dynare_out.mat', 'endo_names', 'sim_data');
    writetable(array2table(sim_data, 'VariableNames', endo_names), 'endval_no_steady_dynare_out.csv');
    disp('DYNARE ENDVAL NO STEADY COMPLETED SUCCESSFULLY');
catch ME
    disp(['ERROR in ENDVAL NO STEADY: ', ME.message]);
end

disp('=== ALL DYNARE RUNS FINISHED ===');
