"""Exercise the model inspection/audit interfaces in a real browser kernel."""
import json

import numpy as np
import pytest


@pytest.mark.pyodide_smoke
def test_ordered_mod_and_fixed_steady_state_audit():
    from puremacro.dsge import inspect_mod, load_mod
    from puremacro.validation import audit_result

    source = """
var y;
varexo e;
model; y = .5*y(-1) + .25*STEADY_STATE(y) + .5 + e; end;
initval; y=2; end;
shocks; var e; stderr .01; end;
stoch_simul(order=1,irf=2) y;
stoch_simul(order=1,irf=4);
"""
    inspection = inspect_mod(source, ordered=True)
    assert inspection.exit_code == 0
    assert json.loads(inspection.to_json())["coverage"] == "complete"
    model = load_mod(source, command_mode="ordered")
    commands = model.execute_mod_commands(ordered=True)
    assert len(commands.invocations) == 2
    assert [len(entry.result.irfs["y_e"]) for entry in commands.invocations] == [3, 5]
    np.testing.assert_allclose(commands.invocations[0].result.irfs["y_e"],
                               .01 * .5 ** np.arange(3), atol=1e-12)
    report = audit_result(model, checks=("solver", "equations", "derivatives"))
    checks = {row.check: row for row in report.records}
    assert report.solver_converged is True
    assert checks["steady_state_residual"].status == "passed"
    assert checks["decision_rule_residual"].status == "passed"
    assert checks["jacobian_current"].status == "passed"
    assert json.loads(report.to_json())["solver_converged"] is True


@pytest.mark.pyodide_smoke
def test_one_asset_budget_and_distribution_audit():
    from puremacro import dp
    from puremacro.validation import audit_result

    model = dp.Model("browser household audit")
    model.state("assets", np.linspace(0, 4, 30))
    model.parameters(beta=.8, r=.01, wage=1.)
    model.local("c=(1+r)*assets+wage-assets(+1)")
    model.reward("log(c)").subject_to("c>0")
    solution = model.solve(method="egm", tol=1e-9)
    report = audit_result(solution)
    checks = {row.check: row for row in report.records}
    assert checks["budget_residual"].status == "passed"
    assert checks["mass_error"].status == "passed"
    assert checks["stationarity_residual"].status == "passed"
    assert np.all(np.isfinite(report.maps["euler_kkt_a"]))
    assert checks["refinement"].status == "not_checked"
    assert "evidence_source" in report.to_dataframe()
