"""Three offline workflow templates with recorded assumptions and inputs.

Run ``python -m puremacro.examples.study_workflows --output DIRECTORY``.
Mexico uses frozen OECD/FRED observations and a declared recursive VAR.
Household transfers are an illustrative partial-equilibrium model experiment.
Tariffs use a conserving aggregation of the bundled 2019 OECD ICIO table.
None is a replication of a published causal finding.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.resources import files
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from puremacro import dp
from puremacro.study import StudySpec, VariableSpec, run_study


HOUSEHOLD = """
parameters beta gamma r transfer n_a;
beta=.96; gamma=2; r=.01; transfer=0; n_a=100;
state a = 40*linspace(0,1,n_a)^2;
exogenous z = rouwenhorst(rho=.9,sigma=.15,n=5);
model(name='Household transfer experiment');
 c=(1+r)*a+exp(z)+transfer-a(+1);
 reward=crra(c,gamma);
end;
constraints; c>0; end;
aggregates; C=c; A=a; end;
solve(method=egm,tol=1e-10);
"""


def mexico_data() -> pd.DataFrame:
    """Frozen monthly observations on an explicit 2000–2023 common window."""
    root = files("puremacro.examples").joinpath("_study_data")
    manifest = json.loads(root.joinpath("manifest.json").read_text())
    series = []
    for code, name in (("CPALTT01MXM659N", "inflation_yoy"),
                       ("IR3TIB01MXM156N", "rate_3m")):
        payload = root.joinpath(code + ".csv").read_bytes()
        if hashlib.sha256(payload).hexdigest() != manifest["files"][code + ".csv"]["sha256"]:
            raise ValueError("frozen data checksum mismatch: " + code)
        frame = pd.read_csv(io.BytesIO(payload), parse_dates=["observation_date"])
        s = frame.set_index("observation_date")[code].astype(float)
        s.index = s.index.to_period("M")
        series.append(s.rename(name))
    data = pd.concat(series, axis=1).loc["2000-01":"2023-12"].copy()
    data.attrs = {"source": manifest, "evidence_kind": "observational_recursive_VAR",
                  "country": "MEX", "sample_policy": "2000-01 through 2023-12; no missing-row deletion",
                  "identification_note": "Inflation precedes the interbank rate. Inflation does not react within the month to the recursively identified rate innovation. This is an assumption, not an externally identified policy surprise."}
    return data


def run_mexico(output: str | Path | None = None, *, n_boot: int = 199):
    """25 bp annualized rate innovation; monthly level responses in pp."""
    spec = StudySpec(
        name="Mexico recursive rate innovation", method="var_cholesky",
        variables=(VariableSpec("inflation_yoy", "percentage points"),
                   VariableSpec("rate_3m", "percentage points per annum")),
        responses=("inflation_yoy", "rate_3m"), shock="rate_3m", frequency="M",
        identification="recursive", ordering=("inflation_yoy", "rate_3m"),
        lags=6, horizon=18, shock_size=.25, n_boot=n_boot, seed=734,
    )
    result = run_study(mexico_data(), spec)
    if output is not None:
        result.write_report(output)
    return result


def run_household(output: str | Path | None = None, *, amount: float = .05,
                  horizon: int = 40, n_a: int = 100):
    """Unanticipated date-0 lump-sum transfer at fixed prices, externally funded."""
    from puremacro.models.hank_sequence_space import dp_nonlinear_transition
    from puremacro.study import ScenarioSpec, record_scenario
    from puremacro.validation import audit_result

    if not np.isfinite(amount) or amount <= 0 or isinstance(amount, bool):
        raise ValueError("amount must be finite and positive")
    if type(horizon) is not int or horizon < 2 or type(n_a) is not int or n_a < 10:
        raise ValueError("horizon >= 2 and n_a >= 10 must be integers")
    sol = dp.solve_mod(HOUSEHOLD, params={"n_a": n_a})
    shock = np.zeros(horizon)
    shock[0] = amount
    zero = dp_nonlinear_transition(sol, {"transfer": np.zeros(horizon)}, outputs=["C", "A"])
    shocked = dp_nonlinear_transition(sol, {"transfer": shock}, outputs=["C", "A"])
    # Compare paths computed with identical terminal values and starting mass.
    changes = {k: shocked[k] - zero[k] for k in zero}
    table = pd.DataFrame([{"response": k, "horizon": t, "estimate": float(v)}
                          for k, seq in changes.items() for t, v in enumerate(seq)])
    states = pd.DataFrame({"a": np.repeat(sol.grids["a"], len(sol.grids["z"])),
                           "z": np.tile(sol.grids["z"], len(sol.grids["a"])),
                           "mass": sol.distribution.ravel(), "consumption": sol.policy("c").ravel(),
                           "saving": sol.policy("a(+1)").ravel()})
    audit = audit_result(sol)
    audit_frame = pd.DataFrame([{"check": f"{i}: {row['check']}", "status": row["status"],
        "detail": json.dumps(row, sort_keys=True, allow_nan=False)}
        for i, row in enumerate(audit.to_dict()["records"])])
    result = record_scenario(
        table, ScenarioSpec(name="Household transfer", model="DP EGM household",
            variables=(VariableSpec("C", "consumption units"), VariableSpec("A", "asset units"),
                       VariableSpec("transfer", "consumption units per household")),
            responses=("C", "A"), shock="transfer", frequency="Q",
            identification="calibrated_model_experiment", response_convention="level_deviation",
            horizon=horizon - 1, shock_size=amount),
        inputs=states, model_source=HOUSEHOLD,
        parameters={"beta": .96, "gamma": 2., "r": .01, "n_a": n_a, "amount": amount},
        audits=audit_frame,
        provenance={"evidence_kind": "illustrative_calibrated_model", "audit": audit.to_dict(),
            "financing": "external transfer; no government budget or tax response modeled",
            "prices": "fixed quarterly interest rate; fixed earnings process",
            "timing": "initial stationary mass; announced at date 0; terminal steady-state value; A is entering assets",
            "normalization": "shocked minus zero-shock path under identical finite horizon",
            "limitations": ["Not fitted to household survey data", "No general-equilibrium price response",
                            "Grid/horizon accuracy requires explicit refinement; audit failures are retained"]},
        evidence={"shock_path": pd.DataFrame({"transfer": shock}),
                  "zero_path": pd.DataFrame(zero), "shocked_path": pd.DataFrame(shocked),
                  "shock_transition": pd.DataFrame(sol.P)},
    )
    if output is not None:
        result.write_report(output)
    return result


def tariff_table():
    """Conserve all flows; USA/MEX/rest × resources/manufacturing/other."""
    from puremacro.trade.data import load_icio_data
    bundle = load_icio_data(return_structured=True)
    countries = ("USA", "MEX", "REST")
    sectors = ("RESOURCES", "MANUFACTURING", "OTHER_INDUSTRIES")
    ci = np.array([countries.index(c) if c in countries else 2 for c in bundle.country_codes])
    si = np.array([0 if s in ("A", "B") else 1 if s == "C" else 2 for s in bundle.sector_codes])
    node = (ci[:, None]*3 + si[None, :]).ravel()
    n, k = len(node), len(bundle.fd_codes)
    P = np.zeros((9, n)); P[node, np.arange(n)] = 1
    fd = (ci[:, None]*k + np.arange(k)[None, :]).ravel()
    Q = np.zeros((3*k, len(fd))); Q[fd, np.arange(len(fd))] = 1
    raw = bundle.matrix
    small = np.block([[P @ raw[:n, :n] @ P.T, P @ raw[:n, n:] @ Q.T],
                      [raw[n:, :n] @ P.T, raw[n:, n:] @ Q.T]])
    if not np.isclose(small.sum(), raw.sum(), rtol=1e-13):
        raise RuntimeError("aggregation did not conserve world flows")
    metadata = {key: value for key, value in bundle.metadata.items() if key != "file_path"}
    metadata["country_mapping"] = dict(zip(bundle.country_codes, (countries[i] for i in ci)))
    metadata["sector_mapping"] = dict(zip(bundle.sector_codes, (sectors[i] for i in si)))
    metadata["original_matrix_sha256"] = hashlib.sha256(raw.astype("<f8").tobytes()).hexdigest()
    return small, countries, sectors, metadata


def run_tariff(output: str | Path | None = None, *, rate: float = .05):
    """Illustrative US tariff on Mexican manufacturing; static equilibrium."""
    from puremacro.study import ScenarioSpec, record_scenario
    from puremacro.trade import calibrate_trade_model, TariffScenario, run_scenario_batch

    small, countries, sectors, metadata = tariff_table()
    calibration = calibrate_trade_model(small, country_codes=countries, sector_codes=sectors)
    scenario = TariffScenario("USA", {("MEX", "MANUFACTURING"): rate}, name="tariff")
    if rate == 0:
        raise ValueError("rate must be positive for this experiment")
    batch = run_scenario_batch(calibration, [scenario], compute_gk=False, tol=1e-5)
    baseline, alternative = batch.scenarios["baseline"], batch.scenarios["tariff"]
    if not baseline.converged or not alternative.converged:
        raise RuntimeError("tariff workflow requires both equilibria to converge")
    # Base prices equal one: sum output quantities at baseline prices, not GDP.
    responses = tuple(f"output_{c}" for c in countries)
    changes = 100*(alternative.y_sol.sum(axis=1).ravel()/baseline.y_sol.sum(axis=1).ravel() - 1)
    table = pd.DataFrame({"response": responses, "horizon": 0, "estimate": changes})
    rows = []
    accounting_evidence = {}
    for name, res in batch.scenarios.items():
        rows.append({"check": name + " equations",
                     "status": "passed" if res.max_residual <= 1e-5 else "failed",
                     "detail": f"Maximum reported residual {res.max_residual:.12g} in mixed native equation units; tolerance 1e-5. Named accounting checks and physical-equation residuals are archived as evidence."})
        ledger = res.metadata["account_residuals"]
        accounting_evidence[name + "_accounting_checks"] = pd.DataFrame({
            "check": list(ledger), "max_absolute_residual": list(ledger.values()),
            "unit": ["mixed native equation units" if key == "physical_equations"
                     else "native account units" for key in ledger],
        })
        # Preserve the full independent residual vector, including the goods
        # equation replaced by the numeraire in the solver's own equations.
        families = (["goods"]*9 + ["prices"]*9 + ["labor"]*3 + ["capital"]*3
                    + ["foreign_balance"]*3 + ["government_budget"]*3)
        accounting_evidence[name + "_physical_residuals"] = pd.DataFrame({
            "equation_family": families,
            "residual": np.asarray(res.metadata["physical_residuals"]).ravel(),
        })
    spec = ScenarioSpec(name="Manufacturing tariff", model="Static production network",
        variables=tuple(VariableSpec(v, "percent of baseline gross output") for v in responses)
                  + (VariableSpec("tariff", "ad valorem fraction"),),
        responses=responses, shock="tariff", frequency="static",
        identification="calibrated_model_experiment", response_convention="static_percent_change",
        horizon=0, shock_size=rate)
    result = record_scenario(table, spec, inputs=pd.DataFrame(small),
        model_source="puremacro.trade: Cobb-Douglas value added, fixed input coefficients, lump-sum duty rebate, fixed foreign saving in numeraire units; USA resources producer price = 1.",
        parameters={"tariff": rate, "tol": 1e-5}, audits=pd.DataFrame(rows),
        provenance={"evidence_kind": "OECD_calibrated_counterfactual", "data": metadata,
            "scenario": {"importer": "USA", "exporter": "MEX", "sector": "MANUFACTURING"},
            "max_residual": {k: float(v.max_residual) for k, v in batch.scenarios.items()},
            "residual_units": "mixed native equation units: quantities, relative prices, factor amounts and nominal budgets; not a single monetary unit",
            "limitations": ["Stylised static technology and fiscal closure", "No parameter uncertainty or causal policy prediction",
                            "Gross output is not GDP or a welfare measure", "OECD value added is split 2/3 labor and 1/3 capital by assumption"]},
        evidence={"baseline_output": pd.DataFrame(baseline.y_sol[0], index=sectors, columns=countries),
                  "tariff_output": pd.DataFrame(alternative.y_sol[0], index=sectors, columns=countries),
                  "baseline_prices": pd.DataFrame(baseline.p_sol[0], index=sectors, columns=countries),
                  "tariff_prices": pd.DataFrame(alternative.p_sol[0], index=sectors, columns=countries),
                  **accounting_evidence},
    )
    if output is not None:
        result.write_report(output)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workflow", choices=("all", "mexico", "household", "tariff"), default="all")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, run in (("mexico", run_mexico), ("household", run_household), ("tariff", run_tariff)):
        if args.workflow in ("all", name):
            result = run(args.output / name)
            print(name, result.to_frame().shape)


if __name__ == "__main__":
    main()
