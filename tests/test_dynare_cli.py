"""Tests for puremacro-dynare CLI runner."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from puremacro.dsge.cli import create_parser, run_cli


def test_cli_parser_creation():
    parser = create_parser()
    assert parser.prog == "puremacro-dynare"
    args = parser.parse_args(["mymodel.mod", "--order", "2", "--irf", "20", "--fevd", "--plot"])
    assert args.model == "mymodel.mod"
    assert args.order == 2
    assert args.irf == 20
    assert args.fevd is True
    assert args.plot is True


def test_cli_file_not_found():
    code = run_cli(["non_existent_model_12345.mod", "--quiet"])
    assert code == 1


def test_cli_steady_only():
    ref_mod = Path("puremacro/dsge/_references/sw07_pfeifer.mod")
    assert ref_mod.is_file()
    code = run_cli([str(ref_mod), "--steady-only", "--quiet"])
    assert code == 0


def test_cli_full_solve_and_export(tmp_path: Path):
    ref_mod = Path("puremacro/dsge/_references/sw07_pfeifer.mod")
    out_dir = tmp_path / "results"
    
    code = run_cli([
        str(ref_mod),
        "--order", "1",
        "--irf", "10",
        "--fevd",
        "--outdir", str(out_dir),
        "--format", "all",
        "--quiet",
    ])
    assert code == 0
    
    # Check exported artifacts
    assert (out_dir / "model_moments.md").is_file()
    assert (out_dir / "model_moments.tex").is_file()
    assert (out_dir / "model_moments.typ").is_file()
    assert (out_dir / "irfs.csv").is_file()
    assert (out_dir / "fevd.md").is_file()
    assert (out_dir / "fevd.tex").is_file()
    assert (out_dir / "fevd.typ").is_file()
    assert (out_dir / "fevd.csv").is_file()
    
    # Check irfs.csv content
    df_irfs = pd.read_csv(out_dir / "irfs.csv", index_col=0)
    assert df_irfs.shape[0] == 11  # horizon 0 to 10


def test_cli_with_plots(tmp_path: Path):
    ref_mod = Path("puremacro/dsge/_references/sw07_pfeifer.mod")
    out_dir = tmp_path / "plots_out"
    
    code = run_cli([
        str(ref_mod),
        "--order", "1",
        "--irf", "5",
        "--fevd",
        "--plot",
        "--outdir", str(out_dir),
        "--quiet",
    ])
    assert code == 0
    assert (out_dir / "irfs.png").is_file()
    assert (out_dir / "fevd.png").is_file()


def test_cli_with_shock_decomposition(tmp_path: Path):
    from puremacro.dsge import load_mod
    ref_mod = Path("puremacro/dsge/_references/sw07_pfeifer.mod")
    m = load_mod(ref_mod)
    
    # Generate a small sample dataset matching observables
    np.random.seed(42)
    T_sim = 15
    data_dict = {v: np.random.randn(T_sim) * 0.01 for v in m.variables[:5]}
    data_df = pd.DataFrame(data_dict)
    csv_path = tmp_path / "data.csv"
    data_df.to_csv(csv_path)
    
    out_dir = tmp_path / "decomp_out"
    code = run_cli([
        str(ref_mod),
        "--shock-decomp", str(csv_path),
        "--outdir", str(out_dir),
        "--quiet",
    ])
    assert code == 0
    assert (out_dir / "shock_decomposition.md").is_file()
    assert (out_dir / "shock_decomposition.tex").is_file()
    assert (out_dir / "shock_decomposition.typ").is_file()


def test_cli_ordered_execute_finite_deterministic_without_stable_qz_solution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A finite deterministic .mod program with no stable local QZ solution executes via ordered CLI."""
    import json
    from puremacro.dsge import BlanchardKahnError, load_mod, read_mod
    import puremacro.dsge.klein as klein_mod

    # Purely forward model p_t = 2 * p_{t+1} + e_t: eigenvalue is 0.5 on forward jump
    # (or 2.0 backward), which violates Blanchard-Kahn if solved locally via load_mod
    # when declared predetermined, or has no stable local solution when 0 stable eigenvalues
    # match 1 predetermined state: x_t = 2.0 * x_{t-1} + e_t.
    mod_file = tmp_path / "explosive_deterministic.mod"
    mod_file.write_text(
        """
        var x;
        varexo e;
        parameters rho;
        rho = 2.0;
        model;
            x = rho * x(-1) + e;
        end;
        initval;
            x = 0;
            e = 0;
        end;
        shocks;
            var e; periods 1; values 1.0;
        end;
        simul(periods=4);
        rho = 1.5;
        simul(periods=4);
        """,
        encoding="utf-8",
    )

    # Confirm that load_mod fails with BlanchardKahnError (no stable local QZ solution)
    with pytest.raises(BlanchardKahnError):
        load_mod(mod_file)

    # Guard klein_solve to prove ordered CLI never runs QZ
    def forbid_qz(*args, **kwargs):
        raise AssertionError("klein_solve QZ must not be called for a deterministic ordered CLI run")

    monkeypatch.setattr(klein_mod, "klein_solve", forbid_qz)

    out_dir_sub = tmp_path / "ordered_sub"
    code_sub = run_cli(["execute", str(mod_file), "--outdir", str(out_dir_sub), "--quiet"])
    assert code_sub == 0

    out_dir_flag = tmp_path / "ordered_flag"
    code_flag = run_cli([str(mod_file), "--ordered", "--outdir", str(out_dir_flag), "--quiet"])
    assert code_flag == 0

    py_results = read_mod(mod_file).execute_mod_commands()
    for out_dir in (out_dir_sub, out_dir_flag):
        summary_json = json.loads((out_dir / "command_summary.json").read_text(encoding="utf-8"))
        assert summary_json["status"] == "success"
        assert summary_json["commands_executed"] == ["simul", "simul"]
        assert len(summary_json["invocations"]) == 2
        assert (out_dir / "command_summary.md").is_file()
        assert (out_dir / "command_summary.tex").is_file()
        assert (out_dir / "command_summary.typ").is_file()

        df_0 = pd.read_csv(out_dir / "invocation_0_simul.csv", index_col=0)
        df_1 = pd.read_csv(out_dir / "invocation_1_simul.csv", index_col=0)
        df_last = pd.read_csv(out_dir / "perfect_foresight_path.csv", index_col=0)

        np.testing.assert_allclose(df_0["x"], [1.0, 2.0, 4.0, 8.0], atol=1e-10)
        np.testing.assert_allclose(df_1["x"], [1.0, 1.5, 2.25, 3.375], atol=1e-10)
        pd.testing.assert_frame_equal(df_0, py_results.invocations[0].result.path, check_index_type=False)
        pd.testing.assert_frame_equal(df_1, py_results.invocations[1].result.path, check_index_type=False)
        pd.testing.assert_frame_equal(df_last, py_results.perfect_foresight.path, check_index_type=False)


def test_cli_ordered_execute_exit_codes(tmp_path: Path):
    """Ordered CLI returns meaningful exit codes for missing, incompatible, incomplete, and failed runs."""
    import json

    # 1. Missing file -> 1
    assert run_cli(["execute", str(tmp_path / "missing.mod"), "--quiet"]) == 1

    # 2. Incompatible program (perfect_foresight_solver without setup) -> 1
    incompat_mod = tmp_path / "incompat.mod"
    incompat_mod.write_text(
        "var x; varexo e; model; x = 0.5*x(-1) + e; end; initval; x=0; e=0; end; perfect_foresight_solver;",
        encoding="utf-8",
    )
    assert run_cli(["execute", str(incompat_mod), "--outdir", str(tmp_path / "out1"), "--quiet"]) == 1

    # 3. Incomplete/malformed syntax -> 2
    malformed_mod = tmp_path / "malformed.mod"
    malformed_mod.write_text(
        "var x; model; x = 0.5*x(-1); /* unclosed comment",
        encoding="utf-8",
    )
    assert run_cli(["execute", str(malformed_mod), "--outdir", str(tmp_path / "out2"), "--quiet"]) == 2

    # 4. Numerical failure (no real steady state) -> 3 and records failed status in command_summary.json
    fail_mod = tmp_path / "fail_steady.mod"
    fail_mod.write_text(
        "var x; model; x^2 + 1 = 0; end; initval; x=0; end; steady; simul(periods=3);",
        encoding="utf-8",
    )
    out_fail = tmp_path / "out3"
    assert run_cli(["execute", str(fail_mod), "--outdir", str(out_fail), "--quiet"]) == 3
    summary = json.loads((out_fail / "command_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["invocations"][0]["status"] == "failed"
