"""Small frozen external references; no MATLAB or archive download in ordinary CI."""
import json

import pytest

from tools.reference_validation.validate_dynare import FIXTURES as DYNARE_FIXTURES, compare_case

CASES = sorted(json.loads((DYNARE_FIXTURES / "manifest.json").read_text(encoding="utf-8"))["cases"])


@pytest.mark.parametrize("case", CASES)
def test_live_dynare_tensors_risk_and_common_innovation_moments(case):
    result = compare_case(case)
    assert result["passed"], result
