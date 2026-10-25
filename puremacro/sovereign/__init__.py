"""Quantitative Sovereign Default, Endogenous Maturity & Sudden-Stop Suite.

Implements:
1. Arellano (2008) & Chatterjee-Eyigungor (2012) endogenous default with
   short-term and long-term maturing coupon bonds, extreme-value taste shocks,
   and DualArray bond-price schedule derivatives dq/db'.
2. Mendoza (2010) sudden-stop collateral constraints with Fisherian debt-deflation
   fire-sale amplification and precautionary deleveraging.
3. Sovereign spread term-structure decomposition and debt restructuring
   counterfactuals (haircuts and maturity extensions).
"""
from puremacro.sovereign._results import (
    RestructuringResult,
    SovereignDefaultResult,
    SuddenStopResult,
)
from puremacro.sovereign.arellano import solve_sovereign_default
from puremacro.sovereign.mendoza import solve_sudden_stop
from puremacro.sovereign.restructuring import (
    decompose_spread_term_structure,
    evaluate_debt_restructuring,
)

__all__ = [
    "SovereignDefaultResult",
    "SuddenStopResult",
    "RestructuringResult",
    "solve_sovereign_default",
    "solve_sudden_stop",
    "decompose_spread_term_structure",
    "evaluate_debt_restructuring",
]
