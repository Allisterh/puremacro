"""Climate and environmental macroeconomics for puremacro.

Contains:
- Forward simulator of the DICE-2016R model (Nordhaus 2017); nothing is optimised.
- Social cost of carbon (SCC) on a simulated path under a user-supplied carbon-tax path.
- Multi-reservoir carbon cycle and climate temperature anomaly dynamics.
"""
from puremacro.climate.dice import DICEResult, simulate_dice_model
from puremacro.climate.spatial_iam import (
    SpatialClimateIAM,
    SpatialClimateIAMResult,
    solve_spatial_climate_iam,
)

__all__ = [
    "DICEResult",
    "simulate_dice_model",
    "SpatialClimateIAM",
    "SpatialClimateIAMResult",
    "solve_spatial_climate_iam",
]
