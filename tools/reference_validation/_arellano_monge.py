"""Exact discrete policy search using concavity and ordered cash proceeds.

For u(c)=-1/c, u(w+x_high)-u(w+x_low) decreases with wealth w. Sort
actions by descending cash proceeds x, then optimal column indices increase
with w, regardless of continuation values. Divide-and-conquer searches this
monotone matrix in O(n_assets * n_income * log(n_assets)) candidate evaluations.
The solver separately checks the returned solution by exhaustive search.
"""
from __future__ import annotations

import numpy as np


def maximize(assets: np.ndarray, income: np.ndarray, cash: np.ndarray,
             continuation: np.ndarray, order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_b, n_y = cash.shape
    sorted_cash = np.take_along_axis(cash, order, axis=0)
    sorted_value = np.take_along_axis(continuation, order, axis=0)
    if np.min(income[None, :]+assets[0]+cash) <= 0:
        raise ValueError("Monotone reference search requires positive consumption for every candidate")
    value = np.empty_like(cash)
    policy = np.empty((n_b, n_y), dtype=int)
    row_low, row_high = np.array([0]), np.array([n_b-1])
    col_low = np.zeros((1, n_y), dtype=int)
    col_high = np.full((1, n_y), n_b-1, dtype=int)
    while len(row_low):
        middle = (row_low+row_high)//2
        lengths = (col_high-col_low+1).ravel()
        starts = np.r_[0, np.cumsum(lengths)[:-1]]
        owner = np.repeat(np.arange(len(lengths)), lengths)
        column = np.arange(lengths.sum())-starts[owner]+col_low.ravel()[owner]
        shock = owner % n_y
        row = middle[owner//n_y]
        original = order[column, shock]
        objective = -1/(income[shock]+assets[row]+sorted_cash[column, shock])+sorted_value[column, shock]
        best = np.maximum.reduceat(objective, starts)
        # Preserve the archived highest-asset-index tie convention, including
        # duplicate cash/continuation choices in the fully-defaulting region.
        chosen = np.maximum.reduceat(np.where(objective == best[owner], original, -1), starts)
        best_column = np.maximum.reduceat(np.where(original == chosen[owner], column, -1), starts)
        best_column = best_column.reshape(len(middle), n_y)
        value[middle] = best.reshape(len(middle), n_y)
        policy[middle] = chosen.reshape(len(middle), n_y)
        left, right = row_low < middle, middle < row_high
        row_low, row_high = (np.r_[row_low[left], middle[right]+1],
                             np.r_[middle[left]-1, row_high[right]])
        col_low, col_high = (np.concatenate((col_low[left], best_column[right])),
                             np.concatenate((best_column[left], col_high[right])))
    return value, policy
