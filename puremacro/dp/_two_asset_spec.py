"""Structural recognition of the supported two-asset resource equation.

Finite probes cannot prove a budget identity. This small symbolic polynomial
normalizer inlines locals and compares the declaration to supported forms;
registered arbitrary functions are opaque and cannot pass as the budget.
It needs no symbolic-algebra dependency.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import product

from puremacro.dp._expr import ModelSpecError, parse, substitute, symbols
from puremacro.dsge._ast import BinOp, Call, Const, Param, UnaryOp, Var

_ALIASES = {
    "r_b": ("r_b", "rb", "r", "r_b_ss", "r_ss", "r_liquid"),
    "r_a": ("r_a", "ra", "r_a_ss", "r_k", "r_illiquid"),
    "w": ("w", "w_ss", "z", "y"),
    "chi_0": ("chi_0",), "chi_1": ("chi_1",), "a_bar": ("a_bar", "b_bar"),
}


def _key(node):
    """Canonical polynomial with non-polynomial expressions as exact atoms."""
    def atom(key):
        return {((key, 1),): Fraction(1)}

    def tidy(poly):
        return {m: c for m, c in poly.items() if c}

    def add(a, b):
        out = dict(a)
        for m, c in b.items():
            out[m] = out.get(m, Fraction(0)) + c
        return tidy(out)

    def mul(a, b):
        out = {}
        for (m, c), (n, d) in product(a.items(), b.items()):
            powers = dict(m)
            for term, exponent in n:
                powers[term] = powers.get(term, 0) + exponent
            key = tuple(sorted(((term, exp) for term, exp in powers.items() if exp), key=repr))
            out[key] = out.get(key, Fraction(0)) + c * d
        if len(out) > 256:
            raise ModelSpecError("two-asset EGM budget is too complex for structural recognition")
        return tidy(out)

    def key(poly):
        return tuple(sorted(((m, (c.numerator, c.denominator)) for m, c in poly.items()), key=repr))

    def reciprocal(n):
        if isinstance(n, BinOp) and n.op == "*":
            return mul(reciprocal(n.left), reciprocal(n.right))
        if isinstance(n, BinOp) and n.op == "/":
            return mul(reciprocal(n.left), visit(n.right))
        poly = visit(n)
        if len(poly) == 1:
            (monomial, coefficient), = poly.items()
            return {tuple((term, -exp) for term, exp in monomial): 1 / coefficient}
        return {((("sum", key(poly)), -1),): Fraction(1)}

    def visit(n):
        if isinstance(n, Const):
            return {(): Fraction(str(n.value))} if n.value else {}
        if isinstance(n, Param):
            return atom(("parameter", n.name))
        if isinstance(n, Var):
            return atom(("variable", n.name, n.lead))
        if isinstance(n, UnaryOp):
            return mul({(): Fraction(-1 if n.op == "-" else 1)}, visit(n.expr))
        if isinstance(n, Call):
            return atom(("function", n.func, tuple(key(visit(a)) for a in n.args)))
        if isinstance(n, BinOp):
            a, b = visit(n.left), visit(n.right)
            if n.op == "+":
                return add(a, b)
            if n.op == "-":
                return add(a, mul({(): Fraction(-1)}, b))
            if n.op == "*":
                return mul(a, b)
            if n.op == "/":
                return mul(a, reciprocal(n.right))
            if n.op == "^" and isinstance(n.right, Const) and n.right.value in (0, 1, 2, 3, 4):
                out = {(): Fraction(1)}
                for _ in range(int(n.right.value)):
                    out = mul(out, a)
                return out
            return atom(("operation", n.op, key(a), key(b)))
        raise ModelSpecError("two-asset EGM budget contains an unsupported expression")
    return key(visit(node))


def recognize(comp, cname):
    """Return proven asset roles and parameter bindings or reject the budget.

    Commutative reordering, parentheses, expanded products, renamed assets,
    parameter aliases and intermediate locals are accepted structurally.
    """
    expanded = {}
    for name, node in comp.locals:
        expanded[name] = substitute(node, expanded)
    budget = expanded[cname]
    def calls(node):
        if isinstance(node, Call):
            return {node.func} | set().union(*(calls(a) for a in node.args))
        if isinstance(node, BinOp):
            return calls(node.left) | calls(node.right)
        if isinstance(node, UnaryOp):
            return calls(node.expr)
        return set()
    functions = calls(budget)
    if functions - {"abs"} or functions.intersection(comp.model._functions):
        raise ModelSpecError("two-asset EGM structurally recognized budgets cannot use registered arbitrary functions")
    if isinstance(comp.reward, Call) and comp.reward.func in comp.model._functions:
        raise ModelSpecError("two-asset EGM requires built-in crra/log utility, not an overridden function")
    used = {name for name, _ in symbols(budget)}
    bindings = {}
    replacements = {}
    for role, candidates in _ALIASES.items():
        matches = [name for name in candidates if name in used and name in comp.param_names]
        if len(matches) != 1:
            raise ModelSpecError(f"two-asset EGM binding for {role} requires exactly one budget parameter "
                                 f"from {candidates}; found {matches}")
        bindings[role] = matches[0]
        replacements[matches[0]] = Param(role)
    canonical_budget = substitute(budget, replacements)
    actual = _key(canonical_budget)
    for illiquid, liquid in (comp.state_names, comp.state_names[::-1]):
        income = f"w*{comp.shock_names[0]}" if comp.shock_names else "w"
        d = f"({illiquid}(+1)-(1+r_a)*{illiquid})"
        den = f"({illiquid}+a_bar)"
        costs = [f"chi_0/(1+chi_1)*abs({d})*(abs({d})/{den})^chi_1",
                 f"chi_0/(1+chi_1)*abs({d})^(1+chi_1)/{den}^chi_1",
                 f"chi_0/(1+chi_1)*(abs({d})/{den})^(1+chi_1)*{den}"]
        for cost in costs:
            expected = parse(f"(1+r_b)*{liquid}+(1+r_a)*{illiquid}+{income}"
                             f"-{illiquid}(+1)-{liquid}(+1)-({cost})",
                             comp.state_names + comp.shock_names)
            if actual == _key(expected):
                _check_domain(canonical_budget, illiquid, comp)
                return illiquid, liquid, bindings
    raise ModelSpecError("two-asset EGM requires the structurally recognized canonical resource equation "
                         "and convex adjustment cost; arbitrary budget functions and changed asset timing are unsupported")


def _check_domain(node, illiquid, comp):
    """Do not cancel a singular denominator or undefined real power away."""
    positive = {_key(parse(text, comp.state_names + comp.shock_names)) for text in
                (f"{illiquid}+a_bar", "1+chi_1", "chi_0", "a_bar", "chi_1")}

    def positive_denominator(n):
        if isinstance(n, Const):
            return n.value > 0
        if _key(n) in positive:
            return True
        if isinstance(n, BinOp) and n.op in ("*", "/"):
            return positive_denominator(n.left) and positive_denominator(n.right)
        if isinstance(n, BinOp) and n.op == "^":
            return positive_denominator(n.left)
        return False

    def nonnegative(n):
        if isinstance(n, Const):
            return n.value >= 0
        if isinstance(n, Call) and n.func == "abs":
            return True
        if _key(n) in positive:
            return True
        if isinstance(n, BinOp) and n.op in ("*", "/"):
            return nonnegative(n.left) and positive_denominator(n.right)
        return False

    if isinstance(node, BinOp):
        if node.op == "/" and not positive_denominator(node.right):
            raise ModelSpecError("two-asset EGM unsupported denominator in canonical budget; cancellation cannot prove its domain")
        if node.op == "^":
            integer_power = (isinstance(node.right, Const) and node.right.value >= 0
                             and float(node.right.value).is_integer())
            if not integer_power and not nonnegative(node.left):
                raise ModelSpecError("two-asset EGM unsupported real-power domain in canonical budget")
        _check_domain(node.left, illiquid, comp)
        _check_domain(node.right, illiquid, comp)
    elif isinstance(node, UnaryOp):
        _check_domain(node.expr, illiquid, comp)
    elif isinstance(node, Call):
        for arg in node.args:
            _check_domain(arg, illiquid, comp)
