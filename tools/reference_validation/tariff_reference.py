"""Independent Leontief equilibrium for the bundled trade calibration.

Only calibration arrays are shared with PureMacro. No native solver, residual,
accounting, Jacobian or welfare function is used. Uniform capital shares imply
equal wage/rental changes within each country. Conditional on country wages,
prices, final quantities and rebated fiscal income solve linear systems; only
relative country wages require a nonlinear root.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve
from scipy.optimize import root


class LeontiefReference:
    def __init__(self, calib, *, importer='USA', exporter='CHN', tariff=0.,
                 domestic_taxes=True, foreign_balances=True):
        self.nc, self.ns, self.nf = calib.nc, calib.ns, calib.n_final_demand
        nc, ns, nf = self.nc, self.ns, self.nf
        n = nc * ns
        self.countries = list(calib.country_codes)
        self.a = np.asarray(calib.a).reshape(n, n, order='F').copy()
        self.af = np.asarray(calib.afd).reshape(n, nc * nf, order='F').copy()
        self.theta = np.asarray(calib.theta).ravel(order='F').copy()
        alpha = np.asarray(calib.alpha).ravel(order='F')
        self.va = alpha ** -alpha * (1 - alpha) ** -(1 - alpha) / np.asarray(calib.beta).ravel(order='F')
        l, k = np.asarray(calib.l_endow).ravel(), np.asarray(calib.k_endow).ravel()
        if not np.allclose(alpha.reshape(nc, ns), (k / (l + k))[:, None], atol=1e-13, rtol=0):
            raise ValueError('independent reduction requires uniform factor shares within each country')
        self.endowment = l + k
        self.balances = np.asarray(calib.invforT).ravel().copy() if foreign_balances else np.zeros(nc)
        self.country_goods = np.repeat(np.eye(nc), ns, axis=1)
        self.country_final = np.repeat(np.eye(nc), nf, axis=1)
        self.tax = np.asarray(calib.tax).ravel(order='F').copy() if domestic_taxes else np.zeros(n)
        data = np.asarray(calib.data_calibra)
        basic = data[:n, n:].sum(axis=0)
        spending = basic + data[n, n:]
        self.fd_tax = np.divide(data[n, n:], spending, out=np.zeros_like(spending), where=spending != 0)
        if not domestic_taxes:
            self.fd_tax[:] = 0
        self.active = basic > 0
        self.ti, self.tf = np.ones((n, n)), np.ones((n, nc * nf))
        i, e = self.countries.index(importer), self.countries.index(exporter)
        self.ti[e * ns:(e + 1) * ns, i * ns:(i + 1) * ns] = 1 + tariff
        self.tf[e * ns:(e + 1) * ns, i * nf:(i + 1) * nf] = 1 + tariff
        self.price_map = solve(np.diag(1 - self.tax) - (self.a * self.ti).T,
                                self.va[:, None] * self.country_goods.T)
        self.quantity_map = solve(np.eye(n) - self.a, self.af)
        # Tax-free factor-resource accounting prices, not consumer prices or
        # welfare weights. Their total final value equals fixed factor supply.
        self.resource_prices = solve(np.eye(n) - self.a.T, self.va)

    def evaluate(self, log_wages):
        w = np.exp(log_wages)
        p = self.price_map @ w
        factor_income = self.endowment * w
        b = self.balances * factor_income.sum() / self.endowment.sum()
        price_fd = np.sum(self.af * self.tf * p[:, None], axis=0) / (1 - self.fd_tax)
        price_fd = np.where(self.active, price_fd, 1.)
        q_income = self.theta[:, None] * self.country_final.T / price_fd[:, None]
        q_balance = np.zeros(self.nc * self.nf)
        q_balance[np.arange(self.nc) * self.nf + 1] = -b / price_fd[np.arange(self.nc) * self.nf + 1]
        revenue_per_output = self.tax * p + np.sum((self.ti - 1) * p[:, None] * self.a, axis=0)
        revenue_per_final = self.fd_tax * price_fd + np.sum((self.tf - 1) * p[:, None] * self.af, axis=0)
        revenue_map = ((self.country_goods * revenue_per_output[None, :]) @ self.quantity_map
                        + self.country_final * revenue_per_final[None, :])
        income = solve(np.eye(self.nc) - revenue_map @ q_income,
                        factor_income + revenue_map @ q_balance)
        c = q_income @ income + q_balance
        y = self.quantity_map @ c
        factor_gap = self.country_goods @ (self.va * y) - self.endowment
        residual = factor_gap / self.endowment
        residual[0] = np.log(p[0])
        return dict(w=w, p=p, y=y, P=price_fd.reshape(self.nc, self.nf),
                    c=c.reshape(self.nc, self.nf), income=income,
                    transfers=income - factor_income, balances=b,
                    factor_gap=factor_gap, residual=residual,
                    government=revenue_map @ c,
                    physical_goods_gap=y - self.a @ y - self.af @ c)

    def solve(self):
        fit = root(lambda x: self.evaluate(x)['residual'], np.zeros(self.nc), tol=1e-11)
        result = self.evaluate(fit.x)
        if np.max(np.abs(result['residual'])) > 1e-10:
            raise RuntimeError(f'independent equilibrium did not converge: {fit.message}')
        result['evaluations'] = int(fit.nfev)
        return result


def category_changes(before, after):
    """Exact single-basket EV for C; baseline-price quantity values for I/Cx."""
    return before['P'] * (after['c'] - before['c'])


def resource_changes(reference, before, after):
    """Final-use quantity changes valued at factor-resource accounting prices."""
    prices = (reference.resource_prices @ reference.af).reshape(reference.nc, reference.nf)
    return prices * (after['c'] - before['c'])


def audit_state(calib, state, *, sigma, tariff=0., importer='USA', exporter='CHN'):
    """Reconstruct flows and budgets from returned states, without native helpers.

    This evaluator uses two-dimensional source-by-buyer matrices. The CES
    composite covers all country-sector inputs, not origins within each sector.
    Every residual is scaled by the relevant physical or financial quantity.
    """
    nc, ns, nf = calib.nc, calib.ns, calib.n_final_demand
    n = nc * ns
    a = np.asarray(calib.a).reshape(n, n, order='F')
    af = np.asarray(calib.afd).reshape(n, nc * nf, order='F')
    p, y = np.ravel(state.p_sol, order='F'), np.ravel(state.y_sol, order='F')
    w, r, transfers = np.ravel(state.w_sol), np.ravel(state.r_sol), np.ravel(state.T_sol)
    quantity = np.ravel(state.c_sol, order='F')
    final_prices = np.ravel(state.Pfd_final, order='F')
    codes = list(calib.country_codes)
    destination, source = codes.index(importer), codes.index(exporter)
    ti, tf = np.ones_like(a), np.ones_like(af)
    ti[source*ns:(source+1)*ns, destination*ns:(destination+1)*ns] += tariff
    tf[source*ns:(source+1)*ns, destination*nf:(destination+1)*nf] += tariff
    purchaser = p[:, None] * ti
    amount = a.sum(axis=0)
    weights = np.divide(a, amount, out=np.zeros_like(a), where=amount > 0)
    if sigma == 0:
        index = np.sum(weights * purchaser, axis=0)
        intermediate = a * y[None, :]
    else:
        if sigma == 1:
            index = np.exp(np.sum(weights * np.log(purchaser), axis=0))
        else:
            inner = np.sum(weights * purchaser**(1-sigma), axis=0)
            index = np.where(amount > 0, inner, 1.)**(1/(1-sigma))
        intermediate = a * y[None, :] * (index[None, :] / purchaser)**sigma
    final = af * quantity[None, :]
    alpha, beta = np.ravel(calib.alpha, order='F'), np.ravel(calib.beta, order='F')
    wage, rental = np.repeat(w, ns), np.repeat(r, ns)
    unit_va = (rental / alpha)**alpha * (wage / (1-alpha))**(1-alpha) / beta
    production_tax = np.ravel(calib.tax, order='F')
    data = np.asarray(calib.data_calibra)
    spending0 = data[:n, n:].sum(axis=0) + data[n, n:]
    fd_tax = np.divide(data[n, n:], spending0, out=np.zeros(nc*nf), where=spending0 != 0)
    predicted_prices = (af * (p[:, None] * tf)).sum(axis=0) / (1-fd_tax)
    predicted_prices = np.where(spending0 > 0, predicted_prices, 1.)
    endow_l, endow_k = np.ravel(calib.l_endow), np.ravel(calib.k_endow)
    factor_income = w * endow_l + r * endow_k
    balances = np.ravel(calib.invforT) * factor_income.sum() / (endow_l + endow_k).sum()
    income = factor_income + transfers
    allocated = np.ravel(calib.theta, order='F') * np.repeat(income, nf)
    allocated[np.arange(nc)*nf + 1] -= balances
    duties_i = ((ti-1)*p[:, None]*intermediate).sum(axis=0).reshape(nc, ns).sum(axis=1)
    duties_f = ((tf-1)*p[:, None]*final).sum(axis=0).reshape(nc, nf).sum(axis=1)
    government = ((production_tax*p*y).reshape(nc, ns).sum(axis=1)
                  + (fd_tax*final_prices*quantity).reshape(nc, nf).sum(axis=1)
                  + duties_i + duties_f)
    bilateral = ((p[:, None]*intermediate).reshape(nc, ns, nc, ns).sum(axis=(1, 3))
                 + (p[:, None]*final).reshape(nc, ns, nc, nf).sum(axis=(1, 3)))
    net_exports = bilateral.sum(axis=1) - bilateral.sum(axis=0)
    def error(gap, scale):
        return float(np.max(np.abs(gap) / np.maximum(1., np.abs(scale))))
    errors = {
        'goods': error(y-intermediate.sum(axis=1)-final.sum(axis=1), y),
        'zero_profit': error((1-production_tax)*p-unit_va-amount*index, p),
        'labor': error(((1-alpha)*unit_va*y/wage).reshape(nc, ns).sum(axis=1)-endow_l, endow_l),
        'capital': error((alpha*unit_va*y/rental).reshape(nc, ns).sum(axis=1)-endow_k, endow_k),
        'final_prices': error(final_prices-predicted_prices, final_prices),
        'household_budget': error(final_prices*quantity-allocated, np.repeat(income, nf)),
        'government_budget': error(transfers-government, income),
        'external_balance': error(net_exports-balances, factor_income),
    }
    return dict(errors=errors, maximum_normalized_error=max(errors.values()),
                income=income, P=final_prices.reshape(nc, nf), c=quantity.reshape(nc, nf))
