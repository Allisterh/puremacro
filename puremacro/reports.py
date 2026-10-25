"""Markdown / LaTeX / Typst output formatters for puremacro results.

Aimed at iPad researchers writing in Working Copy / Pages / Quarto:
turn a result dict (from any LP, VAR, GARCH, etc. function) into a
publication-ready table without leaving the notebook.

Also exposes light dataclass wrappers for the most common result
shapes so users get tab-completion in iPad IDEs and a uniform
``.to_markdown()`` method.

Every renderer here is what the result classes' ``to_markdown`` /
``to_latex`` / ``to_typst`` methods call, so the three guarantees below
hold package-wide:

* LaTeX special characters (``\\ & % $ # _ { } ~ ^``) are escaped in
  headers and cells, so a column named ``lo_95%`` compiles.
* Typst markup characters (``* _ # $ @ [ ] < > ~`` and the backslash) are
  escaped inside content cells, so ``0.452***`` stays literal.
* Floats are printed with at most six decimals unless ``digits`` is
  given, so ``0.30000000000000004`` never reaches a manuscript. An
  unnamed ``RangeIndex`` (pandas' default row numbering) is not
  rendered as a column; any other index is.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Escaping and cell formatting
# ---------------------------------------------------------------------------
_LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

# Characters that start markup inside a Typst content block ``[...]``.
_TYPST_SPECIALS = set("\\*_#$@[]<>~`")


def latex_escape(text: str) -> str:
    """Escape every LaTeX special character in ``text``."""
    return "".join(_LATEX_SPECIALS.get(ch, ch) for ch in str(text))


def typst_escape(text: str) -> str:
    """Escape every character that Typst would read as markup in ``text``."""
    return "".join(("\\" + ch) if ch in _TYPST_SPECIALS else ch for ch in str(text))


def _fmt_cell(v: Any, digits: int | None = None) -> str:
    """One cell as text: NaN/None -> '', floats to ``digits`` decimals.

    With ``digits=None`` a float is printed positionally with at most six
    decimals (trailing zeros dropped), and values below 1e-4 in magnitude
    switch to ``%g`` so they are not rendered as ``0``.
    """
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)):
        x = float(v)
        if np.isnan(x):
            return ""
        if not np.isfinite(x):
            return str(x)
        if digits is not None:
            return f"{x:.{digits}f}"
        if x == 0.0:
            return "0"
        if abs(x) < 1e-4:
            return f"{x:.6g}"
        return np.format_float_positional(x, precision=6, unique=True, trim="-")
    if np.ndim(v) == 0:
        try:
            if pd.isna(v):
                return ""
        except (TypeError, ValueError):
            pass
    return str(v)


def _prepare(df: pd.DataFrame, index: bool | None) -> pd.DataFrame:
    """Return ``df`` with the index promoted to a column when it carries information.

    ``index=None`` (the default) keeps the index unless it is an unnamed
    ``RangeIndex`` — pandas' default row numbering, which would otherwise
    appear as a meaningless ``index`` column. ``True`` / ``False`` force
    either behaviour.
    """
    if index is None:
        index = not (isinstance(df.index, pd.RangeIndex) and df.index.name is None)
    return df.reset_index() if index else df


def _cells(df: pd.DataFrame, digits: int | None) -> tuple[list[str], list[list[str]]]:
    cols = [str(c) for c in df.columns]
    rows = [[_fmt_cell(v, digits) for v in row]
            for row in df.itertuples(index=False, name=None)]
    return cols, rows


def _df_to_markdown(df: pd.DataFrame, index: bool | None = None, *,
                    digits: int | None = None) -> str:
    """Self-contained markdown renderer (avoids the ``tabulate`` dependency).

    Supported internal API: result containers across puremacro (for example
    ``puremacro.trade.caliendo_parro`` and ``puremacro.models.trade_policy``)
    render their ``to_markdown``/``to_latex``/``to_typst`` tables with
    :func:`_df_to_markdown`, :func:`_df_to_latex` and :func:`_df_to_typst`;
    their signatures are kept stable for those callers.

    Parameters
    ----------
    index : bool, optional
        Include the index as the first column. Default: yes, unless it is
        an unnamed ``RangeIndex``.
    digits : int, optional
        Fixed number of decimals for float cells; default at most six.
    """
    df = _prepare(df, index)
    cols, rows = _cells(df, digits)
    cols = [c.replace("|", r"\|") for c in cols]
    rows = [[c.replace("|", r"\|") for c in r] for r in rows]
    widths = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c)
              for i, c in enumerate(cols)]

    def fmt(vals: Sequence[str]) -> str:
        return "| " + " | ".join(v.rjust(w) for v, w in zip(vals, widths)) + " |"

    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    out = [fmt(cols), sep]
    for r in rows:
        out.append(fmt(r))
    return "\n".join(out)


def _df_to_latex(df: pd.DataFrame, index: bool | None = None, *,
                 digits: int | None = None) -> str:
    """Self-contained LaTeX ``tabular`` renderer.

    Every header and cell is passed through :func:`latex_escape`, so
    ``%``, ``&``, ``_`` and the other specials compile as literals.
    Parameters as for :func:`_df_to_markdown`.
    """
    df = _prepare(df, index)
    cols, rows = _cells(df, digits)
    align = "l" + "r" * max(len(cols) - 1, 0)
    header = " & ".join(latex_escape(c) for c in cols) + " \\\\\n"
    body = "".join(
        " & ".join(latex_escape(c) for c in r) + " \\\\\n" for r in rows
    )
    return (
        "\\begin{tabular}{" + align + "}\n"
        + header
        + "\\hline\n"
        + body
        + "\\end{tabular}"
    )


def _df_to_typst(df: pd.DataFrame, index: bool | None = None, *,
                 digits: int | None = None) -> str:
    """Self-contained Typst ``#table`` renderer.

    Cell text is passed through :func:`typst_escape`, so significance
    stars and underscores are literal rather than emphasis markup.
    Parameters as for :func:`_df_to_markdown`.
    """
    df = _prepare(df, index)
    cols, rows = _cells(df, digits)
    n_cols = len(cols)
    header_cells = [f"  [* {typst_escape(c)} *]" for c in cols]
    data_cells = [f"  [{typst_escape(v)}]" for r in rows for v in r]
    all_cells = ",\n".join(header_cells + data_cells)
    return f"#table(\n  columns: {n_cols},\n{all_cells},\n)"


def _p_stars(p: float) -> str:
    """Conventional academic significance stars."""
    if np.isnan(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


# ---------------------------------------------------------------------------
# Coefficient tables
# ---------------------------------------------------------------------------
def coef_table(
    beta: np.ndarray,
    se: np.ndarray,
    names: Sequence[str] | None = None,
    *,
    fmt: str = "markdown",
    digits: int = 3,
    include_t: bool = True,
    include_p: bool = True,
    include_ci: bool = True,
    alpha: float = 0.05,
    stars: bool = False,
) -> str:
    """Render a coefficient table.

    Parameters
    ----------
    beta : (k,) — point estimates.
    se   : (k,) — standard errors.
    names : optional regressor names; defaults to ``x0, x1, ...``.
    fmt  : ``"markdown"``, ``"latex"``, ``"typst"``, or ``"plain"`` (whitespace).
    digits : int, default 3
        Decimals shown for every numeric column.
    include_t, include_p, include_ci : bool
        Add the t statistic, the two-sided normal p-value and the
        ``lo_<level>%`` / ``hi_<level>%`` confidence bounds.
    alpha : float, default 0.05
        Size of the confidence interval (``1 - alpha`` coverage).
    stars : bool, default False
        If True, append significance stars (* p<0.10, ** p<0.05, *** p<0.01) to coef.

    Returns
    -------
    String table. In LaTeX and Typst every special character in the
    names and headers is escaped, so the output compiles as-is.
    """
    beta = np.asarray(beta, dtype=float).ravel()
    se = np.asarray(se, dtype=float).ravel()
    k = len(beta)
    if names is None:
        names = [f"x{i}" for i in range(k)]
    z = norm.ppf(1 - alpha / 2)
    rows = []
    for i in range(k):
        tt = beta[i] / se[i] if se[i] > 0 else np.nan
        p_val = 2.0 * norm.sf(abs(tt)) if not np.isnan(tt) else np.nan
        coef_str = f"{beta[i]:.{digits}f}"
        if stars:
            coef_str += _p_stars(p_val)
        row = {"variable": names[i],
               "coef": coef_str if stars else beta[i],
               "se": se[i]}
        if include_t:
            row["t"] = tt
        if include_p:
            row["p"] = p_val
        if include_ci:
            row[f"lo_{int((1-alpha)*100)}%"] = beta[i] - z * se[i]
            row[f"hi_{int((1-alpha)*100)}%"] = beta[i] + z * se[i]
        rows.append(row)
    df = pd.DataFrame(rows).set_index("variable")
    if not stars:
        df = df.round(digits)
    else:
        # round numeric columns only
        num_cols = [c for c in df.columns if c != "coef"]
        df[num_cols] = df[num_cols].round(digits)
    if fmt == "markdown":
        return _df_to_markdown(df, digits=digits)
    if fmt == "latex":
        return _df_to_latex(df, digits=digits)
    if fmt == "typst":
        return _df_to_typst(df, digits=digits)
    return df.to_string()


def irf_to_dataframe(
    point: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    *,
    h_axis: Iterable[int] | None = None,
    var_names: Sequence[str] | None = None,
    digits: int = 3,
) -> pd.DataFrame:
    """Convert IRF matrices to formatted pandas DataFrame."""
    point = np.asarray(point, dtype=float)
    if point.ndim == 3:
        # If user passes (H+1, n_resp, n_shock), pick the first shock by default.
        point = point[:, :, 0]
        if lower is not None:
            lower = np.asarray(lower, dtype=float)[:, :, 0]
        if upper is not None:
            upper = np.asarray(upper, dtype=float)[:, :, 0]
    H_plus, n = point.shape
    if h_axis is None:
        h_axis = range(H_plus)
    if var_names is None:
        var_names = [f"y{i}" for i in range(n)]

    rows = []
    for h_idx, h in enumerate(h_axis):
        row: Dict[str, Union[int, str]] = {"h": h}
        for i, name in enumerate(var_names):
            cell = f"{point[h_idx, i]:+.{digits}f}"
            if lower is not None and upper is not None:
                cell += (f"  [{lower[h_idx, i]:+.{digits}f}, "
                         f"{upper[h_idx, i]:+.{digits}f}]")
            row[name] = cell
        rows.append(row)
    return pd.DataFrame(rows).set_index("h")


def irf_to_markdown(
    point: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    *,
    h_axis: Iterable[int] | None = None,
    var_names: Sequence[str] | None = None,
    digits: int = 3,
) -> str:
    """Render a (H+1) × n IRF matrix as a markdown table.

    If ``lower`` / ``upper`` are given, columns alternate point estimate
    and bracketed band, e.g. ``+0.123 [-0.045, +0.301]``.
    """
    df = irf_to_dataframe(
        point, lower, upper, h_axis=h_axis, var_names=var_names, digits=digits
    )
    return _df_to_markdown(df)


def irf_to_latex(
    point: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    *,
    h_axis: Iterable[int] | None = None,
    var_names: Sequence[str] | None = None,
    digits: int = 3,
) -> str:
    """Render a (H+1) × n IRF matrix as a LaTeX tabular."""
    df = irf_to_dataframe(
        point, lower, upper, h_axis=h_axis, var_names=var_names, digits=digits
    )
    return _df_to_latex(df)


def irf_to_typst(
    point: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    *,
    h_axis: Iterable[int] | None = None,
    var_names: Sequence[str] | None = None,
    digits: int = 3,
) -> str:
    """Render a (H+1) × n IRF matrix as a Typst table."""
    df = irf_to_dataframe(
        point, lower, upper, h_axis=h_axis, var_names=var_names, digits=digits
    )
    return _df_to_typst(df)


def summary_to_dataframe(d: dict) -> pd.DataFrame:
    """Format flat result dictionary into a 2-column DataFrame."""
    rows = []
    for key, val in d.items():
        if isinstance(val, (int, float, np.floating, np.integer)):
            rows.append({"key": key, "value": f"{val:.4g}"})
        elif isinstance(val, str):
            rows.append({"key": key, "value": val})
        elif isinstance(val, np.ndarray):
            rows.append({"key": key, "value": f"ndarray{tuple(val.shape)}"})
        elif isinstance(val, (pd.Series, pd.DataFrame)):
            rows.append({"key": key, "value": f"{type(val).__name__}{tuple(val.shape)}"})
        elif isinstance(val, (list, tuple)):
            rows.append({"key": key, "value": f"{type(val).__name__} (len {len(val)})"})
        else:
            rows.append({"key": key, "value": str(val)[:40]})
    return pd.DataFrame(rows, columns=["key", "value"])


def summary_to_markdown(d: dict, *, title: str = "Result") -> str:
    """Quick markdown summary for a flat result dict.

    Numeric scalars become a key/value table. NumPy arrays and pandas
    objects are summarised by shape.
    """
    df = summary_to_dataframe(d)
    return f"### {title}\n\n" + _df_to_markdown(df, index=False)


def summary_to_latex(d: dict, *, title: str = "Result") -> str:
    """Quick LaTeX tabular summary for a flat result dict.

    The title is emitted as a ``%`` comment line above the table; keys
    and values are escaped, so ``share_%`` renders as ``share\\_\\%``.
    """
    df = summary_to_dataframe(d)
    return f"% {title}\n" + _df_to_latex(df, index=False)


def summary_to_typst(d: dict, *, title: str = "Result") -> str:
    """Quick Typst table summary for a flat result dict."""
    df = summary_to_dataframe(d)
    return f"// {title}\n" + _df_to_typst(df, index=False)


# ---------------------------------------------------------------------------
# Lightweight dataclass wrappers
# ---------------------------------------------------------------------------
@dataclass
class IRFResult:
    """Wraps a triple (point, lower, upper) of IRF arrays (H+1, n_resp, n_shock).

    ``to_markdown`` / ``to_latex`` / ``to_typst`` all render the same
    table (one column per response variable, first shock, bands in
    brackets) and all accept ``digits=`` and ``h_axis=``.
    """
    point: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    var_names: tuple = ()
    method: str = ""
    alpha: float = 0.10

    def to_frame(self, shock_idx: int = 0) -> pd.DataFrame:
        point = np.asarray(self.point, dtype=float)
        if point.ndim == 3:
            p_slice = point[:, :, shock_idx]
            l_slice = self.lower[:, :, shock_idx] if self.lower is not None else None
            u_slice = self.upper[:, :, shock_idx] if self.upper is not None else None
        else:
            p_slice = point
            l_slice = self.lower
            u_slice = self.upper
        H_plus, n = p_slice.shape
        names = list(self.var_names) if self.var_names else [f"y{i}" for i in range(n)]
        data = {"h": np.arange(H_plus)}
        for i, name in enumerate(names):
            data[name] = p_slice[:, i]
            if l_slice is not None and u_slice is not None:
                data[f"{name}_lower"] = l_slice[:, i]
                data[f"{name}_upper"] = u_slice[:, i]
        return pd.DataFrame(data).set_index("h")

    def to_markdown(self, **kwargs) -> str:
        return irf_to_markdown(
            self.point, self.lower, self.upper,
            var_names=self.var_names or None, **kwargs
        )

    def to_latex(self, **kwargs) -> str:
        return irf_to_latex(
            self.point, self.lower, self.upper,
            var_names=self.var_names or None, **kwargs
        )

    def to_typst(self, **kwargs) -> str:
        return irf_to_typst(
            self.point, self.lower, self.upper,
            var_names=self.var_names or None, **kwargs
        )

    def at(self, h: int):
        """Return a (n_resp, n_shock) slice at horizon h with bands."""
        return {
            "point": self.point[h],
            "lower": None if self.lower is None else self.lower[h],
            "upper": None if self.upper is None else self.upper[h],
        }


@dataclass
class LPResult:
    """Wraps the DataFrame returned by ``lp_hac``, ``panel_lp``, etc."""
    df: pd.DataFrame
    method: str = "lp"
    n_obs: int | None = None
    alpha: float = 0.10

    def __getitem__(self, key):
        return self.df[key]

    def __len__(self):
        return len(self.df)

    @property
    def horizons(self):
        return self.df.index if self.df.index.name == "h" else self.df["h"].values

    def to_markdown(self, **kwargs) -> str:
        return _df_to_markdown(self.df, **kwargs)

    def to_latex(self, **kwargs) -> str:
        return _df_to_latex(self.df, **kwargs)

    def to_typst(self, **kwargs) -> str:
        return _df_to_typst(self.df, **kwargs)


df_to_markdown = _df_to_markdown
df_to_latex = _df_to_latex
df_to_typst = _df_to_typst


def _extract_result_dataframe(result_or_df: Any) -> pd.DataFrame:
    if isinstance(result_or_df, pd.DataFrame):
        return result_or_df.copy()
    if isinstance(result_or_df, pd.Series):
        return result_or_df.to_frame()
    for attr in ("to_dataframe", "to_frame", "simulated_path", "df", "summary"):
        val = getattr(result_or_df, attr, None)
        if callable(val):
            try:
                out = val()
                if isinstance(out, pd.DataFrame):
                    return out.copy()
            except Exception:
                continue
        elif isinstance(val, pd.DataFrame):
            return val.copy()
    if isinstance(result_or_df, dict):
        return summary_to_dataframe(result_or_df)
    raise TypeError(f"Cannot extract DataFrame from object of type {type(result_or_df).__name__}")


def result_to_interactive_html(
    result_or_df: Any,
    *,
    title: str = "PureMacro Interactive Explorer",
    subtitle: str | None = None,
    audit_badge: str | None = None,
    digits: int = 4,
) -> str:
    """Render a self-contained, zero-server HTML5 + inline SVG explorer for any PureMacro result or DataFrame."""
    import html as _html
    import json as _json

    df = _extract_result_dataframe(result_or_df)
    if audit_badge is None:
        conv = getattr(result_or_df, "converged", None)
        if conv is True:
            audit_badge = "PASS"
        elif conv is False:
            audit_badge = "FAIL"

    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] == 0:
        # Try coercing numeric columns if formatted as strings
        coerced = df.apply(pd.to_numeric, errors="coerce")
        num_df = coerced.dropna(axis=1, how="all")

    palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf", "#e377c2"]
    n_rows = max(1, len(df))
    svg_w, svg_h = 760, 260
    pad_l, pad_r, pad_t, pad_b = 56, 24, 20, 32
    plot_w = svg_w - pad_l - pad_r
    plot_h = svg_h - pad_t - pad_b

    polylines: list[str] = []
    checkboxes: list[str] = []
    if num_df.shape[1] > 0 and len(num_df) > 0:
        vals = num_df.to_numpy(dtype=float)
        finite_vals = vals[np.isfinite(vals)]
        y_min = float(np.min(finite_vals)) if finite_vals.size > 0 else -1.0
        y_max = float(np.max(finite_vals)) if finite_vals.size > 0 else 1.0
        if abs(y_max - y_min) < 1e-12:
            y_min -= 1.0
            y_max += 1.0

        def _y_px(v: float) -> float:
            return pad_t + plot_h * (1.0 - (v - y_min) / (y_max - y_min))

        def _x_px(i: int) -> float:
            if len(num_df) <= 1:
                return pad_l + 0.5 * plot_w
            return pad_l + plot_w * (i / (len(num_df) - 1))

        if y_min <= 0.0 <= y_max:
            y0 = _y_px(0.0)
            polylines.append(
                f'<line x1="{pad_l}" y1="{y0:.1f}" x2="{svg_w - pad_r}" y2="{y0:.1f}" '
                f'stroke="#888" stroke-dasharray="4,3" stroke-width="1" />'
            )

        for col_idx, col_name in enumerate(num_df.columns):
            color = palette[col_idx % len(palette)]
            safe_col = _html.escape(str(col_name))
            pts = []
            for r_i, v in enumerate(num_df[col_name].to_numpy(dtype=float)):
                if np.isfinite(v):
                    pts.append(f"{_x_px(r_i):.1f},{_y_px(float(v)):.1f}")
            pts_str = " ".join(pts)
            polylines.append(
                f'<polyline class="pm-series" data-series="{safe_col}" '
                f'fill="none" stroke="{color}" stroke-width="2.2" points="{pts_str}">'
                f"<title>{safe_col}</title></polyline>"
            )
            checkboxes.append(
                f'<label style="margin-right:12px;font-size:13px;cursor:pointer;">'
                f'<input type="checkbox" class="pm-toggle" data-series="{safe_col}" checked '
                f'onchange="pmToggleSeries(this)" /> '
                f'<span style="color:{color};font-weight:600;">■</span> {safe_col}</label>'
            )

    badge_html = ""
    if audit_badge:
        b_up = audit_badge.upper()
        b_col = "#1b7f37" if "PASS" in b_up else ("#b35900" if "WARN" in b_up else "#b91c1c")
        badge_html = (
            f'<span class="pm-audit-badge" style="background:{b_col};color:#fff;'
            f'padding:3px 10px;border-radius:12px;font-size:12px;font-weight:700;">'
            f"{_html.escape(audit_badge)}</span>"
        )

    sub_html = f'<p style="margin:4px 0 0 0;color:#555;font-size:13px;">{_html.escape(subtitle)}</p>' if subtitle else ""

    df_prep = _prepare(df, None)
    cols, rows = _cells(df_prep, digits)
    th_html = "".join(
        f'<th data-col="{_html.escape(c)}" style="padding:6px 10px;border-bottom:2px solid #ccc;text-align:right;">'
        f"{_html.escape(c)}</th>"
        for c in cols
    )
    tr_list: list[str] = []
    for r_idx, r in enumerate(rows):
        tds = "".join(
            f'<td data-col="{_html.escape(cols[c_i])}" style="padding:5px 10px;border-bottom:1px solid #eee;text-align:right;">'
            f"{_html.escape(cell)}</td>"
            for c_i, cell in enumerate(r)
        )
        tr_list.append(f'<tr class="pm-row" data-row="{r_idx}">{tds}</tr>')

    payload_json = _json.dumps({"columns": cols, "row_count": len(rows)})

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{_html.escape(title)}</title>
</head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;margin:16px;color:#111;">
<div class="pm-explorer" style="max-width:860px;border:1px solid #ddd;border-radius:8px;padding:16px;background:#fff;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
    <div>
      <h3 style="margin:0;font-size:18px;">{_html.escape(title)}</h3>
      {sub_html}
    </div>
    {badge_html}
  </div>
  <div class="pm-controls" style="margin-bottom:10px;padding:8px 10px;background:#f8f9fa;border-radius:6px;">
    <div style="margin-bottom:6px;">{"".join(checkboxes)}</div>
    <div style="font-size:12px;color:#444;">
      Horizon / Row Cutoff: <input type="range" id="pm-horizon-slider" min="1" max="{n_rows}" value="{n_rows}" oninput="pmFilterRows(this.value)" />
      <span id="pm-horizon-label">{n_rows} / {n_rows}</span>
    </div>
  </div>
  <svg class="pm-chart" viewBox="0 0 {svg_w} {svg_h}" style="width:100%;height:auto;border:1px solid #eee;border-radius:4px;background:#fcfcfc;margin-bottom:12px;">
    {"".join(polylines)}
  </svg>
  <div style="overflow-x:auto;max-height:320px;">
    <table class="pm-table" style="width:100%;border-collapse:collapse;font-size:12.5px;">
      <thead><tr>{th_html}</tr></thead>
      <tbody>{"".join(tr_list)}</tbody>
    </table>
  </div>
  <script type="application/json" id="pm-metadata">{payload_json}</script>
  <script>
  function pmToggleSeries(cb) {{
    var s = cb.getAttribute('data-series');
    var show = cb.checked;
    document.querySelectorAll('.pm-series[data-series="' + s + '"]').forEach(function(el) {{
      el.style.display = show ? '' : 'none';
    }});
    document.querySelectorAll('[data-col="' + s + '"]').forEach(function(el) {{
      el.style.display = show ? '' : 'none';
    }});
  }}
  function pmFilterRows(val) {{
    var maxRow = parseInt(val, 10);
    var lbl = document.getElementById('pm-horizon-label');
    if (lbl) lbl.textContent = maxRow + ' / {n_rows}';
    document.querySelectorAll('.pm-row').forEach(function(tr) {{
      var idx = parseInt(tr.getAttribute('data-row'), 10);
      tr.style.display = (idx < maxRow) ? '' : 'none';
    }});
  }}
  </script>
</div>
</body>
</html>"""


__all__ = [
    "coef_table",
    "df_to_latex",
    "df_to_markdown",
    "df_to_typst",
    "irf_to_dataframe",
    "irf_to_latex",
    "irf_to_markdown",
    "irf_to_typst",
    "latex_escape",
    "typst_escape",
    "summary_to_dataframe",
    "summary_to_latex",
    "summary_to_markdown",
    "summary_to_typst",
    "result_to_interactive_html",
    "IRFResult",
    "LPResult",
]
