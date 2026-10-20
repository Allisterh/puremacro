# puremacro browser playground (JupyterLite)

Explore `puremacro`'s numerical core in your browser via JupyterLite + Pyodide
(WebAssembly), without a local Python install or a hosted compute session.
Loading the site and its packages requires network access. Browser execution
is best-effort; a local workstation install is the supported target.

## Build

```bash
# Run from the puremacro project root.
pip install -e ".[playground,notebooks]"
bash playground/build_playground.sh
python -m http.server -d playground/dist 8000   # then open http://localhost:8000/lab
```

`build_playground.sh` builds the `puremacro` wheel, pins it via
`PipliteAddon.piplite_urls` (so `%pip install puremacro` resolves to the wheel
served with the site), copies the showcase notebooks (`notebooks/NN_*.ipynb`)
and companion lessons (`notebooks/course/*.ipynb`) plus `_nbstyle.py` / `_tutor.py`
into `content/`, injects a `%pip install puremacro` bootstrap cell into **every**
`NN…`-prefixed notebook it copied (`%pip install puremacro pyarrow` in the ones
whose code reads parquet), and runs `jupyter lite build` → `dist/`.
It also builds the documentation into `dist/docs/`.

These companion lessons are distinct from the MAV notebooks in
[`curso/notebooks/`](../curso/notebooks/README.md), whose course edition retains
`puremacro>=4.8.1,<5`. The Pages workflow separately copies the static
HTML/JavaScript course labs into `dist/curso/`; it does not run them against
the wheel built for the playground.

## Deploy

`.github/workflows/pages.yml` runs `build_playground.sh` on every push to `main`
and publishes `playground/dist/` to GitHub Pages:
<https://jalonso1979.github.io/puremacro/lab/index.html>.

The browser install works only because every base dependency of `puremacro`
ships with the Pyodide distribution: the playground disables PyPI fallback
(`content_static/jupyter-lite.json`), so a base dependency that Pyodide lacks
makes the first cell of every notebook fail. Through 3.2.1 that is exactly what
happened (the wheel required `openpyxl` and `pyarrow`).
`tests/test_pyodide_compat.py::test_runtime_deps_ship_with_pyodide` now guards
it, and the opt-in Pyodide gate (`tools/pyodide/runner.js`) performs the same
dependency-resolving install. After a deploy, run the first cells of one
showcase notebook and of one parquet lesson (08 or 24) to confirm.

The build copies `notebooks/course/data/` into `content/course/data/`, matching
the companion lessons' `DATA = _nb/"course"/"data"` layout. Showcase notebooks
include synthetic and empirical examples; some need additional repository
files, optional packages or services. Copying a notebook into the playground
does not establish browser compatibility. Start with `00_start_here`, and use
the [local notebook setup](../docs/notebooks.md) when an example's data or
dependencies are unavailable in the browser. Live data access also depends on
the provider's authentication and browser cross-origin policy.
