# xclimatology

[![Documentation](https://readthedocs.org/projects/xclimatology/badge/?version=latest)](https://xclimatology.readthedocs.io/en/latest/?badge=latest)
[![Code of Conduct](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](CODE_OF_CONDUCT.md)

xclimatology computes climatologies, such as the seasonal or daily cycle, for xarray
DataArrays and Datasets at every grid point at once. One call, `data.climatology()`,
returns the climatology on the same time axis, so `data - data.climatology()` gives
the anomaly.

## Quick start

Install from GitHub with pip:

```bash
pip install git+https://github.com/anthony-meza/xclimatology
```

As an example, the harmonic method fits the annual and semiannual cycles of sea
surface temperature together with a linear trend:

```python
import xarray as xr
import xclimatology  # registers the .climatology accessor

sst = xr.tutorial.load_dataset("ersstv5")["sst"]
model = dict(method="harmonic", n_harmonics=2, detrend_degree=1)

clim = sst.climatology(**model)
anomaly = sst - clim
coefs = sst.climatology.coefficients(**model)
```

`clim` is the climatology on the original time axis. `coefs` holds the fitted
coefficients, including the amplitude and phase of each harmonic. See the
[documentation](https://xclimatology.readthedocs.io) for worked examples.

## Initializing this repository

Steps to set up a similar package. Replace `mypackage` with its name. Based on
pyOpenSci's
[Create a Python package](https://www.pyopensci.org/python-package-guide/tutorials/create-python-package.html).

**1. Generate from the template**

```bash
copier copy gh:pyopensci/pyos-package-template .
git init && git add . && git commit -m "Initial commit"
pixi init --format pyproject   # optional
```

Add dependencies by editing `[dependency-groups]` in `pyproject.toml`. pip, Hatch, and
Read the Docs read it, while `pixi add --feature` writes elsewhere.

**2. Docs dependencies** (`pyproject.toml`)

```toml
[dependency-groups]
docs = [
    # ...template entries...
    "myst-nb",
    "sphinx-math-dollar",
]
```

To build the docs in a conda environment, and record it in `docs/environment.yml`:

```bash
mamba create -n mypackage_docs python=3.12 pip -c conda-forge -y
mamba activate mypackage_docs
python -m pip install -e . --group docs
conda env export -n mypackage_docs > docs/environment.yml
```

**3. Sphinx settings** (`docs/conf.py`)

```python
extensions = [
    "myst_nb",             # replaces "myst_parser"
    # ...template entries...
    "sphinx_math_dollar",  # $...$ math in docstrings
]
myst_enable_extensions = [
    # ...template entries...
    "dollarmath",
    "amsmath",
]
exclude_patterns = ["_build", "jupyter_execute", "Thumbs.db", ".DS_Store"]
```

Also add `docs/jupyter_execute/` to `.gitignore`. Docstrings with math must be raw
strings (`r"""`), and subscripts that start with a backslash need braces
(`t_{\mathrm{mid}}`).

**4. Pages** (`docs/index.md` and `docs/examples/index.md`)

```markdown
:::{toctree}
:maxdepth: 2
:hidden:

Home <self>
Examples <examples/index>
API reference <autoapi/index>
:::
```

```markdown
# Examples

:::{toctree}
:maxdepth: 1

basic_usage
:::
```

Then add `docs/examples/basic_usage.ipynb`. In notebooks, an import that only registers
an accessor needs `# noqa: F401`, or `ruff check --fix` deletes it.

**5. Read the Docs** (`.readthedocs.yaml`)

```yaml
build:
  os: ubuntu-lts-latest
  tools:
    python: "3.12"
  commands:
    - pip install --upgrade pip   # --group needs pip >= 25.1
    - pip install -e . --group docs
    - sphinx-build -b html docs $READTHEDOCS_OUTPUT/html
```

**6. Lint rules** (`pyproject.toml`)

```toml
[tool.ruff.lint]
select = ["E4", "E7", "E9", "F", "I"]

[tool.hatch.envs.style.scripts]
code = "ruff check --fix {args}"
format = "ruff format {args}"
check = ["format", "code"]
```

Delete `lines-after-imports = 1` from `[tool.ruff.lint.isort]`, since `ruff format`
undoes it. To keep the template's docstring check, add
`docstrings = "pydoclint src/ tests/"` to the scripts and `arg-type-hints-in-signature = true`
under `[tool.pydoclint]`.

## Copyright

- Copyright © 2026 Anthony Meza.
- Free software distributed under the [MIT License](./LICENSE).
