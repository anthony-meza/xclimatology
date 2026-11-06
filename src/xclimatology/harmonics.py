r"""Harmonic climatology: a polynomial in time plus harmonics of a period.

The climatology of each time series is the weighted least-squares fit

$$
\begin{aligned}
\bar{y}(\mathbf{x}, t) &= \sum_{p=0}^{P} c_p(\mathbf{x}) \, (t - t_{\mathrm{mid}})^p
+ \sum_{k=1}^{K} \left[ a_k(\mathbf{x}) \cos(k\omega t)
+ b_k(\mathbf{x}) \sin(k\omega t) \right], \\
\omega &= \frac{2\pi}{T},
\end{aligned}
$$

with time $t$ in ``period_units``, $t_{\mathrm{mid}}$ the middle of the record,
and period $T$. The polynomial has degree $P$, and its constant term $c_0$ is the
intercept at $t_{\mathrm{mid}}$. Harmonic $k$ is the sinusoid of period $T/k$, so
$T = 1$ year and $K = 2$ give the annual and semiannual cycles.

At each $\mathbf{x}$, $\bar{y} = X\beta$. Each column of the design matrix $X$
is one term of the fit, and $\beta$ holds the $m = P + 1 + 2K$ coefficients. $X$
depends only on $t$, so every time series shares it. $\beta$ minimizes
$y'^\mathsf{T} W y'$, where the diagonal matrix $W$ holds the weights and is zero
at missing samples.

The polynomial columns of $X$ are powers of $\tau = (t - t_{\mathrm{mid}})/L$,
which is $t$ rescaled to $[-1, 1]$ by the half-length $L$ of the record. Powers
of $\tau$ stay of order 1, while powers of $t$ do not ($t \approx 2000$ in
years), so $X^\mathsf{T} W X$ stays well conditioned. :py:func:`coefficients`
divides the coefficient of $\tau^p$ by $L^p$ to give $c_p$.
"""

from __future__ import annotations

import cftime
import numpy as np
import pandas as pd
import xarray as xr
import xarray.ufuncs as xu
from xarray_einstats import linalg

# For the m × m matrix XᵀWX. output_dtypes stops dask probing inv with a singular
# dummy matrix.
_MATRIX_OPTIONS = {
    "dims": ["term", "term_"],
    "dask": "parallelized",
    "dask_gufunc_kwargs": {"allow_rechunk": True},
    "output_dtypes": [float],
}


def _time_in(time: xr.DataArray, units: str) -> xr.DataArray:
    """Convert a datetime coordinate to t in ``units``.

    ``"year"`` gives the calendar decimal year, which is an integer on January 1
    in every calendar, so annual phases are measured from January 1. Other units
    give elapsed time since 1970-01-01 00:00, so diurnal phases are measured from
    midnight.
    """
    if units == "year":
        return time.dt.decimal_year
    if time.dtype == object:
        epoch = cftime.datetime(1970, 1, 1, calendar=time.dt.calendar)
    else:
        epoch = np.datetime64("1970-01-01")
    return (time - epoch) / pd.Timedelta(1, unit=units)


def climatology(
    obj: xr.DataArray | xr.Dataset,
    n_harmonics: int = 2,
    *,
    period: float = 1.0,
    period_units: str = "year",
    detrend_degree: int = 1,
    time_dim: str = "time",
    weights: xr.DataArray | None = None,
) -> xr.DataArray | xr.Dataset:
    r"""Return the harmonic climatology $\bar{y}$ of ``obj`` on its time axis.

    ``obj - climatology(obj)`` is the anomaly $y'$.

    Parameters
    ----------
    obj : DataArray or Dataset
        Data with dimension ``time_dim`` and any other dimensions. For a Dataset,
        every variable is fit.
    n_harmonics : int, default: 2
        Number of harmonics $K$. With ``0``, only the polynomial is fit.
    period : float, default: 1.0
        Period $T$, in ``period_units``.
    period_units : str, default: "year"
        Unit of $t$ and $T$, either ``"year"`` or a :py:class:`pandas.Timedelta`
        unit such as ``"day"``, ``"hour"``, or ``"minute"``.
    detrend_degree : int, default: 1
        Degree $P$ of the polynomial. ``0`` fits only the intercept $c_0$, and
        ``1`` adds a linear trend. The polynomial is fit jointly with the
        harmonics so that a trend does not leak into the low harmonics.
    time_dim : str, default: "time"
        Name of the time dimension. Its coordinate must hold datetimes, so decode
        numeric CF times with :py:func:`xarray.decode_cf` first.
    weights : DataArray, optional
        Positive weights along ``time_dim``, the diagonal of $W$.

    Returns
    -------
    DataArray or Dataset
        $\bar{y}$, with the dimensions, name, and attributes of ``obj``.

    See Also
    --------
    coefficients

    Examples
    --------
    >>> clim = ds.climatology()
    >>> anomaly = ds - clim
    """

    def ybar_of(da):
        beta, X, _ = _fit(
            da, n_harmonics, period, period_units, detrend_degree, time_dim, weights
        )
        ybar = xr.dot(beta, X, dim="term")  # ȳ = Xβ
        return ybar.transpose(*da.dims).rename(da.name).assign_attrs(da.attrs)

    return obj.map(ybar_of) if isinstance(obj, xr.Dataset) else ybar_of(obj)


def coefficients(
    da: xr.DataArray,
    n_harmonics: int = 2,
    *,
    period: float = 1.0,
    period_units: str = "year",
    detrend_degree: int = 1,
    time_dim: str = "time",
    weights: xr.DataArray | None = None,
) -> xr.Dataset:
    r"""Fit the harmonic climatology $\bar{y}$ and return its coefficients.

    Parameters
    ----------
    da : DataArray
        Data with dimension ``time_dim`` and any other dimensions.
    n_harmonics : int, default: 2
        Number of harmonics $K$. With ``0``, only the polynomial is fit.
    period : float, default: 1.0
        Period $T$, in ``period_units``.
    period_units : str, default: "year"
        Unit of $t$ and $T$, either ``"year"`` or a :py:class:`pandas.Timedelta`
        unit such as ``"day"``, ``"hour"``, or ``"minute"``.
    detrend_degree : int, default: 1
        Degree $P$ of the polynomial. ``0`` fits only the intercept $c_0$, and
        ``1`` adds a linear trend. The polynomial is fit jointly with the
        harmonics so that a trend does not leak into the low harmonics.
    time_dim : str, default: "time"
        Name of the time dimension. Its coordinate must hold datetimes, so decode
        numeric CF times with :py:func:`xarray.decode_cf` first.
    weights : DataArray, optional
        Positive weights along ``time_dim``, the diagonal of $W$.

    Returns
    -------
    Dataset
        ``polynomial`` holds $c_p$ along ``degree``, with $t_{\mathrm{mid}}$ in
        ``attrs["t_mid"]``. Along ``harmonic``, ``cos`` and ``sin`` hold $a_k$
        and $b_k$, ``amplitude`` holds $\sqrt{a_k^2 + b_k^2}$, and ``phase``
        holds $\mathrm{atan2}(b_k, a_k)$ in radians. A time series is NaN if its
        samples cannot determine every coefficient, for example with fewer valid
        samples than coefficients, or with samples only on January 1, where
        $\cos\omega t = 1$ and $a_1$ cannot be told apart from $c_0$.

    See Also
    --------
    climatology
    """
    if not isinstance(da, xr.DataArray):
        raise TypeError("coefficients takes a DataArray; select one variable.")
    beta, _, model = _fit(
        da, n_harmonics, period, period_units, detrend_degree, time_dim, weights
    )

    def select_terms(name, orders, dim):
        """β for the terms ``{name}{order}``, indexed by order along ``dim``."""
        terms = beta.sel(term=[f"{name}{order}" for order in orders])
        return terms.assign_coords(term=list(orders)).rename(term=dim)

    units = {"units": da.attrs["units"]} if "units" in da.attrs else {}
    # τᵖ = (t - t_mid)ᵖ / Lᵖ, so cₚ is the τᵖ coefficient over Lᵖ.
    c = select_terms("poly", range(detrend_degree + 1), "degree")
    c = (c / model["L"] ** c["degree"]).assign_attrs(
        long_name=f"coefficient of (t - t_mid)**degree, t in {period_units}"
    )
    a = select_terms("cos", range(1, n_harmonics + 1), "harmonic")
    b = select_terms("sin", range(1, n_harmonics + 1), "harmonic")
    variables = {
        "polynomial": c,
        "cos": a.assign_attrs(units),
        "sin": b.assign_attrs(units),
        "amplitude": xu.hypot(a, b).assign_attrs(units),
        "phase": xu.arctan2(b, a).assign_attrs(units="radian"),
    }
    keep = ("period", "period_units", "n_harmonics", "detrend_degree", "t_mid")
    return xr.Dataset(variables, attrs={key: model[key] for key in keep})


def _fit(da, n_harmonics, period, period_units, detrend_degree, time_dim, weights):
    """Return β (term), NaN where a time series cannot be fit, X, and the model."""
    # type(), since bool subclasses int.
    if type(detrend_degree) is not int or detrend_degree < 0:
        raise ValueError("detrend_degree must be a non-negative integer.")
    t = _time_in(da[time_dim], period_units)
    mean_interval = float(t.diff(time_dim).mean())
    if 2 * mean_interval * n_harmonics > period:
        raise ValueError(
            "n_harmonics exceeds the Nyquist limit: "
            "the period T/K must span two sampling intervals."
        )

    model = {
        "period": period,
        "period_units": period_units,
        "n_harmonics": n_harmonics,
        "detrend_degree": detrend_degree,
        "t_mid": float(t.max() + t.min()) / 2,
        "L": float(t.max() - t.min()) / 2 or 1.0,
    }
    X = _design_matrix(t, model)
    # W = weights × valid-sample mask.
    w = 1 if weights is None else weights
    valid, y = da.notnull(), da.fillna(0)
    m = X.sizes["term"]

    # β minimizes (y - Xβ)ᵀW(y - Xβ), so (XᵀWX) β = XᵀWy at every x. y is zero where
    # missing, so XᵀWy needs no mask. With weights along time only, y and the mask
    # are the only arrays as large as the data.
    XtWX = xr.dot(valid, w * X * X.rename(term="term_"), dim=time_dim)
    XtWy = xr.dot(y, w * X, dim=time_dim)
    # Full rank m means the samples determine every coefficient (NumPy's tolerance,
    # σ_max · m · eps).
    is_full_rank = linalg.matrix_rank(XtWX, **_MATRIX_OPTIONS) == m

    # The identity stands in where rank is deficient, so the batched inverse never
    # fails. (XᵀWX)⁻¹ is symmetric, so either of its dims can be contracted.
    identity = xr.DataArray(np.eye(m), dims=("term", "term_"))
    XtWX_inv = linalg.inv(XtWX.where(is_full_rank, identity), **_MATRIX_OPTIONS)
    beta = xr.dot(XtWX_inv, XtWy, dim="term").rename(term_="term")
    return beta.where(is_full_rank), X, model


def _design_matrix(t: xr.DataArray, model: dict) -> xr.DataArray:
    """Return the design matrix X with dimensions (term, time)."""
    omega_t = 2 * np.pi * t / model["period"]
    # τ in [-1, 1] keeps cond(XᵀWX) under ~100 for P ≤ 3. Powers of t ≈ 2000 years
    # would push it past 10¹⁰.
    tau = (t - model["t_mid"]) / model["L"]
    columns = {f"poly{p}": tau**p for p in range(model["detrend_degree"] + 1)}
    for k in range(1, model["n_harmonics"] + 1):
        columns[f"cos{k}"] = xu.cos(k * omega_t)
        columns[f"sin{k}"] = xu.sin(k * omega_t)
    return xr.Dataset(columns).to_dataarray("term")
