r"""Climatology of xarray objects, with a choice of method.

Each time series is split into a climatology $\bar{y}$ and an anomaly $y'$,

$$
y(\mathbf{x}, t) = \bar{y}(\mathbf{x}, t) + y'(\mathbf{x}, t),
$$

where $t$ is time and $\mathbf{x}$ stands for every other dimension, such as
latitude, longitude, or depth. A method fits $\bar{y}$ to $y$ along $t$ at every
$\mathbf{x}$, and $y'$ is the residual. :py:func:`climatology` returns $\bar{y}$
and :py:func:`coefficients` returns the parameters of the fit.

Each method is a module with its own ``climatology`` and ``coefficients``.
Adding a method means adding a module and a branch in ``_method``.
"""

from __future__ import annotations

from types import ModuleType

import xarray as xr

from . import harmonics

def _method(name: str) -> ModuleType:
    if name == "harmonic":
        return harmonics
    raise ValueError(f"Unknown method {name!r}. Available methods: 'harmonic'.")


def climatology(
    obj: xr.DataArray | xr.Dataset, *, method: str = "harmonic", **kwargs
) -> xr.DataArray | xr.Dataset:
    r"""Return the climatology $\bar{y}$ of ``obj`` on its own time axis.

    Parameters
    ----------
    obj : DataArray or Dataset
        Data with a time dimension. For a Dataset, every variable is used.
    method : {"harmonic"}, default: "harmonic"
        ``"harmonic"`` fits a polynomial in time plus harmonics of a period.
        See :py:func:`xclimatology.harmonics.climatology`.
    **kwargs
        Arguments of the method, such as ``n_harmonics`` or ``period``.

    Returns
    -------
    DataArray or Dataset
        $\bar{y}$, with the dimensions, name, and attributes of ``obj``.

    See Also
    --------
    coefficients

    Examples
    --------
    >>> anomaly = ds - ds.climatology()
    >>> ds.climatology(n_harmonics=3, detrend_degree=0)
    """
    return _method(method).climatology(obj, **kwargs)


def coefficients(da: xr.DataArray, *, method: str = "harmonic", **kwargs) -> xr.Dataset:
    r"""Return the parameters of the climatology $\bar{y}$ of ``da``.

    Parameters
    ----------
    da : DataArray
        Data with a time dimension.
    method : {"harmonic"}, default: "harmonic"
        ``"harmonic"``. See :py:func:`xclimatology.harmonics.coefficients`.
    **kwargs
        Arguments of the method.

    Returns
    -------
    Dataset
        Parameters of the fit, named by the method.

    See Also
    --------
    climatology
    """
    return _method(method).coefficients(da, **kwargs)


@xr.register_dataarray_accessor("climatology")
@xr.register_dataset_accessor("climatology")
class _ClimatologyAccessor:
    """``obj.climatology(...)`` and ``obj.climatology.coefficients(...)``.

    These call :py:func:`climatology` and :py:func:`coefficients` with ``obj`` as
    the data.
    """

    def __init__(self, obj: xr.DataArray | xr.Dataset) -> None:
        self._obj = obj

    def __call__(self, **kwargs) -> xr.DataArray | xr.Dataset:
        return climatology(self._obj, **kwargs)

    def coefficients(self, **kwargs) -> xr.Dataset:
        return coefficients(self._obj, **kwargs)
