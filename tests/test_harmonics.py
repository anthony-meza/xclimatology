import numpy as np
import pandas as pd
import xarray as xr

import xclimatology  # noqa: F401 (registers the .climatology accessor)

def test_recovers_known_seasonal_cycle_and_trend():
    time = xr.DataArray(pd.date_range("2000-01-15", periods=120, freq="MS"), dims="time")
    t = time.dt.decimal_year
    y = 10 + 0.5 * (t - 2005) + 3 * np.cos(2 * np.pi * t - 1.0)

    coefs = y.climatology.coefficients(n_harmonics=1, detrend_degree=1)
    annual = coefs.sel(harmonic=1)

    np.testing.assert_allclose(annual["amplitude"], 3)
    np.testing.assert_allclose(annual["phase"], 1.0)
    np.testing.assert_allclose(coefs["polynomial"].sel(degree=1), 0.5)
    np.testing.assert_allclose(y - y.climatology(n_harmonics=1), 0, atol=1e-10)
