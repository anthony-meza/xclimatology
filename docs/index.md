# xclimatology

:::{toctree}
:maxdepth: 2
:hidden:
:caption: Contents:

Home <self>
Examples <examples/index>
API reference <autoapi/index>
:::

xclimatology fits climatologies to xarray objects. Each time series is split
into a climatology $\bar{y}$ and an anomaly $y'$,

$$
y(\mathbf{x}, t) = \bar{y}(\mathbf{x}, t) + y'(\mathbf{x}, t),
$$

where $t$ is time and $\mathbf{x}$ stands for every other dimension, such as
latitude, longitude, or depth. The harmonic method fits $\bar{y}$ as a polynomial in time
plus harmonics of a period, at every $\mathbf{x}$ at once. See
{py:mod}`xclimatology.harmonics` for the model.

## Quick start

```python
import xarray as xr
import xclimatology  # registers the .climatology accessor

sst = xr.tutorial.load_dataset("ersstv5")["sst"]
clim = sst.climatology(n_harmonics=2, detrend_degree=1)
anomaly = sst - clim
coefs = sst.climatology.coefficients(n_harmonics=2, detrend_degree=1)
```

## Copyright

- Copyright © 2026 Anthony Meza.
- Free software distributed under the MIT License.
