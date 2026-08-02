# Data

## Contents

| File | Description |
|---|---|
| `marine/wind_gulf_of_oman.npz` | Processed 48 h normalized power profiles, Gulf of Oman (25.00 N, 58.00 E) |
| `marine/wind_persian_gulf.npz` | Processed 48 h normalized power profiles, Persian Gulf (26.50 N, 52.00 E) |
| `marine/manifest*.json` | Coordinates, years, event-label counts, turbine curve, split boundaries |
| `input/bus_data.csv`, `input/branch_data.csv` | Shipboard microgrid network parameters (companion sizing study) |

Each site provides N = 3,650 overlapping 48 h profiles at 5-minute resolution
(T = 576 steps), derived from ten years (2014-2023) of hourly 100 m wind speed
and mapped to normalized power through a piecewise turbine curve
(cut-in 3 m/s, rated 12 m/s, cut-out 25 m/s).

The raw Open-Meteo response cache is not committed. It is re-fetchable from the
coordinates and years recorded in the manifests via
`python experiments/run_marine.py`.

## Source and licence

The underlying hourly wind fields are **ERA5 reanalysis**, produced by ECMWF and
distributed through the Copernicus Climate Data Store. They were retrieved here
through the **Open-Meteo ERA5 archive endpoint**
(`https://archive-api.open-meteo.com/v1/era5`), which serves the same reanalysis
without requiring CDS credentials. Both route points fall on ERA5's native
0.25 degree grid, so no horizontal interpolation is applied.

> This work contains modified Copernicus Climate Change Service information
> (2014-2023). Neither the European Commission nor ECMWF is responsible for any
> use that may be made of the Copernicus information or data it contains.

ERA5 is published under the Copernicus Licence, which permits redistribution and
adaptation provided the source is acknowledged and modification is indicated.
The arrays here are modified: interpolated to 5-minute resolution, converted to
turbine power, normalized, windowed, and labelled.

If you use these data, please cite ERA5 as well as this repository:

> Hersbach, H., et al. (2020). The ERA5 global reanalysis.
> *Quarterly Journal of the Royal Meteorological Society*, 146(730), 1999-2049.
> https://doi.org/10.1002/qj.3803

The MIT licence in the repository root applies to the **code**, not to these
data files.
