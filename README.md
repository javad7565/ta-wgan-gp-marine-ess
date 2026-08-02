# TA-WGAN-GP: Temporal-Aware Wasserstein GAN for Marine Wind-Power Scenarios

Code and data accompanying the paper:

> **Temporal-Aware Wasserstein GAN with Self-Attention for Marine Wind-Power
> Scenario Generation**
> Javad Azarakhsh, Mahmoud Oukati Sadegh, Abdolhossein Mohammadrahimi
> *Submitted to Ocean Engineering (Elsevier), 2026.*


## Results at a glance



## Data and provenance

The hourly 100 m wind series were retrieved from the **Open-Meteo ERA5 archive
endpoint**, which serves ECMWF ERA5 reanalysis without a credential step. They
were *not* downloaded from the Copernicus Climate Data Store directly, though
the same series can be rebuilt from either source. Both route points lie on
ERA5's native 0.25 degree grid, so no horizontal interpolation is involved.

| | Gulf of Oman | Persian Gulf |
|---|---|---|
| Latitude / Longitude | 25.00 N, 58.00 E | 26.50 N, 52.00 E |
| Years | 2014-2023 | 2014-2023 |
| Hourly samples | 87,648 | 87,648 |
| 48 h profiles (N) | 3,650 | 3,650 |

Processed arrays live in `data/marine/*.npz` (~2 MB each); the coordinates,
years, label counts, and turbine curve that define them are in the companion
`manifest_*.json` files. The raw API response cache is not committed since it is
re-fetchable from the manifests.

**Attribution.** This work contains modified Copernicus Climate Change Service
information (2014-2023). Neither the European Commission nor ECMWF is
responsible for any use of that information. See [`data/README.md`](data/README.md).

## Install

Requires Python 3.12+.

```bash
git clone https://github.com/javad7565/ta-wgan-gp-marine-ess.git
cd ta-wgan-gp-marine-ess
pip install -r requirements.txt
pip install -e .
```

## Reproduce

Training, sampling, and metric evaluation all run from fixed seeds (0, 1, 2).

```bash
# Rebuild the marine dataset from the Open-Meteo ERA5 archive
python experiments/run_marine.py

# Main comparison: TA-WGAN-GP vs. four baselines, three seeds
python experiments/run_engine_a_main.py

# Component ablation (temporal critic, attention, spectral norm)
python experiments/run_engine_a_ablation.py

# Per-event-class fidelity breakdown
python experiments/run_engine_a_per_class.py

# Regenerate every LaTeX table and inline number in the manuscript
python -m ship_ess.engine_a.build_paper_a
```

Run the test suite with `pytest`.

Trained generator weights (~39 MB) are attached to the tagged release rather
than committed, and are regenerable from the code and configuration above.

## Repository layout

```
src/ship_ess/engine_a/   TA-WGAN-GP: generator, critics, training, metrics
src/ship_ess/engine_b/   ESRDC 14-bus testbed and NSGA-II ESS sizing
data/marine/             Processed per-site datasets and manifests
experiments/             Runnable scripts for every reported result
paper_a/                 Manuscript source, figures, generated tables
tests/                   Test suite
docs/project_structure.md  Detailed pipeline notes
```

## Companion study

The ESS-sizing numbers quoted in the paper's discussion come from a companion
shipboard-resilience study that drives NSGA-II sizing over an ESRDC-derived
14-bus testbed using these scenarios. That code is in
`src/ship_ess/engine_b/`; the manuscript is in preparation and is not included
here.

## Citing

See [`CITATION.cff`](CITATION.cff), or use the GitHub "Cite this repository"
button. Please cite the Ocean Engineering paper.

## License

Code is released under the MIT License ([`LICENSE`](LICENSE)). The ERA5-derived
data in `data/marine/` carries Copernicus terms and its own attribution
requirement, described in [`data/README.md`](data/README.md).
