# ps131-iceroute

Code for the paper *Sensor-error-aware ice routing using shipborne imagery and AMSR2: a simulation
study based on Polarstern cruise PS131* (B. Danilov, J. Dong, A. Said Ben Salah).

Archived version used in the paper: v1.0.0, doi:[10.5281/zenodo.23476243](https://doi.org/10.5281/zenodo.23476243).
Result files of the paper: doi:[10.5281/zenodo.23476269](https://doi.org/10.5281/zenodo.23476269).

Camera frames of RV Polarstern cruise PS131 (July–August 2022) are matched to the bridge synoptic
observations, AMSR2 ASI sea ice concentration and ERA5. Three ice classifiers (AMSR2, camera,
early fusion of both) are evaluated by cross-validation over hourly episodes. Their measured errors
then drive a routing simulation: real out-of-fold frames from other episodes are drawn as
observations, and planners that use hard labels, the expected cost or a cell-wise CVaR surrogate
are compared by regret against an oracle, with an AMSR2-based true field and with DMI ice charts as
an alternative reference. A residual PPO agent is trained on the same simulation.

## Data

All data are open. Two services need a free account; the keys stay on your computer and are never
stored in this repository.

| Data | Source | Used for | Account |
|---|---|---|---|
| PS131 camera frames (GoPro, 5 s) | PANGAEA, doi:10.1594/PANGAEA.967580 | images, GPS, frame quality | no |
| PS131 synoptic observations | PANGAEA, doi:10.1594/PANGAEA.987909 | labels: visibility (WMO 4377), ice (WMO 0639) | no |
| AMSR2 ASI 6.25 km, v5.4 | University of Bremen, data.seaice.uni-bremen.de | product, true field of the simulation | no |
| ERA5 hourly single levels | Copernicus CDS, doi:10.24381/cds.adbb2d47 | weather at the frames | CDS key in `~/.cdsapirc` |
| DMI ice charts | Copernicus Marine, SEAICE_ARC_SEAICE_L4_NRT_OBSERVATIONS_011_002 | alternative reference | `copernicusmarine login` |

Only 56 of the 665 hourly camera archives were used (about evenly spread over the cruise); their
names are in `data_manifest/ps131_camera_archives.txt`, and `download_ps131_full.py` downloads
exactly these (about 100–150 GB) together with the synoptic observations. An optional proxy can be
given with the environment variable `PS131_PROXY`. AMSR2 and ERA5 are downloaded by
`collocate_ps131.py`, the DMI charts by `download_dmi_charts_ps131.py`. All data folders are
created next to the scripts.

## Installation

Python 3.10 or newer.

    conda install -c conda-forge pyhdf
    pip install -r requirements.txt

## Running

`run_all.bat` (Windows, Anaconda Prompt) runs everything in order; the commands also work in any
other shell. In short:

    python download_ps131_full.py              # camera archives + synoptic observations
    python build_ps131_index.py                # frame table: time, GPS, quality metrics
    python collocate_ps131.py                  # + AMSR2 and ERA5 (downloads them)
    python label_ps131_synop.py                # + labels from the bridge observations
    python prepare_ps131_dataset.py --max-dt 180 --cache     # 1024x768 frame cache
    python train_ps131_cv.py --mode probe      # ConvNeXt-Tiny features + linear probe, 5 folds by episode
    python fusion_ps131.py                     # AMSR2 / camera / fusion -> runs/fusion_*/fusion_oof.csv
    python cv_repeat_ps131.py --oof <fusion_oof.csv> --repeats 10   # repeated CV, calibration
    python route_ps131_v3.py --oof <fusion_oof.csv> --matrix-cv 5 --obs-model resample --trials 20 --tag main32
    python drl_route_ps131_v3.py --oof <fusion_oof.csv> --train-seed 0 --hours 6

All routing variants of the paper are listed in `run_all.bat` with their tags. Each run writes a
dated folder `runs/<name>_<date>/` with `log.txt`, the arguments and the results; the figure and
summary scripts take the latest folder of each tag. `--smoke` runs a short version of the routing
and PPO scripts with a time estimate.

Computing time on a desktop PC: the ConvNeXt features need a GPU for about half an hour (everything
after that runs on the CPU); one routing run takes 40–50 min (about 19 h for all 25), one PPO
seed about 7 h.

## Result files

`paper_results.zip` (Zenodo, doi:[10.5281/zenodo.23476269](https://doi.org/10.5281/zenodo.23476269)) holds the results of the full run behind the paper:
the matched frame table with labels (`ps131_index/ps131_labeled.csv`), the out-of-fold
predictions of all classifiers (`runs/fusion_*/fusion_oof.csv`, `runs/cv_*/oof_predictions.csv`),
the DMI chart windows, and the folders of all routing and PPO runs (per-trial results, summaries,
PPO policies and training curves). Unzip it into this folder to rebuild the tables and figures
without recomputing: `reviewer_analyses_ps131.py`, `drl_seeds_summary.py`, `make_figures_shrink.py`
and `make_supplement_en.py` need only the archive; `make_figures_article3.py` also needs the AMSR2
maps and, for Fig. 2, the camera frames. The runs of the paper were made with an earlier version of
these scripts that wrote its reports in Russian; the text reports in the archive (`summary.txt`,
`*_metrics.txt`, `calibration.txt`, `checks_summary.txt` and the like) were translated to English
with all numbers unchanged, and the run logs (`log.txt`) are not included. `pack_results.py` builds
the same archive from a recomputation with this code.

## Where the numbers of the paper come from

| Paper (S = supplementary material) | Script | File |
|---|---|---|
| Fig. 1, 2 | `make_figures_article3.py` | `runs/article3_figs_en/fig1_track_map.png`, `fig2_frames_grid.png` |
| Table 1, Tables S6–S7 (WMO codes) | `label_ps131_synop.py` | code mapping in the script; counts in `runs/checks_*/codes_*.csv` |
| Table 2, Fig. 3 | `cv_repeat_ps131.py`, `fusion_ps131.py` | `runs/cvrep_*/cv_repeat_summary.txt`, `fig3_confusion.png` |
| Fine-tuning check (Sect. 4.1) | `train_ps131_cv.py --mode finetune` | `runs/cv_finetune_*/cv_metrics.txt` |
| Table 3 | `check_codes_temporal_ps131.py` | `runs/checks_*/checks_summary.txt` |
| Table 4, Fig. 4 | `cv_repeat_ps131.py` | `runs/cvrep_*/calibration.txt`, `fig4_reliability.png` |
| Fig. 5 (scheme) | `make_figures_shrink.py` | `fig5_scheme.png` |
| Tables 5a, 5b, Fig. 6, Fig. 7 | `route_ps131_v3.py --tag main32` | `runs/routev3_main32_*/summary.txt`, `fig5_regret.png`, `fig7_route_example.png` |
| Fig. 8, Tables S3–S4 (Holm, tails) | `make_figures_shrink.py`, `reviewer_analyses_ps131.py` | `fig8_tails.png`, `reviewer_analyses_ps131.txt` |
| Fig. 9, Table S1 (robustness) | `make_figures_shrink.py` | `fig9_robustness.png`; `summary.txt` of the runs `matrix`, `floe3`, `floe12`, `thr25_75`, `thr45_90`, `cost148`, `cost1212`, `cost124`, `range2`, `block1`, `block6`, `donor32`, `cvar050`, `cvar090`, `lab60` |
| Table 6 (shift between cruise halves) | `route_ps131_v3.py --time-split` | `runs/routev3_timesplit32_*`, `runs/routev3_timesplit_matrix_*` |
| Table 7 (DMI reference) | `route_ps131_v3.py --truth dmi` | `runs/routev3_dmi70_*`, `runs/routev3_dmi90_*`; coverage and lag in `08_icecharts_2022/dmi_coverage.csv` |
| Table S5 (DMI no-data cells, lag) | `route_ps131_v3.py`, `reviewer_analyses_ps131.py` | runs `dmi70water`, `dmi70heavy`, `dmi70amsr`, `dmi50amsr`, `dmi70donor`; `reviewer_analyses_ps131.txt` |
| Table S2 (error autocorrelation), label time offsets | `reviewer_analyses_ps131.py` | `reviewer_analyses_ps131.txt` |
| Table 8, Fig. S1 (PPO) | `drl_route_ps131_v3.py`, `drl_seeds_summary.py`, `make_supplement_en.py` | `drl_seeds_summary.txt`, `runs/drlv3_*/test_results.csv`, `curves.csv` |

The values in the tables of the supplementary material are transcribed in `make_supplement_en.py`
from the files in this map.

## Layout

    download_ps131_full.py         camera archives and synoptic observations (PANGAEA)
    build_ps131_index.py           frame table: time, EXIF GPS, image quality metrics
    collocate_ps131.py             AMSR2 and ERA5 at every frame (download and matching)
    label_ps131_synop.py           labels from the synoptic reports (WMO codes)
    prepare_ps131_dataset.py       label windows, frame cache
    train_ps131_camera.py          shared code: data set, ConvNeXt model, metrics, log
    train_ps131_cv.py              cross-validation by episode: linear probe and fine-tuning
    fusion_ps131.py                AMSR2 / camera / early and late fusion, out-of-fold predictions
    cv_repeat_ps131.py             repeated cross-validation, temperature scaling, reliability
    check_codes_temporal_ps131.py  ambiguous WMO ice codes, transfer between cruise halves
    download_dmi_charts_ps131.py   DMI ice charts (Copernicus Marine)
    check_dmi_coverage_ps131.py    DMI chart windows around the episodes, coverage and time lag
    route_ice_sources_ps131.py     shared routing code: true field, sensors, MCP planner
    route_ps131_v3.py              routing simulation of the paper (all variants)
    drl_route_ps131.py             shared PPO code: policy, PPO, critic warm-up
    drl_route_ps131_v3.py          residual PPO agent in the routing simulation
    drl_seeds_summary.py           PPO over training seeds
    reviewer_analyses_ps131.py     tail metrics, Holm correction, label offsets, error autocorrelation
    make_figures_article3.py       Figs. 1–4, 6, 7
    make_figures_shrink.py         Figs. 5, 8, 9
    make_supplement_en.py          Fig. S1 and the supplementary material (docx)
    pack_results.py                results archive for Zenodo

## Notes

- The unit of independence is the hourly episode: folds, bootstrap confidence intervals and the
  observation pools are built by episode, and the sensor error tables of an episode are always
  estimated from other episodes.
- Random seeds are fixed (routing `--seed 0`, PPO `--train-seed`), so the same data give the same
  numbers; PyTorch on another GPU, CPU or version may change the last digits of the features and
  of the trained policies.
- The run tags keep the names of the development history (`main32` is the main run of version 3.2
  of the routing script, later versions only added options). The second value of Table 6
  (`timesplit_matrix`) was computed with version 3.1 of `route_ps131_v3.py`.
- The paper's seed-0 PPO run is `runs/drlv3_20261012_2324` (before the `--train-seed` option); a new
  run of seed 0 writes `runs/drlv3_s0_<date>`, which the summary scripts also find.

## Licence

Code: MIT, see `LICENSE`. The data keep their own licences (PANGAEA data CC-BY-4.0; AMSR2, ERA5
and Copernicus Marine products under the terms of their providers). Please cite the paper and the
data sets if you use the code (`CITATION.cff`).
