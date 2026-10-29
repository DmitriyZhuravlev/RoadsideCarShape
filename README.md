# Learning-Free 3D Car Shapes from a Single Roadside Video

<p align="center"><img src="figures/turntable.gif" width="560" alt="A reconstructed car model rotating"></p>

<p align="center"><i>A van reconstructed from one fixed roadside camera of DAIR-V2X-C, without learning and without
labels (clip 106_0; solid IoU 0.79 against its LiDAR, length 4.5 m against 4.5 m labelled). The colours are the
original pixels of the video.</i></p>

Monocular 3D perception for roadside cameras is almost always a learned box detector. This work recovers the 3D
**shape** of the cars that pass a fixed, calibrated camera with classical geometry only: in short windows of the video
every moving car is triangulated and its planar motion and the depth of every vertex are solved jointly; over the
whole video the windows of one car are solved together as one rigid car, mirrored about its symmetry plane, averaged
as a robust height field and completed by a visual hull of the silhouettes; a car whose length the video never
measured takes its body type's length and profile. LiDAR and labels are used for evaluation only.

| Data (cars with LiDAR) | ours, closed LiDAR | labelled box | ours, raw LiDAR | labelled box |
| :--- | :---: | :---: | :---: | :---: |
| V2X-Seq-SPD example, intersection A (19; tuning) | 0.617 | 0.572 | 0.403 | 0.376 |
| DAIR-V2X-C example, intersection B (7) | 0.571 | 0.546 | 0.404 | 0.378 |
| DAIR-V2X-C, 20 clips, 7 intersections (109) | 0.612 | 0.618 | 0.463 | 0.465 |
| V2X-Seq-SPD, 12 sequences, 6 intersections (104) | 0.618 | 0.636 | 0.450 | 0.488 |

Mean solid IoU of the height fields against the cars' accumulated LiDAR (raw, and with the gaps between scan lines
closed), the labelled boxes scored the same way on the same cars. On the two larger evaluations the paired difference
to the labelled boxes is not separated from zero (-0.006 [-0.025, +0.012] and -0.018 [-0.041, +0.004], 95 %
bootstrap intervals). The main remaining errors are the position along the viewing ray and the coverage: about half
of the visible moving cars get a model. Nearby pedestrians and cyclists are reconstructed from their silhouettes
alone (coarse models; they need a good tracker).

## Contents

- `main.tex` - the article (LNCS), `springer/` - the version for EURASIP Journal on Image and Video Processing
  (Springer Nature template, `scripts/make_springer.py`), `references.bib`, `figures/`, `tables/`.
- `code/` - a snapshot of the pipeline (from the research repository 3dseg), with its README.
- `scripts/` - tables and figures of the article from the results of the code.

Build: `latexmk -pdf main.tex`; the journal version: `python scripts/make_springer.py`, then `latexmk -pdf main.tex`
in `springer/`.

## Reproducing the tables and figures

The numbers come from 3dseg commit `85785c2`; the same code is in `code/`.

- Tables: `scripts/make_tables.py` (do not edit `tables/*.tex` by hand; `tables/compare.tex` is written by hand) and
  `scripts/paired_ci.py` (`tables/paired_ci.tex`).
  - Ablation and intersections A, B: `scripts/rerun_ablation.py` runs the six steps on both intersections (A: the
    V2X-Seq-SPD example; B: the DAIR-V2X-C example converted by `code/coop_to_spd.py`), with the windows of the current
    code (`whole_scene.py` with `DUMP=1`); `scripts/box_scores.py` scores the labelled box and a box of our own size
    on the same cars.
  - Larger evaluation: `code/experiments/run_subset.py` makes the windows of every clip (DAIR-V2X-C converted by
    `coop_to_spd.py --clips`, V2X-Seq-SPD split by `spd_split.py`); `whole_scene_cars.py` per clip writes
    `docs/whole_scene_cars_all_<clip>.json`; `pool.py` scores the labelled boxes (`boxes.json`).
- Figures of intersection A: `scripts/capture_models.py` (with `DUMPS=docs/ba_objects/A`) keeps the models with their
  poses in `figures/models_A.pkl` (large, regenerable, not in git); then `cars_figure.py` (`cars_best.png`),
  `scene_figure.py` (`scene_models.jpg`), `method_figure.py 003972 003890 004006` (`method_steps.png`, with the same
  `DUMPS`), `lidar_reference_figure.py` (`lidar_reference.png`).
- `figures/best_models.png` (the best models of all datasets with their video frames) and `figures/vru_compare.png`
  (pedestrians and cyclists with our own and with the labelled 2D boxes as tracking): `scripts/best_models_figure.py`
  and `scripts/vru_compare_figure.py`; `figures/turntable.gif`: `scripts/gif_variants.py`.
- Copied from the 3dseg outputs: `window.jpg`, `triangulation_B.png` (an earlier version's models of B, kept as the
  failure the prior fixes), `moving_camera.png`.
