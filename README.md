# Learning-Free 3D Car Shapes from a Single Roadside Video

<p align="center"><img src="figures/turntable.gif" width="560" alt="A reconstructed car model rotating"></p>

<p align="center"><i>A car reconstructed from one fixed roadside camera of V2X-Seq-SPD, without learning and without
labels (sequence 0002, 40 m from the camera; solid IoU 0.80 against its LiDAR, length 4.4 m against 4.3 m labelled).
The colours are the original pixels of the video.</i></p>

Monocular 3D perception for roadside cameras is almost always a learned box detector. This work recovers the 3D
**shape** of the cars that pass a fixed, calibrated camera with classical geometry only: in short windows of the video
every moving car is triangulated and its planar motion and the depth of every vertex are solved jointly; over the
whole video the windows of one car are fused in a common car frame, mirrored about its symmetry plane, averaged as a
robust height field and completed by a visual hull of the silhouettes. LiDAR and labels are used for evaluation only.

| Data (cars with LiDAR) | ours, closed LiDAR | labelled box | ours, raw LiDAR | labelled box |
| :--- | :---: | :---: | :---: | :---: |
| V2X-Seq-SPD example, intersection A (20; tuning) | 0.582 | 0.572 | 0.398 | 0.378 |
| DAIR-V2X-C example, intersection B (7) | 0.386 | 0.546 | 0.287 | 0.378 |
| DAIR-V2X-C, 20 clips, 7 intersections (130) | 0.539 | 0.619 | 0.423 | 0.465 |
| V2X-Seq-SPD, 12 sequences, 6 intersections (128) | 0.525 | 0.635 | 0.417 | 0.505 |

Mean solid IoU of the height fields against the cars' accumulated LiDAR (raw, and with the gaps between scan lines
closed), the labelled boxes scored the same way. The models match the labelled boxes on the intersection the method
was developed on and stay 0.08 to 0.11 below them elsewhere; the main error is the length along the viewing ray.
Nearby pedestrians and cyclists are reconstructed from their silhouettes alone (coarse models).

## Contents

- `main.tex` - the article (LNCS), `springer/` - the version for EURASIP Journal on Image and Video Processing
  (Springer Nature template, `scripts/make_springer.py`), `references.bib`, `figures/`, `tables/`.
- `code/` - a snapshot of the pipeline (from the research repository 3dseg), with its README.
- `scripts/` - tables and figures of the article from the results of the code.

Build: `latexmk -pdf main.tex`; the journal version: `python scripts/make_springer.py`, then `latexmk -pdf main.tex`
in `springer/`.

## Reproducing the tables and figures

The numbers come from 3dseg commit `dec7965` (code and the ablation results `docs/whole_scene_cars_abl_*.json`); the
same code is in `code/`.

- Tables: `scripts/make_tables.py` (do not edit `tables/*.tex` by hand; `tables/compare.tex` is written by hand).
  - Ablation and intersections A, B: `scripts/rerun_ablation.py` runs the 14 configurations (A: the V2X-Seq-SPD
    example; B: DAIR-V2X-C example converted by `code/coop_to_spd.py`; rows 0-4 with the windows of v6, 5-6 with the
    dense windows of v7); `scripts/box_scores.py` scores the labelled box and a box of our own size on the same cars.
  - Larger evaluation: `code/experiments/run_subset.py` (clips of DAIR-V2X-C converted by `coop_to_spd.py --clips`,
    sequences of V2X-Seq-SPD split by `spd_split.py`), then `pool.py` for the pooled table and the box scores.
- Figures of intersection A: `scripts/capture_models.py` (run with `DUMPS=docs/whole_scene_objects_dense`) keeps the
  models with their poses in `figures/models_A.pkl` (large, regenerable, not in git); then `cars_figure.py`
  (`cars_best.png`), `scene_figure.py` (`scene_models.jpg`), `method_figure.py` (`method_steps.png`),
  `all_cars_figure.py` (`all_cars.png`, `turntable.png`), `lidar_reference_figure.py` (`lidar_reference.png`).
- `scripts/large_models_figure.py` - `large_models.png`, the best models of the larger evaluation;
  `figures/vru_models.png` - pedestrians and cyclists (`code/vru_shape.py`, `code/experiments/run_vru.sh`);
  `scripts/turntable_gif.py` - the animation above.
- Copied from the 3dseg outputs: `window.jpg`, `gallery_A.png`, `triangulation_B.png`, `moving_camera.png`.
