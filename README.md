# Learning-Free 3D Triangulation of Moving Objects from a Single Roadside Camera

<p align="center"><img src="figures/turntable.gif" width="560" alt="A reconstructed car model rotating"></p>

<p align="center"><i>A car of DAIR-V2X-C (clip 114_0) reconstructed from one fixed roadside camera. The colours are
the original pixels of the video.</i></p>

Monocular 3D perception for roadside cameras is almost always a learned box detector. This work recovers the 3D
**shape** of the cars that pass a fixed, calibrated camera with classical geometry only. A classical method finds
the mask and the 3D cuboid of every moving car in the video (background subtraction, optical flow, tangents from the
vanishing points). In short windows of the video every car is then triangulated and its planar motion and the depth
of every vertex are solved jointly; over the whole video the windows of one car are solved together as one rigid
car, mirrored about its symmetry plane, averaged as a robust height field and completed by a visual hull of the
masks. No step is learned; LiDAR and 3D labels are used for evaluation only.

| Data (cars scored) | ours | labelled box | input cuboid | best single window |
| :--- | :---: | :---: | :---: | :---: |
| V2X-Seq-SPD example, intersection A (8; tuning) | 0.589 | 0.572 | 0.552 | 0.275 |
| DAIR-V2X-C example, intersection B (9) | 0.507 | 0.508 | 0.504 | 0.401 |
| DAIR-V2X-C, 3 clips (24) | 0.584 | 0.637 | 0.599 | 0.223 |
| V2X-Seq-SPD, 4 sequences (30) | 0.554 | 0.649 | 0.617 | 0.292 |

Mean solid IoU of the height fields against the cars' accumulated LiDAR with the gaps between scan lines closed; the
labelled boxes, the input cuboids of the classical method and the best single window are scored the same way on the
same cars. The models are level with the labelled boxes on A and B (paired difference +0.017 [-0.071, +0.093] and
-0.001 [-0.080, +0.058], 95 % bootstrap intervals) and below them on the further clips (-0.053 [-0.103, -0.003] and
-0.095 [-0.158, -0.039]); they do not yet improve on their own input cuboids. The main limits are the classical
masks and cuboids and the coverage: 8 of the 24 cars of A with a LiDAR reference get a model.

## Contents

- `main.tex` - the article (LNCS), `springer/` - the version for EURASIP Journal on Image and Video Processing
  (Springer Nature template, `scripts/sync_springer.py`), `references.bib`, `figures/`, `tables/`, `tables_cls/`.
- `code/` - a snapshot of the pipeline (from the research repository 3dseg), with its README.
- `scripts/` - tables and figures of the article from the results of the code.

Build: `latexmk -pdf main.tex`; the journal version: `python scripts/sync_springer.py` (copies the abstract and body of `main.tex` into `springer/main.tex`), then `latexmk -pdf main.tex`
in `springer/`.

## Reproducing the tables and figures

- Tables of the article: `scripts/make_tables_cls.py` writes `tables_cls/main_table.tex`, `tables_cls/clips_table.tex`
  and `tables_cls/numbers.tex` from the per-clip results of the research repository 3dseg
  (`docs/whole_scene_cars_ann_<run>_<tag>.json`); do not edit them by hand. `tables/classical_input.tex` and
  `tables/compare.tex` are also used by the article.
- Figures of the article, all from the classical input: `scripts/teaser_figure.py` (`teaser.png`),
  `scripts/heading_figure.py` (`heading.png`), `scripts/method_figure.py` (`method_steps.png`),
  `scripts/cars_figure.py` (`cars_best.png`), `scripts/best_models_classical.py` (`best_models.png`),
  `scripts/lidar_reference_figure.py` (`lidar_reference.png`). The figures of intersection A need the captured models
  `figures/models_A_cls.pkl` (large, regenerable, not in git).
- Not used by the article any more: the other files of `tables/`, the figures `*_SLAM.png` and `vru_compare.png`, and
  their scripts (`make_tables.py`, `paired_ci.py`, `rerun_ablation.py`, `best_models_figure.py`,
  `vru_compare_figure.py`) belong to earlier experiments with masks and cuboids derived from the labels and LiDAR.
