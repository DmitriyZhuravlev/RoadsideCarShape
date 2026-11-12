# Code of the article

A snapshot of the parts of the research repository `3dseg` (commit `bada3ea`) that the article uses: the reconstruction
pipeline, its evaluation, the moving-camera variant and the dataset converter. Everything is classical (no learned
models); LiDAR and labels are read for evaluation only.

    pip install -r requirements.txt

## Entry points

| Script | What it does |
| :--- | :--- |
| `../classical_input/run_example.py` | The classical input: a mask and a 3D cuboid of every moving car per frame, from the video alone (see `../classical_input/README.md`). |
| `classical_to_ann.py OUT NAME` | The classical input's masks and cuboids in the layout the fusion reads (`docs/precise_ann_NAME`). |
| `make_dumps.py NAME` | The windows of every car track: vertices seeded inside the mask, tracked, and solved jointly for planar motion and per-vertex depth by `whole_scene.py` (section "Triangulation of one window"); written to `docs/ann_objects_NAME`. The reported runs use `STRIDE=1`. |
| `whole_scene_cars.py` | The whole video: the windows of a car aligned in one car frame, symmetry, height field, visual hull of the silhouettes, body-type priors where the length was never measured, regularised surface and Poisson surface via `spsr.py`, and the LiDAR evaluation. Results in `docs/whole_scene_cars$OUT_SUFFIX.json`. |
| `whole_scene.py K...` | One window per reference frame K with the pipeline's own segmentation (used by `make_dumps.py` for tracking and the joint solve). |
| `car_ba.py` | The joint track solve (`BA=1`): tried and left out of the reported method; it appears only as an ablation row. |
| `spsr.py` | Poisson surface reconstruction (plain, or screened with `alpha > 0`) on a regular grid, for the shown car models. |
| `ego_sequence.py` | The moving-camera variant (`CAMERA=moving`): planar visual odometry and the ego-motion-compensated background. |
| `vru_shape.py ID...` | Pedestrians, cyclists, motorcyclists: silhouettes, ground path, symmetric visual hull with a 0.6 m elliptic width prior, hold-out silhouette check; `TRACK=label` takes the labelled 2D boxes as the tracker. |
| `coop_to_spd.py` | DAIR-V2X-C infrastructure side to the V2X-Seq-SPD layout; `--clips` splits it at breaks in the recording. |

The dataset folder is given by `V2X_SEQ_DIR` (a V2X-Seq-SPD-style sequence). The reported fusion runs are

    ANN_DIR=docs/precise_ann_NAME DUMPS=docs/ann_objects_NAME OUT_SUFFIX=_ann_NAME_fb_noba \
    SIL_FILL=3 ANN_BOUND=1 BA=0 ANN_VOTE=0.75 ANN_VIS=0.35 ROOF=1 LIDAR_BOX=0 VIDEO=0 MATCH_GATE=G \
    python whole_scene_cars.py

with `MATCH_GATE` 3 for intersection A, 5 for intersection B and the V2X-Seq-SPD sequences, 6 for the DAIR-V2X-C
clips. `SIL_FILL=3` takes the convex hull of each mask as the silhouette, `ANN_BOUND=1` confines the model to the
input cuboid, `BA=0` aligns the windows without the joint track solve. The ablation rows change one setting each
(`SIL_FILL=0`, `ANN_BOUND=0`, `CABIN=0`, `SYMMETRY=0`, `ROOF=0`, `BOX_PRIOR=0`, `BA=1`). Other options are environment
variables documented at the top of each script.

`experiments/` holds the runners of the larger experiments (the DAIR-V2X-C clips with `pool.py` for the pooled table,
the labelled-box scores and the figures, `recolour.py` to redo the car stage, and the test with frames removed);
their dataset and code paths are those of the author's machine and must be adapted.

`tests/` are unit tests of the shared geometry and video helpers (`pytest tests`).
