# Code of the article

A snapshot of the parts of the research repository `3dseg` (commit `da5608c`) that the article uses: the reconstruction
pipeline, its evaluation, the moving-camera variant and the dataset converter. Everything is classical (no learned
models); LiDAR and labels are read for evaluation only.

    pip install -r requirements.txt

## Entry points

| Script | What it does |
| :--- | :--- |
| `whole_scene.py K...` | One window per reference frame K: segmentation, triangulation, joint solve of planar motion and per-vertex depth (Sect. 4.1). With `DUMP=1` it writes the window's cars to `docs/$DUMP_DIR/K.pkl`. |
| `whole_scene_cars.py` | The whole video: per-car fusion of the dumped windows (car frame, ICP, symmetry, height field, visual hull) and the LiDAR evaluation (Sects. 4.2, 5). Results in `docs/whole_scene_cars$OUT_SUFFIX.json`. |
| `ego_sequence.py` | The moving-camera variant (`CAMERA=moving`): planar visual odometry and the ego-motion-compensated background. |
| `vru_shape.py ID...` | Pedestrians, cyclists, motorcyclists: silhouettes, ground path, symmetric visual hull with a 0.6 m elliptic width prior, hold-out silhouette check; `TRACK=label` takes the labelled 2D boxes as the tracker. |
| `coop_to_spd.py` | DAIR-V2X-C infrastructure side to the V2X-Seq-SPD layout; `--clips` splits it at breaks in the recording. |

The dataset folder is given by `V2X_SEQ_DIR` (a V2X-Seq-SPD-style sequence). The settings of the article are the
defaults; the runs used `SEG=mask MIRROR=0 TYRES=0 FIGURES=0 DUMP=1`. Other options are environment variables
documented at the top of each script.

`experiments/` holds the runners of the larger experiments (the DAIR-V2X-C clips with `pool.py` for the pooled table,
the labelled-box scores and the figures, `recolour.py` to redo the car stage, and the test with frames removed);
their dataset and code paths are those of the author's machine and must be adapted.

`tests/` are unit tests of the shared geometry and video helpers (`pytest tests`).
