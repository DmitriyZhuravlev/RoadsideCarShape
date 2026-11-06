# Classical masks and cuboids: optical-flow clustering

Masks and 3D cuboids of moving objects from one fixed, calibrated camera, with no learned component, after

> Zhuravlev, D.: Moving object 3D detection and segmentation using optical flow clustering. In: Silhavy, R.,
> Silhavy, P. (eds.) Artificial Intelligence Application in Networks and Systems (CSOC 2023). Lecture Notes in
> Networks and Systems, vol. 724, pp. 426–443. Springer, Cham (2023). https://doi.org/10.1007/978-3-031-35314-7_38

This is a new implementation written from the paper alone (Alg. 1 and Sect. 4 there). It is the kind of input the
reconstruction of the article takes (Sect. 4.1).

## Method (`ofc.py`)

| Step | Paper | Here |
| :--- | :--- | :--- |
| 1 | dense optical flow (any method) | Farnebäck, half resolution |
| 2 | segments: flow coded in HSV (hue = direction, value = magnitude, fixed saturation), `KMEANS` colour quantisation, regions of one colour | three segmentations of the frame: **background**, the mean of the whole video, a pixel is foreground where the frame differs from it by more than 25 grey levels; **colour**, k-means (k = 12) on the frame's Lab colours; **flow**, k-means (k = 16) on the HSV-coded flow, colours clustered as points (v cos h, v sin h) of the HSV cone. A **segment** is a connected region that is foreground and lies wholly inside one colour segment and one moving flow segment. A segment smaller than 2 % of the image area a car would have at its distance joins its largest neighbour moving alike |
| 3 | `LIFT`: the 2D box of a segment mapped onto the road by IPM; a footprint of the type's average size and the object's orientation inscribed in the quadrangle; the height so that the top face touches it | **the 3D box around the silhouette, after Dubská et al. (see below)**: the size is measured, not assumed. Before it, the object's mirror image on the wet road is cut off at the contact line |
| 4 | type `t = argmin ‖d_t − d̂‖`; merge neighbouring segments while the error is large; keep the best fit | the type whose average size the measured size deviates least from (relative RMS deviation = the error); neighbours moving within 35° are merged one at a time, each time the one whose merge fits best; of all merges the one with the smallest error is kept (of near-equal ones of the same type, the larger), accepted if the deviation is ≤ 30 % |
| 5 | post-processing: footprints cannot intersect on the road, a segment must be sufficiently convex | both; an object cut by the image border is skipped; two overlapping objects moving alike are merged if their union fits as well; an object inside a larger one, or whose bottom (its ground contact) is hidden inside a nearer object, is dropped; so is a mirror image below a car. **Two cars in one object** are split (see below) |
| — | tracking and type persistence proposed as an improvement | **linking in the image**: each track's 2D box moved by its mean optical flow, objects assigned by IoU (Hungarian, ≥ 0.3), a track survives 3 frames without an object. Per track of ≥ 3 objects: the heading from the tangent of the smoothed path where it moves ≥ 0.3 m/frame, the size as the median of the measured sizes, the type from that size, and the centre solved again in every frame with this heading and size. **Cars that merge in some frames are apart in others**: every track is predicted into its span and extrapolated 5 frames beyond it, and merged objects are split by the predictions (see below) |

**The cuboid around the silhouette (`lift()`, after Dubská et al. 2014).** The box's three directions are the
direction of motion u (from the flow), the lateral direction v and the vertical z; each has a vanishing point. The
two tangents from a vanishing point to the convex hull of the object's mask are the images of two planes through
the camera centre that contain that direction, and each such plane touches one edge of the box parallel to it. With
the calibrated camera every tangency is one linear equation in the box's five unknowns, its extents along u and v
(s0, s1, r0, r1) and its height h; which edge a tangent touches follows from the view and is re-assigned from the
current box. Six tangents, five unknowns: least squares, iterated six times. The hull is undistorted first (the lens
is strongly distorted, and vanishing-point geometry needs straight lines). The type's average size enters with a small
weight (0.1), only so that a size the view leaves undetermined, such as the width of a car crossing far away, takes a
sensible value. The size **l × w × h is measured**; the type is then the one whose average size it deviates least
from (Car 4.5 × 1.8 × 1.5, Van 5.0 × 1.95 × 1.95, Truck 8.0 × 2.5 × 3.0, Bus 11.0 × 2.55 × 3.2, Cyclist
1.7 × 0.6 × 1.7, Pedestrian 0.6 × 0.6 × 1.7 m).

**Two cars in one object.** A car consists of segments of different colours (body, windows, tyres), so the merging
cannot refuse other colours as such. Three tests keep two cars apart:

* *while merging*, a neighbour whose largest segment is large (≥ 30 % of the object's body), clearly of another colour
  (Lab distance > 30) and lies beside the object or widens it by more than 25 % is another car's body;
* *in one frame*, the pixels of an accepted object are split into two by 2-means of their image speed (two cars at
  60 m differ by 15–25 %, one rigid car by about its length over its distance; the parts must differ in colour too)
  or of their colour (the parts must lie side by side; a car's windows lie within its body's columns). Each part is
  fitted on its own, the tangents along the border with the other left out; an object that holds two cars but whose
  parts fit no type is dropped rather than shown as one box;
* *over the clip*, an object whose box holds the predicted boxes of two or more tracks (its own included; half of a
  prediction's columns, overlapping a third of its height) is divided among them: each pixel goes to the predicted
  silhouette it lies in (the nearer car where two overlap). A part is placed by its tangents with the track's size
  and heading; a part covering less than a quarter of its prediction (a thin visible strip of a hidden car) keeps the
  prediction; a part of a track with its own object elsewhere is cut away. A frame inside a track's span without an
  object of the track gets the predicted cuboid. In `cuboids/<fid>.json`, `from_track` says which objects were
  detected, split by the tracks or predicted.

The flow at the border of two cars blends their speeds, and the pixel speeds of two touching cars are spatially
mixed; the tracks, built from the frames where the cars are apart, are what separates them reliably.

**Contact line.** On the wet road a car's reflection moves with it and differs from the background as much as the
car. Below the line where the tyres meet the road the image is the car mirrored: of the rows in the lower half of the
mask, the one whose rows below correlate best (NCC ≥ 0.55) with the rows above, mirrored, is the contact line, and the
mask is cut there.

**Why not the paper's LIFT.** The paper fixes the length, inscribes the footprint into the IPM quadrangle (near side
and both lateral sides) to get the centre and the width, and raises the height until the top face touches the box.
For an object crossing the view far away, the quadrangle's lateral sides run almost along the viewing ray, the
direction of the object's width, so the width is undetermined: at 60 m it came out 7.9 m and every car was rejected.

**Tried and dropped.** A shadow test (darker than the background with its hue) also removes dark cars and tyres on
the grey road, and the box loses its bottom; larger flow windows or k = 32 do not separate cars moving alike.

## Camera and road plane (`camera.py`)

* The dataset's calibration: intrinsics with the plumb-bob distortion (k1 = −0.43, strong) and the virtual-LiDAR →
  camera transform. Its "rotation" is **not orthonormal** (row norms 1.00 and 0.84 on V2X-Seq-SPD), so it is used as
  the linear map it is; turning it into a rotation vector moves the projections by up to 35 px.
* The road plane is fitted once by RANSAC on one LiDAR sweep, a calibration step equivalent to measuring the
  camera's height and tilt. On intersection A it gives a camera 8.44 m above the road, tilted 15.1° down.
* Labels are never read by `camera.py` or `ofc.py`.

## Run

    pip install opencv-python numpy scipy matplotlib
    python run_example.py [DATA_DIR] [FIRST] [N]

The defaults are the V2X-Seq-SPD example (intersection A of the article), frames 20–44 (2.5 s), a few seconds per
frame on one CPU core. Outputs in `out/`:

* `masks/<fid>.png`: uint16 instance masks;
* `cuboids/<fid>.json`: per object, its track, type (of the track) and type in this frame, centre in the road frame
  and in the virtual-LiDAR frame, yaw, measured l/w/h, 2D box, size deviation from the type, RMS of the tangents and
  `from_track` (detected / split / predicted);
* `steps.png`: the steps on one frame;
* `frames.png`: masks and cuboids on four frames;
* `metrics.json`: the comparison with the labels.

## Result on the example (frames 010720–010744)

Evaluation only: labelled objects that move in the frame (≥ 1 m/s), within 80 m and not truncated, matched to ours
per frame by footprint centre (Hungarian, ≤ 3 m). 230 labelled object-frames.

| | flow only, paper's LIFT | + background and colour | + average-size box fit | + tangent box (measured size) | **+ two-car tests, tracks** |
| :--- | ---: | ---: | ---: | ---: | ---: |
| our objects / matched | 73 / 28 | 76 / 41 | 146 / 73 | 101 / 71 | 121 / 80 |
| recall / precision | 0.12 / 0.38 | 0.18 / 0.54 | 0.32 / 0.50 | 0.31 / 0.70 | **0.35 / 0.66** |
| median centre error | 1.78 m | 1.58 m | 1.68 m | 1.57 m | 1.65 m |
| median yaw error (mod 180°) | 3.5° | 3.9° | 4.3° | 6.3° | 6.9° |
| median 2D box IoU | 0.55 | 0.33 | 0.53 | 0.59 | 0.58 |
| median size error l / w / h | 0.33 / 0.22 / 0.11 m | 1.13 / 0.37 / 0.28 m | 0.51 / 0.18 / 0.22 m | 0.82 / 0.20 / 0.18 m | 0.69 / 0.18 / 0.18 m |
| size | type average | type average | type average | measured | measured |
| class agreement (car / large / two-wheeler / pedestrian) | 0.93 | 0.39 | 0.62 | 0.80 | 0.80 |

Of the last column's 134 objects (before the 80 m limit of the evaluation), 97 are detected, 13 split from a merged
object by the tracks and 24 predicted from the tracks. The two cars of different colour that merged in a dozen frames
(a beige sedan and a white car partly behind it) are now apart in every frame, and from about the middle of the clip
so are a white SUV and a white van driving side by side.

The flow-only figures are kept as `out/steps_flow_only.png` and `out/frames_flow_only.png`.

Remaining failure modes:

* **Cars merged from the start of their tracks:** the white SUV and van are one object until their tracks, which
  begin where they drive apart, reach back (5 frames); before that they are one "bus", and their track's type is
  pulled to it.
* **Dense far traffic:** at 60–80 m cars, cyclists and motorcyclists occlude each other; about two thirds of them are
  still missed.
* **Heading at the start of motion:** a car that is just starting keeps one frame's flow direction, which is noisy.
* **Stopped objects** have no flow and are not detected, by design.
