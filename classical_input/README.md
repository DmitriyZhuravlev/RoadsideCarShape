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
| 2 | segments: flow coded in HSV (hue = direction, value = magnitude, fixed saturation), `KMEANS` colour quantisation, regions of one colour | three segmentations of the frame: **background subtraction** (`foreground.py`, see below; GSOC by default); **colour**, k-means (k = 12) on the frame's Lab colours; **flow**, k-means (k = 16) on the HSV-coded flow, colours clustered as points (v cos h, v sin h) of the HSV cone. A **segment** is a connected region that is foreground and lies wholly inside one colour segment and one moving flow segment. A segment smaller than 2 % of the image area a car would have at its distance joins its largest neighbour moving alike |
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

**Background subtraction (`foreground.py`, `FG=...`).** The best classical background subtractors on
ChangeDetection.net 2014 are SuBSENSE and PAWCS (St-Charles et al.): per pixel a set of samples of colour and local
binary similarity patterns, with thresholds adapted by feedback. They are in BGSLibrary, which does not build against
OpenCV 5 here; OpenCV contrib has **GSOC**, built on the same ideas, and LSBP. The online models see the whole video
twice (the first pass builds the model, the second gives the masks). Compared on frames 010720–010744:

| foreground | recall / precision | centre error | 2D box IoU | length error | masks |
| :--- | ---: | ---: | ---: | ---: | :--- |
| mean of the video | 0.35 / 0.66 | 1.65 m | 0.58 | 0.69 m | a ghost where a car waited at the light |
| median of the video | 0.32 / 0.66 | 1.58 m | 0.52 | 0.58 m | no ghost, far cars break up |
| **GSOC** | 0.34 / 0.64 | 1.64 m | 0.59 | **0.45 m** | no ghost, filled bodies, least noise |
| LSBP | 0.36 / 0.63 | 1.40 m | 0.57 | 0.45 m | noisier, extra blobs |

GSOC is randomised (it samples its model); its numbers vary by a few points from run to run.

## Camera and road plane (`camera.py`)

* The dataset's calibration: intrinsics with the plumb-bob distortion (k1 = −0.43, strong) and the virtual-LiDAR →
  camera transform. Its "rotation" is **not orthonormal** (row norms 1.00 and 0.84 on V2X-Seq-SPD), so it is used as
  the linear map it is; turning it into a rotation vector moves the projections by up to 35 px.
* The road plane is fitted once by RANSAC on one LiDAR sweep, a calibration step equivalent to measuring the
  camera's height and tilt. On intersection A it gives a camera 8.44 m above the road, tilted 15.1° down.
* Labels are never read by `camera.py` or `ofc.py`.

## Run

    pip install opencv-contrib-python numpy scipy matplotlib pillow
    FG=gsoc python run_example.py [DATA_DIR] [FIRST] [N]

The defaults are the V2X-Seq-SPD example (intersection A of the article), frames 20–44 (2.5 s); the outputs in
`out/` are of `FG=gsoc python run_example.py <example> 20 60`, frames 010719–010778 (6 s). A few seconds per frame on
one CPU core. Outputs in `out/`:

* `masks/<fid>.png`: uint16 instance masks;
* `cuboids/<fid>.json`: per object, its track, type (of the track) and type in this frame, centre in the road frame
  and in the virtual-LiDAR frame, yaw, measured l/w/h, 2D box, size deviation from the type, RMS of the tangents and
  `from_track` (detected / split / predicted);
* `steps.png`: the steps on one frame;
* `frames.png`: masks and cuboids on four frames;
* `metrics.json`: the comparison with the labels;
* `video.mp4`, `video.gif`: the clip with masks, cuboids and track ids ("(track)": an object placed by its track).

## Result on the example

Evaluation only: labelled objects that move in the frame (≥ 1 m/s), within 80 m and not truncated, matched to ours
per frame by footprint centre (Hungarian, ≤ 3 m).

**Development, frames 010720–010744 (230 labelled object-frames)**, the method step by step:

| | flow only, paper's LIFT | + background and colour | + average-size box fit | + tangent box (measured size) | + two-car tests, tracks | **+ GSOC** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| recall / precision | 0.12 / 0.38 | 0.18 / 0.54 | 0.32 / 0.50 | 0.31 / 0.70 | 0.35 / 0.66 | 0.34–0.38 / 0.64–0.69; with the depth fix 0.43 / 0.81 |
| median centre error | 1.78 m | 1.58 m | 1.68 m | 1.57 m | 1.65 m | 1.47–1.64 m |
| median 2D box IoU | 0.55 | 0.33 | 0.53 | 0.59 | 0.58 | 0.56–0.59 |
| median size error l / w / h | 0.33 / 0.22 / 0.11 m | 1.13 / 0.37 / 0.28 m | 0.51 / 0.18 / 0.22 m | 0.82 / 0.20 / 0.18 m | 0.69 / 0.18 / 0.18 m | 0.45–0.53 / – / 0.13 m |
| size | type average | type average | type average | measured | measured | measured |

**The 6-second clip, frames 010719–010778 (559 labelled object-frames), GSOC:**

| | before the depth fix | after it | **+ heading along the track, solid masks** |
| :--- | ---: | ---: | ---: |
| our objects / matched | 358 / 171 | 343 / 240 | 365 / 252 |
| recall / precision | 0.31 / 0.48 | 0.43 / 0.70 | **0.45 / 0.69** |
| median centre error | 1.49 m | 1.60 m | 1.55 m |
| median yaw error (mod 180°) | 8.2° | 7.0° | **5.1°** |
| median 2D box IoU | 0.62 | 0.71 | **0.72** |
| median size error l / w / h | 0.30 / 0.18 / 0.18 m | 0.46 / 0.21 / 0.17 m | 0.40 / 0.19 / 0.14 m |
| class agreement (car / large / two-wheeler / pedestrian) | 0.79 | 0.79 | 0.80 |
| along-ray bias of the cars at 60–80 m | +2.0 m | +0.7 m | |

**Heading along the track.** The track refinement took the heading from the tangent of the track's path; a far
car's centre jumps along the viewing ray from frame to frame, and the tangent with it (10–20° on the first 25
frames). The heading is now the flow's direction of motion averaged over ±3 frames of the track (modulo 180°).
**Solid masks:** an object's mask is the union of its segments, with gaps and holes (windows, lights that joined no
segment); it is now closed and its holes filled. The cuboid is fitted to the convex hull, which this does not change.

**The masks of the research code (`master_input.py`).** The research code's classical pipeline (`code/whole_scene.py`
with `SEG=mask`: median background, reflection cut by edge energy, flow and colour split, merge by one affine motion,
HSV shadow test, full resolution) gives better masks of single cars: against the labelled 2D boxes of moving cars
(frames 010719–010743) box IoU 0.44 against 0.35, found (IoU ≥ 0.5) 0.38 against 0.31, mask below the box (shadow,
reflection) 3.7 % against 5.3 %. It merges far cars more often and its thresholds (1.5 px/frame, 600 px) drop small far
objects. Combined with this module, on the 60 frames:

| combination | recall / precision | yaw | 2D box IoU | length error |
| :--- | ---: | ---: | ---: | ---: |
| its instances, lifted and split here | 0.29 / 0.73 (25 frames) | | | |
| its foreground into the segmentation here | 0.28 / 0.74 (25 frames) | | | |
| detection here, contours from it where one car (`CONTOURS=master`) | 0.40 / 0.62 | 5.0° | 0.69 | 0.46 m |
| **detection and contours here** (default) | **0.43 / 0.70** | **4.1°** | **0.71** | **0.42 m** |

Its contours are smoother, but the cuboids fitted to them are not better; the default keeps this module's. Tried and
dropped as well: cutting the dark, flat band at the bottom of a mask as shadow (precision 0.83 → 0.74 on 25 frames: a
car's own shadow under its body holds the tyre contact, and the cut lifts the box).

**The depth bias and its fix.** Before the fix, cars crossing the view at 60–80 m stood a median 2.1 m too far along
the viewing ray, growing over the clip to +4.0 m, and three quarters of the "false" detections were such cars 3–6 m
from their label. The crops showed boxes standing above the wheels: the contact-line cut, which removes a car's
mirror image on the wet road, found a "mirror" at a far car's sill and cut its wheels off, so the bottom tangents
touched the body and the box went back along the ray. A reflection is blurred, wheels and sills are sharp: the cut
now also needs the edges below the line to be at most 0.7 of those above (`MIRROR_BLUR`). On the last 15 frames the
bias of the detections fell from +3.5 to +1.1 m (with no cut at all: +0.7 m, but on the wet first frames the
reflection then pulls the boxes 1.35 m too near).

What is left, +1.5 to +2.3 m in the second half of the clip (−1.0 m in the first): about 0.35 m from the width along
the ray, measured 0.7 m too large (the size a high camera sees worst for a crossing car), the rest from cars whose
lower part is hidden behind a car in the near lane. Such tangents are now left out where the silhouette borders a
detected object in front (`relift_hidden`), but the car in front is often not detected itself. Tried and dropped: a
firmer width prior (weight 0.5): the late bias rose to +1.7 m and the boxes fitted worse.

**Against the LiDAR, as in the article's Table 1** (`eval_lidar.py`, written from the article's Sect. 5, not from its
code; `out/lidar_eval.json`). Per labelled car, its LiDAR points over the 60 frames in its box frame, mirrored, a
height field of 5 cm columns, gaps closed with a radius of 15 cm; a box is compared as the height field of its
footprint, both centred. Our cuboid of a car is the median of our cuboids matched to it (≤ 3 m, ≥ 3 frames). Cars of
intersection A within 80 m:

| solid IoU | raw | gap-closed |
| :--- | ---: | ---: |
| our classical cuboid (8 cars) | 0.36 | 0.48 |
| labelled box, same 8 cars | 0.41 | 0.57 |
| our cuboid, the 6 cars it gives a car's size | 0.40 | 0.52 |
| labelled box, the same 6 cars | 0.40 | 0.54 |
| *article, A (24 cars, whole sequence): labelled box / box of the shape model's size / shape model* | *0.35 / 0.35 / 0.41* | *0.55 / 0.59 / 0.65* |

Paired, gap-closed, ours − labelled box: −0.09 [−0.18, −0.01] on the 8 cars. Where the classical cuboid has a car's
size it is about as good a box as the labelled one (−0.02); the loss comes from the cars whose cuboid is too long (a
neighbour merged in, or a hidden part).
The sample is small and the protocol is a re-implementation; the shape models themselves are not compared here.

Remaining failure modes:

* **Hidden ground contact in dense far traffic:** see above.
* **Cars merged from the start of their tracks:** the white SUV and van are one object until their tracks, which
  begin where they drive apart, reach back (5 frames); before that they are one "bus".
* **Heading at the start of motion:** a car that is just starting keeps one frame's flow direction, which is noisy.
* **Stopped objects** have no flow and are not detected, by design.
* **Labels on the LiDAR's clock:** the labelled boxes trail the cars in the image by a fraction of a second along
  their motion, which adds to the measured centre error of every moving car.

The figures of the flow-only first version are in the commit that added this folder (`out/steps_flow_only.png`,
`out/frames_flow_only.png`).
