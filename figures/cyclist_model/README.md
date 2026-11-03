# Cyclist 003921 (V2X-Seq-SPD seq_0059): carve + triangulated surface

Chosen model (2026-11-02): the silhouette carve bounded by the surface triangulated over the whole pass, mirrored.

Rebuild (from code/3dseg, GPU venv for SAM):

    V2X_SEQ_DIR=<dataset>/spd_seq/seq_0059 SHOW_K=51 ~/.venvs/gpu/bin/python ../../RoadsideCarShape/scripts/vru_lidar.py OUT 003921
    V2X_SEQ_DIR=<dataset>/spd_seq/seq_0059 python3 ../../RoadsideCarShape/scripts/tri_vru.py OUT/vru_003921.pkl OUT/tri_003921.png
    OCC_OVERRIDE=OUT/tri_003921_occ.npy V2X_SEQ_DIR=... SHOW_K=51 ~/.venvs/gpu/bin/python ../../RoadsideCarShape/scripts/vru_lidar.py OUT2 003921
    python3 ../../RoadsideCarShape/scripts/turn.py turntable.jpg "label" OUT/vru_003921.pkl "label" OUT2/vru_003921.pkl

Numbers: 61 masks (SAM 2.1 on a crop + torso point), rigid camera (CAM_ORTHO); 774 tracks, 546 rigid points
(median reprojection 0.69 px, median 9 cm from its LiDAR); 37 % of the side-plane cells measured, the rest nearest
within 0.2 m or a 0.1 m thin prior; size 1.60 x 0.78 x 1.56 m (LiDAR box 1.63 x 0.67 x 1.50); IoU vs its LiDAR 0.640
(carve alone 0.636, 40 % more volume).

Files: vru_003921.pkl (mesh V/F/col, occupancy, poses, masks), tri_003921_s12_occ.npy (the bounded occupancy),
cyclist_turn.jpg (8 views, carve vs carve + surface), vru_003921.jpg (frame, mask, LiDAR box, 3 views),
tri_003921_s12.png (the triangulated points vs the LiDAR).
