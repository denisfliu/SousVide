# Coordinate Transform Summary

This document describes the full chain of coordinate transforms used to convert drone positions and orientations from the NED simulation frame through to the Nerfstudio training frame used by the Gaussian splat renderer.

---

## Coordinate Frames

| Frame | Convention |
|---|---|
| **NED** | North-East-Down aviation convention. +X = North (forward), +Y = East (right), +Z = Down. Used internally by the SousVide simulator and MPC controller. |
| **MOCAP** | Real-world motion capture frame. +Z = Up (standard vision/Z-up convention). Waypoints and ground-truth positions are defined in this frame. |
| **COLMAP** | Photogrammetric reconstruction frame produced by COLMAP. Related to MOCAP via a Sim(3) alignment (scale + rotation + translation). |
| **Nerfstudio** | Normalized training frame used during Gaussian splat model training. Related to COLMAP via the `dataparser_transforms.json` file. |

---

## Stage 1 — NED Position → MOCAP Position

**Function:** `convert_from_ned_to_standard(pos_ned, permutation)`

The active permutation throughout this codebase is **permutation 5 (`mirror_y`)**.

For permutation 5, the mapping is:
```
x_mocap =  x_ned
y_mocap = -y_ned
z_mocap = -z_ned
```

This negates Z (reversing the down convention) and negates Y (mirroring across the XZ plane).

The inverse direction (MOCAP → NED, used when defining waypoints) is handled by `convert_to_ned_coordinates(pos, permutation=5)`:
```
x_ned =  x_mocap
y_ned = -y_mocap
z_ned = -z_mocap
```

---

## Stage 2 — NED Rotation → MOCAP Rotation

When transforming a full 4×4 body pose (not just a position), the rotation must also be re-expressed. A coordinate frame change matrix is built for permutation 5:

```
T_ned_to_mocap[:3, :3] = [[ 1,  0,  0],
                           [ 0, -1,  0],
                           [ 0,  0, -1]]
```

The rotation matrix is transformed via similarity:
```
R_mocap = T_ned_to_mocap @ R_ned @ T_ned_to_mocap.T
```

The full body-to-world pose in MOCAP is then:
```
Tb2w_mocap[:3, :3] = R_mocap
Tb2w_mocap[:3,  3] = pos_mocap   # from Stage 1
```

The NED body pose `Tb2w_ned` itself is obtained from the simulation state vector via `th.x_to_T(xcr)`, which extracts position from `state[0:3]` and builds a rotation matrix from the quaternion at `state[6:10]` (xyzw order, via scipy `Rotation.from_quat`).

---

## Stage 3 — MOCAP → COLMAP (Sim(3) Inverse)

**Class:** `CoordinateTransformer` in `coordinate_transform.py`  
**Data file:** `colmap_to_mocap_sim3.json` (loaded per scene)

The file stores the **forward** Sim(3) transform from COLMAP to MOCAP:
```
p_mocap = s * R @ p_colmap + t
```

The transformer pre-computes the **inverse** (MOCAP → COLMAP):
```
s_inv = 1 / s
R_inv = R^T
t_inv = -s_inv * R_inv @ t
```

**Position transform:**
```
p_colmap = s_inv * (R_inv @ p_mocap) + t_inv
```

**Full 4×4 pose transform** (`transformer.mocap_to_colmap_pose(T_mocap)`):
```
R_colmap = R_inv @ R_mocap
t_colmap = s_inv * (R_inv @ t_mocap) + t_inv
```

The scale `s_inv` is applied only to the translation, not the rotation. The scene used in this codebase is `left_gate`, which loads from:
```
/home/jatucker/data/data/left_gate_9_24_2025_alignment/colmap_to_mocap_sim3.json
```

---

## Stage 4 — COLMAP → Nerfstudio (Dataparser Transform)

**Data file:** `dataparser_transforms.json` (located inside the GSplat model directory)

The transform is loaded as a 3×4 matrix and padded to 4×4 by appending `[0, 0, 0, 1]` as the last row:

```python
dataparser_transform = np.eye(4)
dataparser_transform[:3, :] = np.array(dataparser_data['transform'])
```

It is applied as a left-multiplication:
```
T_nerf = dataparser_transform @ T_colmap
```

This encodes a rotation and translation (no scale in this matrix — the scale field in the JSON is handled separately inside `gsplat.py`). It is the same normalization that was applied to all training images during model training, so poses passed to the renderer must be in this frame.

---

## Full Chain

```
NED pose (Tb2w_ned)
    │
    │  T_ned_to_mocap = diag(1, -1, -1)
    │  pos:  x_m =  x_n,  y_m = -y_n,  z_m = -z_n
    │  rot:  R_m = T_ned_to_mocap @ R_n @ T_ned_to_mocap.T
    ▼
MOCAP pose (Tb2w_mocap)
    │
    │  Sim(3) inverse from colmap_to_mocap_sim3.json:
    │  R_c = R_inv @ R_m
    │  t_c = s_inv * (R_inv @ t_m) + t_inv
    ▼
COLMAP pose
    │
    │  dataparser_transform (4×4, from dataparser_transforms.json)
    │  T_nerf = dataparser_transform @ T_colmap
    ▼
Nerfstudio pose
```
