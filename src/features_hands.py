import numpy as np

DIST_PAIRS = [(4,8),(4,12),(8,12),(8,16),(0,9),(0,5),(0,17)]
ANGLE_TRIPLETS = [(6,5,7),(10,9,11),(14,13,15),(18,17,19),(3,2,4)]

def _angle(a,b,c):
    ba=a-b; bc=c-b
    cosang = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-9)
    cosang = np.clip(cosang,-1.0,1.0)
    return np.degrees(np.arccos(cosang))

def _normalize_xy(pts):
    pts = pts.astype(np.float32).copy()
    if pts.shape[1]==3: pts = pts[:,:2]
    wrist = pts[0]
    pts -= wrist
    scale = np.linalg.norm(pts[9]) + 1e-9
    pts /= scale
    return pts

def hand_features(landmarks21, use_z=False):
    if landmarks21 is None:
        base_len = 21*(3 if use_z else 2)
        return np.zeros(base_len + len(DIST_PAIRS) + len(ANGLE_TRIPLETS), np.float32)

    pts = landmarks21.astype(np.float32)
    has_z = (pts.shape[1]==3)
    z = pts[:,2].copy() if (has_z and use_z) else None
    xy = _normalize_xy(pts)

    coords = [xy]
    if z is not None:
        z = (z - z[0]) / (np.linalg.norm(xy[9]) + 1e-9)
        coords.append(z.reshape(-1,1))
    coords = np.concatenate(coords, axis=1).flatten()

    dists = [np.linalg.norm(xy[i]-xy[j]) for (i,j) in DIST_PAIRS]
    angs  = [_angle(xy[a], xy[c], xy[b])/180.0 for (c,a,b) in ANGLE_TRIPLETS]
    return np.concatenate([coords, np.array(dists, np.float32), np.array(angs, np.float32)]).astype(np.float32)

def concat_bihand_top_bottom(left_pts, right_pts, use_z=False):
    base_len = 21*(3 if use_z else 2) + len(DIST_PAIRS) + len(ANGLE_TRIPLETS)

    # 0 manos → ceros
    if left_pts is None and right_pts is None:
        return np.zeros(base_len*2, np.float32)

    # 1 mano → TOP = esa mano, BOTTOM = ceros
    if left_pts is None:
        f_top = hand_features(right_pts, use_z)
        f_bot = np.zeros(base_len, np.float32)
        return np.concatenate([f_top, f_bot]).astype(np.float32)
    if right_pts is None:
        f_top = hand_features(left_pts, use_z)
        f_bot = np.zeros(base_len, np.float32)
        return np.concatenate([f_top, f_bot]).astype(np.float32)

    # 2 manos → ordenar por y del wrist (menor y = más arriba)
    yL, yR = float(left_pts[0,1]), float(right_pts[0,1])
    top_pts, bot_pts = (left_pts, right_pts) if yL < yR else (right_pts, left_pts)

    f_top = hand_features(top_pts, use_z)
    f_bot = hand_features(bot_pts, use_z)
    return np.concatenate([f_top, f_bot]).astype(np.float32)

# Alias estable (SIN pasar use_z desde el llamador, si preferís)
def concat_bihand(left_pts, right_pts):
    return concat_bihand_top_bottom(left_pts, right_pts, use_z=False)
