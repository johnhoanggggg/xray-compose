#!/usr/bin/env python3
"""Trace the bodies in embrace photos as floating veins on a white background.

For each photo in data/hugs/:

1. MediaPipe's multiclass selfie segmenter marks the people: hair, skin and
   clothes, but not background.
2. MediaPipe's pose landmarker finds the shoulders, elbows, wrists, hips,
   knees, ankles, ears and mouth of each person it can see. Close hugs hide
   much of this, so the pose is used wherever it is found and nowhere else.
3. From the landmarks, the main named veins and nerves are laid down along
   their real paths: venae cavae, jugulars, subclavian, cephalic, basilic and
   median cubital, iliac, femoral, great and small saphenous, facial and
   superficial temporal veins; spinal cord, median, ulnar, radial, sciatic,
   femoral and facial nerves.
4. Space colonisation grows the smaller veins (and a sparser web of nerves)
   out from those trunks until they fill the silhouette. Where no pose was
   found, the growth starts from a seed near each person's chest.
5. Vessel width follows Murray's law: each segment's width grows with the
   number of branch tips it drains, so the trunks come out thick and the
   tips hair-thin.

Each person's veins get their own colour, so you can see where the two
networks meet. The result is drawn on white with a soft drop shadow.

    python3 tools/veins.py                 # every photo in data/hugs/
    python3 tools/veins.py data/hugs/x.jpg --seed 3 --no-nerves
"""
import argparse, glob, os
import numpy as np
import cv2
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "out", "veins")

LONG = 1400        # working resolution, long side in px
SS = 3             # supersampling factor for drawing
STEP = 4.0         # growth step in px
VEIN_COLORS = [(148, 22, 44), (24, 44, 112), (30, 96, 88), (112, 50, 120)]  # RGB per person
NERVE_COLOR = (212, 160, 60)


# ---------------------------------------------------------------- perception

def load_models():
    from mediapipe.tasks.python import vision, BaseOptions
    m = os.path.join(DATA, "models")
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(m, "pose_landmarker_heavy.task")),
        num_poses=4, min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3))
    seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(m, "selfie_multiclass_256x256.tflite")),
        output_confidence_masks=True))
    face = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(m, "blaze_face_short_range.tflite")),
        min_detection_confidence=0.4))
    return pose, seg, face


def perceive(models, rgb):
    """Return (body mask, face mask, list of poses). A pose is a (33, 3) array of x, y, visibility.

    The pose landmarker struggles with tight hugs: it often reports one person
    twice, or one skeleton stretched across both bodies. The face detector is
    far more reliable, so every face becomes a person. A pose is only trusted
    when its nose lands on a face and its shoulders fit that face's size; a
    face with no trusted pose gets a head-and-shoulders pose built from the
    face alone.
    """
    import mediapipe as mp
    pose, seg, facedet = models
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    conf = np.stack([np.squeeze(m.numpy_view()) for m in seg.segment(img).confidence_masks])
    label = conf.argmax(0)
    body = np.isin(label, (1, 2, 3, 4)).astype(np.uint8)     # hair, skin, clothes: the whole silhouette
    face = (label == 3).astype(np.uint8)
    body = cv2.morphologyEx(body, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    body = drop_small(body, 0.004)

    h, w = body.shape
    found = []
    for lm in pose.detect(img).pose_landmarks:
        p = np.array([(q.x * w, q.y * h, q.visibility) for q in lm], np.float32)
        if p[[11, 12], 2].min() > 0.5:
            found.append(p)
    found.sort(key=lambda p: -p[:25, 2].mean())

    poses = []
    for d in facedet.detect(img).detections:
        b = d.bounding_box
        x0, y0, s = b.origin_x, b.origin_y, max(b.width, b.height)
        match = None
        for p in found:
            sw = np.linalg.norm(p[11, :2] - p[12, :2])
            nx, ny, nv = p[0]
            if nv > 0.5 and x0 - 0.3 * s < nx < x0 + 1.3 * s and y0 - 0.3 * s < ny < y0 + 1.3 * s and 1.2 * s < sw < 4.5 * s:
                match = p
                break
        if match is not None:
            found = [p for p in found if p is not match]
            poses.append(match)
        else:
            poses.append(pose_from_face(d, w, h))
    if not poses:       # nobody's face is visible: fall back to distinct confident poses
        for p in found:
            sw = np.linalg.norm(p[11, :2] - p[12, :2])
            if all(np.linalg.norm((p[11, :2] + p[12, :2]) - (q[11, :2] + q[12, :2])) > sw for q in poses):
                poses.append(p)
    return body, face, poses[:4]


def pose_from_face(det, w, h):
    """A head-and-shoulders pose (MediaPipe pose indices) estimated from a face detection."""
    k = np.array([(q.x * w, q.y * h) for q in det.keypoints])     # r eye, l eye, nose, mouth, r ear, l ear
    s = max(det.bounding_box.width, det.bounding_box.height)
    eyes = (k[0] + k[1]) / 2
    down = k[3] - eyes
    down /= np.linalg.norm(down) + 1e-6
    left = np.array([down[1], -down[0]])              # toward the subject's left, perpendicular to down
    if np.dot(left, k[1] - k[0]) < 0:
        left = -left
    neck = k[3] + down * 0.95 * s
    p = np.zeros((33, 3), np.float32)
    for i, xy in ((0, k[2]), (2, k[1]), (5, k[0]), (7, k[5]), (8, k[4]),
                  (9, k[3] + left * 0.15 * s), (10, k[3] - left * 0.15 * s),
                  (11, neck + left * 1.25 * s + down * 0.25 * s), (12, neck - left * 1.25 * s + down * 0.25 * s)):
        p[i] = (xy[0], xy[1], 1.0)
    return p


def drop_small(mask, frac):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= frac * mask.size
    return keep[lab].astype(np.uint8)


# ---------------------------------------------------------------- anatomy

def anatomy(p, body):
    """Named veins and nerves for one pose, as lists of (polyline, start width, end width)."""
    vis = lambda *ix: all(p[i, 2] > 0.5 for i in ix)
    P = lambda i: p[i, :2].astype(np.float64)
    lerp = lambda a, b, t: a + (b - a) * t

    def perp(a, b, s):
        d = b - a
        n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-6)
        return n * s * np.linalg.norm(d)

    if not vis(11, 12):
        return [], []
    ls, rs = P(11), P(12)
    sw = np.linalg.norm(ls - rs)
    neck = (ls + rs) / 2
    down = np.array([0.0, 1.0])
    if vis(0):
        v = neck - P(0)
        if np.linalg.norm(v) > 1:
            down = v / np.linalg.norm(v)
    hips = vis(23, 24)
    pelvis = (P(23) + P(24)) / 2 if hips else neck + down * sw * 1.5
    heart = lerp(neck, pelvis, 0.28) + (ls - rs) * 0.08      # a little to the person's left
    head = P(0) if vis(0) else neck - down * sw * 0.6

    veins, nerves = [], []
    V = lambda pts, w0, w1: veins.append((np.array(pts), w0, w1))
    N = lambda pts, w0, w1: nerves.append((np.array(pts), w0, w1))

    # venae cavae and the neck
    bl, br = lerp(neck, ls, 0.22) + down * sw * 0.08, lerp(neck, rs, 0.22) + down * sw * 0.08
    V([heart, lerp(heart, neck, 0.6), bl], 7.0, 5.5)                 # SVC, left brachiocephalic
    V([lerp(heart, neck, 0.6), br], 5.5, 5.0)                        # right brachiocephalic
    for side, b, ear, mouth in ((0, bl, 7, 9), (1, br, 8, 10)):
        top = lerp(P(ear), P(mouth), 0.35) if vis(ear, mouth) else lerp(neck, head, 0.7) + (ls - rs) * (0.18 if side == 0 else -0.18)
        V([b, lerp(b, top, 0.5) + (ls - rs) * (0.03 if side == 0 else -0.03), top], 4.0, 2.4)   # internal jugular
        V([lerp(b, P(11 + side), 0.35), lerp(b, top, 0.55) + (ls - rs) * (0.12 if side == 0 else -0.12), top], 2.0, 1.2)  # external jugular
        if vis(ear, mouth):
            V([top, lerp(P(mouth), P(2 + 3 * side), 0.6)], 1.6, 0.8)   # facial vein toward the eye
            V([P(ear), P(ear) - down * sw * 0.35], 1.4, 0.6)            # superficial temporal
    V([heart, pelvis], 6.5, 5.5)                                       # IVC

    # arms
    for sh, el, wr, thumb, pinky, b in ((11, 13, 15, 21, 17, bl), (12, 14, 16, 22, 18, br)):
        if not vis(sh, el):
            continue
        S, E = P(sh), P(el)
        axilla = lerp(S, pelvis, 0.12)
        V([b, lerp(b, S, 0.6) + down * sw * 0.05, axilla], 4.5, 3.6)   # subclavian, axillary
        V([axilla, lerp(axilla, E, 0.5) + perp(S, E, 0.04), E + perp(S, E, 0.08)], 3.4, 2.4)   # basilic / brachial
        V([S + perp(S, E, -0.05), lerp(S, E, 0.5) + perp(S, E, -0.09), E + perp(S, E, -0.1)], 2.6, 2.0)  # cephalic
        V([E + perp(S, E, -0.1), E + perp(S, E, 0.08)], 1.8, 1.8)       # median cubital
        if vis(wr):
            W = P(wr)
            rad = P(thumb) if vis(thumb) else W + perp(E, W, -0.15)
            uln = P(pinky) if vis(pinky) else W + perp(E, W, 0.15)
            V([E + perp(E, W, -0.1), lerp(E, W, 0.5) + perp(E, W, -0.1), W + perp(E, W, -0.08), rad], 2.0, 1.0)   # cephalic, forearm
            V([E + perp(E, W, 0.08), lerp(E, W, 0.5) + perp(E, W, 0.1), W + perp(E, W, 0.08), uln], 2.0, 1.0)    # basilic, forearm
            mid = (rad + uln) / 2
            N([lerp(neck, S, 0.4), S + perp(S, E, 0.02), E, W, mid + (mid - W) * 0.6], 1.6, 0.8)    # median nerve
            N([S, E + perp(S, E, 0.13), W + perp(E, W, 0.1), uln + (uln - W) * 0.5], 1.2, 0.6)      # ulnar nerve
            N([S, lerp(S, E, 0.5) + perp(S, E, -0.1), E + perp(S, E, -0.12), W + perp(E, W, -0.1), rad + (rad - W) * 0.5], 1.2, 0.6)  # radial nerve
        else:
            N([lerp(neck, S, 0.4), S, E], 1.6, 1.0)

    # legs
    if hips:
        for hip, kn, an, foot in ((23, 25, 27, 31), (24, 26, 28, 32)):
            H = P(hip)
            V([pelvis, lerp(pelvis, H, 0.6) + down * sw * 0.1, H + down * sw * 0.15], 5.0, 4.2)   # common/external iliac
            N([pelvis, lerp(pelvis, H, 0.5), H], 1.6, 1.4)
            if not vis(kn):
                continue
            K = P(kn)
            med = 1 if hip == 23 else -1
            V([H + down * sw * 0.15, lerp(H, K, 0.5), K], 4.2, 3.0)                              # femoral / popliteal
            V([H + down * sw * 0.15, lerp(H, K, 0.5) + perp(H, K, 0.08 * med), K + perp(H, K, 0.1 * med)], 2.4, 2.0)  # great saphenous
            N([H, lerp(H, K, 0.5) + perp(H, K, -0.04 * med), K], 2.0, 1.6)                       # sciatic
            N([H + perp(H, K, 0.05 * med), lerp(H, K, 0.3) + perp(H, K, 0.06 * med)], 1.2, 0.8)  # femoral nerve
            if vis(an):
                A = P(an)
                V([K + perp(H, K, 0.1 * med), lerp(K, A, 0.5) + perp(K, A, 0.09 * med), A + perp(K, A, 0.1 * med)], 2.0, 1.4)   # great saphenous, calf
                V([K, lerp(K, A, 0.5) + perp(K, A, -0.06 * med), A + perp(K, A, -0.08 * med)], 1.6, 1.0)                        # small saphenous
                N([K, lerp(K, A, 0.5), A, P(foot) if vis(foot) else A + down * sw * 0.2], 1.4, 0.8)                              # tibial nerve

    # spinal cord and facial nerve
    N([lerp(neck, head, 0.45), neck, lerp(neck, pelvis, 0.5), pelvis], 2.2, 1.6)
    for ear, mouth, eye in ((7, 9, 2), (8, 10, 5)):
        if vis(ear, mouth, eye):
            for tgt in (P(eye), lerp(P(eye), P(mouth), 0.5), P(mouth)):
                N([P(ear), lerp(P(ear), tgt, 0.85)], 1.0, 0.5)
    return veins, nerves


# ---------------------------------------------------------------- vessel trees

class Tree:
    def __init__(self):
        self.pos, self.parent, self.base, self.owner = [], [], [], []

    def add(self, xy, parent, base, owner):
        self.pos.append(xy); self.parent.append(parent); self.base.append(base); self.owner.append(owner)
        return len(self.pos) - 1

    def arrays(self):
        return (np.array(self.pos, np.float64).reshape(-1, 2), np.array(self.parent, np.int64),
                np.array(self.base, np.float64), np.array(self.owner, np.int64))


def resample(pts, step):
    """Smooth a polyline with Chaikin, then resample it at a fixed step."""
    for _ in range(3):
        q = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            q += [0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b]
        q.append(pts[-1])
        pts = np.array(q)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.arange(0, s[-1] + 1e-6, step)
    return np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], 1), t / max(s[-1], 1e-6)


def wobble(pts, rng, amount):
    """Bend a resampled polyline sideways with low-frequency noise, so trunks are not ruler-straight."""
    if len(pts) < 4:
        return pts
    d = np.gradient(pts, axis=0)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1) / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    k = np.linspace(0, 1, len(pts))
    off = sum(rng.normal(0, amount / f) * np.sin(np.pi * f * k + rng.random() * 6.28) for f in (1, 2, 3, 5))
    off *= np.sin(np.pi * k) ** 0.5       # keep both ends where the anatomy puts them
    return pts + nrm * off[:, None]


def lay_paths(tree, paths, owner, inside, rng):
    """Add anatomical polylines to the tree. Each attaches to the nearest node of the same owner."""
    for pts, w0, w1 in paths:
        pts, t = resample(pts, STEP)
        pts = wobble(pts, rng, 0.05 * len(pts) * STEP)
        pos, _, _, own = tree.arrays()
        mine = np.where(own == owner)[0]
        if len(mine):
            d = np.linalg.norm(pos[mine] - pts[0], axis=1)
            parent = int(mine[d.argmin()]) if d.min() < 40 else -1
        else:
            parent = -1
        for xy, tt in zip(pts, t):
            if not inside(xy):       # skip the part that leaves the body; rejoin where it comes back
                parent = None
                continue
            if parent is None:
                pos, _, _, own = tree.arrays()
                mine = np.where(own == owner)[0]
                d = np.linalg.norm(pos[mine] - xy, axis=1) if len(mine) else np.array([np.inf])
                parent = int(mine[d.argmin()]) if d.min() < 3 * STEP else -1
            parent = tree.add(xy, parent, (w0 + (w1 - w0) * tt) * 0.55, owner)


def chest_seeds(body, face, poses):
    """Roots for bodies no pose reached: below each skin-coloured face when there are no poses
    at all, and at the widest point of any piece of body still without a root."""
    dt = cv2.distanceTransform(body, cv2.DIST_L2, 5)
    h, w = body.shape
    taken = [((p[11, :2] + p[12, :2]) / 2) for p in poses if p[[11, 12], 2].min() > 0.5]
    heads = [p[i, :2] for p in poses for i in (0, 7, 8) if p[i, 2] > 0.5]
    seeds = []
    n, lab, stats, cent = cv2.connectedComponentsWithStats(drop_small(face, 0.002))
    faces = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:4] if not poses else []
    for i in faces:
        cx, cy = cent[i]
        fh = stats[i, cv2.CC_STAT_HEIGHT]
        x, y, bw, bh = stats[i, :4]
        if any(x - 0.2 * bw <= hx <= x + 1.2 * bw and y - 0.2 * bh <= hy <= y + 1.2 * bh for hx, hy in heads):
            continue
        y0, y1 = int(min(h - 1, cy + 0.8 * fh)), int(min(h, cy + 2.8 * fh))
        x0, x1 = int(max(0, cx - 1.5 * fh)), int(min(w, cx + 1.5 * fh))
        win = dt[y0:y1, x0:x1]
        if win.size and win.max() > 0:
            yy, xx = np.unravel_index(win.argmax(), win.shape)
            seeds.append(np.array([x0 + xx, y0 + yy], np.float64))
        elif body[int(cy), int(cx)]:
            seeds.append(np.array([cx, cy], np.float64))
    # every sizable piece of body needs a root somewhere
    n, lab = cv2.connectedComponents(body)
    for i in range(1, n):
        m = lab == i
        pts = taken + seeds
        if any(m[int(min(h - 1, max(0, y))), int(min(w - 1, max(0, x)))] for x, y in pts):
            continue
        d = np.where(m, dt, 0)
        yy, xx = np.unravel_index(d.argmax(), d.shape)
        seeds.append(np.array([xx, yy], np.float64))
    return seeds


def colonise(tree, body, spacing, influence, kill, rng, edge=True, momentum=0.8, max_iter=800):
    """Grow the tree toward attractor points scattered through the body mask."""
    h, w = body.shape
    ys, xs = np.nonzero(body)
    n = int(len(xs) / spacing ** 2)
    pick = rng.choice(len(xs), n, replace=False)
    att = np.stack([xs[pick], ys[pick]], 1).astype(np.float64) + rng.random((n, 2))
    if edge:   # a denser band of attractors just inside the outline traces the silhouette
        inner = cv2.erode(body, np.ones((5, 5), np.uint8))
        cs, _ = cv2.findContours(inner, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        ring = np.concatenate([c[::max(1, int(spacing * 1.2)), 0] for c in cs if len(c) > 20]).astype(np.float64)
        att = np.concatenate([att, ring])
    inside = lambda q: body[np.clip(q[:, 1].astype(int), 0, h - 1), np.clip(q[:, 0].astype(int), 0, w - 1)] > 0

    pos, parent, base, owner = tree.arrays()
    if len(pos) == 0:
        return
    d, _ = cKDTree(pos).query(att)
    att = att[d > kill]
    for _ in range(max_iter):
        if len(att) == 0:
            break
        kd = cKDTree(pos)
        d, idx = kd.query(att, distance_upper_bound=influence)
        ok = np.isfinite(d)
        if not ok.any():
            break
        v = att[ok] - pos[idx[ok]]
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        acc = np.zeros_like(pos)
        np.add.at(acc, idx[ok], v)
        grow = np.unique(idx[ok])
        dirs = acc[grow]
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9
        has = parent[grow] >= 0     # carry on roughly the way the vessel was already heading
        prev = pos[grow[has]] - pos[parent[grow[has]]]
        dirs[has] += momentum * prev / (np.linalg.norm(prev, axis=1, keepdims=True) + 1e-9)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9
        dirs += rng.normal(0, 0.3, dirs.shape)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9
        new = pos[grow] + STEP * dirs
        good = inside(new)
        dn, _ = kd.query(new)
        good &= dn > STEP * 0.55
        if not good.any():
            break
        new, grow = new[good], grow[good]
        pos = np.concatenate([pos, new])
        parent = np.concatenate([parent, grow])
        base = np.concatenate([base, np.zeros(len(new))])
        owner = np.concatenate([owner, owner[grow]])
        dk, _ = cKDTree(new).query(att)
        att = att[dk > kill]
    tree.pos, tree.parent, tree.base, tree.owner = list(pos), list(parent), list(base), list(owner)


def widths(parent, base, tip, gain, exponent, cap):
    """Murray's law: width grows with the number of tips drained, raised to 1/exponent."""
    n = len(parent)
    tips = np.ones(n)
    for i in range(n - 1, -1, -1):           # children always come after their parent
        if parent[i] >= 0:
            tips[parent[i]] += tips[i]
    return np.maximum(base, np.minimum(cap, tip + gain * (tips ** (1 / exponent) - 1)))


# ---------------------------------------------------------------- drawing

def draw(trees, shape):
    """trees: list of (pos, parent, width, rgb per node). Returns an RGB image on white."""
    h, w = shape
    H, W = h * SS, w * SS
    col = np.zeros((H, W, 3), np.float32)
    alpha = np.zeros((H, W), np.float32)
    segs = []
    for pos, parent, wid, rgb in trees:
        for i in np.where(parent >= 0)[0]:
            segs.append((wid[i], pos[parent[i]], pos[i], rgb[i]))
    segs.sort(key=lambda s: s[0])
    for wd, a, b, c in segs:
        t = max(1, int(round(wd * SS)))
        pa = tuple(int(round(v * SS)) for v in a)
        pb = tuple(int(round(v * SS)) for v in b)
        cv2.line(col, pa, pb, tuple(float(x) for x in c), t, cv2.LINE_AA)
        cv2.line(alpha, pa, pb, 1.0, t, cv2.LINE_AA)
    col = cv2.resize(col, (w, h), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_AREA)[..., None]
    shadow = cv2.GaussianBlur(alpha[..., 0], (0, 0), 5)
    shadow = cv2.warpAffine(shadow, np.float32([[1, 0, 6], [0, 1, 9]]), (w, h))[..., None] * 0.22
    out = np.full((h, w, 3), 255, np.float32) * (1 - shadow) + np.array([150, 150, 165], np.float32) * shadow
    out = out * (1 - alpha) + col * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def shade(owner, wid, palette, wmax):
    """Thick vessels in the full colour, thin ones paler."""
    base = np.array(palette, np.float32)[owner % len(palette)]
    t = np.clip(wid / wmax, 0, 1)[:, None] ** 0.5
    pale = base + (255 - base) * 0.45
    return pale + (base - pale) * t


# ---------------------------------------------------------------- main

def render(models, path, seed, nerves_on):
    rng = np.random.default_rng(seed)
    bgr = cv2.imread(path)
    s = LONG / max(bgr.shape[:2])
    bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    body, face, poses = perceive(models, rgb)
    h, w = body.shape
    loose = cv2.dilate(body, np.ones((9, 9), np.uint8))
    inside = lambda q: loose[int(np.clip(q[1], 0, h - 1)), int(np.clip(q[0], 0, w - 1))] > 0

    poses.sort(key=lambda p: (p[11, 0] + p[12, 0]) / 2)      # left to right, for stable colours
    vt, nt = Tree(), Tree()
    owner = 0
    for p in poses:
        v, n = anatomy(p, body)
        if not v:
            continue
        lay_paths(vt, v, owner, inside, rng)
        lay_paths(nt, n, owner, inside, rng)
        owner += 1
    for xy in chest_seeds(body, face, poses):
        vt.add(xy, -1, 0.0, owner)
        nt.add(xy, -1, 0.0, owner)
        owner += 1

    scale = np.sqrt(body.sum()) / 1000            # bigger bodies in frame get coarser branching
    k = max(scale, 0.6)
    # coarse pass: a few long tributaries that give the network its hierarchy
    n0 = len(vt.pos)
    colonise(vt, body, spacing=38 * k, influence=170 * k, kill=34 * k, rng=rng, edge=False, momentum=1.2)
    pos, parent, base, own = vt.arrays()
    coarse = widths(parent, base, tip=0.9, gain=0.55, exponent=2.4, cap=4.0)
    vt.base = list(np.where(np.arange(len(pos)) >= n0, coarse, base))
    # fine pass: twigs that fill the silhouette
    colonise(vt, body, spacing=9 * k, influence=45 * k, kill=10, rng=rng)
    pos, parent, base, own = vt.arrays()
    vw = widths(parent, base, tip=0.4, gain=0.22, exponent=2.8, cap=6.0)
    trees = [(pos, parent, vw, shade(own, vw, VEIN_COLORS, 5.0))]
    if nerves_on:
        colonise(nt, body, spacing=26 * max(scale, 0.6), influence=110 * max(scale, 0.6), kill=14, rng=rng, edge=False)
        pos, parent, base, own = nt.arrays()
        nw = widths(parent, base, tip=0.35, gain=0.12, exponent=3.0, cap=2.4)
        rgb_n = np.tile(np.array(NERVE_COLOR, np.float32), (len(pos), 1))
        trees.insert(0, (pos, parent, nw, rgb_n))
    out = draw(trees, (h, w))
    return rgb, out, len(poses)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("images", nargs="*")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-nerves", action="store_true")
    a = ap.parse_args()
    images = a.images or sorted(glob.glob(os.path.join(DATA, "hugs", "*.jpg")))
    os.makedirs(OUT, exist_ok=True)
    models = load_models()
    for f in images:
        name = os.path.splitext(os.path.basename(f))[0]
        photo, veins, npose = render(models, f, a.seed, not a.no_nerves)
        cv2.imwrite(os.path.join(OUT, name + ".png"), cv2.cvtColor(veins, cv2.COLOR_RGB2BGR))
        pair = np.concatenate([photo, np.full((photo.shape[0], 24, 3), 255, np.uint8), veins], 1)
        pair = cv2.resize(pair, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(OUT, name + "_pair.jpg"), cv2.cvtColor(pair, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print("wrote", os.path.join(OUT, name + ".png"), f"({npose} poses)")


if __name__ == "__main__":
    main()
