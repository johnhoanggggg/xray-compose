#!/usr/bin/env python3
"""Trace the bodies in embrace photos as floating veins on a white background.

For each photo in data/hugs/:

1. People. YOLO11 instance segmentation gives one mask per person, and
   YOLO11 pose gives each person's 17 COCO keypoints. A pose belongs to the
   mask that holds most of its keypoints.
2. 3D blockout. Each person is blocked out as capsules: a head, a neck, a
   torso (three capsules side by side), upper arms, forearms, hands, thighs
   and shins. Radii come from body proportions scaled to the shoulder width.
   A part's depth comes from how confidently its joints were seen: YOLO
   reports occluded joints with low confidence, so those parts sit further
   back. A person with no usable pose becomes a single body part.
3. Visibility. The capsules are rasterised into a z-buffer inside each
   person's mask. Each pixel keeps only the frontmost part. Pixels where
   another person's mask is in front belong to that person, so the hidden
   stretches of a limb are never used. Mask pixels that no capsule covers
   join the nearest part.
4. Centre veins. Each part's axis (the line through its centre of mass) is
   kept only where that part is the visible one. A visible piece the axis
   never crosses gets its principal axis instead.
5. Child veins. A Gray-Scott reaction-diffusion labyrinth grows out from the
   centre veins to fill each visible part. Narrow walls between parts keep
   the forms apart, and the body outline is a no-flux boundary so it never
   seeds stripes of its own. Short stubs one period apart along each centre
   vein start the child veins off at right angles. Diffusion is stronger
   along each part's axis than across it, which turns the stripes across the
   axis, so they branch out sideways instead of running alongside the vein.
6. Equal widths. The labyrinth is thinned to centre lines. The period it
   actually grew is the body area divided by the total line length, and each
   line is redrawn at exactly half that, so every child vein is as wide as
   the gap beside it. Centre veins are drawn 30% wider. Child veins
   that end next to a centre vein are joined onto it.

Each person's veins get their own colour, and the drawing gets a soft drop
shadow so it reads as floating on the white.

    python3 tools/veins.py                       # every photo in data/hugs/
    python3 tools/veins.py data/hugs/x.jpg --period 20 --debug
"""
import argparse, glob, os, time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage.morphology import skeletonize

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "out", "veins")

LONG = 1400                    # working resolution, long side in px
NATURAL_PERIOD = 19.0          # rough stripe period of the labyrinth below on its own grid, in px
FEED, KILL = 0.029, 0.057      # Gray-Scott parameters in the labyrinth regime
CENTRE_SCALE = 1.3             # centre veins are 30% wider than child veins
COLORS = [(148, 22, 44), (24, 44, 112), (30, 96, 88), (112, 50, 120)]   # RGB per person
KP_OK = 0.4

# COCO keypoints
NOSE, LEYE, REYE, LEAR, REAR = 0, 1, 2, 3, 4
LSH, RSH, LEL, REL, LWR, RWR, LHIP, RHIP, LKN, RKN, LAN, RAN = range(5, 17)


# ---------------------------------------------------------------- perception

def load_models():
    from ultralytics import YOLO
    m = os.path.join(DATA, "models")
    return YOLO(os.path.join(m, "yolo11x-seg.pt")), YOLO(os.path.join(m, "yolo11x-pose.pt"))


def perceive(models, bgr):
    """Return a person-label image (-1 = background) and one (17, 3) keypoint array or None per person."""
    seg, pose = models
    h, w = bgr.shape[:2]
    rs = seg(bgr, verbose=False, conf=0.25, classes=[0], retina_masks=True)[0]
    if rs.masks is None:
        return np.full((h, w), -1, np.int32), []
    masks = [m > 0.5 for m in rs.masks.data.cpu().numpy()]
    masks = [m for m in masks if m.sum() > 0.02 * m.size]
    if not masks:
        return np.full((h, w), -1, np.int32), []
    # where masks overlap, a pixel goes to the mask it sits deepest inside
    depth = np.stack([cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5) for m in masks])
    label = np.where(depth.max(0) > 0, depth.argmax(0), -1).astype(np.int32)
    label = clean_labels(label, len(masks))

    kps = pose(bgr, verbose=False, conf=0.25)[0].keypoints
    kps = kps.data.cpu().numpy() if kps is not None else np.zeros((0, 17, 3))
    owner = [None] * len(masks)
    score = np.zeros((len(kps), len(masks)))
    for i, k in enumerate(kps):
        ok = k[:, 2] > KP_OK
        xy = k[ok, :2].astype(int).clip(0, [w - 1, h - 1])
        for j in range(len(masks)):
            score[i, j] = (label[xy[:, 1], xy[:, 0]] == j).sum() if len(xy) else 0
    while score.size and score.max() >= 3:           # greedy one-to-one matching
        i, j = np.unravel_index(score.argmax(), score.shape)
        owner[j] = kps[i]
        score[i, :] = 0
        score[:, j] = 0
    return label, owner


def clean_labels(label, n):
    """Drop specks: keep only each person's sizable pieces."""
    out = np.full_like(label, -1)
    for j in range(n):
        m = (label == j).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        k, lab, st, _ = cv2.connectedComponentsWithStats(m)
        keep = [i for i in range(1, k) if st[i, cv2.CC_STAT_AREA] > 0.004 * m.size]
        out[np.isin(lab, keep) & (out < 0)] = j
    return out


# ---------------------------------------------------------------- 3D blockout

class Part:
    def __init__(self, name, capsules, axis, conf):
        self.name = name
        self.capsules = capsules        # list of (a, b, radius) in image px
        self.axis = axis                # (a, b): the centre-of-mass line
        self.conf = conf
        self.depth = 0.0


def blockout(k, region):
    """Capsule parts for one person from COCO keypoints, or None if the shoulders weren't seen."""
    ok = lambda *ix: all(k[i, 2] > KP_OK for i in ix)
    P = lambda i: k[i, :2].astype(np.float64)
    c = lambda *ix: float(np.mean([k[i, 2] for i in ix]))
    if not ok(LSH, RSH):
        return None
    sm = (P(LSH) + P(RSH)) / 2
    sw = np.linalg.norm(P(LSH) - P(RSH))
    face = [i for i in (NOSE, LEYE, REYE, LEAR, REAR) if ok(i)]
    sw = max(sw, 0.25 * np.sqrt(region.sum()))       # a shoulder line seen side-on is too short
    up = (np.mean([P(i) for i in face], 0) - sm) if face else np.array([0.0, -1.0])
    up /= np.linalg.norm(up) + 1e-9
    hm = (P(LHIP) + P(RHIP)) / 2 if ok(LHIP, RHIP) else sm - up * 1.6 * sw
    parts = []

    if face:
        hr = 0.3 * sw
        hc = np.mean([P(i) for i in face], 0)
        top, chin = hc + up * 0.55 * hr, hc - up * 0.7 * hr
        parts.append(Part("head", [(top, chin, hr)], (top, chin), c(*face)))
        parts.append(Part("neck", [(sm, chin, 0.14 * sw)], (sm, chin), c(*face, LSH, RSH)))
    tc = c(LSH, RSH, LHIP, RHIP) if ok(LHIP, RHIP) else c(LSH, RSH) * 0.8
    lh = P(LHIP) if ok(LHIP) else hm + (P(LSH) - sm) * 0.7
    rh = P(RHIP) if ok(RHIP) else hm + (P(RSH) - sm) * 0.7
    parts.append(Part("torso", [(sm, hm, 0.3 * sw), (P(LSH), lh, 0.2 * sw), (P(RSH), rh, 0.2 * sw)], (sm, hm), tc))

    for side, (sh, el, wr) in (("l", (LSH, LEL, LWR)), ("r", (RSH, REL, RWR))):
        if ok(el):
            parts.append(Part(side + "upperarm", [(P(sh), P(el), 0.13 * sw)], (P(sh), P(el)), c(sh, el)))
            if ok(wr):
                parts.append(Part(side + "forearm", [(P(el), P(wr), 0.1 * sw)], (P(el), P(wr)), c(el, wr)))
                hand = P(wr) + (P(wr) - P(el)) * 0.3
                parts.append(Part(side + "hand", [(P(wr), hand, 0.09 * sw)], (P(wr), hand), c(wr)))
    for side, (hp, kn, an) in (("l", (LHIP, LKN, LAN)), ("r", (RHIP, RKN, RAN))):
        if ok(hp, kn):
            parts.append(Part(side + "thigh", [(P(hp), P(kn), 0.18 * sw)], (P(hp), P(kn)), c(hp, kn)))
            if ok(an):
                parts.append(Part(side + "shin", [(P(kn), P(an), 0.13 * sw)], (P(kn), P(an)), c(kn, an)))
    for p in parts:
        # well-seen parts come forward, in proportion to the shoulder width
        p.depth = -p.conf * 0.6 * sw
    return parts


def seg_dist(px, a, b):
    """Distance from each pixel to segment ab."""
    d = b - a
    L2 = max(d @ d, 1e-9)
    t = np.clip(((px - a) @ d) / L2, 0, 1)
    return np.linalg.norm(px - (a + t[:, None] * d), axis=1)


def visible_parts(label, people):
    """Z-buffer the capsules inside each person's mask. Returns a part-label image and the part list."""
    h, w = label.shape
    plabel = np.full((h, w), -1, np.int32)
    allparts = []
    for j, k in enumerate(people):
        region = label == j
        if not region.any():
            continue
        ys, xs = np.nonzero(region)
        px = np.stack([xs, ys], 1).astype(np.float64)
        parts = blockout(k, region) if k is not None else None
        if not parts:
            parts = [Part("body", [], principal_axis(px), 0.5)]
        zbest = np.full(len(px), np.inf)
        rbest = np.full(len(px), np.inf)
        ibest = np.full(len(px), -1)
        inear = np.full(len(px), -1)
        for p in parts:
            p.person = j
            p.id = len(allparts)
            allparts.append(p)
            for a, b, r in p.capsules:
                d = seg_dist(px, a, b)
                z = p.depth - np.sqrt(np.clip(r * r - d * d, 0, None))   # nearer surface = smaller z
                front = (d < r) & (z < zbest)
                zbest[front], ibest[front] = z[front], p.id
                near = d / r < rbest
                rbest[near], inear[near] = (d / r)[near], p.id
        if not parts[0].capsules:
            ibest[:] = parts[0].id
        plabel[ys, xs] = np.where(ibest >= 0, ibest, inear)
    return plabel, allparts


def principal_axis(pts):
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    proj = (pts - c) @ vt[0]
    lo, hi = np.percentile(proj, [4, 96])
    return c + vt[0] * lo, c + vt[0] * hi


# ---------------------------------------------------------------- centre veins

def centre_veins(plabel, parts, inset):
    """For every visible piece of every part, the visible stretch of its axis as polylines."""
    h, w = plabel.shape
    veins = []           # (polyline (n, 2), part id)

    def sample(a, b):
        t = np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / 2)))
        return a + t[:, None] * (b - a)

    def keep(pts, ok):
        xy = np.round(pts).astype(int)
        inb = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        good = np.zeros(len(pts), bool)
        good[inb] = ok[xy[inb, 1], xy[inb, 0]]
        return [pts[r] for r in runs(good) if len(r) * 2 >= 3 * inset]

    for p in parts:
        m = (plabel == p.id).astype(np.uint8)
        if m.sum() < 30:
            continue
        core = cv2.erode(m, np.ones((2 * inset + 1, 2 * inset + 1), np.uint8)) > 0
        n, comp = cv2.connectedComponents(m)
        seeded = set()
        for seg in keep(sample(*p.axis), core):
            veins.append((seg, p.id))
            mid = np.round(seg[len(seg) // 2]).astype(int)
            seeded.add(comp[mid[1], mid[0]])
        # visible pieces the axis never reaches get their own principal axis
        for ci in range(1, n):
            piece = (comp == ci) & core
            if ci in seeded or piece.sum() < 20 * inset:
                continue
            ys, xs = np.nonzero(piece)
            for seg in keep(sample(*principal_axis(np.stack([xs, ys], 1).astype(np.float64))), piece):
                veins.append((seg, p.id))
    return veins


def runs(mask):
    out, cur = [], []
    for i, v in enumerate(mask):
        if v:
            cur.append(i)
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


# ---------------------------------------------------------------- reaction-diffusion

def labyrinth(region, pins, nx, ny, aniso, max_steps, log):
    """Grow a Gray-Scott labyrinth inside `region` out from `pins`. Returns the B field.

    The diffusion operator blends Karl Sims' 9-point Laplacian with the second
    derivative along the local direction (nx, ny), weighted by `aniso`. Extra
    diffusion along a direction turns the stripes across it. The blend keeps
    the stencil's centre weight at -1, which keeps the explicit update stable.
    """
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32))
    reg, pin = T(region), T(pins)
    out = reg == 0
    nxx, nyy, nxy = T(nx * nx), T(ny * ny), T(nx * ny)
    # neighbour weights of four second-difference stencils (their centres are -2)
    kern = torch.tensor([
        [[0, 0, 0], [1, 0, 1], [0, 0, 0]],           # u_xx
        [[0, 1, 0], [0, 0, 0], [0, 1, 0]],           # u_yy
        [[1, 0, 0], [0, 0, 0], [0, 0, 1]],           # along the main diagonal
        [[0, 0, 1], [0, 0, 0], [1, 0, 0]],           # along the anti-diagonal
    ], dtype=torch.float32)[:, None]
    conv = lambda u: F.conv2d(F.pad(u, (1, 1, 1, 1)), kern)
    # no-flux boundary: a stencil only reaches neighbours inside the body, so
    # the outline never seeds stripes of its own and growth starts at the pins
    nreg = conv(reg[None, None])[0]
    A = torch.ones_like(reg)
    B = pin.clone()
    one, zero = torch.ones_like(A), torch.zeros_like(A)
    filled_prev = 0
    for step in range(max_steps):
        u = torch.stack([A, B])[:, None]
        d = conv(u * reg) - u * nreg
        uxx, uyy, dd1, dd2 = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
        iso = 0.2 * (uxx + uyy) + 0.05 * (dd1 + dd2)
        unn = nxx * uxx + nyy * uyy + nxy * (dd1 - dd2) / 2
        lap = (1 - aniso) * iso + aniso * 0.5 * unn    # same centre weight as iso, so it stays stable
        abb = A * B * B
        A = (A + lap[0] - abb + FEED * (1 - A)).clamp_(0, 1)
        B = (B + 0.5 * lap[1] + abb - (KILL + FEED) * B).clamp_(0, 1)
        A = torch.where(out, one, A)
        B = torch.where(out, zero, torch.maximum(B, pin))
        if step % 500 == 499:
            filled = float(((B > 0.2) & ~out).sum() / max(1, int((~out).sum())))
            log(f"    rd step {step + 1}: stripes cover {filled:.1%} of the body")
            if abs(filled - filled_prev) < 0.003 and step > 3000:     # the labyrinth has settled
                break
            filled_prev = filled
    return B.numpy()


# ---------------------------------------------------------------- render

def render(models, path, period, aniso, max_steps, debug, log):
    bgr = cv2.imread(path)
    s = LONG / max(bgr.shape[:2])
    bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    h, w = bgr.shape[:2]
    label, people = perceive(models, bgr)
    plabel, parts = visible_parts(label, people)
    vw = period / 2                                     # child vein width = gap width
    cw = vw * CENTRE_SCALE
    veins = centre_veins(plabel, parts, inset=int(round(vw)))
    log(f"  {len(people)} people, {sum(k is not None for k in people)} with pose, "
        f"{len(parts)} parts, {len(veins)} centre veins")

    # walls between parts so each form fills on its own
    wall = np.zeros((h, w), np.uint8)
    for dy, dx in ((0, 1), (1, 0)):
        a, b = plabel[: h - dy, : w - dx], plabel[dy:, dx:]
        edge = ((a != b) & (a >= 0) & (b >= 0)).astype(np.uint8)
        wall[: h - dy, : w - dx] |= edge
        wall[dy:, dx:] |= edge
    k = max(1, int(vw * 0.5)) | 1
    wall = cv2.dilate(wall, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    region = ((plabel >= 0) & (wall == 0)).astype(np.uint8)

    # the reaction runs on a grid scaled so its natural period becomes `period` image px
    g = NATURAL_PERIOD / period
    gw, gh = int(round(w * g)), int(round(h * g))
    reg_g = cv2.resize(region, (gw, gh), interpolation=cv2.INTER_NEAREST)
    pins = np.zeros((gh, gw), np.uint8)
    for pts, _ in veins:
        q = pts * g
        cv2.polylines(pins, [np.round(q).astype(np.int32)], False, 1, max(1, int(round(cw * g))))
        # short side stubs one period apart start the child veins off at right angles
        d = (q[-1] - q[0]) / (np.linalg.norm(q[-1] - q[0]) + 1e-9)
        n = np.array([-d[1], d[0]])
        L = np.linalg.norm(q[-1] - q[0])
        for t in np.arange(NATURAL_PERIOD / 2, L - NATURAL_PERIOD / 4, NATURAL_PERIOD):
            c = q[0] + d * t
            cv2.line(pins, tuple(np.round(c - n * NATURAL_PERIOD).astype(int)),
                     tuple(np.round(c + n * NATURAL_PERIOD).astype(int)), 1, max(1, int(round(NATURAL_PERIOD / 9))))
    pins &= reg_g
    # unit direction of each part's axis: extra diffusion along it turns the stripes across it
    nxf, nyf = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    for p in parts:
        a, b = p.axis
        d = (b - a) / (np.linalg.norm(b - a) + 1e-9)
        m = plabel == p.id
        nxf[m], nyf[m] = d[0], d[1]
    nx = cv2.resize(nxf, (gw, gh), interpolation=cv2.INTER_NEAREST)
    ny = cv2.resize(nyf, (gw, gh), interpolation=cv2.INTER_NEAREST)
    t0 = time.time()
    ys, xs = np.nonzero(reg_g)                 # only simulate the box around the bodies
    y0, y1, x0, x1 = max(0, ys.min() - 2), ys.max() + 3, max(0, xs.min() - 2), xs.max() + 3
    B = np.zeros((gh, gw), np.float32)
    B[y0:y1, x0:x1] = labyrinth(reg_g[y0:y1, x0:x1], pins[y0:y1, x0:x1], nx[y0:y1, x0:x1], ny[y0:y1, x0:x1],
                                aniso, max_steps, log)
    gw, gh = x1 - x0, y1 - y0
    log(f"    labyrinth on a {gw}x{gh} grid took {time.time() - t0:.0f}s")

    # thin the stripes to centre lines, then redraw them all at one width
    Bf = cv2.resize(B, (w, h), interpolation=cv2.INTER_CUBIC)
    stripes = (Bf > 0.2) & (region > 0)
    centre_mask = np.zeros((h, w), np.uint8)
    for pts, _ in veins:
        cv2.polylines(centre_mask, [np.round(pts).astype(np.int32)], False, 1, max(1, int(round(cw))))
    skel = skeletonize(stripes)
    skel &= cv2.dilate(centre_mask, np.ones((3, 3), np.uint8)) == 0
    skel = drop_short(skel, period)
    # each stripe centre line owns a band one period wide, so area / line length is the
    # period actually grown; veins and gaps are each drawn at exactly half of it
    period = float(stripes.sum() + (~stripes & (region > 0)).sum()) / max(1, int(skel.sum()))
    vw = period / 2
    cw = vw * CENTRE_SCALE
    log(f"    grown period {period:.1f}px: child veins {vw:.1f}px, centre veins {cw:.1f}px")
    links = join_to_centres(skel, centre_mask, period)

    # each pixel takes the colour of the person it belongs to (or the nearest one)
    person_of_part = np.array([p.person for p in parts] + [0])
    owner = np.where(plabel >= 0, person_of_part[plabel], -1)
    if (owner < 0).all():
        owner[:] = 0
    _, (iy, ix) = ndimage.distance_transform_edt(owner < 0, return_indices=True)
    owner = owner[iy, ix]

    # soft-edged child veins from the distance to their centre lines; centre veins drawn supersampled
    d_child = ndimage.distance_transform_edt((skel.astype(np.uint8) | links) == 0)
    alpha_child = np.clip(vw / 2 - d_child + 0.5, 0, 1)
    SS = 3
    cen = np.zeros((h * SS, w * SS), np.float32)
    for pts, _ in veins:
        cv2.polylines(cen, [np.round(pts * SS * 16).astype(np.int32)], False, 1.0,
                      max(1, int(round(cw * SS))), cv2.LINE_AA, shift=4)
    alpha_centre = cv2.resize(cen, (w, h), interpolation=cv2.INTER_AREA)
    alpha = np.maximum(alpha_child, alpha_centre)[..., None]

    col = np.array(COLORS, np.float32)[owner % len(COLORS)]
    pale = col + (255 - col) * 0.25
    col = np.where(alpha_centre[..., None] > 0.01, col, pale)
    shadow = cv2.GaussianBlur(alpha[..., 0], (0, 0), 5)
    shadow = cv2.warpAffine(shadow, np.float32([[1, 0, 6], [0, 1, 9]]), (w, h))[..., None] * 0.2
    out = np.full((h, w, 3), 255, np.float32) * (1 - shadow) + np.array([150, 150, 165], np.float32) * shadow
    out = out * (1 - alpha) + col * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)

    dbg = blockout_view(bgr, plabel, parts, veins) if debug else None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), out, dbg


def drop_short(skel, minlen):
    n, lab, st, _ = cv2.connectedComponentsWithStats(skel.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= minlen
    return keep[lab]


def join_to_centres(skel, centre_mask, period):
    """Short connectors from child-vein ends that stop next to a centre vein onto that vein."""
    links = np.zeros(skel.shape, np.uint8)
    if not centre_mask.any():
        return links
    nb = cv2.filter2D(skel.astype(np.uint8), -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT)
    ends = skel & (nb == 2)                              # the pixel itself plus one neighbour
    d, (iy, ix) = ndimage.distance_transform_edt(centre_mask == 0, return_indices=True)
    ys, xs = np.nonzero(ends & (d < period * 1.1))
    for y, x in zip(ys, xs):
        cv2.line(links, (int(x), int(y)), (int(ix[y, x]), int(iy[y, x])), 1, 1)
    return links


def blockout_view(bgr, plabel, parts, veins):
    """Debug image: the visible part map over the photo, the capsule outlines and the centre veins."""
    colors = np.random.default_rng(1).integers(60, 255, (len(parts) + 1, 3))
    vis = (bgr * 0.35).astype(np.uint8)
    m = plabel >= 0
    vis[m] = (vis[m] * 0.3 + colors[plabel[m]] * 0.7).astype(np.uint8)
    for p in parts:
        for a, b, r in p.capsules:
            d = b - a
            n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9) * r
            cv2.polylines(vis, [np.array([a + n, b + n, b - n, a - n]).astype(np.int32)], True, (255, 255, 255), 1, cv2.LINE_AA)
    for pts, _ in veins:
        cv2.polylines(vis, [np.round(pts).astype(np.int32)], False, (0, 0, 255), 4, cv2.LINE_AA)
    return vis


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("images", nargs="*")
    ap.add_argument("--period", type=float, default=18, help="labyrinth period in px at 1400 px; veins are half this wide")
    ap.add_argument("--aniso", type=float, default=0.4, help="share of diffusion directed along each part's axis (0-0.6)")
    ap.add_argument("--steps", type=int, default=12000, help="maximum reaction-diffusion steps")
    ap.add_argument("--debug", action="store_true", help="also write the blockout and visibility map")
    a = ap.parse_args()
    images = a.images or sorted(glob.glob(os.path.join(DATA, "hugs", "*.jpg")))
    os.makedirs(OUT, exist_ok=True)
    models = load_models()
    for f in images:
        name = os.path.splitext(os.path.basename(f))[0]
        print(name, flush=True)
        photo, veins, dbg = render(models, f, a.period, a.aniso, a.steps, a.debug, lambda s: print(s, flush=True))
        cv2.imwrite(os.path.join(OUT, name + ".png"), cv2.cvtColor(veins, cv2.COLOR_RGB2BGR))
        pair = np.concatenate([photo, np.full((photo.shape[0], 24, 3), 255, np.uint8), veins], 1)
        pair = cv2.resize(pair, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(OUT, name + "_pair.jpg"), cv2.cvtColor(pair, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        if dbg is not None:
            cv2.imwrite(os.path.join(OUT, name + "_blockout.jpg"), dbg, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print("  wrote", os.path.join(OUT, name + ".png"), flush=True)


if __name__ == "__main__":
    main()
