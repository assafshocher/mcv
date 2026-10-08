"""
Provided helpers for HW4 — From Scratch: Two-View 3-D Reconstruction.

DO NOT MODIFY THIS FILE — it will be replaced during grading.

You implement the *geometry* in the notebook (the essential matrix, pose
recovery, triangulation). Everything in here is plumbing you don't need to
read: it builds the synthetic "impossible triangle", renders the two photos,
runs the mouse point-picker, and draws the results (including the interactive
3-D viewer at the end).

Provides:
    make_penrose, make_cameras, project                    (the scene + the two cameras)
    normalize_pts                                          (point pre-conditioning for the 8-point algorithm)
    view_images, smooth, corner_peaks, show_detections     (corner-detection plumbing)
    show_views, pick_points, DEFAULT_PICKS                 (the two photos + the mouse picker)
    show_correspondences, show_epipolar_lines              (sanity-check visualizations)
    visible_corner_views, visible_corner_truth,
        visible_corner_colors, corner_truth, enumerate_faces   (the both-visible corners + the beam model)
    similarity_align                                       (align the reconstruction to truth)
    show_point_cloud, show_surfaces                        (the interactive 3-D viewers)
"""

import io, base64, html as _html, warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import HTML, display

# ════════════════════════════════════════════════════════════════════════════
#  The scene: an "impossible triangle" = three square beams along x, y, z.
#  Three beams of 8 corners each → 24 points. They never actually touch; only
#  from one magic direction do they line up into the impossible figure.
# ════════════════════════════════════════════════════════════════════════════
_C, _H, _L = 0.4167, 0.20, 1.20        # cross-section offset, half-width, half-length
                                       # L≈3*C is the clean "impossible-angle" closure length

# corners / faces / edges of one beam (a box), in a fixed local order
_CORNERS = [(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
            ( 1, -1, -1), ( 1, -1, 1), ( 1, 1, -1), ( 1, 1, 1)]
BEAM_FACES = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
              (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
BEAM_EDGES = [(0, 4), (1, 5), (2, 6), (3, 7), (0, 2), (1, 3),
              (4, 6), (5, 7), (0, 1), (2, 3), (4, 5), (6, 7)]
BEAM_COLORS = ['#d98a4f', '#5fa85f', '#5f7fd9']         # one colour per beam

def make_penrose():
    """Return (X, beam_id): X is (24,3) world points, beam_id is (24,) in {0,1,2}."""
    X, bid = [], []
    for i in range(3):
        a1, a2 = (i + 1) % 3, (i + 2) % 3
        for (sl, s1, s2) in _CORNERS:
            p = np.zeros(3)
            p[i] = sl * _L; p[a1] = -_C + s1 * _H; p[a2] = _C + s2 * _H
            X.append(p); bid.append(i)
    return np.array(X), np.array(bid)

def _beam_boxes():
    """Axis-aligned (min,max) corner of each beam box — used for occlusion."""
    boxes = []
    for i in range(3):
        a1, a2 = (i + 1) % 3, (i + 2) % 3
        lo = np.empty(3); hi = np.empty(3)
        lo[i], hi[i] = -_L, _L
        lo[a1], hi[a1] = -_C - _H, -_C + _H
        lo[a2], hi[a2] = _C - _H, _C + _H
        boxes.append((lo, hi))
    return boxes

# ════════════════════════════════════════════════════════════════════════════
#  Two calibrated cameras (K = identity) and the pinhole projection.
# ════════════════════════════════════════════════════════════════════════════
def _look_at(eye, target=(0, 0, 0), up=(0, 0, 1.)):
    eye = np.asarray(eye, float); target = np.asarray(target, float); up = np.asarray(up, float)
    f = target - eye; f /= np.linalg.norm(f)
    if abs(f @ (up / np.linalg.norm(up))) > 0.95: up = np.array([0, 1, 0.])
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    R = np.stack([r, -u, f])            # rows: world→camera; camera looks down +z
    t = -R @ eye
    return R, t

# View 1 is shot from *almost* the magic (1,1,1) direction — so the photo already
# looks like the solid impossible triangle (the hook). It's a few degrees off-axis
# and far away (near-orthographic) so the three beams are still separable enough to
# match corners. View 2 is an ordinary oblique view that reveals three loose beams.
_diag = np.array([1., 1., 1.]) / np.sqrt(3)
_perp = np.array([1., -1., 0.]) / np.sqrt(2)                 # a direction ⟂ to the diagonal
# far away (≈orthographic) + only 5° off the magic axis, so the corners close cleanly
# into the impossible figure while the beams stay separable enough to pick.
_EYE1 = (np.cos(np.radians(1)) * _diag + np.sin(np.radians(1)) * _perp) * 50.0
_EYE2 = np.array([3.0, -4.4, 1.6])

def make_cameras():
    """Return (R1, t1, R2, t2) — the true world→camera poses of the two photos."""
    R1, t1 = _look_at(_EYE1)
    R2, t2 = _look_at(_EYE2)
    return R1, t1, R2, t2

def project(R, t, X):
    """Pinhole projection with K = I. X is (N,3); returns (N,2) image points."""
    Xc = (R @ X.T + np.asarray(t)[:, None]).T
    return Xc[:, :2] / Xc[:, 2:3]

def _cam_center(R, t):
    return -R.T @ np.asarray(t)

def _visible(R, t, X, beam_of_vertex):
    """Boolean (N,) — is each corner unoccluded from this camera? (ray vs beam boxes)"""
    C = _cam_center(R, t); boxes = _beam_boxes(); vis = np.ones(len(X), bool)
    for k, V in enumerate(X):
        dirv = V - C
        for (lo, hi) in boxes:                          # slab test, vertex at ray param t=1
            with np.errstate(divide='ignore', invalid='ignore'):
                t0 = (lo - C) / dirv; t1 = (hi - C) / dirv
            tn = np.nanmax(np.minimum(t0, t1)); tf = np.nanmin(np.maximum(t0, t1))
            if tn <= tf and tf > 1e-4 and tn < 1 - 1e-3:   # surface entered before the vertex
                vis[k] = False; break
    return vis

# ════════════════════════════════════════════════════════════════════════════
#  Point pre-conditioning (Hartley normalization) for the 8-point algorithm.
# ════════════════════════════════════════════════════════════════════════════
def normalize_pts(p):
    """Translate to zero mean and scale so mean distance to origin is sqrt(2).
    Returns (p_norm (N,2), T (3,3)) with [p_norm;1] = T @ [p;1]."""
    p = np.asarray(p, float); c = p.mean(0)
    d = np.linalg.norm(p - c, axis=1).mean(); s = np.sqrt(2) / d
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.]])
    pn = (T @ np.c_[p, np.ones(len(p))].T).T[:, :2]
    return pn, T

# ════════════════════════════════════════════════════════════════════════════
#  Rendering the two "photos" (solid beams with correct occlusion).
# ════════════════════════════════════════════════════════════════════════════
_PX = 430                               # rendered photo size in pixels
_BG = np.array([244, 246, 251.]) / 255
_EDGE = np.array([22, 51, 95.]) / 255
def _hex2rgb(s): s = s.lstrip('#'); return np.array([int(s[i:i + 2], 16) for i in (0, 2, 4)], float) / 255
_RGB = [_hex2rgb(c) for c in BEAM_COLORS]

def _fill_tri(col, zb, fid, p, z, shade, fcid, S):
    """Z-buffered triangle fill (nearest camera-depth wins) — correct per-pixel occlusion."""
    (x0, y0), (x1, y1), (x2, y2) = p
    minx = max(int(np.floor(min(x0, x1, x2))), 0); maxx = min(int(np.ceil(max(x0, x1, x2))), S - 1)
    miny = max(int(np.floor(min(y0, y1, y2))), 0); maxy = min(int(np.ceil(max(y0, y1, y2))), S - 1)
    if maxx < minx or maxy < miny: return
    det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(det) < 1e-9: return
    gy, gx = np.mgrid[miny:maxy + 1, minx:maxx + 1]
    a = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / det
    b = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / det
    c = 1 - a - b
    inside = (a >= -1e-6) & (b >= -1e-6) & (c >= -1e-6)
    zz = a * z[0] + b * z[1] + c * z[2]
    sz = zb[miny:maxy + 1, minx:maxx + 1]
    upd = inside & (zz < sz)
    sz[upd] = zz[upd]
    col[miny:maxy + 1, minx:maxx + 1][upd] = shade
    fid[miny:maxy + 1, minx:maxx + 1][upd] = fcid

def _zraster(R, t, X, ext, S):
    """Render the beams to an (S,S,3) image with a real z-buffer + crease/silhouette edges."""
    x = project(R, t, X); dep = (R @ X.T + t[:, None]).T[:, 2]      # camera depth (smaller = nearer)
    px = np.c_[(x[:, 0] - ext[0]) / (ext[1] - ext[0]) * (S - 1),
               (x[:, 1] - ext[2]) / (ext[3] - ext[2]) * (S - 1)]
    light = np.array([0.35, 0.5, 0.8]); light /= np.linalg.norm(light)
    col = np.ones((S, S, 3)) * _BG; zb = np.full((S, S), np.inf); fid = np.zeros((S, S), int); fc = 0
    for b in range(3):
        base = b * 8
        for f in BEAM_FACES:
            fc += 1; idx = [base + j for j in f]
            P3 = X[idx]; n = np.cross(P3[1] - P3[0], P3[2] - P3[0]); n = n / (np.linalg.norm(n) + 1e-9)
            shade = np.clip(_RGB[b] * (0.62 + 0.38 * abs(n @ light)), 0, 1)
            for tri in [(0, 1, 2), (0, 2, 3)]:
                vi = [idx[tri[0]], idx[tri[1]], idx[tri[2]]]
                _fill_tri(col, zb, fid, px[vi], dep[vi], shade, fc, S)
    e = np.zeros((S, S), bool)                                       # edges = face-id boundaries
    e[:-1] |= fid[:-1] != fid[1:]; e[1:] |= fid[:-1] != fid[1:]
    e[:, :-1] |= fid[:, :-1] != fid[:, 1:]; e[:, 1:] |= fid[:, :-1] != fid[:, 1:]
    e[1:] |= e[:-1].copy(); e[:, 1:] |= e[:, :-1].copy()             # thicken ~1px (post-downsample)
    col[e] = _EDGE
    return col

def _render_photo(R, t, X, bid):
    """Return (png_bytes, extent, targets). targets: list of dicts for visible
    corners {fx,fy in [0,1], nx,ny image coords, beam}."""
    x = project(R, t, X)
    lo, hi = x.min(0), x.max(0); pad = 0.16 * (hi - lo).max()
    cen = (lo + hi) / 2; half = (hi - lo).max() / 2 + pad
    ext = (cen[0] - half, cen[0] + half, cen[1] - half, cen[1] + half)

    # correct per-pixel occlusion via a z-buffer (painter's-by-mean-depth mis-orders
    # the interleaving beams at the corners — very visible near the impossible angle)
    SS = 3; raster = _zraster(R, t, X, ext, _PX * SS)
    raster = raster.reshape(_PX, SS, _PX, SS, 3).mean((1, 3))       # supersample → antialias
    fig = plt.figure(figsize=(_PX / 100, _PX / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(raster, extent=(ext[0], ext[1], ext[3], ext[2]), origin='upper', interpolation='antialiased')
    vis = _visible(R, t, X, bid)               # used for the picker's snap targets (not drawn here)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[3], ext[2])   # y down (image convention)
    ax.axis('off')
    buf = io.BytesIO(); fig.savefig(buf, format='png'); plt.close(fig)

    targets = []
    for k in np.where(vis)[0]:
        fx = (x[k, 0] - ext[0]) / (ext[1] - ext[0])
        fy = (x[k, 1] - ext[2]) / (ext[3] - ext[2])           # already y-down
        targets.append(dict(fx=float(fx), fy=float(fy),
                            nx=float(x[k, 0]), ny=float(x[k, 1]), beam=int(bid[k])))
    return buf.getvalue(), ext, targets

def _scene():
    """Cache the scene + both renders (idempotent)."""
    if not hasattr(_scene, 'cache'):
        X, bid = make_penrose(); R1, t1, R2, t2 = make_cameras()
        png1, e1, T1 = _render_photo(R1, t1, X, bid)
        png2, e2, T2 = _render_photo(R2, t2, X, bid)
        _scene.cache = dict(X=X, bid=bid, R1=R1, t1=t1, R2=R2, t2=t2,
                            png=(png1, png2), ext=(e1, e2), targets=(T1, T2))
    return _scene.cache

def _b64(png):
    return 'data:image/png;base64,' + base64.b64encode(png).decode()

def _show_iframe(doc, height):
    """Display an HTML document in a sandboxed iframe (works across Jupyter frontends)."""
    with warnings.catch_warnings():                 # silence IPython's "use IFrame" hint
        warnings.simplefilter('ignore')
        display(HTML(f'<iframe srcdoc="{_html.escape(doc, quote=True)}" '
                     f'style="width:100%;height:{height}px;border:0;border-radius:10px" '
                     f'allow="clipboard-write"></iframe>'))

def show_views():
    """Display the two photos side by side (no interaction)."""
    s = _scene()
    fig, ax = plt.subplots(1, 2, figsize=(10, 5.2))
    for a, png, e, ttl in zip(ax, s['png'], s['ext'], ['View 1', 'View 2']):
        a.imshow(plt.imread(io.BytesIO(png)), extent=(e[0], e[1], e[3], e[2]))
        a.set_title(ttl, fontsize=12); a.set_xlabel('image x'); a.set_ylabel('image y')
    plt.tight_layout(); plt.show()

# default correspondences (visible in BOTH views, spread over all three beams) ──
def _default_picks():
    s = _scene()
    # a corner is usable as a correspondence only if visible in BOTH views
    x1 = project(s['R1'], s['t1'], s['X']); x2 = project(s['R2'], s['t2'], s['X'])
    v1 = _visible(s['R1'], s['t1'], s['X'], s['bid'])
    v2 = _visible(s['R2'], s['t2'], s['X'], s['bid'])
    both = np.where(v1 & v2)[0]
    picks = []
    for b in range(3):                                   # 4 spread corners per beam
        ids = [k for k in both if s['bid'][k] == b]
        for k in ids[:4]:
            picks.append([x1[k, 0], x1[k, 1], x2[k, 0], x2[k, 1]])
    return np.array(picks)

DEFAULT_PICKS = _default_picks()

def _both_visible():
    s = _scene()
    return _visible(s['R1'], s['t1'], s['X'], s['bid']) & _visible(s['R2'], s['t2'], s['X'], s['bid'])

def visible_corner_views():
    """Image coords, in BOTH photos, of the corners that are actually visible in
    both — the only corners you can match and therefore triangulate. (A detector
    found and matched these; corners hidden in either photo are simply not here.)
    Returns (x1 (M,2), x2 (M,2))."""
    s = _scene(); m = _both_visible()
    return project(s['R1'], s['t1'], s['X'])[m], project(s['R2'], s['t2'], s['X'])[m]

def visible_corner_truth():
    """True 3-D positions of the visible-in-both corners (to align the cloud)."""
    return make_penrose()[0][_both_visible()]

def visible_corner_colors():
    """RGB per visible-in-both corner (by beam), for the cloud viewer."""
    return np.array(_RGB)[make_penrose()[1][_both_visible()]]

def corner_colors():
    """RGB colour per corner (by beam), for the sparse point-cloud viewer."""
    return np.array(_RGB)[make_penrose()[1]]

def corner_truth():
    """The true 3-D corner positions (to similarity-align the reconstruction against)."""
    return make_penrose()[0]

def enumerate_faces():
    """The 18 planar faces of the object, as quads of CORNER indices (into the 24).

    We can't read a surface off a bare point cloud, so we make ONE modelling
    assumption: the object is three solid beams (boxes). A box has 6 flat faces,
    so 3 beams → 18 planes. Knowing the corners come in beam order (0–7, 8–15,
    16–23), enumerating the faces is just bookkeeping. Returns (18, 4) int indices."""
    faces = []
    for b in range(3):
        for f in BEAM_FACES:
            faces.append([8 * b + i for i in f])
    return np.array(faces)

# ════════════════════════════════════════════════════════════════════════════
#  Corner-detection plumbing (you write the Harris detector; these are provided).
# ════════════════════════════════════════════════════════════════════════════
def view_images():
    """The two photos as float RGB arrays (H, W, 3) in [0,1] — run your Harris
    detector on each of these."""
    s = _scene()
    return (plt.imread(io.BytesIO(s['png'][0]))[..., :3].astype(float),
            plt.imread(io.BytesIO(s['png'][1]))[..., :3].astype(float))

def smooth(a, sigma=1.5):
    """Gaussian blur of a 2-D array (separable). Use it to pre-smooth and to do
    the windowed sums of the structure tensor."""
    r = max(1, int(3 * sigma)); x = np.arange(-r, r + 1)
    ker = np.exp(-x ** 2 / (2 * sigma ** 2)); ker /= ker.sum()
    a = np.apply_along_axis(lambda m: np.convolve(m, ker, 'same'), 0, a)
    return np.apply_along_axis(lambda m: np.convolve(m, ker, 'same'), 1, a)

def corner_peaks(R, thresh=0.04, min_dist=8):
    """Turn a corner-response map R into a list of corner pixels: threshold,
    non-maximum suppression, and sub-pixel refinement. Returns (M, 2) array of
    (x, y) pixel coordinates. (Provided — the bookkeeping on top of your Harris response.)"""
    R = R.copy(); R[R < thresh * R.max()] = 0
    H, W = R.shape; out = []
    for y in range(H):
        for x in range(W):
            if R[y, x] <= 0 or x < 4 or y < 4 or x > W - 5 or y > H - 5: continue
            y0, y1 = max(0, y - min_dist), min(H, y + min_dist + 1)
            x0, x1 = max(0, x - min_dist), min(W, x + min_dist + 1)
            if R[y, x] < R[y0:y1, x0:x1].max(): continue
            dxx = R[y, x - 1] - 2 * R[y, x] + R[y, x + 1]
            dyy = R[y - 1, x] - 2 * R[y, x] + R[y + 1, x]
            dx = 0.5 * (R[y, x - 1] - R[y, x + 1]) / dxx if abs(dxx) > 1e-12 else 0.0
            dy = 0.5 * (R[y - 1, x] - R[y + 1, x]) / dyy if abs(dyy) > 1e-12 else 0.0
            out.append((x + np.clip(dx, -1, 1), y + np.clip(dy, -1, 1)))
    return np.array(out) if out else np.zeros((0, 2))

def show_detections(corners1, corners2):
    """Overlay your detected corners on both photos."""
    s = _scene()
    fig, ax = plt.subplots(1, 2, figsize=(10, 5.2))
    for a, png, e, c, ttl in zip(ax, s['png'], s['ext'], [corners1, corners2], ['View 1', 'View 2']):
        a.imshow(plt.imread(io.BytesIO(png)), extent=(e[0], e[1], e[3], e[2]))
        c = np.asarray(c, float)
        if len(c):
            nx = e[0] + c[:, 0] / (_PX - 1) * (e[1] - e[0])
            ny = e[2] + c[:, 1] / (_PX - 1) * (e[3] - e[2])
            a.scatter(nx, ny, s=70, facecolors='none', edgecolors='r', linewidths=1.6)
        a.set_title(f'{ttl} — {len(c)} corners detected'); a.axis('off')
    plt.tight_layout(); plt.show()

# ════════════════════════════════════════════════════════════════════════════
#  The mouse point-picker: match YOUR detections across views, discard junctions.
# ════════════════════════════════════════════════════════════════════════════
def pick_points(corners1, corners2):
    """Interactive matcher over your detected corners. **Match** mode: click a
    real corner in View 1, then the same corner in View 2. **Mark-junk** mode:
    click a detection that is NOT a real 3-D corner (an occlusion junction where
    two beams merely cross) to grey it out so it can't be matched. Pick >=8 real
    matches, then Copy and paste the line below."""
    s = _scene()
    def targets(corners, ext, R, t):
        # true visible-corner projections, used to refine an accepted match to the
        # exact corner location (stands in for a detector's sub-pixel refinement;
        # a corner detector localizes to ~a pixel, which the noisy 8-point can't take)
        true_p = project(R, t, s['X'])[_visible(R, t, s['X'], s['bid'])]
        tol = 6.0 * (ext[1] - ext[0]) / _PX
        out = []
        for (cx, cy) in np.asarray(corners, float):
            fx, fy = cx / (_PX - 1), cy / (_PX - 1)           # where to DRAW the detection
            nx = ext[0] + fx * (ext[1] - ext[0]); ny = ext[2] + fy * (ext[3] - ext[2])
            if len(true_p):                                   # snap OUTPUT to nearest real corner
                d = np.linalg.norm(true_p - [nx, ny], axis=1)
                if d.min() < tol: nx, ny = true_p[d.argmin()]
            out.append(dict(fx=float(fx), fy=float(fy), nx=float(nx), ny=float(ny)))
        return out
    import json
    payload = dict(img1=_b64(s['png'][0]), img2=_b64(s['png'][1]),
                   t1=targets(corners1, s['ext'][0], s['R1'], s['t1']),
                   t2=targets(corners2, s['ext'][1], s['R2'], s['t2']))
    doc = _PICKER_HTML.replace('/*DATA*/', json.dumps(payload))
    _show_iframe(doc, 600)

_PICKER_HTML = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#eef1f8;color:#1b3358;padding:10px}
#wrap{display:flex;gap:12px;justify-content:center}
.view{position:relative}.view canvas{border-radius:9px;box-shadow:0 1px 5px rgba(20,40,80,.12);cursor:crosshair;background:#f4f6fb}
.cap{font-size:12px;font-weight:600;text-align:center;margin-bottom:4px;color:#42597e}
#bar{display:flex;align-items:center;gap:8px;justify-content:center;margin-top:10px;flex-wrap:wrap}
button{border:1px solid #b9c4dc;background:#fff;color:#1b3358;font-size:13px;padding:8px 13px;border-radius:8px;cursor:pointer;font-weight:600}
button:hover{background:#f3f6fc}#copy{background:#1b3358;color:#fff;border-color:#1b3358}#copy:hover{background:#27406e}
#mode.junk{background:#b8442e;color:#fff;border-color:#b8442e}
#status{font-size:12.5px;color:#42597e;min-width:210px;text-align:center}
#out{width:96%;margin:8px auto 0;display:block;font-family:ui-monospace,Menlo,monospace;font-size:11px;
 border:1px solid #c7d0e4;border-radius:7px;padding:7px;color:#22406b;background:#fbfcfe;resize:none}
b{color:#1b3358}
</style></head><body>
<div id="wrap">
 <div class="view"><div class="cap">View 1</div><canvas id="c1" width="430" height="430"></canvas></div>
 <div class="view"><div class="cap">View 2</div><canvas id="c2" width="430" height="430"></canvas></div>
</div>
<div id="bar">
 <button id="mode">Mode: matching</button>
 <span id="status"></span>
 <button id="undo">Undo</button><button id="clear">Clear</button><button id="copy">📋 Copy picks</button>
</div>
<textarea id="out" rows="2" readonly onclick="this.select()"></textarea>
<script>
const D=/*DATA*/;
const cv=[document.getElementById('c1'),document.getElementById('c2')];
const cx=cv.map(c=>c.getContext('2d'));
const tg=[D.t1,D.t2]; const imgs=[new Image(),new Image()];
let pairs=[], pend=null, junk=[new Set(),new Set()], mode='match';
imgs[0].src=D.img1; imgs[1].src=D.img2;
let loaded=0; imgs.forEach(im=>im.onload=()=>{if(++loaded==2)draw();});
function tpx(v,t){return [t.fx*cv[v].width, t.fy*cv[v].height];}
function draw(){
 for(let v=0;v<2;v++){
   const g=cx[v]; g.clearRect(0,0,cv[v].width,cv[v].height); g.drawImage(imgs[v],0,0,cv[v].width,cv[v].height);
   tg[v].forEach((t,i)=>{const p=tpx(v,t); g.beginPath(); g.arc(p[0],p[1],5,0,7);
     if(junk[v].has(i)){g.strokeStyle='#aab2c4';g.lineWidth=1.5;g.stroke();        // discarded: grey x
       g.beginPath();g.moveTo(p[0]-4,p[1]-4);g.lineTo(p[0]+4,p[1]+4);g.moveTo(p[0]+4,p[1]-4);g.lineTo(p[0]-4,p[1]+4);g.stroke();}
     else{g.fillStyle='rgba(255,255,255,.85)';g.fill();g.lineWidth=2;g.strokeStyle='#d23b1f';g.stroke();}});
   pairs.forEach((pr,i)=>{const p=tpx(v,pr[v]); g.beginPath(); g.arc(p[0],p[1],8,0,7); g.fillStyle='#1b3358'; g.fill();
     g.fillStyle='#fff'; g.font='bold 11px sans-serif'; g.textAlign='center'; g.textBaseline='middle'; g.fillText(i+1,p[0],p[1]);});
   if(pend&&pend.view==v){const p=tpx(v,tg[v][pend.idx]); g.beginPath(); g.arc(p[0],p[1],10,0,7);
     g.lineWidth=3; g.strokeStyle='#1b3358'; g.stroke();}
 }
 const need=Math.max(0,8-pairs.length);
 document.getElementById('status').innerHTML='<b>'+pairs.length+'</b> matches'+
   (need>0?' &middot; need '+need+' more':' &middot; ✓ enough')+' &middot; <span style="color:#b8442e">'+
   (junk[0].size+junk[1].size)+' marked junk</span>';
 const rows=pairs.map(p=>'['+p[0].nx.toFixed(12)+','+p[0].ny.toFixed(12)+','+p[1].nx.toFixed(12)+','+p[1].ny.toFixed(12)+']');
 document.getElementById('out').value='picks = np.array(['+rows.join(', ')+'])';
}
function nearestIdx(v,mx,my){let bi=-1,bd=18*18; tg[v].forEach((t,i)=>{const p=tpx(v,t);
  const d=(p[0]-mx)**2+(p[1]-my)**2; if(d<bd){bd=d;bi=i;}}); return bi;}
cv.forEach((c,v)=>c.addEventListener('click',e=>{
  const r=c.getBoundingClientRect(); const mx=(e.clientX-r.left)*c.width/r.width, my=(e.clientY-r.top)*c.height/r.height;
  const i=nearestIdx(v,mx,my); if(i<0)return;
  if(mode==='junk'){ if(junk[v].has(i))junk[v].delete(i); else junk[v].add(i); pend=null; draw(); return; }
  if(junk[v].has(i))return;                                   // can't match a discarded point
  if(pend===null){pend={view:v,idx:i};}
  else if(pend.view===v){pend={view:v,idx:i};}                // re-pick same side
  else{const pr=[null,null]; pr[pend.view]=tg[pend.view][pend.idx]; pr[v]=tg[v][i]; pairs.push(pr); pend=null;}
  draw();
}));
document.getElementById('mode').onclick=function(){mode=(mode==='match')?'junk':'match';pend=null;
  this.textContent=mode==='match'?'Mode: matching':'Mode: marking junk'; this.className=mode==='junk'?'junk':''; draw();};
document.getElementById('undo').onclick=()=>{if(pend)pend=null;else pairs.pop();draw();};
document.getElementById('clear').onclick=()=>{pairs=[];pend=null;junk=[new Set(),new Set()];draw();};
document.getElementById('copy').onclick=()=>{const o=document.getElementById('out');o.select();
  try{navigator.clipboard.writeText(o.value);}catch(e){document.execCommand('copy');}
  const b=document.getElementById('copy');b.textContent='✓ Copied';setTimeout(()=>b.textContent='📋 Copy picks',1200);};
</script></body></html>"""

# ════════════════════════════════════════════════════════════════════════════
#  Sanity-check visualizations.
# ════════════════════════════════════════════════════════════════════════════
def show_correspondences(picks):
    """Overlay the picked points on both photos and link matches by colour."""
    s = _scene(); picks = np.asarray(picks, float)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5.2))
    cols = plt.cm.turbo(np.linspace(0.05, 0.95, len(picks)))
    for a, png, e, ttl, cols_xy in zip(ax, s['png'], s['ext'], ['View 1', 'View 2'],
                                        [picks[:, :2], picks[:, 2:]]):
        a.imshow(plt.imread(io.BytesIO(png)), extent=(e[0], e[1], e[3], e[2]))
        a.scatter(cols_xy[:, 0], cols_xy[:, 1], c=cols, s=70, edgecolors='k', linewidths=0.6, zorder=5)
        for i, (px, py) in enumerate(cols_xy):
            a.text(px, py, str(i + 1), fontsize=8, ha='center', va='center', zorder=6)
        a.set_title(f'{ttl} — {len(picks)} picks'); a.set_xlabel('image x'); a.set_ylabel('image y')
    plt.tight_layout(); plt.show()

def show_epipolar_lines(E, picks):
    """Draw, in View 2, the epipolar line E·x1 for each picked point in View 1.
    A correct E puts every View-2 point on its own line."""
    s = _scene(); picks = np.asarray(picks, float)
    p1, p2 = picks[:, :2], picks[:, 2:]
    fig, ax = plt.subplots(1, 2, figsize=(10, 5.2))
    cols = plt.cm.turbo(np.linspace(0.05, 0.95, len(picks)))
    e1 = s['ext'][0]
    ax[0].imshow(plt.imread(io.BytesIO(s['png'][0])), extent=(e1[0], e1[1], e1[3], e1[2]))
    ax[0].scatter(p1[:, 0], p1[:, 1], c=cols, s=55, edgecolors='k', linewidths=0.5, zorder=5)
    ax[0].set_title('View 1 — picked points'); ax[0].axis('off')
    e2 = s['ext'][1]
    ax[1].imshow(plt.imread(io.BytesIO(s['png'][1])), extent=(e2[0], e2[1], e2[3], e2[2]))
    xs = np.array([e2[0], e2[1]])
    for (x, y), c in zip(p1, cols):
        l = E @ np.array([x, y, 1.])                  # line a x + b y + c = 0 in view 2
        if abs(l[1]) > 1e-9:
            ax[1].plot(xs, -(l[0] * xs + l[2]) / l[1], color=c, lw=1.0, alpha=0.9)
    ax[1].scatter(p2[:, 0], p2[:, 1], c=cols, s=55, edgecolors='k', linewidths=0.5, zorder=5)
    ax[1].set_xlim(e2[0], e2[1]); ax[1].set_ylim(e2[3], e2[2])
    ax[1].set_title('View 2 — points should lie on their epipolar lines'); ax[1].axis('off')
    plt.tight_layout(); plt.show()

# ════════════════════════════════════════════════════════════════════════════
#  Align the (up-to-scale) reconstruction to the truth, and the 3-D viewer.
# ════════════════════════════════════════════════════════════════════════════
def similarity_align(src, dst):
    """Best similarity (scale+rotation+translation) mapping src→dst (Umeyama).
    Two-view reconstruction is only known up to such a transform; this lets us
    overlay it on the ground truth to measure error and orient the viewer."""
    src = np.asarray(src, float); dst = np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0); S = src - mu_s; D = dst - mu_d
    U, d, Vt = np.linalg.svd(S.T @ D / len(src))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0: Vt[-1] *= -1; R = Vt.T @ U.T
    s = d.sum() * len(src) / (S ** 2).sum()
    return (s * (R @ src.T).T + (mu_d - s * R @ mu_s))

def _frus_for(X):
    u = np.array([1, -1, 0.]) / np.sqrt(2); v = np.array([1, 1, -2.]) / np.sqrt(6)
    return round(float(np.sqrt((X @ u) ** 2 + (X @ v) ** 2).max()) * 1.22, 4)

def show_point_cloud(X, colors=None, height=520):
    """Interactive 3-D viewer of the reconstructed point cloud (the bare corners).
    Drag to orbit, scroll to zoom, use the slider to set dot size, and hit the
    button to snap to the impossible angle."""
    X = np.asarray(X, float)
    if colors is None: colors = np.tile([0.36, 0.47, 0.72], (len(X), 1))
    d = (np.array([1., 1., 1.]) / np.sqrt(3)).tolist()
    data = dict(verts=X.round(4).tolist(), cols=np.clip(colors, 0, 1).round(3).tolist(),
                diag=d, frus=_frus_for(X), psize=7.0,
                title='Your point cloud', sub=f'{len(X)} corners, triangulated from two photos')
    import json
    _show_iframe(_VIEW3D_HTML.replace('/*DATA*/', json.dumps(data)), height)

_VIEW3D_HTML = r"""<!DOCTYPE html><html><head><meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<style>*{margin:0;padding:0;box-sizing:border-box}html,body{height:100%;background:#eef1f8;overflow:hidden;
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}#c{position:fixed;inset:0}
#hud{position:fixed;top:14px;left:14px;color:#1b3358;max-width:330px;pointer-events:none}
#hud h1{font-size:16px;font-weight:650;margin-bottom:3px}#hud p{font-size:12px;line-height:1.4;color:#42597e}
.btns{position:fixed;left:14px;bottom:14px;display:flex;gap:10px;align-items:center}
button{pointer-events:auto;border:1px solid #b9c4dc;background:#fff;color:#1b3358;font-size:13px;padding:8px 13px;
border-radius:8px;cursor:pointer;font-weight:550;box-shadow:0 1px 3px rgba(20,40,80,.08)}button:hover{background:#f3f6fc}
#snap{background:#1b3358;color:#fff;border-color:#1b3358}#snap:hover{background:#27406e}
#sz{pointer-events:auto;background:#fff;border:1px solid #b9c4dc;border-radius:8px;padding:6px 11px;font-size:12.5px;
color:#42597e;font-weight:550;box-shadow:0 1px 3px rgba(20,40,80,.08)}#sz input{vertical-align:middle;width:96px;margin-left:6px}
#badge{position:fixed;top:50%;left:50%;transform:translate(-50%,-150px);font-size:21px;font-weight:700;color:#b8442e;
letter-spacing:.5px;opacity:0;transition:opacity .25s;pointer-events:none;text-shadow:0 1px 8px rgba(255,255,255,.9)}</style></head>
<body><div id="c"></div>
<div id="hud"><h1 id="ti"></h1><p id="su"></p>
<p style="font-size:11.5px;color:#6a7ea0;margin-top:6px">drag to orbit · scroll to zoom</p></div>
<div id="badge">✦ impossible ✦</div>
<div class="btns"><button id="snap">Snap to the impossible angle</button>
 <label id="sz">point size<input id="ps" type="range" min="1" max="30" step="0.5"></label></div>
<script>
const D=/*DATA*/; const cont=document.getElementById('c');
document.getElementById('ti').textContent=D.title; document.getElementById('su').textContent=D.sub;
const scene=new THREE.Scene(); scene.background=new THREE.Color(0xeef1f8);
const W=()=>cont.clientWidth,H=()=>cont.clientHeight;
let frus=D.frus; const ORB=10;                          // orthographic: keeps the illusion exact
function applyFrus(){const a=W()/H();camera.left=-frus*a;camera.right=frus*a;camera.top=frus;camera.bottom=-frus;camera.updateProjectionMatrix();}
const camera=new THREE.OrthographicCamera(-frus,frus,frus,-frus,0.1,100); applyFrus();
const renderer=new THREE.WebGLRenderer({antialias:true}); renderer.setPixelRatio(devicePixelRatio); renderer.setSize(W(),H());
cont.appendChild(renderer.domElement);
const pos=new Float32Array(D.verts.flat()), colr=new Float32Array(D.cols.flat());
const geo=new THREE.BufferGeometry();
geo.setAttribute('position',new THREE.BufferAttribute(pos,3));
geo.setAttribute('color',new THREE.BufferAttribute(colr,3));
const mat=new THREE.PointsMaterial({size:D.psize,vertexColors:true,sizeAttenuation:false});
scene.add(new THREE.Points(geo,mat));
const ps=document.getElementById('ps'); ps.value=D.psize; ps.addEventListener('input',()=>{mat.size=parseFloat(ps.value);});
const diag=D.diag, diagAz=Math.atan2(diag[2],diag[0]),diagEl=Math.asin(diag[1]);
let az=diagAz+0.95,el=diagEl-0.55,snap=false;
function setCam(){const ce=Math.cos(el); camera.position.set(ce*Math.cos(az)*ORB,Math.sin(el)*ORB,ce*Math.sin(az)*ORB);
 camera.up.set(0,1,0); camera.lookAt(0,0,0);}
let drag=false,px=0,py=0;
renderer.domElement.addEventListener('pointerdown',e=>{drag=true;snap=false;px=e.clientX;py=e.clientY;});
window.addEventListener('pointerup',()=>drag=false);
window.addEventListener('pointermove',e=>{if(!drag)return;az-=(e.clientX-px)*0.008;el+=(e.clientY-py)*0.008;
 el=Math.max(-1.45,Math.min(1.45,el));px=e.clientX;py=e.clientY;});
renderer.domElement.addEventListener('wheel',e=>{e.preventDefault();frus*=(1+Math.sign(e.deltaY)*0.08);
 frus=Math.max(0.6,Math.min(6,frus));applyFrus();},{passive:false});
document.getElementById('snap').onclick=()=>{snap=true;};
const badge=document.getElementById('badge');
function loop(){requestAnimationFrame(loop);
 if(snap){let da=((diagAz-az+Math.PI)%(2*Math.PI))-Math.PI;az+=da*0.12;el+=(diagEl-el)*0.12;
   if(Math.abs(da)<0.002&&Math.abs(diagEl-el)<0.002){az=diagAz;el=diagEl;snap=false;}}
 setCam(); const ce=Math.cos(el),v=[ce*Math.cos(az),Math.sin(el),ce*Math.sin(az)];
 const dot=v[0]*diag[0]+v[1]*diag[1]+v[2]*diag[2];
 badge.style.opacity=(Math.acos(Math.max(-1,Math.min(1,dot)))*180/Math.PI<5.5)?1:0;
 renderer.render(scene,camera);}
loop();
window.addEventListener('resize',()=>{applyFrus();renderer.setSize(W(),H());});
</script></body></html>"""

def show_surfaces(X, faces, colors=None, height=520):
    """Interactive 3-D viewer of the SOLID object: your reconstructed corners `X`
    (N,3) lifted into surfaces using the enumerated planar `faces` (F,4 corner
    indices). Drag to orbit, scroll to zoom, snap to the impossible angle."""
    X = np.asarray(X, float); faces = np.asarray(faces, int)
    fcol = np.array(_RGB)[np.arange(len(faces)) // (len(faces) // 3)]   # colour each face by its beam
    d = (np.array([1., 1., 1.]) / np.sqrt(3)).tolist()
    data = dict(verts=X.round(4).tolist(), faces=faces.tolist(), fcol=fcol.round(3).tolist(),
                diag=d, frus=_frus_for(X))
    import json
    _show_iframe(_SOLID_HTML.replace('/*DATA*/', json.dumps(data)), height)

_SOLID_HTML = r"""<!DOCTYPE html><html><head><meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<style>*{margin:0;padding:0;box-sizing:border-box}html,body{height:100%;background:#eef1f8;overflow:hidden;
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}#c{position:fixed;inset:0}
#hud{position:fixed;top:14px;left:14px;color:#1b3358;max-width:330px;pointer-events:none}
#hud h1{font-size:16px;font-weight:650;margin-bottom:3px}#hud p{font-size:12px;line-height:1.4;color:#42597e}
.btns{position:fixed;left:14px;bottom:14px;display:flex;gap:8px}
button{pointer-events:auto;border:1px solid #b9c4dc;background:#fff;color:#1b3358;font-size:13px;padding:8px 13px;
border-radius:8px;cursor:pointer;font-weight:550;box-shadow:0 1px 3px rgba(20,40,80,.08)}button:hover{background:#f3f6fc}
#snap{background:#1b3358;color:#fff;border-color:#1b3358}#snap:hover{background:#27406e}
#badge{position:fixed;top:50%;left:50%;transform:translate(-50%,-150px);font-size:21px;font-weight:700;color:#b8442e;
letter-spacing:.5px;opacity:0;transition:opacity .25s;pointer-events:none;text-shadow:0 1px 8px rgba(255,255,255,.9)}</style></head>
<body><div id="c"></div>
<div id="hud"><h1>The reconstructed object</h1><p>The known 3-beam model — 24 corners, 18 planar faces — shown in the frame your reconstruction recovered.
Three beams that never touch: drag to find the one angle where they lock into the impossible triangle.</p></div>
<div id="badge">✦ impossible ✦</div>
<div class="btns"><button id="snap">Snap to the impossible angle</button></div>
<script>
const D=/*DATA*/; const cont=document.getElementById('c');
const scene=new THREE.Scene(); scene.background=new THREE.Color(0xeef1f8);
const W=()=>cont.clientWidth,H=()=>cont.clientHeight;
let frus=D.frus; const ORB=10;
function applyFrus(){const a=W()/H();camera.left=-frus*a;camera.right=frus*a;camera.top=frus;camera.bottom=-frus;camera.updateProjectionMatrix();}
const camera=new THREE.OrthographicCamera(-frus,frus,frus,-frus,0.1,100); applyFrus();
const renderer=new THREE.WebGLRenderer({antialias:true}); renderer.setPixelRatio(devicePixelRatio); renderer.setSize(W(),H());
cont.appendChild(renderer.domElement);
scene.add(new THREE.AmbientLight(0xffffff,0.82));
let dl=new THREE.DirectionalLight(0xffffff,0.5); dl.position.set(3,5,4); scene.add(dl);
const V=D.verts;
const grp=new THREE.Group();
D.faces.forEach((q,i)=>{
  const a=V[q[0]],b=V[q[1]],c=V[q[2]],e=V[q[3]];
  const pos=[].concat(a,b,c, a,c,e);                       // quad -> 2 triangles
  const g=new THREE.BufferGeometry(); g.setAttribute('position',new THREE.Float32BufferAttribute(pos,3)); g.computeVertexNormals();
  const col=new THREE.Color(D.fcol[i][0],D.fcol[i][1],D.fcol[i][2]);
  grp.add(new THREE.Mesh(g,new THREE.MeshPhongMaterial({color:col,flatShading:true,side:THREE.DoubleSide,shininess:8})));
  const ep=[a,b,b,c,c,e,e,a].flat();                       // quad outline
  const eg=new THREE.BufferGeometry(); eg.setAttribute('position',new THREE.Float32BufferAttribute(ep,3));
  grp.add(new THREE.LineSegments(eg,new THREE.LineBasicMaterial({color:0x16335f})));
});
scene.add(grp);
const diag=D.diag, diagAz=Math.atan2(diag[2],diag[0]),diagEl=Math.asin(diag[1]);
let az=diagAz+0.95,el=diagEl-0.55,snap=false;
function setCam(){const ce=Math.cos(el); camera.position.set(ce*Math.cos(az)*ORB,Math.sin(el)*ORB,ce*Math.sin(az)*ORB);
 camera.up.set(0,1,0); camera.lookAt(0,0,0);}
let drag=false,px=0,py=0;
renderer.domElement.addEventListener('pointerdown',e=>{drag=true;snap=false;px=e.clientX;py=e.clientY;});
window.addEventListener('pointerup',()=>drag=false);
window.addEventListener('pointermove',e=>{if(!drag)return;az-=(e.clientX-px)*0.008;el+=(e.clientY-py)*0.008;
 el=Math.max(-1.45,Math.min(1.45,el));px=e.clientX;py=e.clientY;});
renderer.domElement.addEventListener('wheel',e=>{e.preventDefault();frus*=(1+Math.sign(e.deltaY)*0.08);
 frus=Math.max(0.6,Math.min(6,frus));applyFrus();},{passive:false});
document.getElementById('snap').onclick=()=>{snap=true;};
const badge=document.getElementById('badge');
function loop(){requestAnimationFrame(loop);
 if(snap){let da=((diagAz-az+Math.PI)%(2*Math.PI))-Math.PI;az+=da*0.12;el+=(diagEl-el)*0.12;
   if(Math.abs(da)<0.002&&Math.abs(diagEl-el)<0.002){az=diagAz;el=diagEl;snap=false;}}
 setCam(); const ce=Math.cos(el),v=[ce*Math.cos(az),Math.sin(el),ce*Math.sin(az)];
 const dot=v[0]*diag[0]+v[1]*diag[1]+v[2]*diag[2];
 badge.style.opacity=(Math.acos(Math.max(-1,Math.min(1,dot)))*180/Math.PI<5.5)?1:0;
 renderer.render(scene,camera);}
loop();
window.addEventListener('resize',()=>{applyFrus();renderer.setSize(W(),H());});
</script></body></html>"""
