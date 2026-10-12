import numpy as np
from shapely.geometry import Polygon, Point

def point_to_segment_dist_batch(points, seg_start, seg_end):
    """Vectorized distance from array of points (N,2) to a single segment."""
    p = points
    a = seg_start
    b = seg_end
    ab = b - a
    ab_len2 = np.dot(ab, ab)
    if ab_len2 < 1e-12:
        return np.linalg.norm(p - a, axis=1)
    t = np.clip(((p - a) @ ab) / ab_len2, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(p - proj, axis=1)

def calculate_coverage_true(field, boundary, opw, ngrid=40, obstacles=None):
    """
    True area-coverage fraction: sample a grid of points inside the
    boundary polygon (excluding any obstacle holes), and mark a point
    'covered' if it lies within opw/2 of any planned track segment.
    Coverage = covered / total, where "total" only counts points in
    the actually-coverable region (AOI minus obstacles).
    """
    poly = Polygon(boundary)
    if not poly.is_valid or poly.area <= 0:
        return 0.0
    minx, miny, maxx, maxy = poly.bounds
    xs = np.linspace(minx, maxx, ngrid)
    ys = np.linspace(miny, maxy, ngrid)
    GX, GY = np.meshgrid(xs, ys)
    grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=1)

    # keep only points actually inside the AOI polygon
    inside_mask = np.array([poly.contains(Point(p)) for p in grid_pts])
    interior_pts = grid_pts[inside_mask]

    # exclude points that fall inside any obstacle (uncoverable by definition)
    if obstacles:
        obs_polys = [Polygon(o) if not isinstance(o, Polygon) else o for o in obstacles]
        keep = np.array([not any(op.contains(Point(p)) for op in obs_polys) for p in interior_pts])
        interior_pts = interior_pts[keep]

    if len(interior_pts) == 0:
        return 0.0

    half_w = opw / 2.0
    covered = np.zeros(len(interior_pts), dtype=bool)
    for track in field.tracks:
        a = np.array(track[0], dtype=float)
        b = np.array(track[1], dtype=float)
        d = point_to_segment_dist_batch(interior_pts, a, b)
        covered |= (d <= half_w)
        if covered.all():
            break

    return float(covered.sum()) / float(len(interior_pts))
