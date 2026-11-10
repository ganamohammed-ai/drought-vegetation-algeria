r"""Comptage des classes de couvert par cellule TerraClimate pour un GeoTIFF uint8 tuile (Esri ou Dynamic World).
Usage: python3 -I cover_counts.py <tif> <group_meta.csv> <out_csv> <year> <label>"""
import sys, os, zlib, numpy as np, pandas as pd, tifffile
from multiprocessing import Pool
TIF, GM, OUT, YEAR, LABEL = sys.argv[1:6]
cs = 1/24
m = pd.read_csv(GM); m = m[m.keep]; c = m.groupby("tc_cell")[["lon","lat"]].first().reset_index()
ci = np.floor(c.lon.values/cs).astype(int); cj = np.floor(c.lat.values/cs).astype(int)
i0, j0 = ci.min(), cj.min(); NI, NJ = ci.max()-i0+1, cj.max()-j0+1
LUT = np.full((NJ, NI), -1, np.int32); LUT[cj-j0, ci-i0] = np.arange(len(c))
P = tifffile.TiffFile(TIF).pages[0]
W, H, T = P.imagewidth, P.imagelength, P.tilewidth; PXX, PXY = P.tags[33550].value[:2]; X0, Y0 = P.tags[33922].value[3:5]
PRED = P.predictor; NT = len(P.dataoffsets); NTX = -(-W//T)
def tile_ids(tx, ty):
    px = tx*T + np.arange(T); py = ty*T + np.arange(T)
    lon = X0 + (px+0.5)*PXX; lat = Y0 - (py+0.5)*PXY
    cx = np.floor(lon/cs).astype(int)-i0; cy = np.floor(lat/cs).astype(int)-j0
    okx = (cx >= 0) & (cx < NI) & (px < W); oky = (cy >= 0) & (cy < NJ) & (py < H)
    if not (okx.any() and oky.any()): return None
    sub = LUT[np.clip(cy, 0, NJ-1)[:, None], np.clip(cx, 0, NI-1)[None, :]]
    sub[~oky, :] = -1; sub[:, ~okx] = -1; return sub
def work(r):
    k0, k1 = r; t = tifffile.TiffFile(TIF); p = t.pages[0]; fh = t.filehandle; out = np.zeros((len(c), 12), np.int64)
    for k in range(k0, k1):
        ty, tx = divmod(k, NTX); ids = tile_ids(tx, ty)
        if ids is None or (ids < 0).all(): continue
        fh.seek(p.dataoffsets[k]); a = np.frombuffer(zlib.decompress(fh.read(p.databytecounts[k])), np.uint8).reshape(T, T)
        if PRED == 2: a = np.cumsum(a, axis=1, dtype=np.uint8)
        s = ids >= 0; out += np.bincount(ids[s].astype(np.int64)*12 + np.minimum(a[s], 11), minlength=len(c)*12).reshape(len(c), 12)
    return out
if __name__ == "__main__":
    step = 400
    with Pool(os.cpu_count()) as pool: res = pool.map(work, [(a, min(a+step, NT)) for a in range(0, NT, step)])
    tot = sum(res); print(LABEL, YEAR, "pixels", tot.sum(), "classes", tot.sum(0))
    df = pd.DataFrame(tot, columns=[f"c{k}" for k in range(12)]); df.insert(0, "tc_cell", c.tc_cell.values); df.insert(1, "year", int(YEAR)); df.insert(2, "source", LABEL)
    df.to_csv(OUT, index=False)
