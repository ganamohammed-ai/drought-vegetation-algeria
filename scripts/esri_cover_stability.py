r"""Stabilite du couvert (ESRI LULC 10 m) dans chaque cellule TerraClimate, par annee.
Usage: python3 -I esri_cover_stability.py <dir_tifs> <group_meta.csv> <out_csv> <year> [<year> ...]
Sortie: tc_cell, year, classe ESRI (1 eau, 2 arbres, 4 vegetation inondee, 5 cultures, 7 bati, 8 sol nu, 9 neige, 10 nuages, 11 parcours), n pixels."""
import sys, os, zlib, numpy as np, pandas as pd, tifffile
from multiprocessing import Pool
D, GM, OUT = sys.argv[1:4]; YEARS = [int(a) for a in sys.argv[4:]]
cs = 1/24; PX = 1e-4; X0, Y0 = -2.25, 37.15
m = pd.read_csv(GM); m = m[m.keep]; c = m.groupby("tc_cell")[["lon","lat"]].first().reset_index()
ci = np.floor(c.lon.values/cs).astype(int); cj = np.floor(c.lat.values/cs).astype(int)
i0, j0 = ci.min(), cj.min(); NI, NJ = ci.max()-i0+1, cj.max()-j0+1
LUT = np.full((NJ, NI), -1, np.int32); LUT[cj-j0, ci-i0] = np.arange(len(c))
def tile_cells(tx, ty, W, H, T=512):
    px = tx*T + np.arange(T); py = ty*T + np.arange(T)
    lon = X0 + (px+0.5)*PX; lat = Y0 - (py+0.5)*PX
    cx = np.floor(lon/cs).astype(int)-i0; cy = np.floor(lat/cs).astype(int)-j0
    okx = (cx >= 0) & (cx < NI) & (px < W); oky = (cy >= 0) & (cy < NJ) & (py < H)
    ids = np.full((T, T), -1, np.int32)
    if okx.any() and oky.any():
        sub = LUT[np.clip(cy, 0, NJ-1)[:, None], np.clip(cx, 0, NI-1)[None, :]]
        sub[~oky, :] = -1; sub[:, ~okx] = -1; ids = sub
    return ids
def work(args):
    path, k0, k1 = args
    t = tifffile.TiffFile(path); p = t.pages[0]; W, H = p.imagewidth, p.imagelength; T = p.tilewidth
    ntx = -(-W//T); fh = t.filehandle; out = np.zeros((len(c), 12), np.int64)
    for k in range(k0, k1):
        ty, tx = divmod(k, ntx); ids = tile_cells(tx, ty, W, H, T)
        if (ids < 0).all(): continue
        fh.seek(p.dataoffsets[k]); raw = zlib.decompress(fh.read(p.databytecounts[k]))
        a = np.frombuffer(raw, np.uint8).reshape(T, T)
        s = ids >= 0; out += np.bincount(ids[s].astype(np.int64)*12 + np.minimum(a[s], 11), minlength=len(c)*12).reshape(len(c), 12)
    return out
rows = []
for y in YEARS:
    path = f"{D}/Esri_LULC_10m_{y}_Nord_Algerie.tif"; p = tifffile.TiffFile(path).pages[0]
    n = len(p.dataoffsets); step = 600
    with Pool(os.cpu_count()) as pool:
        res = pool.map(work, [(path, a, min(a+step, n)) for a in range(0, n, step)])
    tot = sum(res); print(y, "ok", tot.sum(), flush=True)
    df = pd.DataFrame(tot, columns=[f"c{k}" for k in range(12)]); df.insert(0, "tc_cell", c.tc_cell.values); df.insert(1, "year", y); rows.append(df)
pd.concat(rows).to_csv(OUT, index=False)
