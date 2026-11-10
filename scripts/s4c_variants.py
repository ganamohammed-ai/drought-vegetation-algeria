"""Task 4: monthly seasonal response with MSAVI2 and with sensitivity to the missing composites.

Usage: python3 -I s4c_variants.py <study_dir> <out_dir>
Re-uses the S4c pipeline (z vs 2001-2020 monthly climatology, event months + 6 months) and computes the seasonal
means for: NDVI (replication), MSAVI2, NDVI with the known missing months set to NaN explicitly,
NDVI without 2025 (censored end of E12), MSAVI2 without 2025.
"""
import os, sys
import numpy as np, pandas as pd

ROOT, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
ORDER = ["arid|steppe", "semi-arid|steppe", "semi-arid|shrubland", "semi-arid|forest", "dry sub-humid|forest", "humid|forest"]
BASE = (2001, 2020); MIN_BASE = 10; MIN_VALID = 0.5
MONTHS = range(1, 13)
SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}

m = pd.read_csv(f"{ROOT}/S4c_Monthly/S4c_monthly_by_cell_cover.csv.gz")
m["cls"] = m.zone + "|" + m.cover
m = m[m.cls.isin(ORDER)].copy()
id_vars = ["tc_cell", "zone", "cover", "cls", "year"]
parts = []
for v in ("ndvi", "msavi2"):
    t = m.melt(id_vars=id_vars, value_vars=[f"{v}_{k:02d}" for k in MONTHS], var_name="vm", value_name=v)
    t["month"] = t.vm.str[-2:].astype(int)
    parts.append(t.drop(columns="vm").set_index(id_vars + ["month"]))
va = m.melt(id_vars=id_vars, value_vars=[f"valid_{k:02d}" for k in MONTHS], var_name="vm", value_name="valid")
va["month"] = va.vm.str[-2:].astype(int)
L = parts[0].join(parts[1]).join(va.drop(columns="vm").set_index(id_vars + ["month"])).reset_index()
del parts, va
for v in ("ndvi", "msavi2"):
    L.loc[L.valid < MIN_VALID, v] = np.nan
key = ["tc_cell", "zone", "cover", "month"]
for v in ("ndvi", "msavi2"):
    b = L[(L.year >= BASE[0]) & (L.year <= BASE[1])].groupby(key)[v].agg(["mean", "std", "count"]).reset_index()
    b.columns = key + [v + "_mu", v + "_sd", v + "_nb"]
    L = L.merge(b, on=key, how="left")
    ok = (L[v + "_nb"] >= MIN_BASE) & (L[v + "_sd"] > 0.002)
    L["z_" + v] = np.where(ok, (L[v] - L[v + "_mu"]) / L[v + "_sd"], np.nan)
    L = L.drop(columns=[v + "_mu", v + "_sd", v + "_nb", v])
L["t"] = L.year * 12 + L.month - 1
L["season"] = L.month.map(SEASON)
# known gaps in the source
missing = ((L.year == 2023) & (L.month == 2)) | ((L.year == 2024) & (L.month == 8)) | ((L.year == 2025) & L.month.between(7, 10))
print("cell-months in the known missing periods:", int(missing.sum()),
      "| of which with a valid NDVI z in the file:", int((missing & L.z_ndvi.notna()).sum()))
L["z_ndvi_nm"] = np.where(missing, np.nan, L.z_ndvi)
L["z_ndvi_no2025"] = np.where(L.year == 2025, np.nan, L.z_ndvi)
L["z_msavi2_no2025"] = np.where(L.year == 2025, np.nan, L.z_msavi2)
ZC = ["z_ndvi", "z_msavi2", "z_ndvi_nm", "z_ndvi_no2025", "z_msavi2_no2025"]
L = L[["tc_cell", "zone", "cover", "t", "month", "season"] + ZC]

ev = pd.read_csv(f"{ROOT}/S4_Response_v2/S4_event_response_by_cell.csv")
ev = ev[(ev.variant == "T80_250m") & (ev.fire_history == 0) & (ev.episode != "E00_local")].copy()
ev["cls"] = ev.zone_name + "|" + ev.cover
ev = ev[ev.cls.isin(ORDER)].reset_index(drop=True)
ev["t0"] = ev.onset.str[:4].astype(int) * 12 + ev.onset.str[5:7].astype(int) - 1
ev["t1"] = ev.end.str[:4].astype(int) * 12 + ev.end.str[5:7].astype(int) - 1
ev["eid"] = np.arange(len(ev))
nm = (ev.t1 - ev.t0 + 7).values
X = pd.DataFrame({"eid": np.repeat(ev.eid.values, nm)})
X["t"] = np.repeat(ev.t0.values, nm) + (np.arange(len(X)) - np.repeat(np.cumsum(nm) - nm, nm))
X = X.merge(ev[["eid", "tc_cell", "zone_name", "cover", "cls", "episode"]].rename(columns={"zone_name": "zone"}), on="eid")
X = X.merge(L, on=["tc_cell", "zone", "cover", "t"], how="left")
print("event-month rows:", len(X))

rows, gaps = [], []
for z in ZC:
    S = X.groupby(["eid", "season"])[z].mean().unstack()
    S["n_win"] = X.groupby("eid")[z].count()
    E = ev[["eid", "tc_cell", "cls", "episode", "n_pixels", "peak_spei"]].set_index("eid").join(S)
    E = E[E.n_win >= 3].copy()
    E["gap"] = E.z_SON - E.z_MAM if "z_SON" in E else np.nan
    E = E.rename(columns={c: c for c in E.columns})
    E["gap"] = E["SON"] - E["MAM"]
    if z == "z_ndvi": E.reset_index().to_csv(f"{OUT}/V0_event_seasonal_z_ndvi.csv", index=False)
    if z == "z_msavi2": E.reset_index().to_csv(f"{OUT}/V0_event_seasonal_z_msavi2.csv", index=False)
    cm = E.groupby("cls")[["DJF", "MAM", "JJA", "SON", "gap"]].median().reindex(ORDER)
    cm.insert(0, "variant", z)
    cm["n"] = E.groupby("cls").size().reindex(ORDER).values
    rows.append(cm.reset_index())
    ge = E.groupby(["episode", "cls"]).gap.median().unstack().reindex(columns=ORDER)
    ge.insert(0, "variant", z)
    gaps.append(ge.reset_index())
R = pd.concat(rows)
R.to_csv(f"{OUT}/V1_class_medians_by_variant.csv", index=False)
G = pd.concat(gaps)
G.to_csv(f"{OUT}/V2_gap_by_episode_class_by_variant.csv", index=False)
pd.set_option("display.width", 200)
print("\nMedian SON - MAM gap by class and variant (n cell-episodes in the last column of the NDVI row):")
print(R.pivot(index="cls", columns="variant", values="gap").reindex(ORDER).round(2).to_string())
print("\nMedian z_SON by class and variant:")
print(R.pivot(index="cls", columns="variant", values="SON").reindex(ORDER).round(2).to_string())
print("\nGap by episode, humid forest and dry sub-humid forest, per variant:")
for z in ZC:
    g = G[G.variant == z].set_index("episode")
    print(z, "| humid:", {e: round(v, 2) for e, v in g["humid|forest"].dropna().items()},
          "| subhumid:", {e: round(v, 2) for e, v in g["dry sub-humid|forest"].dropna().items()})
