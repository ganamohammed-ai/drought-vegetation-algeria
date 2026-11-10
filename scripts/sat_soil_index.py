r"""Anomalies saisonnieres (z) de l'humidite du sol satellitaire (CCI, SMAP) au format de h1_chirps.py (month 5 = MAM, 11 = SON ; positif = humide).
Usage: python3 -I sat_soil_index.py <csv> <colonne> <annee_base_debut> <annee_base_fin> <out_csv> [min_mois=2]"""
import sys, numpy as np, pandas as pd
f, col, b0, b1, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]; minm = int(sys.argv[6]) if len(sys.argv) > 6 else 2
d = pd.read_csv(f)[["tc_cell", "year", "month", col]].dropna(); R = []
for end, ms in ((5, [3, 4, 5]), (11, [9, 10, 11])):
    s = d[d.month.isin(ms)].groupby(["tc_cell", "year"])[col].agg(["mean", "count"]).reset_index(); s = s[s["count"] >= minm]
    b = s[s.year.between(b0, b1)].groupby("tc_cell")["mean"].agg(["mean", "std", "count"]).reset_index().rename(columns={"mean": "bm", "std": "bs", "count": "bn"})
    s = s.merge(b, on="tc_cell"); s = s[(s.bn >= max(6, (b1-b0+1)//2)) & (s.bs > 0)]; s["spei03"] = (s["mean"]-s.bm)/s.bs; s["month"] = end
    R.append(s[["tc_cell", "year", "month", "spei03"]])
pd.concat(R).to_csv(out, index=False); print(out, len(pd.concat(R)))
