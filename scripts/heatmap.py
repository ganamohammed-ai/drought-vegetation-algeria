"""Heatmap class x episode x season of the monthly NDVI anomaly during drought episodes.

Usage: python3 -I heatmap.py <study_dir> <out_dir>
Cell value = median over cells of the seasonal z (event months + 6 months after).
Hatched = 95 % block-bootstrap CI (0.5 deg blocks) of the median includes 0.
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

SD, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)

ORDER = ["arid|steppe", "semi-arid|steppe", "semi-arid|shrubland",
         "semi-arid|forest", "dry sub-humid|forest", "humid|forest"]
LABEL = ["Steppe aride", "Steppe semi-aride", "Matorral semi-aride",
         "Forêt semi-aride", "Forêt subhumide sèche", "Forêt humide"]

m = pd.read_csv(f"{SD}/S4c_Monthly/Results/Event_monthly_response_by_cell.csv")
sens = pd.read_csv(f"{SD}/S4_Response_v2/S4_sensitivity_by_cell.csv")[["tc_cell", "lon", "lat"]]
sens = sens.drop_duplicates("tc_cell")
m = m.merge(sens, on="tc_cell", how="left")
m["block"] = (np.floor(m.lon / 0.5).astype("Int64").astype(str) + "_" +
              np.floor(m.lat / 0.5).astype("Int64").astype(str))
m["gap"] = m.z_SON - m.z_MAM
ep = pd.read_csv(f"{SD}/S2b_Robustness/Drought_episodes_regional.csv").set_index("episode")
EPS = sorted(m.episode.unique())
print("episodes:", EPS, "| rows:", len(m), "| rows without coords:", int(m.lon.isna().sum()))

def block_ci(g, col, B=400):
    """median and 95 % CI, resampling spatial blocks."""
    v = g[col].dropna()
    if len(v) == 0:
        return np.nan, np.nan, np.nan, 0
    blocks = g.loc[v.index, "block"].values
    ub, inv = np.unique(blocks, return_inverse=True)
    groups = [v.values[inv == i] for i in range(len(ub))]
    meds = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        meds[b] = np.median(np.concatenate([groups[i] for i in pick]))
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return float(np.median(v)), lo, hi, len(v)

PANELS = [("z_DJF", "Hiver (DJF)"), ("z_MAM", "Printemps (MAM)"),
          ("z_JJA", "Été (JJA)"), ("z_SON", "Automne (SON)"),
          ("gap", "Automne − Printemps")]
res = {}
rows = []
for col, _ in PANELS:
    M = np.full((len(ORDER), len(EPS)), np.nan)
    LO = M.copy(); HI = M.copy(); N = np.zeros_like(M)
    for i, c in enumerate(ORDER):
        for j, e in enumerate(EPS):
            g = m[(m.cls == c) & (m.episode == e)]
            if len(g) < 30:
                continue
            M[i, j], LO[i, j], HI[i, j], N[i, j] = block_ci(g, col)
            rows.append(dict(season=col, cls=c, episode=e, n_cells=int(N[i, j]),
                             median=M[i, j], ci_low=LO[i, j], ci_high=HI[i, j]))
    res[col] = (M, LO, HI, N)
pd.DataFrame(rows).to_csv(f"{OUT}/heatmap_class_episode_season_values.csv", index=False)
allv = np.concatenate([res[c][0][~np.isnan(res[c][0])] for c, _ in PANELS[:4]])
print("seasonal medians range:", allv.min().round(2), allv.max().round(2))

# ---- colour: diverging blue <-> grey <-> red (reference palette), red = vegetation below normal
NEUTRAL, RED, BLUE = "#f0efec", "#e34948", "#2a78d6"
cmap = LinearSegmentedColormap.from_list("div", [RED, NEUTRAL, BLUE])
INK, INK2, SURF = "#1a1a19", "#55554f", "#fcfcfb"

def lum(rgb):
    r, g, b = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in rgb[:3]]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

fig = plt.figure(figsize=(21, 5.9), facecolor=SURF)
gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1.0, 0.05], wspace=0.07,
                      left=0.115, right=0.955, top=0.78, bottom=0.17)
axes = [fig.add_subplot(gs[0, k]) for k in range(5)]
cax = fig.add_subplot(gs[0, 5])
VMAX = 1.5
norm = Normalize(-VMAX, VMAX)
GAPMAX = 1.5
for k, (col, title) in enumerate(PANELS):
    ax = axes[k]
    M, LO, HI, N = res[col]
    nrm = Normalize(-GAPMAX, GAPMAX) if col == "gap" else norm
    for i in range(len(ORDER)):
        for j in range(len(EPS)):
            v = M[i, j]
            x0, y0 = j, i
            if np.isnan(v):
                ax.add_patch(Rectangle((x0, y0), 1, 1, facecolor=SURF, edgecolor=SURF, lw=2))
                ax.text(x0 + .5, y0 + .5, "–", ha="center", va="center", color="#8a8a85", fontsize=10)
                continue
            fc = cmap(nrm(v))
            ns = (LO[i, j] <= 0 <= HI[i, j])
            ax.add_patch(Rectangle((x0, y0), 1, 1, facecolor=fc, edgecolor=SURF, lw=2.5))
            if ns:
                ax.add_patch(Rectangle((x0, y0), 1, 1, facecolor="none", edgecolor=INK2,
                                       hatch="////", lw=0, alpha=.45))
            ax.text(x0 + .5, y0 + .5, f"{v:+.2f}".replace("-", "−"), ha="center", va="center",
                    fontsize=9.5, color=("#fcfcfb" if lum(fc) < 0.22 else INK),
                    fontweight="normal" if ns else "bold")
    ax.set_xlim(0, len(EPS)); ax.set_ylim(len(ORDER), 0)
    ax.set_xticks(np.arange(len(EPS)) + .5)
    ax.set_xticklabels([f"{e}\n{ep.loc[e,'start'][:4]}" if e in ep.index else e for e in EPS],
                       fontsize=8.5, color=INK2)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=2)
    ax.set_yticks(np.arange(len(ORDER)) + .5)
    if k == 0:
        ax.set_yticklabels(LABEL, fontsize=10.5, color=INK)
    else:
        ax.set_yticklabels([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=12.5, color=INK, pad=34, fontweight="bold", loc="left")
    ax.set_facecolor(SURF)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cb = fig.colorbar(sm, cax=cax)
cb.set_label("Anomalie z du NDVI\n(rouge = végétation sous la normale)", fontsize=9.5, color=INK2)
cb.ax.tick_params(labelsize=9, colors=INK2, length=0)
cb.outline.set_visible(False)
cb.set_ticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
cb.ax.set_yticklabels(["≤ −1,5", "−1", "−0,5", "0", "+0,5", "+1", "≥ +1,5"])

fig.text(0.115, 0.945, "Réponse saisonnière du NDVI aux épisodes de sécheresse, par classe et par épisode",
         fontsize=15, color=INK, fontweight="bold", ha="left")
fig.text(0.115, 0.905,
         "Médiane de l'anomalie z (mois de l'événement + 6 mois suivants), cellules TerraClimate × zone × couvert. "
         "Dernier panneau : automne − printemps (même échelle de couleurs).",
         fontsize=10, color=INK2, ha="left")
fig.text(0.115, 0.115,
         "Gras : IC95 % du bootstrap par blocs de 0,5° exclut 0.   Hachuré, valeur normale : IC inclut 0.   – : classe non éligible à l'épisode (< 30 cellules).   Valeurs hors de ±1,5 : couleur saturée (ex. E08, steppe aride, hiver : +2,10).\n"
         "E05 : 1999–2003, E12 : octobre 2019–décembre 2025 (en cours). Source : MOD13Q1 v061, TerraClimate SPEI ; masque T80, pixels-mois brûlés exclus.",
         fontsize=9, color=INK2, ha="left", va="top", linespacing=1.6)
fig.savefig(f"{OUT}/Fig_heatmap_classe_episode_saison.png", dpi=200, facecolor=SURF)
fig.savefig(f"{OUT}/Fig_heatmap_classe_episode_saison.svg", facecolor=SURF)
print("saved")
