"""Figure 1 (study area + episodes) and Figure 3 (map of the autumn - spring gap).

Usage: python3 -I maps.py <study_dir> <out_dir>
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
from gpkg_lines import read_polys

SD, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
INK, INK2, SURF, GRID = "#1a1a19", "#55554f", "#fcfcfb", "#e4e3df"
polys = read_polys(f"{SD}/S0_Grid/Domain_NordALG_WGS84.gpkg")

def outline(ax, lw=0.6, color="#8a8a85", fill=None):
    for rings in polys:
        x, y = rings[0]
        if fill:
            ax.fill(x, y, color=fill, lw=0, zorder=0)
        ax.plot(x, y, color=color, lw=lw, zorder=3)

# ------------------------------------------------------------ zone raster
with tifffile.TiffFile(f"{SD}/S1_Climate/Aridity_Zones.tif") as t:
    Z = t.asarray().astype(float)
    sx, sy, _ = t.pages[0].tags[33550].value
    _, _, _, x0, y0, _ = t.pages[0].tags[33922].value
ny, nx = Z.shape
extent = (x0, x0 + nx * sx, y0 - ny * sy, y0)
Zm = np.ma.masked_where(Z == 0, Z)
ZONE_COL = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]   # one hue, light = arid, dark = humid
ZONE_LAB = ["Aride", "Semi-aride", "Subhumide sèche", "Humide"]

# ------------------------------------------------------------ episodes + drought share
ep = pd.read_csv(f"{SD}/S2b_Robustness/Drought_episodes_regional.csv")
sh = pd.read_csv(f"{SD}/S2_Drought/Drought_area_share_monthly.csv")
sh["t"] = pd.to_datetime(sh.month + "-01")

fig = plt.figure(figsize=(17, 8.2), facecolor=SURF)
gs = fig.add_gridspec(4, 2, width_ratios=[0.78, 1.5], hspace=0.18, wspace=0.07,
                      left=0.045, right=0.985, top=0.80, bottom=0.085)
axm = fig.add_subplot(gs[:, 0])
axm.set_facecolor(SURF)
outline(axm, fill="#f3f2ee")
axm.imshow(Zm, extent=extent, cmap=ListedColormap(ZONE_COL), vmin=1, vmax=4,
           interpolation="nearest", zorder=1, alpha=.95)
outline(axm, lw=0.5, color="#6f6f6a")
axm.set_xlim(-2.4, 9.1); axm.set_ylim(30.6, 37.4)
axm.set_aspect(1 / np.cos(np.deg2rad(34)))
axm.set_xlabel("Longitude (°E)", fontsize=9.5, color=INK2); axm.set_ylabel("Latitude (°N)", fontsize=9.5, color=INK2)
axm.tick_params(labelsize=9, colors=INK2, length=3)
for s in ("top", "right"):
    axm.spines[s].set_visible(False)
for s in ("left", "bottom"):
    axm.spines[s].set_color(GRID)
axm.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(ZONE_COL, ZONE_LAB)], loc="lower right",
           frameon=False, fontsize=9.5, title="Zone bioclimatique (P/PET)", title_fontsize=9.5)
axm.set_title("a. Domaine d'étude et zones bioclimatiques", loc="left", fontsize=12.5,
              fontweight="bold", color=INK, pad=8)

zones = ["arid", "semi-arid", "dry sub-humid", "humid"]
axs = []
for i, z in enumerate(zones):
    ax = fig.add_subplot(gs[i, 1], sharex=axs[0] if axs else None)
    axs.append(ax)
    ax.set_facecolor(SURF)
    for _, r in ep.iterrows():
        a = pd.to_datetime(r.start + "-01"); b = pd.to_datetime(r.end + "-01") + pd.offsets.MonthEnd(1)
        ax.axvspan(a, b, color="#ecebe6", lw=0, zorder=0)
        if i == 0:
            ax.text(a + (b - a) / 2, 1.13, r.episode.replace("E", "E"), ha="center", va="bottom",
                    fontsize=8.2, color=INK2, clip_on=False)
    ax.fill_between(sh.t, sh[z], color="#e34948", alpha=.88, lw=0, zorder=2)
    ax.axvline(pd.Timestamp("2000-03-01"), color=INK, lw=1, ls=(0, (4, 3)), zorder=3)
    ax.set_ylim(0, 1.0); ax.set_yticks([0, .5, 1]); ax.set_yticklabels(["0", "50 %", "100 %"])
    ax.tick_params(labelsize=8.5, colors=INK2, length=2)
    ax.text(0.006, 0.9, ZONE_LAB[i], transform=ax.transAxes, fontsize=10.5, color=INK, fontweight="bold",
            va="top", bbox=dict(facecolor=SURF, edgecolor="none", alpha=.9, pad=2.5), zorder=5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    if i < 3:
        plt.setp(ax.get_xticklabels(), visible=False)
axs[-1].set_xlim(pd.Timestamp("1986-01-01"), pd.Timestamp("2026-01-01"))
axs[0].set_title("b. Part de la surface de chaque zone en sécheresse (SPEI-12 ≤ −1) et épisodes régionaux",
                 loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=30)
axs[0].text(pd.Timestamp("2000-06-01"), 0.97, "début MODIS\n(mars 2000)", fontsize=8.2, color=INK, va="top", ha="left", bbox=dict(facecolor=SURF, edgecolor="none", alpha=.85, pad=1.5), zorder=6)
fig.text(0.045, 0.955, "Zone d'étude et épisodes de sécheresse régionaux, 1986–2025", fontsize=15,
         fontweight="bold", color=INK, ha="left")
fig.text(0.045, 0.92, "Grille TerraClimate (~4 km), nord de l'Algérie ; zone hyperaride exclue. Bandes grises : épisodes régionaux E01–E12 (SPEI-12).",
         fontsize=10, color=INK2, ha="left")
fig.text(0.045, 0.03, "Sources : TerraClimate (SPEI, P/PET 1991–2020), limites administratives du domaine. E12 est en cours à la fin de la série (décembre 2025).",
         fontsize=9, color=INK2)
fig.savefig(f"{OUT}/Fig1_zone_etude_episodes.png", dpi=200, facecolor=SURF)
fig.savefig(f"{OUT}/Fig1_zone_etude_episodes.svg", facecolor=SURF)
plt.close(fig)

# ------------------------------------------------------------ gap map
mo = pd.read_csv(f"{SD}/S4c_Monthly/Results/Event_monthly_response_by_cell.csv")
sens = pd.read_csv(f"{SD}/S4_Response_v2/S4_sensitivity_by_cell.csv")[["tc_cell", "lon", "lat"]].drop_duplicates("tc_cell")
mo = mo.merge(sens, on="tc_cell", how="left")
mo["gap"] = mo.z_SON - mo.z_MAM
cell = mo.groupby("tc_cell").agg(gap=("gap", "mean"), lon=("lon", "first"), lat=("lat", "first"),
                                 n_events=("gap", "size")).reset_index()
lo = np.round((cell.lon.values - x0) / sx - 0.5).astype(int)
la = np.round((y0 - cell.lat.values) / sy - 0.5).astype(int)
G = np.full(Z.shape, np.nan)
ok = (lo >= 0) & (lo < nx) & (la >= 0) & (la < ny)
G[la[ok], lo[ok]] = cell.gap.values[ok]
print("cells on the map:", int(ok.sum()), "of", len(cell))
cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
fig = plt.figure(figsize=(16, 7.8), facecolor=SURF)
gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 0.9], wspace=0.34, left=0.05, right=0.97, top=0.85, bottom=0.2)
ax = fig.add_subplot(gs[0]); axb = fig.add_subplot(gs[1])
cax = fig.add_axes([0.12, 0.105, 0.30, 0.022])
outline(ax, fill="#f3f2ee")
im = ax.imshow(np.ma.masked_invalid(G), extent=extent, cmap=cmap, norm=Normalize(-1, 1),
               interpolation="nearest", zorder=1)
outline(ax, lw=0.5, color="#6f6f6a")
ax.set_xlim(-2.4, 9.1); ax.set_ylim(30.6, 37.4); ax.set_aspect(1 / np.cos(np.deg2rad(34)))
ax.set_xlabel("Longitude (°E)", fontsize=9.5, color=INK2); ax.set_ylabel("Latitude (°N)", fontsize=9.5, color=INK2)
ax.tick_params(labelsize=9, colors=INK2, length=3)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_label("z automne − z printemps (rouge : plus négatif à l'automne)", fontsize=9.5, color=INK2)
cb.ax.tick_params(labelsize=9, colors=INK2, length=0); cb.outline.set_visible(False)
cb.set_ticks([-1, -.5, 0, .5, 1]); cb.ax.set_xticklabels(["≤ −1", "−0,5", "0", "+0,5", "≥ +1"])
ax.set_title("a. Écart automne − printemps par cellule (moyenne des épisodes)", loc="left", fontsize=12.5,
             fontweight="bold", color=INK, pad=8)

ORDER = ["arid|steppe", "semi-arid|steppe", "semi-arid|shrubland", "semi-arid|forest", "dry sub-humid|forest", "humid|forest"]
LAB = ["Steppe aride", "Steppe semi-aride", "Matorral semi-aride", "Forêt semi-aride", "Forêt subhumide sèche", "Forêt humide"]
cc = mo.groupby(["cls", "tc_cell"]).gap.mean().reset_index()
stats = [cc[cc.cls == c].gap.quantile([.1, .25, .5, .75, .9]).values for c in ORDER]
for i, (st, c) in enumerate(zip(stats, ORDER)):
    axb.plot([st[0], st[4]], [i, i], color="#8a8a85", lw=1.2, zorder=1)
    axb.plot([st[1], st[3]], [i, i], color="#8a8a85", lw=6, solid_capstyle="butt", zorder=2)
    axb.plot(st[2], i, marker="o", ms=9, color="#1a1a19", mec=SURF, mew=2, zorder=3)
    axb.text(1.12, i, f"{st[2]:+.2f}".replace("-", "−"), va="center", fontsize=9.5, color=INK)
axb.axvline(0, color=INK2, lw=.8, zorder=0)
axb.set_ylim(len(ORDER) - .4, -.6); axb.set_xlim(-1.5, 1.35)
axb.set_yticks(range(len(ORDER))); axb.set_yticklabels(LAB, fontsize=10.5, color=INK)
axb.tick_params(axis="x", labelsize=9, colors=INK2); axb.tick_params(axis="y", length=0)
axb.set_xlabel("Écart z automne − printemps (médiane, intervalle 25–75 %, 10–90 %)", fontsize=9.5, color=INK2)
for s in ("top", "right", "left"):
    axb.spines[s].set_visible(False)
axb.spines["bottom"].set_color(GRID)
axb.set_title("b. Distribution par classe", loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=8)
fig.text(0.05, 0.945, "Les forêts du Tell perdent davantage à l'automne qu'au printemps ; les steppes non",
         fontsize=15, fontweight="bold", color=INK, ha="left")
fig.text(0.05, 0.91, "Anomalie z du NDVI (MOD13Q1, moyenne saisonnière pendant l'événement et les 6 mois suivants), cellules TerraClimate × couvert.",
         fontsize=10, color=INK2, ha="left")
fig.text(0.05, 0.012, "Une cellule peut figurer pour plusieurs couverts : la carte montre la moyenne de ses événements ; le panneau b compte chaque couple cellule × classe.",
         fontsize=9, color=INK2)
fig.savefig(f"{OUT}/Fig3_carte_ecart_automne_printemps.png", dpi=200, facecolor=SURF)
fig.savefig(f"{OUT}/Fig3_carte_ecart_automne_printemps.svg", facecolor=SURF)
pd.DataFrame({"cls": LAB, "p10": [s[0] for s in stats], "p25": [s[1] for s in stats], "median": [s[2] for s in stats],
              "p75": [s[3] for s in stats], "p90": [s[4] for s in stats]}).to_csv(f"{OUT}/Fig3_gap_by_class_quantiles.csv", index=False)
print("saved")
