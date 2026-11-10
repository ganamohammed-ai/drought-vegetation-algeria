"""Figure 4: correlation between monthly NDVI z and SPEI-3/6/12, by class and calendar month."""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
SD, OUT = sys.argv[1], sys.argv[2]
INK, INK2, SURF, GRID = "#1a1a19", "#55554f", "#fcfcfb", "#e4e3df"
s = pd.read_csv(f"{SD}/S4c_Monthly/Results/Sensitivity_median_r_class_month.csv")
ORDER = ["arid|steppe", "semi-arid|steppe", "semi-arid|shrubland", "semi-arid|forest", "dry sub-humid|forest", "humid|forest"]
LAB = ["Steppe aride", "Steppe semi-aride", "Matorral semi-aride", "Forêt semi-aride", "Forêt subhumide sèche", "Forêt humide"]
SC = [("spei03", "SPEI-3", "#9ecae1", (0, (1, 1.5))), ("spei06", "SPEI-6", "#4292c6", (0, (5, 2))), ("spei12", "SPEI-12", "#08306b", "-")]
fig, axes = plt.subplots(2, 3, figsize=(15, 8.2), sharey=True, sharex=True, facecolor=SURF)
fig.subplots_adjust(left=0.055, right=0.985, top=0.80, bottom=0.12, hspace=0.3, wspace=0.07)
months = np.arange(1, 13)
for ax, c, lab in zip(axes.ravel(), ORDER, LAB):
    d = s[s.cls == c].sort_values("month")
    ax.set_facecolor(SURF)
    ax.axvspan(2.5, 5.5, color="#ecebe6", lw=0, zorder=0)
    ax.axvspan(8.5, 11.5, color="#f6f5f1", lw=0, zorder=0)
    ax.axhline(0, color=INK2, lw=.7, zorder=1)
    for col, name, color, ls in SC:
        ax.plot(d.month, d[col], color=color, lw=2.2, ls=ls, zorder=3)
        j = d[col].idxmax()
    for col, name, color, ls in SC:
        last = d[d.month == 12][col].values[0]
    ax.set_title(lab, loc="left", fontsize=12, fontweight="bold", color=INK, pad=6)
    ax.set_xticks(months); ax.set_xticklabels(list("JFMAMJJASOND"), fontsize=9.5, color=INK2)
    ax.set_ylim(-0.12, 0.88); ax.set_xlim(.7, 12.3)
    ax.tick_params(labelsize=9.5, colors=INK2, length=2)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID)
for ax in axes[:, 0]: ax.set_ylabel("Corrélation médiane r", fontsize=10, color=INK2)
# direct labels on the first panel + legend
h = [plt.Line2D([0], [0], color=c, lw=2.4, ls=ls, label=n) for _, n, c, ls in SC]
fig.legend(handles=h, loc="upper right", bbox_to_anchor=(0.985, 0.885), ncol=3, frameon=False, fontsize=11,
           handlelength=3.2, columnspacing=1.8)
axes[0, 0].text(4, 0.80, "printemps", ha="center", fontsize=8.5, color=INK2)
axes[0, 0].text(10, 0.80, "automne", ha="center", fontsize=8.5, color=INK2)
fig.text(0.055, 0.945, "Plus la zone est humide, plus l'échelle de temps pertinente est courte en été–automne",
         fontsize=15, fontweight="bold", color=INK, ha="left")
fig.text(0.055, 0.905, "Corrélation médiane entre l'anomalie mensuelle du NDVI et le SPEI à trois échelles, par classe et mois calendaire (2000–2025).",
         fontsize=10, color=INK2, ha="left")
d = s[s.month.between(6, 11)].groupby("cls")[["spei03", "spei06", "spei12"]].mean().reindex(ORDER)
d.to_csv(f"{OUT}/Fig4_mean_r_by_scale_Jun_Nov.csv")
fig.text(0.055, 0.035, "Bandes : printemps (mars–mai) et automne (septembre–novembre). Juin–novembre, r moyen SPEI-3 / SPEI-12 : steppe aride 0,45 / 0,69 ; forêt semi-aride 0,44 / 0,60 ; forêt humide 0,50 / 0,36. Corrélations descriptives (séries autocorrélées).",
         fontsize=9, color=INK2)
fig.savefig(f"{OUT}/Fig4_echelles_SPEI_par_classe_mois.png", dpi=200, facecolor=SURF)
fig.savefig(f"{OUT}/Fig4_echelles_SPEI_par_classe_mois.svg", facecolor=SURF)
print("saved")
