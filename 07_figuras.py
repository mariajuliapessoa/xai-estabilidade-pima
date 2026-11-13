"""Etapa 7 — Figuras do artigo a partir das saídas da etapa 6 (não refaz cálculos).
Figura 1: τ-b médio (a) e % de pares com mudança do atributo principal (b) por σ:
          TreeSHAP (C2a = C2b), LIME C2a (sementes comuns), LIME C2b (sementes independentes)
          e a linha de base do LIME sem perturbação (C1).
Figura 2: L1 normalizada média nos pares com e sem mudança da classe predita (C2a), por método.
Uso: python 07_figuras.py"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import xai_estab as xe

OUT = "resultados"
D = pd.read_csv(xe.caminho(OUT, "tabelas", "descritivas_todas_metricas.csv"))
R = json.load(open(xe.caminho(OUT, "relatorio_resultados.json"), encoding="utf-8"))
os.makedirs(xe.caminho(OUT, "figuras"), exist_ok=True)

AZUL, LARANJA = "#2a78d6", "#eb6834"   # paleta validada (ΔE CVD 24,7)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman"],
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#555555", "axes.labelcolor": "#222222",
                     "xtick.color": "#444444", "ytick.color": "#444444"})
virg = matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:g}".replace(".", ","))
SIG = [0.01, 0.03, 0.05, 0.10]

SERIES = [("TreeSHAP", "C2a", "TreeSHAP", AZUL, "o", "-", True),
          ("LIME", "C2a", "LIME, sementes comuns (C2a)", LARANJA, "s", "-", True),
          ("LIME", "C2b", "LIME, sementes independentes (C2b)", LARANJA, "s", "--", False)]


def serie(met, cond, mt):
    return D[(D.condicao == cond) & (D.metodo == met) & (D.metrica == mt)].sort_values("sigma")


fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.3))
for met, cond, rot, cor, mk, ls, cheio in SERIES:
    g = serie(met, cond, "tau_b")
    axs[0].errorbar(g.sigma, g.media, yerr=[g.media - g.ic_inf, g.ic_sup - g.media], color=cor, marker=mk,
                    ms=6, lw=2, ls=ls, capsize=3, label=rot, mfc=cor if cheio else "white", mew=1.5)
    t = serie(met, cond, "top1")
    axs[1].errorbar(t.sigma, 100 * (1 - t.media), yerr=[100 * (t.ic_sup - t.media), 100 * (t.media - t.ic_inf)],
                    color=cor, marker=mk, ms=6, lw=2, ls=ls, capsize=3, mfc=cor if cheio else "white", mew=1.5)
c1 = D[(D.condicao == "C1") & (D.metodo == "LIME")]
axs[0].axhline(c1[c1.metrica == "tau_b"].media.item(), color="#777777", ls=":", lw=1.3)
axs[0].annotate("LIME sem perturbação (C1)", (0.058, c1[c1.metrica == "tau_b"].media.item()),
                xytext=(0, 4), textcoords="offset points", fontsize=8, color="#444444")
axs[1].axhline(100 * (1 - c1[c1.metrica == "top1"].media.item()), color="#777777", ls=":", lw=1.3)
axs[0].set_ylabel("τ-b médio por instância")
axs[0].set_title("(a) Concordância de ordem (τ-b)", fontsize=10, loc="left")
axs[1].set_ylabel("Pares com mudança do atributo principal (%)")
axs[1].set_title("(b) Mudança do atributo principal", fontsize=10, loc="left")
for ax in axs:
    ax.set_xlabel("Intensidade da perturbação (σ)")
    ax.set_xticks(SIG)
    ax.set_xticklabels(["0,01", "0,03", "0,05", "0,10"])
    ax.set_xlim(0, 0.11)
    ax.grid(axis="y", color="#e5e5e5", lw=0.8)
    ax.yaxis.set_major_formatter(virg)
fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(xe.caminho(OUT, "figuras", "figura1_robustez.png"), dpi=300, facecolor="white")

# ---------------------------------------------------------------- Figura 2 (A3, exploratória)
A3 = json.load(open(xe.caminho(OUT, "analises_complementares.json"), encoding="utf-8"))["A3_mecanismo"]
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
for ax, met, cor in [(axs[0], "TreeSHAP", AZUL), (axs[1], "LIME", LARANJA)]:
    com = [A3[f"{met}_{s}"]["l1_abs_com_mudanca"] for s in [0.01, 0.03, 0.05, 0.1]]
    sem = [A3[f"{met}_{s}"]["l1_abs_sem_mudanca"] for s in [0.01, 0.03, 0.05, 0.1]]
    ax.plot(SIG, com, color=cor, marker="o", lw=2, label="classe predita mudou")
    ax.plot(SIG, sem, color=cor, marker="o", lw=2, ls="--", mfc="white", mew=1.5, label="classe predita mantida")
    ax.set_title(f"({'a' if met == 'TreeSHAP' else 'b'}) {met}", fontsize=10, loc="left")
    ax.set_xlabel("Intensidade da perturbação (σ)")
    ax.set_xticks(SIG)
    ax.set_xticklabels(["0,01", "0,03", "0,05", "0,10"])
    ax.grid(axis="y", color="#e5e5e5", lw=0.8)
    ax.yaxis.set_major_formatter(virg)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
axs[0].set_ylabel("L1 não normalizada média por par")
fig.tight_layout()
fig.savefig(xe.caminho(OUT, "figuras", "figura2_predicao_explicacao.png"), dpi=300, facecolor="white")
print("Figuras salvas em", os.path.join(OUT, "figuras"))
