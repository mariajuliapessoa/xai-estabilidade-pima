"""Etapa 3 — Estudo piloto para definir M (versões perturbadas por instância).
Usa 20 instâncias do TREINO e o modelo de referência, na condição C2a.
Critério: menor M da grade tal que, para todo método e sigma, a variação
relativa da média de rho (Spearman) para qualquer M maior da grade é < 1%.
Uso: python 03_piloto.py [--smoke]"""
import pickle
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import xai_estab as xe

SMOKE = "--smoke" in sys.argv
cfg = xe.carregar_config()
alvo = cfg["dados"]["alvo"]
tr = pd.read_csv(xe.caminho("intermediarios", "treino_imputado.csv"))
nomes = [c for c in tr.columns if c != alvo]
X, y = tr[nomes].values, tr[alvo].values
pre = xe.ler_json("intermediarios", "preprocessamento.json")
with open(xe.caminho("intermediarios", "modelos.pkl"), "rb") as f:
    modelo = pickle.load(f)[cfg["modelo"]["semente_referencia"]]

pc = cfg["piloto"]
n_inst = 3 if SMOKE else pc["n_instancias"]
grade_M = [2, 4] if SMOKE else pc["grade_M"]
Mmax = max(grade_M)
rng = np.random.default_rng(pc["semente_amostra"])
# amostra estratificada por classe
idx = np.concatenate([rng.choice(np.where(y == c)[0], size=int(round(n_inst * np.mean(y == c))), replace=False)
                      for c in (0, 1)])
sigmas = cfg["perturbacao"]["sigmas"]
pb = cfg["perturbacao"]["semente_base"]
lb = cfg["repeticoes"]["semente_base_lime"]


def tarefa(i):
    shap_e = xe.ExplicadorTreeSHAP(modelo, cfg)
    lime_e = xe.ExplicadorLIME(modelo, X, nomes, cfg)
    x = X[i]
    phi_s = shap_e.explicar(x)[0]
    linhas = []
    for si, s in enumerate(sigmas):
        Xp = xe.perturbar(x, s, pre, nomes, cfg, np.random.default_rng(xe.semente(pb, 9, i, si)), Mmax)
        phis_p = shap_e.explicar(Xp)
        for m in range(Mmax):
            r = xe.semente(lb, 9, i, si, m)
            phi_l, _ = lime_e.explicar(x, r)
            phi_lp, _ = lime_e.explicar(Xp[m], r)
            linhas.append({"inst": int(i), "sigma": s, "m": m, "metodo": "TreeSHAP",
                           "spearman": xe.metricas_par(phi_s, phis_p[m])["spearman"]})
            linhas.append({"inst": int(i), "sigma": s, "m": m, "metodo": "LIME",
                           "spearman": xe.metricas_par(phi_l, phi_lp)["spearman"]})
    return linhas


t0 = time.time()
res = Parallel(n_jobs=cfg["n_processos"])(delayed(tarefa)(i) for i in idx)
df = pd.DataFrame([l for r in res for l in r])

curva = []
for (met, s), g in df.groupby(["metodo", "sigma"]):
    for M in grade_M:
        rho = g[g.m < M].groupby("inst")["spearman"].mean().mean()
        curva.append({"metodo": met, "sigma": s, "M": M, "rho_medio": rho})
curva = pd.DataFrame(curva)

tol = pc["tolerancia_relativa"]
M_escolhido = None
for M in grade_M:
    ok = True
    for (met, s), g in curva.groupby(["metodo", "sigma"]):
        base = g.loc[g.M == M, "rho_medio"].item()
        maiores = g.loc[g.M > M, "rho_medio"].values
        if len(maiores) and np.any(np.abs(maiores - base) / max(abs(base), 1e-12) >= tol):
            ok = False
    if ok:
        M_escolhido = M
        break

saida = "piloto_smoke" if SMOKE else "piloto"
df.to_csv(xe.caminho("intermediarios", f"{saida}_pares.csv"), index=False)
curva.to_csv(xe.caminho("intermediarios", f"{saida}_curva.csv"), index=False)
xe.salvar_json({"instancias_treino": idx.tolist(), "grade_M": grade_M, "tolerancia": tol,
                "M_escolhido": M_escolhido, "tempo_s": round(time.time() - t0, 1),
                "data": xe.agora(), "smoke": SMOKE}, "intermediarios", f"{saida}.json")
print(curva.pivot_table(index=["metodo", "sigma"], columns="M", values="rho_medio").round(4))
print("M escolhido:", M_escolhido, "| tempo (s):", round(time.time() - t0, 1))
if not SMOKE:
    print("Copie esse valor para repeticoes.M em config.json antes de congelar o protocolo.")
