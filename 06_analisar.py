"""Etapa 6 — Análise estatística pré-especificada (Seção 6.8 do artigo) e
geração das Tabelas 1 e 2. Unidade de análise: instância (média sobre as M
versões ou as K execuções). Uso: python 06_analisar.py [--smoke]"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

import xai_estab as xe

SMOKE = "--smoke" in sys.argv
cfg = xe.carregar_config()
if not SMOKE:
    xe.exigir_protocolo_congelado(cfg)
OUT = "resultados_smoke" if SMOKE else "resultados"
TAB = os.path.join(OUT, "tabelas")
os.makedirs(xe.caminho(TAB), exist_ok=True)
B, sb, alfa = cfg["analise"]["bootstrap_B"], cfg["analise"]["semente_bootstrap"], cfg["analise"]["alfa"]
if SMOKE:
    B = 200
prim = cfg["analise"]["metrica_primaria"]
sigmas = cfg["perturbacao"]["sigmas"]

C1 = pd.read_csv(xe.caminho(OUT, "C1_variabilidade_intrinseca.csv"))
C2 = pd.read_csv(xe.caminho(OUT, "C2_robustez_pares.csv"))
EF = pd.read_csv(xe.caminho(OUT, "perturbacoes_efetivas.csv"))
desemp = xe.ler_json(OUT, "desempenho.json")
rel = {}

# agregação por instância
agg_cols = xe.METRICAS + ["dif_prob", "mudou_classe"]
I2 = C2.groupby(["condicao", "metodo", "sigma", "inst"])[agg_cols].mean().reset_index()
I1 = C1.groupby(["metodo", "inst"])[xe.METRICAS].mean().reset_index()


def fmt(v, d=3):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}".replace(".", ",")


# ---------------------------------------------------------------- Tabela 1
linhas = ["| Métrica | Valor | IC 95% |", "|:--|:--:|:--:|"]
for k, rot in [("acuracia", "Acurácia"), ("auc_roc", "AUC-ROC"), ("f1", "F1"), ("brier", "Escore de Brier")]:
    v = desemp[k]
    linhas.append(f"| {rot} | {fmt(v['valor'])} | {fmt(v['ic95'][0])} – {fmt(v['ic95'][1])} |")
open(xe.caminho(TAB, "tabela1_desempenho.md"), "w").write("\n".join(linhas) + "\n")

# ---------------------------------------------------------------- Tabela 2 (C2a)
t2 = []
for met in ["TreeSHAP", "LIME"]:
    for s in sigmas:
        g = I2[(I2.condicao == "C2a") & (I2.metodo == met) & (I2.sigma == s)]
        tau = xe.bootstrap_ic(g.tau_b, B, sb)
        jac = xe.bootstrap_ic(g.jaccard3, B, sb)
        t2.append({"metodo": met, "sigma": s, "tau_b": tau, "jaccard3": jac,
                   "mudanca_top1_pct": 100 * (1 - g.top1.mean()), "mudanca_classe_pct": 100 * g.mudou_classe.mean(),
                   "n_inst": int(len(g))})
linhas = ["| Método | σ | τ-b médio (IC 95%) | J@3 médio (IC 95%) | Mudança do atributo principal (%) | Mudança da classe predita (%) |",
          "|:--|:--:|:--:|:--:|:--:|:--:|"]
for r in t2:
    linhas.append(f"| {r['metodo']} | {fmt(r['sigma'], 2)} | {fmt(r['tau_b'][0])} ({fmt(r['tau_b'][1])}–{fmt(r['tau_b'][2])}) | "
                  f"{fmt(r['jaccard3'][0])} ({fmt(r['jaccard3'][1])}–{fmt(r['jaccard3'][2])}) | "
                  f"{fmt(r['mudanca_top1_pct'], 1)} | {fmt(r['mudanca_classe_pct'], 1)} |")
open(xe.caminho(TAB, "tabela2_robustez_C2a.md"), "w").write("\n".join(linhas) + "\n")
rel["tabela2"] = t2

# ------------------------------------------- todas as métricas, todas as condições
desc = []
for (cond, met, s), g in I2.groupby(["condicao", "metodo", "sigma"]):
    for mt in xe.METRICAS:
        v, lo, hi = xe.bootstrap_ic(g[mt], B, sb)
        desc.append({"condicao": cond, "metodo": met, "sigma": s, "metrica": mt, "media": v, "ic_inf": lo, "ic_sup": hi,
                     "n_inst": int(g[mt].notna().sum())})
for met, g in I1.groupby("metodo"):
    for mt in xe.METRICAS:
        v, lo, hi = xe.bootstrap_ic(g[mt], B, sb)
        desc.append({"condicao": "C1", "metodo": met, "sigma": 0, "metrica": mt, "media": v, "ic_inf": lo, "ic_sup": hi,
                     "n_inst": int(g[mt].notna().sum())})
pd.DataFrame(desc).to_csv(xe.caminho(TAB, "descritivas_todas_metricas.csv"), index=False)

# ------------------------------------- H2: determinismo do TreeSHAP
rel["H2_treeshap_max_dif_abs"] = float(C1.loc[C1.metodo == "TreeSHAP", "max_dif_abs"].max())

# ------------------------------------- análise primária (H3): TreeSHAP × LIME em C2a
prim_res = []
for s in sigmas:
    a = I2[(I2.condicao == "C2a") & (I2.metodo == "TreeSHAP") & (I2.sigma == s)].set_index("inst")[prim]
    b = I2[(I2.condicao == "C2a") & (I2.metodo == "LIME") & (I2.sigma == s)].set_index("inst")[prim]
    a, b = a.align(b, join="inner")
    prim_res.append({"sigma": s, **xe.wilcoxon_pareado(a.values, b.values, B, sb)})
aj = xe.holm([r["p"] for r in prim_res])
for r, p in zip(prim_res, aj):
    r["p_holm"] = p
rel["primaria_H3_TreeSHAP_menos_LIME"] = prim_res

# ------------------------------------- H1: tendência com sigma (Friedman + pares Holm)
h1 = {}
for met in ["TreeSHAP", "LIME"]:
    W = I2[(I2.condicao == "C2a") & (I2.metodo == met)].pivot(index="inst", columns="sigma", values=prim).dropna()
    out = {"n": int(len(W))}
    if W.shape[1] >= 3 and len(W) > 2 and W.values.std() > 0:
        fr = stats.friedmanchisquare(*[W[c].values for c in W.columns])
        out.update({"friedman_chi2": float(fr.statistic), "p": float(fr.pvalue)})
    pares = [(W.columns[i], W.columns[j]) for i in range(len(W.columns)) for j in range(i + 1, len(W.columns))]
    pr = [{"sigmas": [float(c1), float(c2)], **xe.wilcoxon_pareado(W[c1].values, W[c2].values, B, sb)} for c1, c2 in pares]
    for r, p in zip(pr, xe.holm([r["p"] for r in pr])):
        r["p_holm"] = p
    out["pares"] = pr
    h1[met] = out
rel["H1_tendencia_sigma"] = h1

# ------------------------------------- excesso de variação do LIME
exc = []
base_c1 = I1[I1.metodo == "LIME"].set_index("inst")
for cond in ["C2b", "C2a"]:
    for s in sigmas:
        g = I2[(I2.condicao == cond) & (I2.metodo == "LIME") & (I2.sigma == s)].set_index("inst")
        for mt in [prim, "l1_norm"]:
            d = (g[mt] - base_c1[mt]).dropna()
            v, lo, hi = xe.bootstrap_ic(d, B, sb)
            exc.append({"comparacao": f"{cond} − C1", "sigma": s, "metrica": mt, "media": v, "ic_inf": lo, "ic_sup": hi})
rel["excesso_LIME"] = exc

# ------------------------------------- H4 (exploratória)
h4 = []
for (met, s), g in C2[C2.condicao == "C2a"].groupby(["metodo", "sigma"]):
    gi = I2[(I2.condicao == "C2a") & (I2.metodo == met) & (I2.sigma == s)]
    rho = stats.spearmanr(gi.dif_prob, gi.l1_norm, nan_policy="omit").statistic if gi.l1_norm.std() > 0 else np.nan
    h4.append({"metodo": met, "sigma": s,
               "l1_medio_com_mudanca": float(g.loc[g.mudou_classe == 1, "l1_norm"].mean()) if (g.mudou_classe == 1).any() else None,
               "l1_medio_sem_mudanca": float(g.loc[g.mudou_classe == 0, "l1_norm"].mean()),
               "n_pares_com_mudanca": int(g.mudou_classe.sum()),
               "n_inst_com_mudanca": int(g.groupby("inst").mudou_classe.max().sum()),
               "spearman_difprob_l1_instancia": float(rho)})
rel["H4_exploratoria"] = h4

# ------------------------------------- perturbações efetivas
rel["perturbacoes_efetivas_pct"] = (EF.groupby("sigma").mean(numeric_only=True).drop(columns="inst") * 100).round(1).to_dict()

# ------------------------------------- C3
if os.path.exists(xe.caminho(OUT, "C3_variabilidade_modelo.csv")):
    C3 = pd.read_csv(xe.caminho(OUT, "C3_variabilidade_modelo.csv"))
    I3 = C3.groupby(["metodo", "inst"])[xe.METRICAS].mean().reset_index()
    rel["C3"] = {met: {mt: xe.bootstrap_ic(g[mt], B, sb) for mt in xe.METRICAS} for met, g in I3.groupby("metodo")}
    rel["C3_desempenho"] = xe.ler_json(OUT, "C3_desempenho_por_semente.json")

# ------------------------------------- fidelidade local do LIME (complementar)
l = C2[(C2.metodo == "LIME") & (C2.condicao == "C2a")]
rel["lime_r2_local"] = {str(s): {"r2_x_medio": float(g.r2_x.mean()), "queda_media": float((g.r2_x - g.r2_xp).mean())}
                        for s, g in l.groupby("sigma")}

rel["desempenho"] = desemp
rel["smoke"] = SMOKE
rel["data"] = xe.agora()
xe.salvar_json(rel, OUT, "relatorio_resultados.json")
print(open(xe.caminho(TAB, "tabela1_desempenho.md")).read())
print(open(xe.caminho(TAB, "tabela2_robustez_C2a.md")).read())
print("Relatório completo:", os.path.join(OUT, "relatorio_resultados.json"))
