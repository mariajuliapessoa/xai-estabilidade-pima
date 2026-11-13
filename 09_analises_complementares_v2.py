"""Etapa 9 — Análises complementares A4 (versão completa), A5 e A6. Definidas DEPOIS
da execução do protocolo congelado; exploratórias. Não alteram nenhum resultado das
etapas 5 e 6 e não fazem parte dos arquivos protegidos pelo congelamento.

A4 LIME na configuração padrão da biblioteca (discretize_continuous=True,
   sample_around_instance=False, feature_selection="auto"), com os parâmetros do
   protocolo: C1 com K = 30 e C2a/C2b com M = 100 nos quatro níveis de σ.
   Substitui a versão reduzida da etapa 8 (K = 10, M = 30, σ ∈ {0,03; 0,10}).
A5 Instâncias com Insulin imputado × observado: robustez (τ-b em C2a) e posição
   de Insulin no ranking da explicação de referência.
A6 Perturbação apenas dos atributos contínuos (Pregnancies e Age mantidos fixos),
   C2a, quatro níveis de σ, com o mesmo ruído e as mesmas sementes do LIME da etapa 5.
Uso: python 09_analises_complementares_v2.py   (requer as etapas 1, 2 e 5)"""
import json
import pickle
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

import xai_estab as xe

t0 = time.time()
cfg = xe.carregar_config()
alvo = cfg["dados"]["alvo"]
tr = pd.read_csv(xe.caminho("intermediarios", "treino_imputado.csv"))
te = pd.read_csv(xe.caminho("intermediarios", "teste_imputado.csv"))
nomes = [c for c in tr.columns if c != alvo]
X_tr = tr[nomes].values
X_te = te[nomes].values
pre = xe.ler_json("intermediarios", "preprocessamento.json")
with open(xe.caminho("intermediarios", "modelos.pkl"), "rb") as f:
    modelo = pickle.load(f)[cfg["modelo"]["semente_referencia"]]
C2 = pd.read_csv(xe.caminho("resultados", "C2_robustez_pares.csv"))
B, sb = cfg["analise"]["bootstrap_B"], cfg["analise"]["semente_bootstrap"]
sigmas = cfg["perturbacao"]["sigmas"]
pb, lb = cfg["perturbacao"]["semente_base"], cfg["repeticoes"]["semente_base_lime"]
M, K = cfg["repeticoes"]["M"], cfg["repeticoes"]["K_lime"]
n = len(X_te)
out = {"aviso": "Análises definidas após a execução do protocolo congelado; exploratórias.",
       "versoes": xe.versoes()}
ts = C2[(C2.condicao == "C2a") & (C2.metodo == "TreeSHAP")]
lm = C2[(C2.condicao == "C2a") & (C2.metodo == "LIME")]


def tau_inst(df, s):
    return df[df.sigma == s].groupby("inst").tau_b.mean().reindex(range(n))


# ------------------------------------------------------------------ A4
cfg_pad = json.loads(json.dumps(cfg))
cfg_pad["lime"].update({"discretize_continuous": True, "sample_around_instance": False, "feature_selection": "auto"})


def a4_inst(i):
    x = X_te[i]
    lime_e = xe.ExplicadorLIME(modelo, X_tr, nomes, cfg_pad)
    runs = [lime_e.explicar(x, xe.semente(lb, 41, i, k))[0] for k in range(K)]
    c1 = [{"inst": i, "k": k, **xe.metricas_par(runs[0], runs[k])} for k in range(1, K)]
    c2 = []
    for si, s in enumerate(sigmas):
        Xp = xe.perturbar(x, s, pre, nomes, cfg, np.random.default_rng(xe.semente(pb, 2, i, si)), M)
        for m in range(M):
            r = xe.semente(lb, 42, i, si, m)
            phi, _ = lime_e.explicar(x, r)
            c2.append({"inst": i, "sigma": s, "m": m, "condicao": "C2a",
                       **xe.metricas_par(phi, lime_e.explicar(Xp[m], r)[0])})
            c2.append({"inst": i, "sigma": s, "m": m, "condicao": "C2b",
                       **xe.metricas_par(phi, lime_e.explicar(Xp[m], xe.semente(lb, 43, i, si, m))[0])})
    return c1, c2


res = Parallel(n_jobs=cfg["n_processos"])(delayed(a4_inst)(i) for i in range(n))
A4c1 = pd.DataFrame([l for r in res for l in r[0]])
A4c2 = pd.DataFrame([l for r in res for l in r[1]])
A4c1.to_csv(xe.caminho("resultados", "A4v2_lime_padrao_C1.csv"), index=False)
A4c2.to_csv(xe.caminho("resultados", "A4v2_lime_padrao_pares.csv"), index=False)
met = ["tau_b", "jaccard3", "top1", "l1_norm", "sinal"]
a4 = {"config": cfg_pad["lime"], "K": K, "M": M, "sigmas": sigmas,
      "C1": {mt: xe.bootstrap_ic(A4c1.groupby("inst")[mt].mean(), B, sb) for mt in met}}
comp = []
for s in sigmas:
    for cond in ["C2a", "C2b"]:
        g = A4c2[(A4c2.sigma == s) & (A4c2.condicao == cond)].groupby("inst")[met].mean()
        a4[f"{cond}_{s}"] = {mt: xe.bootstrap_ic(g[mt], B, sb) for mt in met}
    g = A4c2[(A4c2.sigma == s) & (A4c2.condicao == "C2a")].groupby("inst").tau_b.mean().reindex(range(n))
    comp.append({"sigma": s, **xe.wilcoxon_pareado(tau_inst(ts, s).values, g.values, B, sb)})
for c, p in zip(comp, xe.holm([c["p"] for c in comp])):
    c["p_holm"] = p
a4["TreeSHAP_menos_LIMEpadrao_C2a"] = comp
out["A4_lime_padrao_completa"] = a4

# ------------------------------------------------------------------ A5
te_bruto = pd.read_csv(xe.caminho(cfg["dados"]["arquivo"]), header=None)
idx_te = xe.ler_json("intermediarios", "divisao.json")["teste"]
imput = (te_bruto.iloc[idx_te, nomes.index("Insulin")].values == 0)
j_ins = nomes.index("Insulin")


def ranks_ref(i):
    x = X_te[i]
    phi_s = xe.ExplicadorTreeSHAP(modelo, cfg).explicar(x)[0]
    phi_l, _ = xe.ExplicadorLIME(modelo, X_tr, nomes, cfg).explicar(x, xe.semente(lb, 1, i, 0))
    pos = lambda phi: int(np.where(np.argsort(-np.abs(phi), kind="stable") == j_ins)[0][0]) + 1
    return pos(phi_s), pos(phi_l)


rk = np.array(Parallel(n_jobs=cfg["n_processos"])(delayed(ranks_ref)(i) for i in range(n)))
a5 = {"n_imputado": int(imput.sum()), "n_observado": int((~imput).sum()),
      "obs_ranking": "posição 1 = atributo mais importante; LIME: execução de referência da C1 (k = 0)"}
for mi, mt in enumerate(["TreeSHAP", "LIME"]):
    df = ts if mt == "TreeSHAP" else lm
    for s in [0.05, 0.10]:
        t = tau_inst(df, s).values
        a5[f"{mt}_tau_C2a_{s}"] = {"imputado": xe.bootstrap_ic(t[imput], B, sb),
                                   "observado": xe.bootstrap_ic(t[~imput], B, sb),
                                   "mann_whitney_p": float(stats.mannwhitneyu(t[imput], t[~imput]).pvalue)}
    r = rk[:, mi]
    a5[f"{mt}_posicao_insulin"] = {"imputado_media": float(r[imput].mean()), "observado_media": float(r[~imput].mean()),
                                   "mann_whitney_p": float(stats.mannwhitneyu(r[imput], r[~imput]).pvalue)}
out["A5_insulin_imputado"] = a5

# ------------------------------------------------------------------ A6
fixos = [nomes.index("Pregnancies"), nomes.index("Age")]


def a6_inst(i):
    x = X_te[i]
    shap_e = xe.ExplicadorTreeSHAP(modelo, cfg)
    lime_e = xe.ExplicadorLIME(modelo, X_tr, nomes, cfg)
    phi_s = shap_e.explicar(x)[0]
    linhas = []
    for si, s in enumerate(sigmas):
        Xp = xe.perturbar(x, s, pre, nomes, cfg, np.random.default_rng(xe.semente(pb, 2, i, si)), M)
        Xp[:, fixos] = x[fixos]
        phis_p = shap_e.explicar(Xp)
        for m in range(M):
            r = xe.semente(lb, 2, i, si, m)
            phi_l, _ = lime_e.explicar(x, r)
            linhas.append({"inst": i, "sigma": s, "m": m, "metodo": "TreeSHAP", **xe.metricas_par(phi_s, phis_p[m])})
            linhas.append({"inst": i, "sigma": s, "m": m, "metodo": "LIME",
                           **xe.metricas_par(phi_l, lime_e.explicar(Xp[m], r)[0])})
    return linhas


A6 = pd.DataFrame([l for r in Parallel(n_jobs=cfg["n_processos"])(delayed(a6_inst)(i) for i in range(n)) for l in r])
A6.to_csv(xe.caminho("resultados", "A6_continuos_pares.csv"), index=False)
a6 = {"fixos": ["Pregnancies", "Age"]}
comp = []
for s in sigmas:
    t_s = tau_inst(A6[A6.metodo == "TreeSHAP"], s)
    t_l = tau_inst(A6[A6.metodo == "LIME"], s)
    a6[f"tau_C2a_{s}"] = {"TreeSHAP": xe.bootstrap_ic(t_s, B, sb), "LIME": xe.bootstrap_ic(t_l, B, sb)}
    comp.append({"sigma": s, **xe.wilcoxon_pareado(t_s.values, t_l.values, B, sb)})
for c, p in zip(comp, xe.holm([c["p"] for c in comp])):
    c["p_holm"] = p
a6["TreeSHAP_menos_LIME_C2a"] = comp
out["A6_somente_continuos"] = a6

out["tempo_s"] = round(time.time() - t0, 1)
out["data"] = xe.agora()
xe.salvar_json(out, "resultados", "analises_complementares_v2.json")
print("ok", out["tempo_s"])
