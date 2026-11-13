"""Etapa 8 — Análises complementares, definidas DEPOIS da execução do protocolo
congelado e declaradas como exploratórias no artigo. Não alteram nenhum resultado
das etapas 5 e 6 e não fazem parte dos arquivos protegidos pelo congelamento.

A1 Limiar alternativo escolhido apenas no treino (Youden na validação cruzada).
A2 Estabilidade das explicações em predições corretas e incorretas (σ = 0,10, C2a).
A3 Mecanismo da relação predição × explicação: L1 NÃO normalizada e norma das
   atribuições (reproduz exatamente as sementes da etapa 5).
A4 LIME na configuração padrão da biblioteca (com discretização, amostragem padrão,
   seleção automática), C1 (K = 10) e C2a/C2b (σ ∈ {0,03; 0,10}, M = 30).
Uso: python 08_analises_complementares.py"""
import json
import pickle
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold

import xai_estab as xe

t0 = time.time()
cfg = xe.carregar_config()
alvo = cfg["dados"]["alvo"]
tr = pd.read_csv(xe.caminho("intermediarios", "treino_imputado.csv"))
te = pd.read_csv(xe.caminho("intermediarios", "teste_imputado.csv"))
nomes = [c for c in tr.columns if c != alvo]
X_tr, y_tr = tr[nomes].values, tr[alvo].values
X_te, y_te = te[nomes].values, te[alvo].values
pre = xe.ler_json("intermediarios", "preprocessamento.json")
params = xe.ler_json("intermediarios", "selecao_hiperparametros.json")["selecionado"]
with open(xe.caminho("intermediarios", "modelos.pkl"), "rb") as f:
    modelo = pickle.load(f)[cfg["modelo"]["semente_referencia"]]
C2 = pd.read_csv(xe.caminho("resultados", "C2_robustez_pares.csv"))
B, sb = cfg["analise"]["bootstrap_B"], cfg["analise"]["semente_bootstrap"]
sigmas = cfg["perturbacao"]["sigmas"]
pb, lb = cfg["perturbacao"]["semente_base"], cfg["repeticoes"]["semente_base_lime"]
M = cfg["repeticoes"]["M"]
out = {"aviso": "Análises definidas após a execução do protocolo congelado; exploratórias."}

# ------------------------------------------------------------------ A1
oof = np.zeros(len(y_tr))
for a, b in StratifiedKFold(5, shuffle=True, random_state=cfg["modelo"]["cv_semente"]).split(X_tr, y_tr):
    oof[b] = xe.treinar_catboost(X_tr[a], y_tr[a], params, cfg["modelo"]["semente_referencia"]).predict_proba(X_tr[b])[:, 1]
fpr, tpr, thr = roc_curve(y_tr, oof)
lim = float(thr[np.argmax(tpr - fpr)])
p_te = modelo.predict_proba(X_te)[:, 1]
a1 = {"limiar_youden_treino": lim}
for nome, L in [("0.5", 0.5), ("youden", lim)]:
    yh = (p_te >= L).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, yh, labels=[0, 1]).ravel()
    rng = np.random.default_rng(sb)
    bs = {"sens": [], "espec": []}
    for _ in range(B):
        ii = rng.integers(0, len(y_te), len(y_te))
        t_, p_ = y_te[ii], yh[ii]
        bs["sens"].append(np.mean(p_[t_ == 1]))
        bs["espec"].append(1 - np.mean(p_[t_ == 0]))
    a1[nome] = {"limiar": L, "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
                "sensibilidade": tp / (tp + fn), "especificidade": tn / (tn + fp), "acuracia": (tp + tn) / len(y_te),
                "ic_sens": [float(np.quantile(bs["sens"], .025)), float(np.quantile(bs["sens"], .975))],
                "ic_espec": [float(np.quantile(bs["espec"], .025)), float(np.quantile(bs["espec"], .975))]}
# mudança de classe sob o limiar alternativo (H4 recontada)
c = C2[(C2.condicao == "C2a") & (C2.metodo == "TreeSHAP")]
a1["pares_com_mudanca_limiar_youden_pct"] = {str(s): float(100 * np.mean((g.p_x >= lim) != (g.p_xp >= lim)))
                                             for s, g in c.groupby("sigma")}
out["A1_limiar"] = a1

# ------------------------------------------------------------------ A2
correto = ((p_te >= 0.5).astype(int) == y_te)
dist_limiar = np.abs(p_te - 0.5)
a2 = {"n_corretas": int(correto.sum()), "n_incorretas": int((~correto).sum()),
      "dist_limiar_mediana_corretas": float(np.median(dist_limiar[correto])),
      "dist_limiar_mediana_incorretas": float(np.median(dist_limiar[~correto]))}
for met in ["TreeSHAP", "LIME"]:
    g = C2[(C2.condicao == "C2a") & (C2.metodo == met)]
    for s in [0.03, 0.10]:
        tau = g[g.sigma == s].groupby("inst").tau_b.mean().reindex(range(len(y_te))).values
        v_c, v_i = tau[correto], tau[~correto]
        mw = stats.mannwhitneyu(v_c, v_i, alternative="two-sided")
        rho = stats.spearmanr(dist_limiar, tau).statistic
        a2[f"{met}_{s}"] = {"tau_corretas": xe.bootstrap_ic(v_c, B, sb), "tau_incorretas": xe.bootstrap_ic(v_i, B, sb),
                            "mann_whitney_p": float(mw.pvalue), "spearman_dist_limiar_tau": float(rho)}
out["A2_acertos_erros"] = a2


# ------------------------------------------------------------------ A3
def a3_inst(i):
    x = X_te[i]
    shap_e = xe.ExplicadorTreeSHAP(modelo, cfg)
    lime_e = xe.ExplicadorLIME(modelo, X_tr, nomes, cfg)
    phi_s = shap_e.explicar(x)[0]
    linhas = []
    for si, s in enumerate(sigmas):
        Xp = xe.perturbar(x, s, pre, nomes, cfg, np.random.default_rng(xe.semente(pb, 2, i, si)), M)
        phis_p = shap_e.explicar(Xp)
        pp = modelo.predict_proba(Xp)[:, 1]
        p_x = float(modelo.predict_proba(x[None, :])[0, 1])
        for m in range(M):
            r = xe.semente(lb, 2, i, si, m)
            phi_l, _ = lime_e.explicar(x, r)
            phi_lp, _ = lime_e.explicar(Xp[m], r)
            base = {"inst": i, "sigma": s, "m": m, "dif_prob": abs(pp[m] - p_x),
                    "mudou_classe": int((pp[m] >= 0.5) != (p_x >= 0.5))}
            linhas.append({**base, "metodo": "TreeSHAP", "l1_abs": float(np.abs(phi_s - phis_p[m]).sum()),
                           "norma": float(np.abs(phi_s).sum()),
                           "l1_norm": xe.metricas_par(phi_s, phis_p[m])["l1_norm"]})
            linhas.append({**base, "metodo": "LIME", "l1_abs": float(np.abs(phi_l - phi_lp).sum()),
                           "norma": float(np.abs(phi_l).sum()),
                           "l1_norm": xe.metricas_par(phi_l, phi_lp)["l1_norm"]})
    return linhas


A3 = pd.DataFrame([l for r in Parallel(n_jobs=cfg["n_processos"])(delayed(a3_inst)(i) for i in range(len(X_te))) for l in r])
A3.to_csv(xe.caminho("resultados", "A3_mecanismo_pares.csv"), index=False)
# conferência: a L1 normalizada recalculada deve coincidir com a da etapa 5
ref = C2[(C2.condicao == "C2a")].set_index(["metodo", "sigma", "inst", "m"]).l1_norm
chk = A3.set_index(["metodo", "sigma", "inst", "m"]).l1_norm
a3 = {"max_dif_vs_etapa5": float(np.nanmax(np.abs(chk - ref.reindex(chk.index))))}
for (met, s), g in A3.groupby(["metodo", "sigma"]):
    gi = g.groupby("inst")[["dif_prob", "l1_abs", "norma", "l1_norm"]].mean()
    a3[f"{met}_{s}"] = {
        "spearman_difprob_l1_abs": float(stats.spearmanr(gi.dif_prob, gi.l1_abs).statistic),
        "spearman_difprob_norma": float(stats.spearmanr(gi.dif_prob, gi.norma).statistic),
        "spearman_difprob_l1_norm": float(stats.spearmanr(gi.dif_prob, gi.l1_norm).statistic),
        "l1_abs_com_mudanca": float(g.loc[g.mudou_classe == 1, "l1_abs"].mean()) if g.mudou_classe.any() else None,
        "l1_abs_sem_mudanca": float(g.loc[g.mudou_classe == 0, "l1_abs"].mean()),
        "norma_com_mudanca": float(g.loc[g.mudou_classe == 1, "norma"].mean()) if g.mudou_classe.any() else None,
        "norma_sem_mudanca": float(g.loc[g.mudou_classe == 0, "norma"].mean())}
out["A3_mecanismo"] = a3

# ------------------------------------------------------------------ A4
cfg_pad = json.loads(json.dumps(cfg))
cfg_pad["lime"].update({"discretize_continuous": True, "sample_around_instance": False, "feature_selection": "auto"})
K4, M4, S4 = 10, 30, [0.03, 0.10]


def a4_inst(i):
    x = X_te[i]
    lime_e = xe.ExplicadorLIME(modelo, X_tr, nomes, cfg_pad)
    c1 = []
    runs = [lime_e.explicar(x, xe.semente(lb, 41, i, k))[0] for k in range(K4)]
    for k in range(1, K4):
        c1.append({"inst": i, **xe.metricas_par(runs[0], runs[k])})
    c2 = []
    for s in S4:
        si = sigmas.index(s)
        Xp = xe.perturbar(x, s, pre, nomes, cfg, np.random.default_rng(xe.semente(pb, 2, i, si)), M4)
        for m in range(M4):
            r = xe.semente(lb, 42, i, si, m)
            phi, _ = lime_e.explicar(x, r)
            c2.append({"inst": i, "sigma": s, "condicao": "C2a", **xe.metricas_par(phi, lime_e.explicar(Xp[m], r)[0])})
            c2.append({"inst": i, "sigma": s, "condicao": "C2b",
                       **xe.metricas_par(phi, lime_e.explicar(Xp[m], xe.semente(lb, 43, i, si, m))[0])})
    return c1, c2


res = Parallel(n_jobs=cfg["n_processos"])(delayed(a4_inst)(i) for i in range(len(X_te)))
A4c1 = pd.DataFrame([l for r in res for l in r[0]])
A4c2 = pd.DataFrame([l for r in res for l in r[1]])
A4c2.to_csv(xe.caminho("resultados", "A4_lime_padrao_pares.csv"), index=False)
a4 = {"config": cfg_pad["lime"], "K": K4, "M": M4,
      "C1": {mt: xe.bootstrap_ic(A4c1.groupby("inst")[mt].mean(), B, sb) for mt in ["tau_b", "top1", "l1_norm"]}}
ts = C2[(C2.condicao == "C2a") & (C2.metodo == "TreeSHAP")]
for s in S4:
    for cond in ["C2a", "C2b"]:
        g = A4c2[(A4c2.sigma == s) & (A4c2.condicao == cond)].groupby("inst")[["tau_b", "top1", "l1_norm"]].mean()
        a4[f"{cond}_{s}"] = {mt: xe.bootstrap_ic(g[mt], B, sb) for mt in ["tau_b", "top1", "l1_norm"]}
    # comparação pareada com o TreeSHAP (mesmas instâncias; TreeSHAP com M = 100)
    g = A4c2[(A4c2.sigma == s) & (A4c2.condicao == "C2a")].groupby("inst").tau_b.mean()
    t = ts[ts.sigma == s].groupby("inst").tau_b.mean().reindex(g.index)
    a4[f"TreeSHAP_menos_LIMEpadrao_C2a_{s}"] = xe.wilcoxon_pareado(t.values, g.values, B, sb)
out["A4_lime_padrao"] = a4
out["tempo_s"] = round(time.time() - t0, 1)
out["data"] = xe.agora()
xe.salvar_json(out, "resultados", "analises_complementares.json")
print(json.dumps(out, ensure_ascii=False, indent=1, default=float)[:6000])
