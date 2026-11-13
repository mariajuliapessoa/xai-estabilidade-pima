"""Etapa 10 — Análises complementares A7, A8 e A9 (definidas em 13/11/2026, após a
execução do protocolo congelado; exploratórias; não alteram as etapas 5 e 6).

A7 Largura do kernel do LIME: C1 (K = 10) e C2a (M = 30; sigma em {0,03; 0,10}) com
   larguras 0,5, 1,0 e a padrão (0,75*sqrt(8) ~ 2,12); H3 (TreeSHAP x LIME) por largura.
   As perturbações e as sementes do LIME são as da etapa 5 (primeiras 30 versões).
A8 Fidelidade por curvas de deleção: atributos substituídos, em ordem decrescente de
   |atribuição|, pela mediana do treino; fidelidade = área média sob a curva de
   |p(x_k) - p(x)| (k = 0..8, normalizada), comparada com ordens aleatórias.
   Relação entre fidelidade e robustez (tau-b em C2a, sigma = 0,10) por instância.
A9 Replicação com Random Forest (300 árvores) no mesmo treino: C1 (K = 10) e C2a
   (M = 30; sigma em {0,03; 0,10}), TreeSHAP x LIME.
Uso: python 10_analises_complementares_v3.py   (requer as etapas 1, 2 e 5)"""
import pickle, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import xai_estab as xe

t0 = time.time()
cfg = xe.carregar_config(); alvo = cfg["dados"]["alvo"]
tr = pd.read_csv(xe.caminho("intermediarios", "treino_imputado.csv"))
te = pd.read_csv(xe.caminho("intermediarios", "teste_imputado.csv"))
nomes = [c for c in tr.columns if c != alvo]
X_tr, y_tr, X_te, y_te = tr[nomes].values, tr[alvo].values, te[nomes].values, te[alvo].values
pre = xe.ler_json("intermediarios", "preprocessamento.json")
with open(xe.caminho("intermediarios", "modelos.pkl"), "rb") as f:
    modelo = pickle.load(f)[cfg["modelo"]["semente_referencia"]]
B, sb = cfg["analise"]["bootstrap_B"], cfg["analise"]["semente_bootstrap"]
sig_all = cfg["perturbacao"]["sigmas"]; SIG = [0.03, 0.10]
pb, lb = cfg["perturbacao"]["semente_base"], cfg["repeticoes"]["semente_base_lime"]
M, K, n, d = 30, 10, len(X_te), len(nomes)
LARG = {"0,5": 0.5, "1,0": 1.0, "padrao": None}
out = {"aviso": "Análises definidas em 13/11/2026, após o congelamento; exploratórias.", "M": M, "K": K}

def cfg_k(w):
    c = __import__("json").loads(__import__("json").dumps(cfg)); c["lime"]["kernel_width"] = w; return c

def bloco(mod, i, larguras):
    x = X_te[i]; ts = xe.ExplicadorTreeSHAP(mod, cfg)
    phi_s = ts.explicar(x)[0]; res = []
    Xps = {s: xe.perturbar(x, s, pre, nomes, cfg, np.random.default_rng(xe.semente(pb, 2, i, sig_all.index(s))), M) for s in SIG}
    for s in SIG:
        sp = ts.explicar(Xps[s])
        for m in range(M):
            res.append({"inst": i, "metodo": "TreeSHAP", "largura": "-", "cond": "C2a", "sigma": s, **xe.metricas_par(phi_s, sp[m])})
    for nome, w in larguras.items():
        le = xe.ExplicadorLIME(mod, X_tr, nomes, cfg_k(w))
        runs = [le.explicar(x, xe.semente(lb, 1, i, k))[0] for k in range(K)]
        for k in range(1, K):
            res.append({"inst": i, "metodo": "LIME", "largura": nome, "cond": "C1", "sigma": 0, **xe.metricas_par(runs[0], runs[k])})
        for s in SIG:
            si = sig_all.index(s)
            for m in range(M):
                r = xe.semente(lb, 2, i, si, m)
                res.append({"inst": i, "metodo": "LIME", "largura": nome, "cond": "C2a", "sigma": s,
                            **xe.metricas_par(le.explicar(x, r)[0], le.explicar(Xps[s][m], r)[0])})
    return res

def resumir(df, larguras, rotulo):
    o = {}
    ts = df[df.metodo == "TreeSHAP"]
    for nome in larguras:
        lm = df[(df.metodo == "LIME") & (df.largura == nome)]
        o[nome] = {"C1_tau": xe.bootstrap_ic(lm[lm.cond == "C1"].groupby("inst").tau_b.mean(), B, sb)}
        comp = []
        for s in SIG:
            a = ts[ts.sigma == s].groupby("inst").tau_b.mean().reindex(range(n)).values
            b = lm[(lm.cond == "C2a") & (lm.sigma == s)].groupby("inst").tau_b.mean().reindex(range(n)).values
            o[nome][f"C2a_{s}_tau_LIME"] = xe.bootstrap_ic(b, B, sb)
            o[nome][f"C2a_{s}_top1_LIME"] = float(lm[(lm.cond == "C2a") & (lm.sigma == s)].top1.mean())
            comp.append({"sigma": s, **xe.wilcoxon_pareado(a, b, B, sb)})
        for c, p in zip(comp, xe.holm([c["p"] for c in comp])): c["p_holm"] = p
        o[nome]["TreeSHAP_menos_LIME"] = comp
    for s in SIG:
        o[f"TreeSHAP_C2a_{s}_tau"] = xe.bootstrap_ic(ts[ts.sigma == s].groupby("inst").tau_b.mean(), B, sb)
        o[f"TreeSHAP_C2a_{s}_top1"] = float(ts[ts.sigma == s].top1.mean())
    print(rotulo, "ok", round(time.time() - t0), "s", flush=True)
    return o

# ---------------- A7
r7 = Parallel(n_jobs=cfg["n_processos"])(delayed(bloco)(modelo, i, LARG) for i in range(n))
A7 = pd.DataFrame([l for r in r7 for l in r]); A7.to_csv(xe.caminho("resultados", "A7_largura_kernel.csv"), index=False)
out["A7_largura_kernel"] = resumir(A7, LARG, "A7")

# ---------------- A8 fidelidade (deleção)
med = np.median(X_tr, axis=0)
def curva(f, x, ordem):
    p0 = f(x[None])[0, 1]; xs = [x.copy()]
    for j in ordem:
        z = xs[-1].copy(); z[j] = med[j]; xs.append(z)
    p = f(np.array(xs))[:, 1]
    return float(np.mean(np.abs(p - p0)))
def a8(i):
    x = X_te[i]; f = modelo.predict_proba
    phi_s = xe.ExplicadorTreeSHAP(modelo, cfg).explicar(x)[0]
    phi_l = xe.ExplicadorLIME(modelo, X_tr, nomes, cfg).explicar(x, xe.semente(lb, 1, i, 0))[0]
    rng = np.random.default_rng(xe.semente(pb, 80, i))
    aleat = np.mean([curva(f, x, rng.permutation(d)) for _ in range(20)])
    t = stats.kendalltau(np.abs(phi_s), np.abs(phi_l), variant="b")[0]
    return {"inst": i, "fid_TreeSHAP": curva(f, x, np.argsort(-np.abs(phi_s), kind="stable")),
            "fid_LIME": curva(f, x, np.argsort(-np.abs(phi_l), kind="stable")), "fid_aleatoria": aleat,
            "tau_TreeSHAP_LIME": t}
A8 = pd.DataFrame(Parallel(n_jobs=cfg["n_processos"])(delayed(a8)(i) for i in range(n)))
C2 = pd.read_csv(xe.caminho("resultados", "C2_robustez_pares.csv"))
for mt in ["TreeSHAP", "LIME"]:
    g = C2[(C2.condicao == "C2a") & (C2.metodo == mt) & (C2.sigma == 0.10)].groupby("inst").tau_b.mean().reindex(range(n))
    A8[f"tau_C2a_0.1_{mt}"] = g.values
A8.to_csv(xe.caminho("resultados", "A8_fidelidade_delecao.csv"), index=False)
a8o = {k: xe.bootstrap_ic(A8[k], B, sb) for k in ["fid_TreeSHAP", "fid_LIME", "fid_aleatoria", "tau_TreeSHAP_LIME"]}
a8o["TreeSHAP_menos_LIME_fid"] = xe.wilcoxon_pareado(A8.fid_TreeSHAP.values, A8.fid_LIME.values, B, sb)
a8o["LIME_menos_aleatoria_fid"] = xe.wilcoxon_pareado(A8.fid_LIME.values, A8.fid_aleatoria.values, B, sb)
a8o["TreeSHAP_menos_aleatoria_fid"] = xe.wilcoxon_pareado(A8.fid_TreeSHAP.values, A8.fid_aleatoria.values, B, sb)
for mt in ["TreeSHAP", "LIME"]:
    r, p = stats.spearmanr(A8[f"fid_{mt}"], A8[f"tau_C2a_0.1_{mt}"]); a8o[f"spearman_fid_tau_{mt}"] = {"rho": float(r), "p": float(p)}
out["A8_fidelidade"] = a8o; print("A8 ok", round(time.time() - t0), flush=True)

# ---------------- A9 Random Forest
rf = RandomForestClassifier(n_estimators=300, random_state=cfg["modelo"]["semente_referencia"], n_jobs=1).fit(X_tr, y_tr)
p = rf.predict_proba(X_te)[:, 1]
out["A9_random_forest"] = {"AUC_teste": float(roc_auc_score(y_te, p)), "acuracia": float(((p >= .5) == y_te).mean())}
r9 = Parallel(n_jobs=cfg["n_processos"])(delayed(bloco)(rf, i, {"padrao": None}) for i in range(n))
A9 = pd.DataFrame([l for r in r9 for l in r]); A9.to_csv(xe.caminho("resultados", "A9_random_forest.csv"), index=False)
out["A9_random_forest"].update(resumir(A9, {"padrao": None}, "A9"))
out["tempo_s"] = round(time.time() - t0); out["versoes"] = xe.versoes()
xe.salvar_json(out, "resultados", "analises_complementares_v3.json"); print("fim", out["tempo_s"])
