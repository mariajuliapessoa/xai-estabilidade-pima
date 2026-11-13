"""Etapa 5 — Executa C0 (desempenho e explicações de referência), C1, C2a, C2b e C3
no conjunto de TESTE. Exige protocolo congelado.
Uso: python 05_executar_condicoes.py            (execução real)
     python 05_executar_condicoes.py --smoke    (teste do código com 5 instâncias do
                                                 TREINO e repetições mínimas; não gera resultados)"""
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, f1_score, roc_auc_score

import xai_estab as xe

SMOKE = "--smoke" in sys.argv
cfg = xe.carregar_config()
if not SMOKE:
    xe.exigir_protocolo_congelado(cfg)
OUT = "resultados_smoke" if SMOKE else "resultados"
os.makedirs(xe.caminho(OUT), exist_ok=True)

alvo = cfg["dados"]["alvo"]
tr = pd.read_csv(xe.caminho("intermediarios", "treino_imputado.csv"))
nomes = [c for c in tr.columns if c != alvo]
X_tr = tr[nomes].values
if SMOKE:
    av = tr.sample(5, random_state=1)
    X_av, y_av = av[nomes].values, av[alvo].values
else:
    te = pd.read_csv(xe.caminho("intermediarios", "teste_imputado.csv"))
    X_av, y_av = te[nomes].values, te[alvo].values
pre = xe.ler_json("intermediarios", "preprocessamento.json")
with open(xe.caminho("intermediarios", "modelos.pkl"), "rb") as f:
    modelos = pickle.load(f)
ref = cfg["modelo"]["semente_referencia"]
modelo = modelos[ref]
rp = cfg["repeticoes"]
M = 3 if SMOKE else rp["M"]
K = 3 if SMOKE else rp["K_lime"]
sem_C3 = rp["sementes_modelo_C3"][:2] if SMOKE else rp["sementes_modelo_C3"]
sigmas = cfg["perturbacao"]["sigmas"]
pb, lb = cfg["perturbacao"]["semente_base"], rp["semente_base_lime"]
limiar = cfg["modelo"]["limiar"]
t0 = time.time()

# ---------------------------------------------------------- C0: desempenho
p = modelo.predict_proba(X_av)[:, 1]
yhat = (p >= limiar).astype(int)
B, sb = cfg["analise"]["bootstrap_B"], cfg["analise"]["semente_bootstrap"]
rng = np.random.default_rng(sb)
boot = {"acuracia": [], "auc_roc": [], "f1": [], "brier": []}
for _ in range(B if not SMOKE else 50):
    ii = rng.integers(0, len(y_av), len(y_av))
    if len(np.unique(y_av[ii])) < 2:
        continue
    boot["acuracia"].append(accuracy_score(y_av[ii], yhat[ii]))
    boot["auc_roc"].append(roc_auc_score(y_av[ii], p[ii]))
    boot["f1"].append(f1_score(y_av[ii], yhat[ii], zero_division=0))
    boot["brier"].append(brier_score_loss(y_av[ii], p[ii]))
pontual = {"acuracia": accuracy_score(y_av, yhat), "auc_roc": roc_auc_score(y_av, p) if len(set(y_av)) > 1 else np.nan,
           "f1": f1_score(y_av, yhat, zero_division=0), "brier": brier_score_loss(y_av, p)}
desempenho = {k: {"valor": float(v), "ic95": [float(np.quantile(boot[k], .025)), float(np.quantile(boot[k], .975))]}
              for k, v in pontual.items()}
desempenho["matriz_confusao"] = confusion_matrix(y_av, yhat, labels=[0, 1]).tolist()
desempenho["n"] = int(len(y_av))
desempenho["prevalencia"] = float(np.mean(y_av))
xe.salvar_json(desempenho, OUT, "desempenho.json")


# ------------------------------------------------------- tarefas por instância
def por_instancia(i):
    x = X_av[i]
    shap_e = xe.ExplicadorTreeSHAP(modelo, cfg)
    lime_e = xe.ExplicadorLIME(modelo, X_tr, nomes, cfg)
    p_x = float(modelo.predict_proba(x[None, :])[0, 1])
    c1, c2, efetivas = [], [], []

    # C1 — variabilidade intrínseca: K execuções do LIME com sementes distintas
    sem_k = [xe.semente(lb, 1, i, k) for k in range(K)]
    runs = [lime_e.explicar(x, s) for s in sem_k]
    for k in range(1, K):
        c1.append({"inst": i, "metodo": "LIME", "k": k, **xe.metricas_par(runs[0][0], runs[k][0])})
    phi_s = shap_e.explicar(x)[0]
    phi_s2 = xe.ExplicadorTreeSHAP(modelo, cfg).explicar(x)[0]
    c1.append({"inst": i, "metodo": "TreeSHAP", "k": 1, **xe.metricas_par(phi_s, phi_s2),
               "max_dif_abs": float(np.max(np.abs(phi_s - phi_s2)))})

    # C2a / C2b — robustez à perturbação da entrada
    for si, s in enumerate(sigmas):
        Xp = xe.perturbar(x, s, pre, nomes, cfg, np.random.default_rng(xe.semente(pb, 2, i, si)), M)
        efetivas.append({"inst": i, "sigma": s, **{n: float(np.mean(Xp[:, j] != x[j])) for j, n in enumerate(nomes)}})
        pp = modelo.predict_proba(Xp)[:, 1]
        phis_p = shap_e.explicar(Xp)
        for m in range(M):
            base = {"inst": i, "sigma": s, "m": m, "p_x": p_x, "p_xp": float(pp[m]),
                    "dif_prob": float(abs(pp[m] - p_x)), "mudou_classe": int((pp[m] >= limiar) != (p_x >= limiar))}
            c2.append({**base, "condicao": "C2", "metodo": "TreeSHAP", **xe.metricas_par(phi_s, phis_p[m])})
            r = xe.semente(lb, 2, i, si, m)
            r_ind = xe.semente(lb, 3, i, si, m)
            phi_l, sc_x = lime_e.explicar(x, r)
            phi_lp, sc_xp = lime_e.explicar(Xp[m], r)
            phi_lp_ind, sc_ind = lime_e.explicar(Xp[m], r_ind)
            c2.append({**base, "condicao": "C2a", "metodo": "LIME", **xe.metricas_par(phi_l, phi_lp),
                       "r2_x": sc_x, "r2_xp": sc_xp})
            c2.append({**base, "condicao": "C2b", "metodo": "LIME", **xe.metricas_par(phi_l, phi_lp_ind),
                       "r2_x": sc_x, "r2_xp": sc_ind})
    return c1, c2, efetivas


res = Parallel(n_jobs=cfg["n_processos"])(delayed(por_instancia)(i) for i in range(len(X_av)))
C1 = pd.DataFrame([l for r in res for l in r[0]])
C2 = pd.DataFrame([l for r in res for l in r[1]])
EF = pd.DataFrame([l for r in res for l in r[2]])
# O TreeSHAP não tem semente: C2a e C2b coincidem para ele
tshap = C2[C2.metodo == "TreeSHAP"]
C2 = pd.concat([C2[C2.metodo == "LIME"], tshap.assign(condicao="C2a"), tshap.assign(condicao="C2b")])
C1.to_csv(xe.caminho(OUT, "C1_variabilidade_intrinseca.csv"), index=False)
C2.to_csv(xe.caminho(OUT, "C2_robustez_pares.csv"), index=False)
EF.to_csv(xe.caminho(OUT, "perturbacoes_efetivas.csv"), index=False)

# ------------------------------------------------------- C3: variabilidade do modelo
if rp["executar_C3"]:
    def c3_inst(i):
        x = X_av[i]
        r = xe.semente(lb, 4, i)
        phis = {s: (xe.ExplicadorTreeSHAP(modelos[s], cfg).explicar(x)[0],
                    xe.ExplicadorLIME(modelos[s], X_tr, nomes, cfg).explicar(x, r)[0]) for s in sem_C3}
        out = []
        for s in sem_C3:
            if s == ref:
                continue
            out.append({"inst": i, "semente_modelo": s, "metodo": "TreeSHAP", **xe.metricas_par(phis[ref][0], phis[s][0])})
            out.append({"inst": i, "semente_modelo": s, "metodo": "LIME", **xe.metricas_par(phis[ref][1], phis[s][1])})
        return out
    C3 = pd.DataFrame([l for r in Parallel(n_jobs=cfg["n_processos"])(delayed(c3_inst)(i) for i in range(len(X_av))) for l in r])
    C3.to_csv(xe.caminho(OUT, "C3_variabilidade_modelo.csv"), index=False)
    perf_c3 = {}
    for s in sem_C3:
        ps = modelos[s].predict_proba(X_av)[:, 1]
        perf_c3[str(s)] = {"auc_roc": float(roc_auc_score(y_av, ps)) if len(set(y_av)) > 1 else None,
                           "acuracia": float(accuracy_score(y_av, (ps >= limiar).astype(int)))}
    xe.salvar_json(perf_c3, OUT, "C3_desempenho_por_semente.json")

xe.salvar_json({"data": xe.agora(), "tempo_s": round(time.time() - t0, 1), "smoke": SMOKE, "M": M, "K": K,
                "n_instancias": int(len(X_av)), "versoes": xe.versoes()}, OUT, "log_execucao.json")
print(f"Concluído em {time.time() - t0:.0f} s. Resultados em {OUT}/")
