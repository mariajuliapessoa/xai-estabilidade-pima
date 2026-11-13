"""Etapa 2 — Seleção de hiperparâmetros por validação cruzada estratificada
(5 dobras, AUC-ROC) apenas no treino; reajuste do modelo de referência e
dos modelos da condição C3. Não usa o conjunto de teste."""
import itertools
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import xai_estab as xe

cfg = xe.carregar_config()
alvo = cfg["dados"]["alvo"]
tr = pd.read_csv(xe.caminho("intermediarios", "treino_imputado.csv"))
X, y = tr.drop(columns=alvo).values, tr[alvo].values
mc = cfg["modelo"]
grade = mc["grade"]
chaves = list(grade)
skf = StratifiedKFold(n_splits=mc["cv_dobras"], shuffle=True, random_state=mc["cv_semente"])

resultados = []
for valores in itertools.product(*[grade[k] for k in chaves]):
    params = dict(zip(chaves, valores))
    aucs = []
    for f_tr, f_va in skf.split(X, y):
        m = xe.treinar_catboost(X[f_tr], y[f_tr], params, mc["semente_referencia"])
        aucs.append(roc_auc_score(y[f_va], m.predict_proba(X[f_va])[:, 1]))
    resultados.append({**params, "auc_cv_media": float(np.mean(aucs)), "auc_cv_dp": float(np.std(aucs, ddof=1))})
    print(params, f"AUC-CV = {np.mean(aucs):.4f} ± {np.std(aucs, ddof=1):.4f}")

melhor = max(resultados, key=lambda r: r["auc_cv_media"])
params = {k: melhor[k] for k in chaves}
print("Hiperparâmetros selecionados:", params)

modelos = {}
for s in sorted(set([mc["semente_referencia"]] + cfg["repeticoes"]["sementes_modelo_C3"])):
    modelos[s] = xe.treinar_catboost(X, y, params, s)
with open(xe.caminho("intermediarios", "modelos.pkl"), "wb") as f:
    pickle.dump(modelos, f)
xe.salvar_json({"grade": resultados, "selecionado": params, "data": xe.agora(), "versoes": xe.versoes()},
               "intermediarios", "selecao_hiperparametros.json")
