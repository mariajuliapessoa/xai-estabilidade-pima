"""Funções comuns do protocolo experimental: dados, modelo, explicadores,
perturbações e métricas de estabilidade. Todas as estatísticas de
pré-processamento são calculadas exclusivamente no conjunto de treino."""
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

RAIZ = os.path.dirname(os.path.abspath(__file__))
ARQ_CONGELADO = os.path.join(RAIZ, "protocolo_congelado.json")
ARQUIVOS_PROTOCOLO = [
    "config.json", "xai_estab.py", "01_preparar_dados.py", "02_treinar_modelo.py",
    "03_piloto.py", "05_executar_condicoes.py", "06_analisar.py",
]


# ----------------------------------------------------------------- utilidades
def caminho(*p):
    return os.path.join(RAIZ, *p)


def carregar_config():
    with open(caminho("config.json"), encoding="utf-8") as f:
        return json.load(f)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def versoes():
    import catboost, lime, shap, sklearn, scipy
    return {
        "python": sys.version.split()[0], "plataforma": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__, "catboost": catboost.__version__,
        "shap": shap.__version__, "lime": getattr(lime, "__version__", "0.2.0.1"),
    }


def agora():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def salvar_json(obj, *p):
    os.makedirs(os.path.dirname(caminho(*p)), exist_ok=True)
    with open(caminho(*p), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)


def ler_json(*p):
    with open(caminho(*p), encoding="utf-8") as f:
        return json.load(f)


def hashes_protocolo(cfg):
    h = {a: sha256(caminho(a)) for a in ARQUIVOS_PROTOCOLO}
    h["dados"] = sha256(caminho(cfg["dados"]["arquivo"]))
    return h


def exigir_protocolo_congelado(cfg):
    """Impede o uso do conjunto de teste antes do congelamento do protocolo
    ou se algum arquivo do protocolo foi alterado depois dele."""
    if not os.path.exists(ARQ_CONGELADO):
        sys.exit("ERRO: protocolo não congelado. Confirme as decisões pendentes e rode 04_congelar_protocolo.py.")
    congelado = ler_json("protocolo_congelado.json")["hashes"]
    atual = hashes_protocolo(cfg)
    dif = [k for k in congelado if congelado[k] != atual.get(k)]
    if dif:
        sys.exit(f"ERRO: arquivos alterados após o congelamento: {dif}. Recongele ou desfaça as alterações.")


def semente(*partes):
    """Semente determinística derivada de uma sequência de inteiros."""
    return int(np.random.SeedSequence([int(p) for p in partes]).generate_state(1)[0])


# --------------------------------------------------------------------- dados
def carregar_dados(cfg):
    d = cfg["dados"]
    arq = caminho(d["arquivo"])
    h = sha256(arq)
    if d.get("sha256_esperado") and h != d["sha256_esperado"]:
        print(f"AVISO: SHA-256 do arquivo ({h}) difere do esperado; registre a versão usada.")
    df = pd.read_csv(arq, header=0 if d["tem_cabecalho"] else None)
    df.columns = d["colunas"]
    return df, h


def ajustar_preprocessamento(X_tr, cfg):
    """Mediana do treino (sem os zeros implausíveis) e limites/desvios do treino imputado."""
    zeros = cfg["dados"]["zeros_como_ausentes"]
    medianas = {c: float(X_tr.loc[X_tr[c] != 0, c].median()) for c in zeros}
    X_imp = aplicar_imputacao(X_tr, medianas, zeros)
    pre = {
        "medianas": medianas,
        "desvio_padrao": X_imp.std(ddof=1).to_dict(),
        "minimo": X_imp.min().to_dict(),
        "maximo": X_imp.max().to_dict(),
        "p99_5": X_imp.quantile(0.995).to_dict(),
    }
    return pre


def aplicar_imputacao(X, medianas, zeros):
    X = X.copy().astype(float)
    for c in zeros:
        X.loc[X[c] == 0, c] = medianas[c]
    return X


# -------------------------------------------------------------------- modelo
def treinar_catboost(X, y, params, semente_modelo):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(**params, random_seed=semente_modelo, verbose=False,
                           allow_writing_files=False, thread_count=1)
    m.fit(X, y)
    return m


# --------------------------------------------------------------- explicadores
class ExplicadorTreeSHAP:
    def __init__(self, modelo, cfg):
        import shap
        self.exp = shap.TreeExplainer(modelo, feature_perturbation=cfg["treeshap"]["feature_perturbation"])

    def explicar(self, X):
        X = np.atleast_2d(X)
        sv = self.exp.shap_values(X)
        sv = np.asarray(sv)
        if sv.ndim == 3:  # (n, d, classes) ou (classes, n, d)
            sv = sv[..., 1] if sv.shape[-1] == 2 else sv[1]
        return sv


class ExplicadorLIME:
    """LIME tabular; a semente é definida a cada chamada para permitir
    sementes comuns (C2a) ou independentes (C1, C2b)."""

    def __init__(self, modelo, X_treino, nomes, cfg):
        from lime.lime_tabular import LimeTabularExplainer
        c = cfg["lime"]
        self.c = c
        self.predict = modelo.predict_proba
        self.exp = LimeTabularExplainer(
            np.asarray(X_treino), feature_names=nomes, class_names=["0", "1"], mode="classification",
            discretize_continuous=c["discretize_continuous"], sample_around_instance=c["sample_around_instance"],
            feature_selection=c["feature_selection"], kernel_width=c["kernel_width"], random_state=0)
        self.d = len(nomes)

    def explicar(self, x, semente_lime):
        from sklearn.utils import check_random_state
        rs = check_random_state(int(semente_lime))
        self.exp.random_state = rs
        self.exp.base.random_state = rs
        e = self.exp.explain_instance(np.asarray(x), self.predict, labels=(1,),
                                      num_features=self.c["num_features"], num_samples=self.c["num_samples"])
        phi = np.zeros(self.d)
        for j, w in e.as_map()[1]:
            phi[j] = w
        return phi, float(e.score)


# --------------------------------------------------------------- perturbação
def perturbar(x, sigma, pre, nomes, cfg, rng, M):
    """M versões x' = x + eps, eps_j ~ N(0, (sigma*s_j)^2), com truncamento e arredondamento."""
    p = cfg["perturbacao"]
    s = np.array([pre["desvio_padrao"][n] for n in nomes])
    lo = np.array([pre["minimo"][n] for n in nomes])
    if p["limite_superior"] == "percentil_99_5":
        hi = np.array([pre["p99_5"][n] for n in nomes])
    else:
        hi = np.array([pre["maximo"][n] for n in nomes])
    Xp = x[None, :] + rng.normal(0.0, 1.0, size=(M, len(x))) * (sigma * s)[None, :]
    Xp = np.clip(Xp, lo, hi)
    for n in p["arredondar"]:
        j = nomes.index(n)
        Xp[:, j] = np.round(Xp[:, j])
    return Xp


# ------------------------------------------------------------------ métricas
def metricas_par(phi, phi2, k=3):
    """Seis métricas de estabilidade entre duas explicações do mesmo método."""
    a, b = np.abs(phi), np.abs(phi2)
    if np.allclose(phi, phi2, rtol=0, atol=1e-12):
        tau = rho = 1.0
    else:
        with np.errstate(all="ignore"):
            tau = stats.kendalltau(a, b, variant="b").statistic
            rho = stats.spearmanr(a, b).statistic
    ta = set(np.argsort(-a, kind="stable")[:k])
    tb = set(np.argsort(-b, kind="stable")[:k])
    jac = len(ta & tb) / len(ta | tb)
    top1 = float(np.argmax(a) == np.argmax(b))
    den = np.sum(a)
    l1 = float(np.sum(np.abs(phi - phi2)) / den) if den > 1e-12 else np.nan
    sinal = float(np.mean(np.sign(phi) == np.sign(phi2)))
    return {"tau_b": float(tau), "spearman": float(rho), "jaccard3": jac,
            "top1": top1, "l1_norm": l1, "sinal": sinal}


METRICAS = ["tau_b", "spearman", "jaccard3", "top1", "l1_norm", "sinal"]


# -------------------------------------------------------- estatística
def bootstrap_ic(v, B, seed, func=np.mean, alfa=0.05):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(B, len(v)))
    bs = func(v[idx], axis=1)
    return (float(func(v)), float(np.quantile(bs, alfa / 2)), float(np.quantile(bs, 1 - alfa / 2)))


def wilcoxon_pareado(x, y, B, seed):
    """Wilcoxon de postos sinalizados (x - y), bisserial de postos e IC bootstrap da diferença mediana."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    dlt = x[ok] - y[ok]
    nz = dlt[dlt != 0]
    if len(nz) == 0:
        return {"n": int(ok.sum()), "W": np.nan, "p": 1.0, "r_bisserial": 0.0,
                "dif_mediana": 0.0, "ic_inf": 0.0, "ic_sup": 0.0}
    res = stats.wilcoxon(dlt, zero_method="wilcox", alternative="two-sided")
    postos = stats.rankdata(np.abs(nz))
    wp, wn = postos[nz > 0].sum(), postos[nz < 0].sum()
    r = (wp - wn) / (wp + wn)
    med, lo, hi = bootstrap_ic(dlt, B, seed, func=np.median)
    return {"n": int(ok.sum()), "W": float(res.statistic), "p": float(res.pvalue),
            "r_bisserial": float(r), "dif_mediana": med, "ic_inf": lo, "ic_sup": hi}


def holm(pvals):
    p = np.asarray(pvals, float)
    ordem = np.argsort(p)
    m = len(p)
    aj = np.empty(m)
    acum = 0.0
    for r, i in enumerate(ordem):
        acum = max(acum, (m - r) * p[i])
        aj[i] = min(1.0, acum)
    return aj.tolist()
