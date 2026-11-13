"""Etapa 1 — Divisão estratificada 80/20, imputação pela mediana do treino e
estatísticas de pré-processamento (todas calculadas só no treino)."""
import numpy as np
from sklearn.model_selection import train_test_split

import xai_estab as xe

cfg = xe.carregar_config()
df, h = xe.carregar_dados(cfg)
alvo = cfg["dados"]["alvo"]
nomes = [c for c in cfg["dados"]["colunas"] if c != alvo]
zeros = cfg["dados"]["zeros_como_ausentes"]

eda = {
    "arquivo_sha256": h,
    "n_registros": int(len(df)),
    "classes": df[alvo].value_counts().sort_index().to_dict(),
    "zeros_por_atributo": {c: int((df[c] == 0).sum()) for c in nomes},
    "zeros_por_atributo_pct": {c: round(100 * float((df[c] == 0).mean()), 1) for c in nomes},
    "descritivas": df[nomes].describe().round(3).to_dict(),
}

idx = np.arange(len(df))
i_tr, i_te = train_test_split(idx, test_size=cfg["divisao"]["proporcao_teste"],
                              stratify=df[alvo], random_state=cfg["divisao"]["semente"])
X_tr_bruto = df.loc[i_tr, nomes]
pre = xe.ajustar_preprocessamento(X_tr_bruto, cfg)
X_tr = xe.aplicar_imputacao(X_tr_bruto, pre["medianas"], zeros)
X_te = xe.aplicar_imputacao(df.loc[i_te, nomes], pre["medianas"], zeros)

xe.salvar_json({"treino": i_tr.tolist(), "teste": i_te.tolist()}, "intermediarios", "divisao.json")
xe.salvar_json(pre, "intermediarios", "preprocessamento.json")
xe.salvar_json(eda, "intermediarios", "analise_exploratoria.json")
X_tr.assign(**{alvo: df.loc[i_tr, alvo].values}).to_csv(xe.caminho("intermediarios", "treino_imputado.csv"), index=False)
X_te.assign(**{alvo: df.loc[i_te, alvo].values}).to_csv(xe.caminho("intermediarios", "teste_imputado.csv"), index=False)

print(f"Registros: {len(df)} | treino: {len(i_tr)} | teste: {len(i_te)}")
print("Classes:", eda["classes"])
print("Zeros por atributo (%):", eda["zeros_por_atributo_pct"])
print("Medianas de imputação (treino):", {k: round(v, 2) for k, v in pre["medianas"].items()})
