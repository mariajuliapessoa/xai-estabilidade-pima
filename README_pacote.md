# (Documentação do pacote) Pacote de execução — Estabilidade e robustez das explicações de TreeSHAP e LIME (PIMA, CatBoost)

Este pacote implementa o protocolo da Seção 3.3 do artigo: as quatro condições (C0/C1, C2a/C2b, C3), as seis métricas de estabilidade, a instância como unidade de análise, o Wilcoxon pareado com correção de Holm, Friedman, *bootstrap* e o excesso de variação do LIME.

**Situação:** protocolo congelado em 02/11/2026 (`protocolo_congelado.json`) e executado no mesmo dia; as saídas estão em `resultados/`. Para reproduzir, rode as etapas 1 a 3 e 5 a 7 sem alterar `config.json` nem o código (o congelamento já existe). A etapa 8 (`python 08_analises_complementares.py`) roda as análises complementares A1–A3 (e uma versão reduzida da A4, substituída pela etapa 9); a etapa 9 (`python 09_analises_complementares_v2.py`) roda a A4 completa (K = 30, M = 100, quatro níveis de σ), a A5 e a A6. Todas foram definidas após a execução e são relatadas como exploratórias (Seção 4.4). A etapa 7 (`python 07_figuras.py`, depois da 8) gera as Figuras 1 e 2. A etapa 10 (`python 10_analises_complementares_v3.py`, depois da 5) roda as análises A7 (largura do kernel do LIME), A8 (fidelidade por curvas de deleção) e A9 (replicação com Random Forest), definidas em 13/11/2026, com M = 30, K = 10 e σ ∈ {0,03; 0,10}. As etapas 7 a 10 não são protegidas pelo congelamento.

**Nota sobre o piloto, o congelamento e a numeração.** Na execução registrada de 02/11/2026, o piloto (`intermediarios/piloto.json`, 14:09:14) e o congelamento (`protocolo_congelado.json`, 14:09:29) rodaram em sequência. O piloto apenas confirmou M = 100; as quatro decisões revisadas antes do congelamento (Quadro 3 do artigo) são decisões de desenho e não decorreram do piloto. Os hashes garantem que configuração, código e dados não mudaram desde o congelamento, mas não constituem pré-registro externo. Comentários internos dos scripts e o campo de `config.json` citam a numeração de uma versão anterior do manuscrito (Seções 6 e 7; Quadros 6 e 7); eles não foram editados para não alterar os hashes congelados. Correspondência: Seção 6 → 3.3; Seção 7 → 4.3 e 4.4; Quadro 7 → Quadro 3.


## 1. O que é necessário

- Python 3.10 ou superior (testado com 3.13.17 em Linux).
- As bibliotecas de `requirements.txt`, com versões fixas.
- O arquivo de dados `dados/pima-indians-diabetes.csv` (incluído no pacote). É o CSV sem cabeçalho, com 768 linhas e 9 colunas, obtido em `https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv` em 23/10/2026.
  - SHA-256: `6bfe5d0f379d17a0e0819b996407e3c09bf80febd4287f2ed212190dfff154af`.
  - Se usar outra cópia (por exemplo, a do Kaggle, que tem cabeçalho), ajuste `tem_cabecalho` e `sha256_esperado` em `config.json` e registre a origem.
- Tempo aproximado em 2 núcleos de CPU: etapas 1 e 2, menos de 1 min; etapa 3 (piloto), cerca de 1,5 min; etapa 5 com M = 100, estimativa de 10 a 20 min (ajuste `n_processos`).

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Arquivos

| Arquivo | Função |
|---|---|
| `config.json` | Todos os parâmetros do Quadro 3 do artigo. O campo `decisoes_pendentes` é histórico: as decisões foram confirmadas antes do congelamento e o campo não foi apagado para não alterar o hash congelado |
| `xai_estab.py` | Funções comuns: dados, imputação, CatBoost, TreeSHAP, LIME, perturbação, métricas, testes, trava do protocolo |
| `01_preparar_dados.py` | Divisão 80/20 estratificada, imputação pela mediana do treino, análise exploratória |
| `02_treinar_modelo.py` | Busca em grade com validação cruzada de 5 dobras (AUC-ROC) só no treino; treina o modelo de referência e os da C3 |
| `03_piloto.py` | Piloto com 20 instâncias do treino para definir M |
| `04_congelar_protocolo.py` | Registra os hashes de configuração, código e dados |
| `05_executar_condicoes.py` | Desempenho no teste, C1, C2a, C2b e C3 |
| `06_analisar.py` | Análise pré-especificada e Tabelas 1 e 2 |
| `07_figuras.py` | Figuras 1 e 2 |
| `08_analises_complementares.py` | Análises exploratórias A1–A3 (Seção 4.4) |
| `09_analises_complementares_v2.py` | Análises exploratórias A4 (completa), A5 e A6 (Seção 4.4; Tabela 4) |
| `10_analises_complementares_v3.py` | Análises exploratórias A7, A8 e A9 (Seção 4.4) |

## 3. Passo a passo

Execute os comandos dentro da pasta do pacote.

1. **Preparar os dados:** `python 01_preparar_dados.py`. Gera `intermediarios/divisao.json`, `preprocessamento.json`, `analise_exploratoria.json` e os CSVs imputados.
2. **Selecionar e treinar o modelo:** `python 02_treinar_modelo.py`. Gera `intermediarios/selecao_hiperparametros.json` e `modelos.pkl`.
3. **Rodar o piloto:** `python 03_piloto.py`. Gera `intermediarios/piloto.json` (M mínimo pelo critério) e `piloto_curva.csv`. O piloto usa ρ de Spearman; a métrica primária da análise é o τ-b. M = 100 foi mantido com margem (Seção 3.3 do artigo).
4. **Confirmar as decisões pendentes** com o orientador (lista em `config.json` → `decisoes_pendentes`) e editar `config.json`, se for o caso. Se mudar algo que afete as etapas 1 a 3, rode-as de novo.
5. **Testar o código (opcional):** `python 05_executar_condicoes.py --smoke` e `python 06_analisar.py --smoke`. Usa 5 instâncias do treino e grava em `resultados_smoke/`. Os valores não são resultados.
6. **Congelar o protocolo:** `python 04_congelar_protocolo.py` (mostra a lista) e depois `python 04_congelar_protocolo.py --confirmo`. A partir daqui, não edite `config.json` nem os scripts.
7. **Executar as condições:** `python 05_executar_condicoes.py`.
8. **Analisar:** `python 06_analisar.py`.

## 4. Correspondência com o artigo (Seções 3.3, 4.3 e 4.4)

| Seção do artigo | Fonte |
|---|---|
| 4.3 Dados, modelo, piloto (Tabela 1) | `intermediarios/analise_exploratoria.json`, `preprocessamento.json`, `selecao_hiperparametros.json`, `piloto.json`, `piloto_curva.csv`; `resultados/desempenho.json`, `tabelas/tabela1_desempenho.md`, `C3_desempenho_por_semente.json` |
| 4.3 C1 e H2 | `relatorio_resultados.json` → `H2_treeshap_max_dif_abs`; `tabelas/descritivas_todas_metricas.csv` (condição C1) |
| 4.3 Robustez (Tabelas 2 e 3, Figura 1) | `tabelas/tabela2_robustez_C2a.md`; `relatorio_resultados.json` → `primaria_H3_TreeSHAP_menos_LIME`, `H1_tendencia_sigma`, `excesso_LIME`, `perturbacoes_efetivas_pct` |
| 4.3 Predição × explicação (H4) | `relatorio_resultados.json` → `H4_exploratoria` |
| 4.3 C3 e ajuste local do LIME | `relatorio_resultados.json` → `C3`, `C3_desempenho`, `lime_r2_local` |
| 4.4 A1–A3 (Figura 2) | `resultados/analises_complementares.json` → `A1_limiar`, `A2_acertos_erros`, `A3_mecanismo`; `A3_mecanismo_pares.csv` |
| 4.4 A4–A6 (Tabela 4) | `resultados/analises_complementares_v2.json`; `A4v2_lime_padrao_C1.csv`, `A4v2_lime_padrao_pares.csv`, `A6_continuos_pares.csv` |
| 4.4 A7–A9 | `resultados/analises_complementares_v3.json`; `A7_largura_kernel.csv`, `A8_fidelidade_delecao.csv`, `A9_random_forest.csv` |
| 3.3 Reprodutibilidade | `protocolo_congelado.json` (hashes, versões, configuração) e `resultados/log_execucao.json` |

Para cada teste, registre: n de instâncias, estatística, p bruto, p ajustado por Holm, bisserial de postos e IC 95% da diferença mediana. Relate também as instâncias com L1 normalizada indefinida (soma das atribuições próxima de zero).

## 5. Convenções de cálculo

- **Unidade de análise:** a instância. As métricas são agregadas pela média das M versões (C2) ou das K − 1 comparações com a primeira execução (C1).
- **Métricas:** calculadas sobre |φ|, em `xai_estab.metricas_par`.
  - Kendall τ-b e Spearman ρ sobre os rankings de |φ|.
  - J@3 e Top-1 com desempate estável pelo índice do atributo.
  - L1 normalizada: ‖φ − φ′‖₁ / ‖φ‖₁.
  - Concordância de sinal.
  - Explicações idênticas recebem τ = ρ = 1.
- **H3:** a diferença é TreeSHAP − LIME. Valor positivo indica maior concordância do TreeSHAP.
- **Excesso do LIME:** C2b − C1 (análise principal) e C2a − C1 (descritiva).
- **TreeSHAP:** não tem semente, então seus valores em C2a e C2b são idênticos.
- **Sementes:** derivadas por `numpy.random.SeedSequence` a partir das sementes-base de `config.json`.
