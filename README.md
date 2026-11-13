# Estabilidade e robustez das explicações de SHAP e LIME na classificação do diabetes (PIMA)

Código, dados e resultados do estudo experimental do Trabalho de Conclusão de Curso
**"Estabilidade e robustez das explicações de SHAP e LIME na classificação do diabetes mellitus:
revisão estruturada da literatura e estudo experimental com o conjunto PIMA"**.

- **Autora:** Maria Júlia Pessoa Cunha
- **Orientador:** Eduardo Nascimento de Arruda
- **Curso:** Ciência da Computação, CESAR School, 2026

## O que o experimento faz

Treina um classificador CatBoost no PIMA Indians Diabetes Dataset e compara as explicações
locais do TreeSHAP e do LIME em 154 instâncias de teste, separando três fontes de variação:

| Condição | O que mede |
|---|---|
| C1 | Variabilidade intrínseca: o LIME explica a mesma instância 30 vezes, com sementes diferentes |
| C2a | Robustez à perturbação da entrada, com a mesma semente do LIME para x e x′ (isola o efeito da entrada) |
| C2b | Robustez no uso habitual, com sementes independentes |
| C3 | Variabilidade do modelo: o CatBoost é retreinado com outras quatro sementes |

Ruído gaussiano proporcional ao desvio-padrão de cada atributo, σ ∈ {0,01; 0,03; 0,05; 0,10},
M = 100 versões por instância. Métrica primária: Kendall τ-b; teste de Wilcoxon pareado com correção de Holm.

## Protocolo congelado

O protocolo foi congelado em 02/11/2026, antes do uso do conjunto de teste. O arquivo
`protocolo_congelado.json` guarda o SHA-256 da configuração, dos scripts do protocolo e dos dados.
Os scripts da análise pré-especificada recusam-se a executar se algum desses arquivos for alterado.
O congelamento garante a integridade desde essa data, mas não é um pré-registro externo.

## Como reproduzir

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
gunzip -k resultados/C2_robustez_pares.csv.gz       # necessário só para as etapas 6, 8 e 9 sem rodar a 5
python 01_preparar_dados.py
python 02_treinar_modelo.py
python 03_piloto.py
python 05_executar_condicoes.py    # 10 a 20 min em 2 núcleos
python 06_analisar.py
python 08_analises_complementares.py
python 09_analises_complementares_v2.py
python 07_figuras.py
```

Testado com Python 3.13. Não rode `04_congelar_protocolo.py --confirmo`: o congelamento já existe.

`resultados/C2_robustez_pares.csv` foi compactado (`.csv.gz`) porque o GitHub não aceita arquivos
acima de 25 MB pelo navegador. A etapa 5 gera o arquivo de novo.

## Estrutura

| Pasta ou arquivo | Conteúdo |
|---|---|
| `dados/` | PIMA Indians Diabetes Dataset (768 registros, 8 atributos) |
| `config.json` | Todos os parâmetros do protocolo |
| `xai_estab.py` | Funções comuns: perturbação, explicadores, métricas, testes |
| `01` a `09` | Etapas do pipeline, na ordem acima |
| `intermediarios/` | Divisão, imputação, hiperparâmetros, piloto |
| `resultados/` | Saídas, tabelas e figuras do artigo |
| `README_pacote.md` | Documentação completa, com a correspondência entre cada número do artigo e o arquivo que o gera |

## Dados

O PIMA Indians Diabetes Database foi originalmente disponibilizado pelo National Institute of Diabetes
and Digestive and Kidney Diseases e é distribuído publicamente pelo UCI Machine Learning Repository e pelo Kaggle.
O arquivo está incluído só para reprodutibilidade. A licença MIT deste repositório cobre o código, não os dados.

## Como citar

Veja `CITATION.cff`.
