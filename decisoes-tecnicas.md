# Decisões Técnicas

Registro das decisões tomadas para o projeto do Tech Challenge Fase 3 e a
justificativa por trás de cada uma — especialmente onde optamos por uma versão
mais simples do que seria "o ideal" em produção, e por quê. Complementa o
`architecture.md` (que descreve o *o quê*; este documento foca no *por quê*).

## Dataset e rotulagem de urgência

**Escolha:** Medical Abstracts TC Corpus (Kaggle), 14.438 laudos rotulados em 5
categorias de doença (neoplasias, doenças cardiovasculares, doenças do sistema
nervoso, doenças digestivas, condições patológicas gerais).

**Problema:** o dataset não possui rótulos reais de urgência (normal/atenção/
urgente) — apenas categorias de doença. Treinar diretamente contra rótulos de
urgência sintéticos (inventados por nós) seria cientificamente frágil: as
métricas de avaliação estariam medindo o quão bem o modelo aprendeu nossa
própria heurística, não urgência real.

**Decisão:** treinar o classificador nas 5 categorias reais (ground truth
genuíno) e aplicar uma camada determinística de mapeamento categoria→urgência
por cima da predição, ajustada por palavras-chave presentes no próprio texto do
laudo. Isso mantém a avaliação do modelo honesta (métricas contra rótulos
reais) e documenta a lógica de urgência como uma regra de negócio explícita e
auditável, não como algo "aprendido" de forma opaca.

**Trade-off assumido:** a qualidade da triagem final depende tanto da acurácia
do classificador de categoria quanto da qualidade da regra de mapeamento — um
erro em qualquer uma das duas partes pode produzir uma urgência incorreta. Por
isso a matriz de confusão é tratada como artefato obrigatório em cada
experimento: um laudo cardiovascular classificado erroneamente como "condição
patológica geral", por exemplo, cairia silenciosamente para "normal" a jusante
— esse é o modo de falha mais perigoso do sistema, e é o que a matriz de
confusão existe para detectar.

## Modelos candidatos

**Escolha:** Regressão Logística, LinearSVC, Multinomial Naive Bayes, Complement
Naive Bayes e Random Forest, todos com `class_weight="balanced"` onde suportado.

**Por que não apenas Random Forest** (sugestão literal do PDF do desafio): para
texto TF-IDF de alta dimensionalidade e esparso, modelos lineares tendem a
performar melhor e de forma mais previsível do que ensembles de árvores — e têm
exportação ONNX mais limpa. Mesmo assim, Random Forest entra na comparação:
testá-lo empiricamente e mostrar por que (ou se) ele perde para os modelos
lineares é uma narrativa mais forte do que simplesmente descartá-lo sem
evidência.

**Por que não embeddings (Word2Vec/BERT/ClinicalBERT):** o próprio desafio pede
um "modelo de texto (NLP) leve" — embeddings/transformers trabalhariam contra
esse requisito e contra a história de otimização de latência que construímos em
torno de um modelo linear pequeno. Considerado e descartado deliberadamente, não
por desconhecimento.

## Seleção de modelo: engenharia de features simplificada primeiro

**Decisão:** ao invés de buscar a combinação ótima de configuração de TF-IDF
para cada um dos 5 modelos (busca completa, custosa), testamos apenas 2
configurações representativas ("conservadora" e "rica") em todos os 5 modelos
na etapa de seleção de modelo. A busca completa de features (36 combinações) é
feita depois, apenas no modelo vencedor.

**Por que isso é aceitável:** é uma simplificação pragmática de uma prática já
padrão na indústria — "model bake-off" / "spot-checking algorithms": avaliar
candidatos de forma barata primeiro, investir esforço de tuning pesado só no
vencedor. A versão totalmente automatizada disso (AutoML, busca bayesiana
conjunta sobre modelo+features+hiperparâmetros) exigiria mais infraestrutura do
que duas semanas permitem, e — mais importante — uma busca automatizada conjunta
produz uma história muito menos explicável no README/vídeo do que um pipeline
em etapas claras. Preferimos interpretabilidade à rigor marginal adicional
aqui, e isso é uma escolha deliberada, não uma limitação escondida.

## Métricas de avaliação

**Métrica de decisão: F1-macro.** Faz média não ponderada entre as 5 classes —
um modelo não pode vencer só por ser bom na classe majoritária ("condição
patológica geral", 4.805 exemplos).

**Por que não apenas acurácia:** com desbalanceamento moderado entre classes
(~3,2x entre a maior e a menor), acurácia pode ser enganosa — reportamos por
contexto, mas nunca como critério de decisão.

**Por que recall de cardiovascular é verificado à parte:** a categoria
cardiovascular é nossa linha de base para "urgente". Um falso negativo aqui
(laudo cardiovascular classificado incorretamente) tem custo clínico maior do
que um falso positivo. Por isso, após escolher o vencedor por F1-macro,
confirmamos explicitamente que o recall de cardiovascular não foi sacrificado —
uma checagem em duas etapas deliberada, mais transparente do que tentar
embutir esse peso clínico em uma única métrica composta automática.

**Por que não ROC-AUC/PR-AUC/MCC:** para um problema multiclasse, ROC-AUC exige
decisões adicionais (One-vs-Rest, média macro/weighted) e `predict_proba` — que
o LinearSVC não possui nativamente, exigindo `CalibratedClassifierCV` extra só
para esse candidato. F1-macro + recall por classe + matriz de confusão já
cobrem o mesmo terreno com menos complexidade adicional. Considerado e
descartado conscientemente, não por desconhecimento — inclusive discutido e
avaliado antes da decisão final.

## Divisão treino/validação/teste e vazamento de dados

**Decisão:** conjunto de teste (~15–20%) separado uma única vez no início,
nunca tocado durante as etapas de seleção de modelo, engenharia de features ou
tuning de hiperparâmetros. Todas as etapas de experimentação usam
Stratified K-Fold sobre o restante dos dados. O pipeline final (modelo +
configuração de features + hiperparâmetros vencedores) é retreinado no
conjunto completo de treino+validação e avaliado uma única vez no conjunto de
teste — esse é o número que vai para o README.

**Por que isso importa:** avaliar repetidamente contra o mesmo conjunto de
teste em cada etapa de decisão, mesmo sem treinar diretamente nele, cria
overfitting indireto ao conjunto de teste. Separar um conjunto de teste único e
usá-lo apenas uma vez, no final, evita essa armadilha.

**Uso de `sklearn.pipeline.Pipeline`:** o vetorizador TF-IDF e o classificador
são empacotados em um único objeto `Pipeline`. Isso não é só conveniência de
deploy — evita vazamento de dados estruturalmente: ao passar o `Pipeline` para
`cross_validate`/`GridSearchCV`, o vetorizador é reajustado (`fit`) apenas nos
dados de treino de cada fold, nunca nos dados de validação. O mesmo objeto
`Pipeline` treinado é salvo, carregado pela API FastAPI e exportado para ONNX —
um único artefato, comportamento idêntico em todo lugar, sem risco de
divergência entre treino e produção (training-serving skew).

## Otimização de latência (Etapa 4)

**Decisão:** a técnica aplicada depende de qual modelo vencer a seleção:
- Modelo linear vencedor: exportação ONNX + quantização dinâmica INT8.
- Random Forest vencedor: exportação ONNX + poda por complexidade de custo
  (`ccp_alpha`) e/ou redução de `n_estimators`.

**Por que não forçar quantização em qualquer caso:** quantização reduz a
precisão numérica de matrizes de pesos densas (multiplicação de matrizes) —
não se aplica de forma significativa a ensembles de árvores, que não têm essa
estrutura. Forçar essa técnica em um Random Forest seria um erro de categoria.
Poda de árvore (`ccp_alpha`), por outro lado, é literalmente a origem histórica
do termo "pruning" (anterior ao uso em redes neurais) — mais apropriada, não
uma alternativa inferior.

**Métrica reportada:** P50/P95/P99 sobre o pipeline completo de `/predict`
(pré-processamento + inferência + resposta), não apenas `model.predict()` —
pré-processamento pode representar 40–60% da latência total em um sistema não
otimizado.

## Ferramentas: uv, DVC, MLflow

**uv:** substitui `requirements.txt`/Poetry, sem custo relevante de adoção.

**DVC + S3 (não apenas local):** mesmo com um dataset estático (não muda ao
longo do projeto), optamos por hospedar no S3 via DVC real, não apenas
localmente — reflete como isso é feito de fato na indústria e evita tratar essa
prática como decorativa.

**MLflow, 4 experimentos separados, desde o início:** `model-selection`,
`feature-engineering`, `hyperparameter-tuning`, `latency-optimization`. Rastreamento
local (`mlruns/`, sem servidor dedicado) — visualizado via `mlflow ui` no
navegador. Separar essas etapas em experimentos distintos (prática comum na
indústria) mantém o espaço de busca de cada etapa limpo e comparável.

## CI/CD e AWS

**Push real para o ECR via GitHub Actions:** autenticação por OIDC (federação
com `token.actions.githubusercontent.com`), sem chaves estáticas da AWS
armazenadas como secrets do GitHub — prática de segurança padrão do setor.

**EC2, não Lambda/Batch/SageMaker, para inferência em tempo real:** o cenário
exige resposta imediata (triagem hospitalar). EC2 mantém o modelo carregado em
memória permanentemente, sem o cold-start problemático do Lambda. Justificativa
completa no README.

## Airflow em modo standalone

**Decisão:** `airflow standalone` (backend SQLite), não o docker-compose oficial
de produção (Postgres + Redis + webserver + scheduler + worker).

**Por que:** o desafio pede um DAG "simples" simulando treino/retreino — o
compose oficial de produção é desproporcional para uma demonstração de 3
tasks. O DAG é executado manualmente (não há fluxo real de dados novos
alimentando o projeto), demonstrando a capacidade de orquestração, não uma
necessidade real de retreino recorrente neste contexto específico.

## Frontend Streamlit (extra, não avaliado)

Aplicação simples separada, chamando o endpoint `/predict` da API via HTTP (sem
duplicar lógica do modelo) — adicionada apenas para tornar a demonstração em
vídeo mais visual do que mostrar a documentação Swagger ou um comando curl.
Marcada explicitamente como não obrigatória.

## Fora de escopo (deliberadamente)

Kubernetes/HPA/KEDA, deployment Canary/Shadow, detecção de drift implementada
(PSI/KS — mencionada no README como trabalho futuro, não construída),
embeddings de palavras/transformers, busca automatizada conjunta (AutoML) sobre
modelo+features+hiperparâmetros, ROC-AUC/PR-AUC/MCC, qualquer tráfego real de
produção. Todos considerados e descartados conscientemente pelas razões
descritas acima — não por falta de conhecimento do que existe, mas porque não
se justificam dentro do escopo e prazo deste desafio.
