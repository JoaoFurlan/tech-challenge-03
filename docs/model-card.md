# Model Card — Classificador de Triagem de Laudos Médicos

Segue a estrutura proposta em Mitchell et al., *Model Cards for Model
Reporting* (2019). Não exigido pelo Tech Challenge — incluído porque
documentar um modelo assim é prática profissional padrão para qualquer coisa
que toque decisões clínicas, mesmo em escala de demo/acadêmica. Ver
`architecture.md` e `technical-decisions.md` para o raciocínio completo por
trás de cada decisão de design.

## Detalhes do modelo

- **Desenvolvido por:** projeto solo, Tech Challenge Fase 3 (POS Tech / MLET).
- **Data do modelo:** 2026-09.
- **Tipo do modelo:** classificador de texto — vetorizador TF-IDF (unigrama,
  `max_features=10000`, `min_df=2`, `stop_words="english"`) + **Complement
  Naive Bayes** (`alpha=0.5, norm=True`), escolhido no experimento MLflow
  `model-selection` entre Regressão Logística, LinearSVC, Multinomial NB e
  Random Forest — *não* o líder bruto de F1-macro (LinearSVC), escolhido em
  vez disso por uma taxa de subtriagem substancialmente menor, uma troca
  documentada e explícita de segurança por acurácia. Raciocínio completo em
  `technical-decisions.md`.
- **O que prevê diretamente:** uma de 5 categorias de doença (neoplasms,
  doenças cardiovasculares, doenças do sistema nervoso, doenças digestivas,
  condições patológicas gerais) a partir do texto livre de um laudo médico.
- **O que retorna após pós-processamento:** um nível de urgência — normal /
  atenção / urgente — derivado deterministicamente da categoria prevista mais
  um ajuste por palavra-chave sobre o texto do laudo (ver `architecture.md`
  § Mapeamento de urgência). O nível de urgência **não** é aprendido; só a
  previsão de categoria é uma saída de machine learning. A resposta também
  inclui `low_confidence: bool`, sinalizando quando a previsão de categoria
  não tem evidência real de vocabulário por trás.
- **Otimização:** exportado para ONNX (FP32) — escolhido sobre o pipeline
  sklearn por um ganho de 4,4x em latência P50 sem custo de acurácia. A
  quantização INT8 foi testada e rejeitada: contraintuitivamente mais *lenta*
  que FP32 nessa escala de modelo — um achado negativo documentado. Ver
  `technical-decisions.md` § Otimização de latência.
- **Licença / artigo:** nenhum publicado para este projeto; o dataset
  subjacente é o Medical Abstracts TC Corpus (Kaggle /
  `sebischair/Medical-Abstracts-TC-Corpus`).

## Uso pretendido

- **Uso principal pretendido:** demonstração acadêmica de um pipeline de
  deploy de ML (CI/CD, orquestração, monitoramento, otimização de latência)
  para um desafio de pós-graduação. A ambientação de "triagem hospitalar" é o
  cenário do exercício, não um deploy clínico validado.
- **Usuários principais pretendidos:** o autor do projeto e os avaliadores do
  curso; público secundário é quem revisar o repositório como peça de
  portfólio/referência.
- **Usos fora de escopo:** **este modelo não deve ser usado para triagem
  clínica real ou qualquer decisão real voltada a paciente.** É treinado em
  um dataset público de abstracts de pesquisa, não em texto real de admissão
  hospitalar, não foi validado clinicamente, e a camada de mapeamento de
  urgência é uma heurística documentada, não um sistema de pontuação
  clinicamente derivado.

## Fatores

- **Fatores relevantes:** tamanho e estilo de vocabulário do laudo (o
  dataset é de *abstracts* médicos — escrita acadêmica condensada e técnica —
  que pode diferir sistematicamente de como um laudo hospitalar real é
  redigido).
- **Fatores de avaliação:** o desempenho é avaliado por categoria de doença
  (5 classes), não por nível de urgência, já que urgência não tem ground
  truth neste dataset — ver § Métricas.

## Métricas

- **Métrica principal de seleção de modelo: F1-macro** entre as 5 categorias
  de doença — escolhida especificamente para que um modelo não vença só por
  ser bom na maior categoria. Raciocínio completo em `technical-decisions.md`
  § Métricas de avaliação.
- **Reportado junto de cada modelo:** precisão/recall por classe, matriz de
  confusão, acurácia (só contexto, não decisiva).
- **Checagem secundária motivada clinicamente:** o recall na categoria
  cardiovascular (nossa baseline de "urgente") é confirmado explicitamente
  depois de escolher o vencedor por F1-macro — um falso negativo ali é o modo
  de falha mais custoso, já que cairia silenciosamente para um nível de
  urgência mais baixo.
- **Métricas de latência** (separadas da qualidade de classificação): tempo
  de resposta P50/P95/P99 sobre todo o pipeline de `/predict`, modelo
  original vs. otimizado.
- **Métricas consideradas e não usadas:** ROC-AUC, PR-AUC, Coeficiente de
  Correlação de Matthews — ver `technical-decisions.md` para o porquê.

## Dados de treino

- **Fonte:** Medical Abstracts TC Corpus (Kaggle), 14.438 registros
  rotulados.
- **Distribuição de classes:**

| Categoria | Quantidade |
|---|---|
| Condições patológicas gerais | 4.805 |
| Neoplasms | 3.163 |
| Doenças cardiovasculares | 3.051 |
| Doenças do sistema nervoso | 1.925 |
| Doenças digestivas | 1.494 |

- **Pré-processamento:** vetorização TF-IDF (configuração escolhida via o
  experimento MLflow `feature-engineering`); sem embeddings externos.
- **Split:** ~80–85% usado para seleção de modelo, engenharia de features e
  tuning de hiperparâmetros com validação cruzada (Stratified K-Fold);
  ~15–20% separado como conjunto de teste, intocado até a avaliação final.
  Detalhes em `technical-decisions.md` § Split treino/validação/teste.

## Dados de avaliação

Mesma distribuição de origem dos dados de treino — o conjunto de teste
descrito acima, extraído do mesmo Medical Abstracts TC Corpus via um único
split estratificado. Nenhum conjunto de avaliação fora-de-distribuição
separado é usado; essa é uma limitação conhecida (ver § Caveats).

## Análises quantitativas

Pipeline final (ComplementNB `alpha=0.5, norm=True` + TF-IDF conforme acima),
avaliado uma única vez no conjunto de teste de 1.245 documentos — ver
`architecture.md` e `technical-decisions.md` para o processo completo em
etapas (seleção de modelo → engenharia de features → tuning de
hiperparâmetros) que levou até aqui, incluindo cada alternativa rejeitada e
por quê.

| Métrica | Valor |
|---|---|
| F1-macro | 0,7786 |
| Acurácia | 0,7880 |
| Acurácia de nível (urgência, não só categoria) | 0,8056 |
| Taxa de subtriagem (nível previsto abaixo do real — perigoso) | 0,0498 |
| Taxa de sobretriagem (nível previsto acima do real — custoso, não perigoso) | 0,1446 |

Por classe (conjunto de teste):

| Categoria | Recall | Precisão |
|---|---|---|
| Doenças cardiovasculares | 0,939 | 0,758 |
| Neoplasms | 0,927 | 0,831 |
| Doenças digestivas | 0,781 | 0,788 |
| Doenças do sistema nervoso | 0,709 | 0,747 |
| Condições patológicas gerais | 0,574 | 0,792 |

O recall de cardiovascular (0,939) foi a checagem clinicamente motivada
explícita da § Métricas — confirmado alto, consistente com a escolha de
modelo voltada à segurança. Condições patológicas gerais tem o menor recall
(0,574) mas a maior precisão entre as classes mais fracas (0,792) —
consistente com a inclinação documentada do ComplementNB de evitar prever
com confiança a categoria mapeada para "normal" a menos que a evidência seja
forte, o mesmo mecanismo que guiou a escolha na seleção de modelo.

**Latência** (ONNX FP32 vs. baseline sklearn, `/predict` de documento único,
benchmark de 500 requisições):

| Variante | P50 | P95 | P99 | Tamanho |
|---|---|---|---|---|
| Baseline sklearn | 0,595ms | 0,814ms | 1,040ms | 1.233KB |
| ONNX FP32 (servido) | 0,135ms | 0,263ms | 0,339ms | 413KB |
| ONNX INT8 (testado, não servido) | 0,163ms | 0,281ms | 0,397ms | 267KB |

INT8 foi mais lento que FP32, não mais rápido — ver `technical-decisions.md`
§ Otimização de latência para o porquê.

## Considerações éticas

- **Não é um dispositivo médico validado.** Nenhuma revisão clínica,
  regulatória ou com seres humanos foi realizada. Explicitamente não é para
  uso real em triagem — ver § Uso pretendido.
- **Os níveis de urgência são uma heurística documentada, não ground truth.**
  A tabela de nível base por categoria e as listas de palavras-chave de
  escalada/de-escalada (`architecture.md` § Mapeamento de urgência) foram
  desenhadas pelo autor do projeto com base em raciocínio clínico geral, não
  derivadas de dado real de urgência nem revisadas por um clínico. Devem ser
  lidas como uma regra de negócio transparente e auditável — deliberadamente
  simples e inspecionável — não como um protocolo de triagem validado.
- **Modo de falha de maior preocupação:** um laudo de uma categoria
  genuinamente urgente (cardiovascular) classificado erroneamente em uma
  categoria de urgência menor causaria subtriagem silenciosa. É por isso que
  a revisão da matriz de confusão e o recall de cardiovascular são tratados
  como checagens obrigatórias, não diagnósticos opcionais, ao longo de todo o
  pipeline de modelagem.
- **Origem do dado:** o dado de treino são *abstracts* médicos publicados
  (texto de pesquisa/acadêmico), não registros reais de admissão de paciente
  — não há PII/PHI envolvido, mas isso também significa que o estilo do texto
  pode não transferir bem para a linguagem real de um laudo hospitalar (ver
  § Caveats).

## Caveats e recomendações

- **Risco de mudança de domínio — confirmado via teste no mundo real, não só
  teórico.** Abstracts médicos (registro condensado, em terceira pessoa,
  acadêmico) se lêem de forma diferente da fala real de triagem (curta,
  informal, jargão clínico). Testes manuais pós-deploy confirmaram isso
  concretamente: "Unresponsive, no detectable pulse, non-breathing" — uma
  descrição clássica de parada cardíaca — foi classificado como `normal`. As
  palavras estão individualmente no vocabulário de treino, mas o modelo nunca
  aprendeu a associar esse *registro* à urgência, porque nunca viu texto
  escrito dessa forma. **Isso não é corrigível adicionando mais dados do
  mesmo tipo de treino** — o Medical Abstracts TC Corpus é a única fonte de
  dado disponível, e mais abstracts só melhorariam o desempenho em texto
  estilo abstract, não ensinariam um registro nunca visto. Uma correção real
  precisa de exemplos de treino nesse registro diferente (notas de triagem
  reais ou realisticamente sintetizadas), o que é um aumento genuíno de
  escopo, não um follow-up rápido. Investigação completa, incluindo três
  outros exemplos concretos de erro de classificação e os dois bugs que essa
  investigação revelou e corrigiu, em `technical-decisions.md` § Teste no
  mundo real revelou uma limitação genuína de mudança de domínio.
- **Entradas de baixa confiança agora são sinalizadas, não silenciosamente
  confiadas.** Com a correção acima, `/predict` retorna `low_confidence:
  true` quando a entrada não compartilha nenhum vocabulário com os dados de
  treino (ex.: texto muito curto ou coloquial) — a previsão de categoria é
  então guiada pelo viés estrutural do classificador, não por evidência
  real. A urgência é **fixada em `attention`** nesse caso (não só elevada) —
  um palpite bruto de `urgente` é exatamente tão infundado quanto `normal`
  quando não há evidência real por trás, então nenhum dos extremos é
  confiado; um campo `message` também é retornado, orientando quem chamou a
  API a fornecer mais detalhe. Isso reduz um modo de falha (entradas de sinal
  zero) mas não resolve o risco mais amplo de descompasso de registro acima.
- **Nenhum monitoramento de drift implementado.** Em um deploy real, deriva
  de distribuição de entrada e deriva de conceito precisariam de
  monitoramento ativo; a stack de monitoramento deste projeto
  (Prometheus/Grafana) cobre só métricas operacionais (contagem de
  requisições, latência, taxa de erro), não deriva de qualidade do modelo.
- **Um único conjunto de teste separado.** Os resultados refletem um único
  split estratificado; nenhuma estimativa repetida/bootstrapped do conjunto
  de teste é calculada, então as métricas reportadas carregam alguma
  variância amostral não capturada por uma única estimativa pontual.
- **Recomendação para qualquer adaptação futura ao mundo real:** substituir
  o dado de treino por texto real (desidentificado) de laudo hospitalar, ter
  a regra de mapeamento de urgência revisada e validada por um clínico em vez
  de depender da heurística do autor, e adicionar monitoramento de drift e
  revisão humana no loop antes de considerar qualquer uso clínico real.
