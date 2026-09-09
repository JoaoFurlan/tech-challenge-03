# Decisões Técnicas

Registro das decisões tomadas no projeto do Tech Challenge Fase 3 e o
raciocínio por trás de cada uma — especialmente onde optamos por uma versão
mais simples do que "o ideal" em produção, e por quê. Complementa
`architecture.md` (que descreve o *quê*; este documento foca no *porquê*).

## Dataset e mapeamento de urgência

**Escolha:** Medical Abstracts TC Corpus (Kaggle), 14.438 laudos rotulados em
5 categorias de doença (neoplasms, doenças cardiovasculares, doenças do
sistema nervoso, doenças digestivas, condições patológicas gerais).

**O problema:** o dataset não tem rótulos reais de urgência (normal/atenção/
urgente) — só categorias de doença. Treinar diretamente contra rótulos de
urgência sintéticos (inventados por nós) seria cientificamente frágil: as
métricas de avaliação mediriam o quão bem o modelo aprendeu nossa própria
heurística, não a urgência real.

**Decisão:** treinar o classificador nas 5 categorias reais (ground truth
genuíno) e aplicar por cima uma camada determinística de mapeamento
categoria→urgência, ajustada por palavras-chave presentes no próprio texto do
laudo. Isso mantém a avaliação do modelo honesta (métricas contra rótulos
reais) e documenta a lógica de urgência como uma regra de negócio explícita e
auditável, em vez de algo "aprendido" de forma opaca.

**Trade-off aceito:** a qualidade final da triagem depende tanto da acurácia
do classificador de categoria quanto da qualidade da regra de mapeamento — um
erro em qualquer uma das partes pode gerar uma urgência incorreta. Por isso a
matriz de confusão é tratada como artefato obrigatório em todo experimento: um
laudo cardiovascular classificado erroneamente como "condição patológica
geral", por exemplo, cairia silenciosamente para "normal" no mapeamento —
esse é o modo de falha mais perigoso do sistema, e é o que a matriz de
confusão existe para pegar.

## Novo split do dataset e remoção de rótulos ambíguos

**O problema:** o Kaggle entrega esse corpus já dividido em train (11.550
linhas) e test (2.888 linhas). Combinar os dois para montar nosso próprio
split (em vez de usar o que veio pronto) revelou dois problemas de qualidade
de dado invisíveis em cada arquivo isoladamente:

1. **988 abstracts aparecem em ambos os arquivos**, original de train e de
   test — o split oficial tem vazamento train/test embutido.
2. **2.929 abstracts aparecem mais de uma vez com rótulos de categoria
   conflitantes** — mesmo texto, categoria diferente. Nenhum grupo duplicado
   repete com rótulo igual: toda duplicata é um conflito genuíno.
   "Condições patológicas gerais" está envolvida na grande maioria dos pares
   conflitantes (ex.: `cardiovascular diseases` + `general pathological
   conditions`, 738 pares), consistente com o corpus original tendo
   multi-rotulado alguns documentos, o que essa versão do Kaggle explodiu em
   linhas separadas de rótulo único.

**Por que isso importa além de higiene de dado:** colide diretamente com o
desenho do mapeamento de urgência. "Condições patológicas gerais" mapeia para
**normal**; vários de seus parceiros de conflito mais comuns (`cardiovascular
diseases`) mapeiam para **urgente**. Qual rótulo fosse mantido para um
documento ambíguo decidiria silenciosamente sua urgência — de forma
arbitrária.

**Decisão:** combinar train+test e **descartar por completo os 2.929
documentos ambíguos**, em vez de manter um rótulo por documento. Considerado e
rejeitado: manter a primeira linha ocorrida (mais simples, mas o rótulo
mantido seria incidental à ordem das linhas do Kaggle) e um desempate
enviesado para segurança (amarra a limpeza de dado a comportamento de modelo
disfarçado). Descartar os ambíguos por completo mantém todo rótulo
remanescente como ground truth inequívoco e ainda deixa **8.298 documentos** —
folgadamente acima do piso de 2.000 exigido pelo desafio — com o
desbalanceamento de classes praticamente inalterado (~3,4x vs. ~3,2x
original). O conjunto de teste é então separado desse pool limpo com
`random_state` fixo.

## Modelos candidatos

**Escolha:** Regressão Logística, LinearSVC, Multinomial Naive Bayes,
Complement Naive Bayes e Random Forest, todos com `class_weight="balanced"`
onde suportado.

**Por que não só Random Forest** (sugestão literal do PDF do desafio): para
texto TF-IDF esparso e de alta dimensionalidade, modelos lineares tendem a
performar melhor e de forma mais previsível que ensembles de árvore — e
exportam mais limpo para ONNX. Random Forest ainda entra na comparação:
testá-lo e mostrar por que ele perde (ou não) para os modelos lineares é uma
narrativa mais forte do que descartá-lo sem evidência.

**Por que não embeddings (Word2Vec/BERT/ClinicalBERT):** o próprio desafio
pede um "modelo leve de NLP" — embeddings/transformers iriam contra esse
requisito e contra a história de otimização de latência construída em torno
de um modelo linear pequeno. Considerado e deliberadamente rejeitado, não
esquecido.

## Seleção de modelo: engenharia de features simplificada primeiro

**Decisão:** em vez de buscar a configuração ideal de TF-IDF por modelo (uma
busca completa e cara nos 5 candidatos), testamos apenas 2 configurações
representativas ("conservadora" e "rica") nos 5 modelos na etapa de seleção
de modelo. A busca completa de features (36 combinações) roda depois, só no
modelo vencedor.

**Por que isso é aceitável:** é uma simplificação pragmática de uma prática
já padrão na indústria — "bake-off de modelos": filtrar candidatos de forma
barata primeiro, investir tuning pesado só no vencedor. A versão totalmente
automatizada disso (AutoML, busca conjunta sobre modelo+features+
hiperparâmetros) precisaria de mais infraestrutura do que o prazo permite — e,
mais importante, produziria uma história bem menos explicável para o
README/vídeo do que um pipeline com decisões claras e em etapas.
Interpretabilidade escolhida em vez de rigor marginal adicional,
deliberadamente.

## Métricas de avaliação

**Métrica de decisão: F1-macro.** Média não ponderada entre as 5 classes — um
modelo não pode vencer só por ser bom na classe majoritária ("condição
patológica geral", 4.805 amostras).

**Por que não só acurácia:** com desbalanceamento moderado de classes (~3,2x
entre a maior e a menor classe), a acurácia pode enganar — reportada como
contexto, nunca como critério de decisão.

**Por que o recall de cardiovascular é checado à parte:** cardiovascular é
nossa baseline de "urgente". Um falso negativo aqui tem custo clínico maior
que um falso positivo. Depois de escolher o vencedor por F1-macro,
confirmamos explicitamente que o recall de cardiovascular não foi
sacrificado — uma checagem deliberada em duas etapas, mais transparente do
que embutir essa ponderação clínica em uma única métrica composta.

**Por que não ROC-AUC/PR-AUC/MCC:** em um problema multiclasse, ROC-AUC exige
decisões extras (One-vs-Rest, média macro/ponderada) e `predict_proba`, que o
LinearSVC não tem nativamente. F1-macro + recall por classe + matriz de
confusão já cobrem o mesmo terreno com menos complexidade. Considerado e
conscientemente descartado.

**Emenda: métricas de nível de urgência adicionadas, e usadas como
co-decisivas.** A seleção de modelo revelou um problema estrutural do
F1-macro para este sistema: 3 das 5 categorias (neoplasms, nervous, digestive)
mapeiam para o **mesmo** nível de urgência ("atenção"). O F1-macro penaliza
confundir essas três exatamente como penalizaria um erro genuinamente
perigoso (cardiovascular → geral, queda de 2 níveis), mesmo que o primeiro
caso não afete a triagem real. `training/model_selection.py` agora também
loga `tier_accuracy`, `undertriage_rate` (nível previsto < real — a direção
perigosa) e `overtriage_rate` (nível previsto > real — custoso, mas seguro).

**Resultado da seleção de modelo: ComplementNB (TF-IDF conservador) escolhido
em vez do líder de F1-macro.** LinearSVC/rico teve o melhor F1-macro (0,798),
mas sua margem sobre LogisticRegression/rico (0,793) foi menor que o desvio
padrão fold-a-fold de ambos (~0,009–0,013) — estatisticamente um empate.
ComplementNB/conservador ficou atrás em F1-macro (0,765, diferença real de
~3 pontos) mas teve `undertriage_rate` substancialmente menor (0,053 vs.
0,089 do LinearSVC/rico — redução relativa de ~40%). Verificamos o mecanismo
antes de confiar no número: o recall de cardiovascular do ComplementNB
(0,933) supera bastante sua precisão (0,774), e sua precisão em condições
patológicas gerais (0,783) supera bastante seu recall (0,562) — uma
inclinação consistente para longe do nível "normal" quando incerto, não
superprevisão indiscriminada. Comportamento esperado do Complement Naive
Bayes: diferente do Multinomial NB (que estima probabilidades por classe só
com os dados daquela classe e tende a enviesar para classes majoritárias —
visível aqui no recall baixo de digestive/nervous, 0,18–0,39), o ComplementNB
estima os parâmetros de cada classe a partir de todas as *outras* classes, o
que contrapõe estruturalmente esse viés. **Trade-off aceito, não escondido:**
a taxa total de erro de nível do ComplementNB é na verdade um pouco maior que
a do LinearSVC/rico (19,8% vs. 17,4%) — ele não reduz os erros no total,
redistribui-os para a direção segura (`overtriage_rate` 0,145 vs. 0,085).
Para um sistema de triagem hospitalar, mais falsos alarmes são um custo
operacional aceitável em troca de significativamente menos erros perigosos —
uma troca deliberada de segurança por acurácia, documentada como exatamente
isso.

## Resultado da engenharia de features: só unigramas, confirmado

**Grid de 36 combinações** (`ngram_range` x `max_features` x `min_df` x
`sublinear_tf`) com ComplementNB fixo revelou a mesma tensão
F1-macro-vs-undertriage da seleção de modelo, um nível abaixo: a config de
melhor F1-macro (`ngram_range=(1,2)`, `max_features=20000`, `min_df=1`,
`sublinear_tf=False`, F1=0,782) tem `undertriage_rate=0,066`, pior que a
região só-unigrama (undertriage entre 0,052-0,058 em todo o sub-grid — uma
diferença consistente, não um par escolhido a dedo). Bigramas melhoram o
recall de "condições patológicas gerais" (de ~0,54-0,59 para ~0,60-0,64), que
é exatamente a relutância favorável à segurança que tornou o ComplementNB
atraente na seleção de modelo — bigramas corroem isso. Mantivemos só
unigramas (`ngram_range=(1,1)`) pelo mesmo raciocínio de segurança.

**Sweep complementar** (`max_df`, `stop_words`): `max_df` não teve efeito
mensurável em nenhum valor testado. `stop_words="english"` bateu
`stop_words=None` de forma consistente, tanto em F1-macro quanto em
undertriage_rate (0,052-0,053 vs. 0,055-0,055) — recall de digestive e
nervous caem sem remoção de stop words, e essa perda é o que impulsiona o
undertriage extra. Confirma com evidência real uma escolha que já vínhamos
fazendo por padrão.

**Config final de TF-IDF**: `ngram_range=(1,1)`, `max_features=10000`,
`min_df=2`, `sublinear_tf=False`, `stop_words="english"` — F1-macro 0,773,
undertriage_rate 0,0525. Essencialmente igual à config "conservadora"
original usada na seleção de modelo — o grid search principalmente
*confirmou* que o ponto de partida já estava perto da fronteira de segurança.

## Resultado do tuning de hiperparâmetros

**Achado: `fit_prior` não tem efeito nenhum no ComplementNB.** Todo par
`fit_prior=True`/`False` produziu métricas idênticas nas 12 combinações de
`alpha`/`norm` testadas. Não é um bug do nosso pipeline — a implementação do
`ComplementNB` do sklearn não incorpora o termo de prior de classe na regra
de decisão, conforme a formulação do artigo original. Confirmado
empiricamente, não assumido.

**Resultado: `alpha=0,5, norm=True`** — F1-macro 0,769, `undertriage_rate`
0,048. A mesma tensão F1-macro-vs-undertriage recorreu neste terceiro nível: o
ponto de melhor F1-macro (`alpha=0,1, norm=False`, F1=0,777) tem undertriage
pior que nossa baseline anterior (0,056 vs 0,053). `norm=True` é
consistentemente o que compra melhoria de undertriage em todo o grid, a um
custo real de F1-macro. Dos três pontos na fronteira de troca — máximo F1,
máxima segurança (`alpha=0,05, norm=True`, undertriage 0,038 mas F1 caindo
para 0,751) e este meio-termo — escolhemos o meio-termo: melhoria
significativa de undertriage sobre a baseline pré-tuning (~9% de redução
relativa) sem o custo de F1 mais acentuado do ponto de máxima segurança.

## Split treino/validação/teste e vazamento de dado

**Decisão:** um conjunto de teste (~15–20%) é separado uma única vez no
início, nunca tocado durante seleção de modelo, engenharia de features ou
tuning de hiperparâmetros. Todas as etapas de experimentação usam Stratified
K-Fold sobre o restante dos dados. O pipeline final é retreinado no pool
completo de treino+validação e avaliado uma única vez no conjunto de teste —
esse é o número reportado no README.

**Por que isso importa:** avaliar repetidamente contra o mesmo conjunto de
teste em cada etapa de decisão, mesmo sem treinar diretamente nele, cria
overfitting indireto a esse conjunto.

**Uso de `sklearn.pipeline.Pipeline`:** o vetorizador TF-IDF e o classificador
são agrupados em um único objeto `Pipeline` — não é só conveniência de
deploy, previne estruturalmente vazamento de dado: quando passado para
`cross_validate`/`GridSearchCV`, o vetorizador é reajustado só nos dados de
treino de cada fold. O mesmo objeto `Pipeline` treinado é salvo, carregado
pelo serviço FastAPI e exportado para ONNX — um único artefato, comportamento
idêntico em todo lugar.

## Otimização de latência (Etapa 4)

**Decisão:** a técnica aplicada depende de qual modelo vence a seleção —
modelo linear: exportação ONNX + quantização dinâmica INT8; Random Forest:
exportação ONNX + poda por complexidade de custo (`ccp_alpha`).

**Por que não forçar quantização de qualquer forma:** quantização reduz a
precisão numérica de matrizes de peso densas — não se aplica a ensembles de
árvore, que não têm essa estrutura. Forçar essa técnica em um Random Forest
seria um erro de categoria.

**Métrica reportada:** P50/P95/P99 sobre todo o pipeline de `/predict`
(pré-processamento + inferência + resposta), não só `model.predict()` — o
pré-processamento pode ser 40–60% da latência total em um sistema não
otimizado.

**O ComplementNB não é nenhum dos dois ramos, e foi tratado como o linear.**
O modelo vencedor é Naive Bayes; sua regra de decisão (produto escalar contra
uma matriz de peso densa por classe) é arquitetonicamente da mesma forma que
o `coef_` de um modelo linear, então foi exportado e quantizado da mesma
forma.

**Resultado** (benchmark de 500 requisições, um documento por vez):

| Variante | P50 | P95 | P99 | F1-macro | Tamanho |
|---|---|---|---|---|---|
| Baseline sklearn | 0,595ms | 0,814ms | 1,040ms | 0,7786 | 1.233KB |
| ONNX FP32 | **0,135ms** | **0,263ms** | **0,339ms** | 0,7786 (exato) | 413KB |
| ONNX INT8 (dinâmico) | 0,163ms | 0,281ms | 0,397ms | 0,7803 | 267KB |

**Escolhido: ONNX FP32 como artefato servido.** A exportação ONNX sozinha já
é o ganho dominante — 4,4x mais rápido no P50, 3x menor, e matematicamente
exato. A quantização INT8 é, contraintuitivamente, mais *lenta* que FP32
aqui — nessa escala, o modelo inteiro já roda em uma fração de milissegundo,
então o overhead de desquantização por operação supera a economia de
compute. O que o INT8 entrega de fato é uma redução real de tamanho (35%
menor que FP32) — uma escolha legítima se espaço em disco/memória for a
prioridade, só não é o ganho de latência normalmente buscado com essa
técnica. Reportado como um achado negativo genuíno sobre quantização, não
maquiado como vitória.

## Airflow em modo standalone

**Decisão:** `airflow standalone` (backend SQLite), não o docker-compose
oficial de produção (Postgres + Redis + webserver + scheduler + worker).

**Por quê:** o desafio pede uma DAG "simples" simulando train/retrain — o
compose de produção oficial é desproporcional para uma demo de 3 tarefas. A
DAG é disparada manualmente (não há um fluxo real de novos dados alimentando
este projeto), demonstrando capacidade de orquestração, não uma necessidade
real de retreino recorrente neste contexto.

**3 tarefas** (`ingest` → `train` → `save_model`), rodando em um container
Docker dedicado (Airflow não roda nativamente no Windows). Verificado de
fato, não só assumido: a DAG foi rodada duas vezes contra o dataset e o
remoto S3 reais, e as métricas logadas na segunda run (`f1_macro=0,7786`,
`undertriage=0,0498`) bateram exatamente com os números já reportados no
README — reprodutibilidade real entre ambientes.

## Teste no mundo real revelou uma limitação genuína de mudança de domínio

Depois do deploy, testes manuais com frases de triagem realistas (não no
estilo abstract) revelaram erros concretos de classificação, confirmando um
risco que o `model-card.md` já havia sinalizado teoricamente antes de
qualquer evidência existir. Quatro exemplos, e o que revelaram:

| Entrada | Previsto (antes da correção) | Causa raiz |
|---|---|---|
| "Unresponsive, no detectable pulse, non-breathing." | `normal` | As palavras *estão* no vocabulário, mas o modelo nunca aprendeu a associar esse registro à urgência — treinado só em abstracts formais estilo PubMed, nunca em linguagem clínica informal. |
| "stomachache" | `cardiovascular` / `urgente` | Genuinamente **zero** features TF-IDF — `stomachache` nunca aparece no vocabulário de treino. Sem evidência real, a previsão é guiada inteiramente pelo viés estrutural de classe do ComplementNB aplicado a um vetor vazio. |
| "...severe facial/airway swelling, blood pressure 80/50 mmHg following a bee sting." | `atenção` (anafilaxia, deveria ser `urgente`) | Bug de design no ajuste por palavra-chave: limitado a exatamente um nível independente de quantas palavras de escalada batessem. |
| "Asymptomatic patient requesting a routine prescription renewal for hypertension medication; mild ... rash ..." | `atenção` (deveria ser mais perto de `normal`) | Erro no nível de categoria somado ao mesmo limite de um nível mascarando os acertos de de-escalada que deveriam ter corrigido mais. |

**Duas dessas são bugs genuínos de código — corrigidos, não só documentados:**

1. **O ajuste por palavra-chave agora escala com o placar líquido em vez de
   saturar em um nível** (`app/urgency.py`): `tier = clamp(baseline + net, 0,
   2)` em vez de `tier = baseline ± 1`. Verificado: anafilaxia agora chega a
   `urgente` (era `atenção`); o caso de renovação de rotina agora chega a
   `normal` (era `atenção`).
2. **Uma trava de baixa confiança agora sinaliza entradas com sinal quase
   vazio** (`app/model.py::has_known_vocabulary`, exposta como
   `low_confidence: bool` em `/predict`). **Revisado após o primeiro
   deploy**: a primeira versão só elevava um piso de urgência (subia um
   palpite baixo, mantinha um alto) — para "stomachache", a categoria bruta
   já implicava `urgente`, então o piso não fazia nada e a UI mostrava
   `URGENT` ao lado de um aviso de baixa confiança, contraditório. Corrigido
   para uma **substituição fixa**: `low_confidence` agora sempre força
   `urgency = "attention"`, descartando o palpite bruto por completo — um
   sinal deliberado de "sinalizar para revisão humana". Também foi
   adicionado um campo `message` orientando a reenviar com mais detalhe.

**Uma é uma limitação genuína, não corrigível por nenhuma das duas mudanças —
documentada, não aceita silenciosamente.** O exemplo de parada cardíaca
(`normal`, `low_confidence: false`) falha porque seu vocabulário genuinamente
se sobrepõe aos dados de treino, só que não de um jeito que o modelo aprendeu
a associar com urgência — um descompasso real de registro entre o *Medical
Abstracts TC Corpus* (escrita formal, acadêmica) e como a urgência é de fato
comunicada em uma frase clínica curta de triagem. **Mais dados de treino do
mesmo tipo não resolveriam isso** — o corpus é a única fonte disponível, e
mais abstracts só melhorariam o desempenho em texto estilo abstract. Uma
correção real precisaria de exemplos nesse registro diferente (notas de
triagem reais ou realisticamente sintetizadas) — um aumento genuíno de
escopo. Ver `model-card.md` § Caveats.

## Fora de escopo (deliberadamente)

Kubernetes/HPA/KEDA, deploy Canary/Shadow, detecção de drift implementada
(PSI/KS — mencionada no README como trabalho futuro, não construída),
embeddings de palavra/transformer, busca automatizada conjunta (AutoML)
sobre modelo+features+hiperparâmetros, ROC-AUC/PR-AUC/MCC. Todos
considerados e conscientemente descartados pelos motivos acima — não por
falta de conhecimento do que existe, mas porque não se justificam dentro do
escopo e prazo deste desafio.
