# MedSys — Classificador de Triagem de Laudos Médicos

Tech Challenge de pós-graduação (POS Tech / MLET, Fase 3): triagem automática
de laudos médicos em níveis de urgência — **normal / atenção / urgente** —
servida como uma API REST, com pipeline de CI/CD, stack de monitoramento e
modelo otimizado para baixa latência.

## Situação, Tarefa, Ação e Resultado

**Situação.** Um hospital recebe laudos médicos em texto livre e precisa
priorizá-los por urgência antes que um humano os leia. Um atraso na
identificação de um caso urgente tem custo clínico real; o desafio pede um
classificador de NLP leve, em produção, com pipeline de CI/CD, monitoramento
e otimização de latência — não apenas um notebook de modelagem.

**Tarefa.** Construir e colocar em produção um serviço de classificação de
urgência a partir de texto médico, cobrindo o ciclo completo: dataset →
experimentação de modelo (rastreada em MLflow) → API servida em container →
otimização de latência com números reais de antes/depois → pipeline de
retrain (Airflow) → monitoramento (Prometheus + Grafana) → CI/CD →
deploy real na AWS.

**Ação.**

- Curadoria do dataset (Medical Abstracts TC Corpus): identificação e remoção
  de vazamento train/test e de rótulos conflitantes, resultando em 8.298
  documentos limpos ([§ Dataset](#dataset)).
- 4 experimentos MLflow em etapas — seleção de modelo, engenharia de
  features, tuning de hiperparâmetros, otimização de latência — cada um
  revisado antes de alimentar o próximo (grades completos e raciocínio em
  `docs/technical-decisions.md`).
- Escolha deliberada do modelo por segurança clínica, não apenas pela métrica
  de classificação bruta ([§ Resultados do modelo](#resultados-do-modelo)).
- Exportação para ONNX com benchmark real de latência, comparando baseline,
  FP32 e quantização INT8 ([§ Otimização de latência](#otimização-de-latência)).
- API em FastAPI, containerizada, instrumentada com `prometheus-client`.
- DAG de Airflow simulando o ciclo de retrain (ingest → train → save_model).
- Stack de monitoramento (API + Prometheus + Grafana) via Docker Compose, com
  dashboard provisionado automaticamente.
- CI/CD com 2 workflows independentes no GitHub Actions (lint → test → build
  → smoke test → deploy → rollback automático).
- Deploy real na AWS, com decisão arquitetural documentada (tempo real vs.
  batch) e um pivô de infraestrutura forçado por uma restrição real de custo
  ([§ Arquitetura em nuvem](#arquitetura-em-nuvem-tempo-real-vs-batch)).

**Resultado.** Pipeline final `TF-IDF + ComplementNB` com F1-macro de 0.7786
e taxa de subtriagem (o erro clinicamente perigoso) de apenas 4.98%; a versão
ONNX FP32 servida em produção é 4.4x mais rápida que o baseline sklearn, sem
perda de qualidade. A API está de fato no ar na AWS, com deploy automático a
cada push e monitoramento ao vivo mostrando tráfego real de produção. Uma
limitação real e não trivial foi encontrada e documentada durante o teste
pós-deploy (descasamento de registro entre laudos formais de treino e texto
de triagem informal) — ver [§ Resultados do modelo](#resultados-do-modelo).

**Deploy ao vivo**: http://medsys.us-east-1.elasticbeanstalk.com/ — API de
inferência em tempo real (`/predict`, `/health`, `/metrics`) rodando em AWS
Elastic Beanstalk, deployada automaticamente pela CI a cada push em `main`
(desligada fora da janela de avaliação/demo).

```bash
curl http://medsys.us-east-1.elasticbeanstalk.com/health
curl -X POST http://medsys.us-east-1.elasticbeanstalk.com/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient presents with acute chest pain and shortness of breath, elevated troponin levels observed"}'
```

**Demo frontend** (extra, não avaliado, `frontend/streamlit_app.py`): UI em
Streamlit sobre `/predict`, hospedada separadamente no Streamlit Community
Cloud (gratuito) para não duplicar horas de EC2 do Free Tier com um
componente que é só um cliente HTTP.

## Sumário

- [Resultados do modelo](#resultados-do-modelo)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Como reproduzir tudo localmente](#como-reproduzir-tudo-localmente)
- [Dataset](#dataset)
- [Otimização de latência](#otimização-de-latência)
- [Arquitetura em nuvem: tempo real vs. batch](#arquitetura-em-nuvem-tempo-real-vs-batch)
- [CI/CD](#cicd)

## Resultados do modelo

O pipeline final é **TF-IDF (unigramas, 10.000 features) + Complement Naive
Bayes** (`alpha=0.5, norm=True`). Ele não foi o vencedor bruto na métrica
principal (F1-macro) entre os 4 modelos testados — esse título ficou com
LinearSVC — mas foi o escolhido por ter uma taxa de subtriagem
significativamente menor, um trade-off explícito de segurança clínica em
troca de acurácia bruta. Esse padrão (o vencedor de acurácia não é o vencedor
de segurança) se repetiu em todas as etapas de tuning, não só na escolha do
modelo. O raciocínio completo, com os grids de busca de cada etapa, está em
`docs/technical-decisions.md`.

**O que cada métrica mede:**

- **F1-macro** — média do F1 (harmônica de precisão e recall) calculada por
  categoria de doença e depois tirada a média simples entre as 5 categorias,
  sem ponderar pelo tamanho de cada uma. É a métrica principal de seleção de
  modelo: um modelo não pode vencer só por ir bem na categoria majoritária.
- **Acurácia** — proporção de acertos sobre o total, contexto geral, não é
  usada para decidir entre modelos (é dominada pela categoria majoritária).
- **Taxa de subtriagem (undertriage)** — proporção de casos em que o modelo
  previu uma urgência **abaixo** da real (ex.: um caso urgente classificado
  como normal). É o erro clinicamente perigoso, porque atrasa o atendimento
  de um caso grave.
- **Taxa de sobretriagem (overtriage)** — proporção de casos em que o modelo
  previu uma urgência **acima** da real. Gera custo operacional (atenção
  desnecessária), mas não coloca o paciente em risco.

**Resultado no conjunto de teste** (1.245 documentos, avaliação única):

| Métrica | Valor |
|---|---|
| F1-macro | 0.7786 |
| Acurácia | 0.7880 |
| Taxa de subtriagem (erros perigosos) | 0.0498 |
| Taxa de sobretriagem (falsos alarmes) | 0.1446 |

**Limitação conhecida, confirmada em teste real, não apenas suposta**: o
modelo é treinado sobre *abstracts* médicos formais (registro acadêmico, em
terceira pessoa), não sobre a fala real de triagem. Testar entradas curtas e
informais pós-deploy revelou erros genuínos — ex.: "Unresponsive, no
detectable pulse, non-breathing" classificado como `normal`. Duas correções
já foram aplicadas (a camada de ajuste por palavra-chave agora escala com a
força do sinal em vez de saturar em um nível; `/predict` agora sinaliza
`low_confidence: true` quando não há sobreposição real de vocabulário), mas
o descasamento de registro não é resolvido só com mais dados do mesmo tipo —
exigiria texto real de triagem, um aumento genuíno de escopo. Detalhes em
`docs/technical-decisions.md` e `docs/model-card.md` § Caveats.

## Estrutura do projeto

```
├── app/                    # API FastAPI (inferência)
│   ├── main.py              # endpoints /predict, /health, /metrics
│   ├── model.py              # carregamento e execução do modelo ONNX
│   └── urgency.py            # mapeamento categoria -> nível de urgência
├── training/               # experimentação e treino (rastreados em MLflow)
│   ├── data.py                # carga e split do dataset
│   ├── model_selection.py     # experimento 1: escolha do modelo
│   ├── feature_engineering.py # experimento 2: TF-IDF, n-gramas, etc.
│   ├── hyperparameter_tuning.py # experimento 3: tuning do modelo escolhido
│   ├── train_final.py         # treino final + avaliação no held-out test
│   ├── airflow_tasks.py       # tarefas usadas pela DAG do Airflow
│   └── evaluation.py          # métricas de classificação e triagem
├── optimization/           # experimento 4: exportação ONNX + benchmark
│   └── export_and_benchmark.py
├── dags/                   # DAG de retrain do Airflow
│   └── train_pipeline_dag.py
├── airflow/                # ambiente Docker isolado para rodar a DAG
├── frontend/               # demo Streamlit (não avaliado)
│   └── streamlit_app.py
├── monitoring/             # config do Prometheus + provisionamento do Grafana
├── models/                 # artefatos de modelo (versionados via DVC)
├── data/                   # CSVs brutos (versionados via DVC)
├── tests/                  # testes automatizados (API, urgência, smoke test)
├── docs/
│   ├── architecture.md         # o quê: dataset, pipeline, AWS, CI/CD, Airflow
│   ├── technical-decisions.md  # o porquê: raciocínio de cada decisão não óbvia
│   └── model-card.md           # documentação do modelo (formato Mitchell et al.)
├── Dockerfile              # imagem da API servida em produção
└── docker-compose.yml      # stack local: api + prometheus + grafana
```

## Como reproduzir tudo localmente

**Pré-requisitos**: Python 3.11+, [`uv`](https://docs.astral.sh/uv/), Docker
Desktop (com `buildx`), AWS CLI configurado (só necessário para puxar
artefatos do DVC/S3).

**1. Clonar e instalar dependências.** As dependências são divididas em
grupos `uv` para que a imagem Docker servida não instale nada além do
necessário:

```bash
git clone https://github.com/JoaoFurlan/tech-challenge-03
cd tech-challenge-03
uv sync --group dev          # base + dev (fastapi, pytest, ruff, ...)
```

| Necessidade | Comando |
|---|---|
| Rodar/editar a API, rodar testes, lint | `uv sync --group dev` |
| Treinar, reexportar para ONNX, rodar experimentos MLflow | `uv sync --group training` |
| Rodar o frontend Streamlit | `uv sync --extra frontend` |

**2. Puxar dados e artefatos de modelo (DVC/S3)** — não estão no git:

```bash
uv run --group training dvc pull
```

**3. Rodar os testes e o lint** (o mesmo que a CI roda):

```bash
uv run pytest -q
uv run ruff check .
```

**4. Treinar o modelo do zero** (opcional — os artefatos já vêm do DVC no
passo 2). Cada etapa é um script simples, na ordem em que foram desenhadas:

```bash
uv run python -m training.model_selection        # experimento: seleção de modelo
uv run python -m training.feature_engineering     # experimento: engenharia de features
uv run python -m training.hyperparameter_tuning   # experimento: tuning
uv run python -m training.train_final             # gera models/pipeline.joblib
uv run python -m optimization.export_and_benchmark # gera models/pipeline_fp32.onnx
```

Inspecionar qualquer run (métricas, parâmetros, matrizes de confusão) na UI
do MLflow, apontada para o SQLite local que todos os scripts usam:

```bash
uv run --group training mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**5. Rodar a API sozinha** (loop mais rápido para editar `app/`):

```bash
uv run uvicorn app.main:app --reload --port 8000
```

`curl http://localhost:8000/health`, ou abra
http://localhost:8000/docs para a documentação interativa (Swagger).

**6. Rodar a stack completa de monitoramento** (API + Prometheus + Grafana):

```bash
docker compose up --build
```

| Serviço | URL |
|---|---|
| API | http://localhost:8000 (`/predict`, `/health`, `/metrics`) |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (acesso anônimo de visualização, sem login) |

Gere tráfego para ver o dashboard populando — ex. `curl -X POST
http://localhost:8000/predict -H "Content-Type: application/json" -d
'{"text": "..."}'` — ou use o Streamlit do passo seguinte.

**7. Rodar o frontend de demo (Streamlit)**, apontado por padrão para
`http://localhost:8000`:

```bash
uv run --extra frontend streamlit run frontend/streamlit_app.py
```

**8. Rodar a DAG de retrain do Airflow.** Roda em Docker, não diretamente no
host — o Airflow não roda nativamente no Windows. É um ambiente separado do
`docker-compose.yml` da raiz (orquestra retreino, não serve/observa a API):

```bash
cd airflow
docker compose up --build -d
```

O container imprime a senha de admin gerada no primeiro boot —
`docker compose logs | Select-String "Password for user"` (PowerShell) ou
`docker compose logs | grep "Password for user"` (bash/zsh). Abra
http://localhost:8081 e entre como `admin` com essa senha, depois dispare a
DAG (`train_pipeline_dag`) pela UI ou via CLI:

```bash
docker compose exec airflow airflow dags unpause train_pipeline_dag
docker compose exec airflow airflow dags trigger train_pipeline_dag
```

O container monta o repositório em modo leitura-escrita, então as tarefas
`ingest`/`train`/`save_model` escrevem de volta em `data/`/`models/` no seu
disco, do mesmo jeito que rodar os scripts de treino diretamente. Ao terminar:

```bash
docker compose down
```

Detalhes de por que o Airflow standalone é usado aqui (e não o
docker-compose de produção) estão em `docs/technical-decisions.md`.

## Dataset

[Medical Abstracts TC Corpus](https://www.kaggle.com/datasets/chaitanyakck/medical-text)
(`sebischair/Medical-Abstracts-TC-Corpus`) — abstracts médicos rotulados em 5
categorias de doença, usados como ground truth do classificador. A urgência é
derivada da categoria prevista por uma camada de mapeamento determinística,
não aprendida diretamente (ver `docs/technical-decisions.md`).

O Kaggle já entrega o corpus dividido em train/test, totalizando 14.438
linhas. Não usamos esse split como veio — combinar os dois arquivos revelou:

- **988 abstracts vazando** entre o split oficial de train/test (o mesmo
  documento nos dois arquivos).
- **2.929 abstracts com rótulos de categoria conflitantes** — o mesmo texto
  atribuído a mais de uma categoria. É um artefato real do corpus original
  (alguns documentos eram multi-rotulados, ex. tanto `cardiovascular
  diseases` quanto o bucket genérico `general pathological conditions`),
  explodido em linhas de rótulo único nesta versão do Kaggle.

Os dois problemas são resolvidos combinando train+test e descartando por
completo todo documento ambíguo, em vez de escolher arbitrariamente um
rótulo por documento — raciocínio completo em `docs/technical-decisions.md`.
Isso deixa:

**8.298 documentos limpos e sem ambiguidade de rótulo**, re-divididos por
nós (estratificado, ~15% como conjunto de teste, `random_state=42` fixo):

| Categoria | Quantidade | Proporção |
|---|---|---|
| General pathological conditions | 2.394 | 28,9% |
| Neoplasms | 2.195 | 26,5% |
| Cardiovascular diseases | 1.961 | 23,6% |
| Nervous system diseases | 1.049 | 12,6% |
| Digestive system diseases | 699 | 8,4% |

Ainda folgadamente acima do mínimo de 2.000 amostras exigido pelo desafio,
com o desbalanceamento de classes (~3,4x) praticamente inalterado em relação
ao corpus bruto (~3,2x).

## Otimização de latência

Exportado para ONNX (`skl2onnx`) e comparado com quantização dinâmica INT8,
medindo a latência real de `/predict` (vetorização TF-IDF + inferência) por
documento único, 500 requisições:

| Variante | P50 | P95 | P99 | F1-macro | Tamanho |
|---|---|---|---|---|---|
| Baseline sklearn | 0.595ms | 0.814ms | 1.040ms | 0.7786 | 1.233KB |
| **ONNX FP32 (servido)** | **0.135ms** | **0.263ms** | **0.339ms** | 0.7786 | 413KB |
| ONNX INT8 | 0.163ms | 0.281ms | 0.397ms | 0.7803 | 267KB |

**Servido em produção: ONNX FP32** — 4,4x mais rápido que o baseline sklearn
no P50, 3x menor, matematicamente exato (F1-macro idêntico, não é uma
aproximação). A quantização INT8 foi na verdade mais *lenta* que FP32 aqui —
nessa escala o modelo já roda em uma fração de milissegundo, então o overhead
de desquantização por chamada supera o ganho de compute. Ainda entrega uma
economia real de tamanho (35% menor que FP32) se espaço em disco for mais
relevante que latência. Raciocínio completo em `docs/technical-decisions.md`.

## Arquitetura em nuvem: tempo real vs. batch

Existem três padrões de deploy possíveis para um modelo como este — a
escolha certa depende do que a carga de trabalho realmente exige:

| | Batch | Tempo real | Serverless |
|---|---|---|---|
| Latência | Alta, tolerada | Baixa, determinística | Variável (cold start) |
| Formato de custo | Concentrado em execuções agendadas | Constante (infra sempre ligada) | Alinhado à demanda |
| Se encaixa aqui? | Não — não há carga recorrente em lote a processar | **Sim** | Não — cold start inaceitável |

**Tempo real, não batch**: um laudo chega e o hospital precisa de uma
resposta normal/atenção/urgente imediatamente, não na próxima execução
agendada — o objetivo da triagem é pegar o caso urgente *agora*. Batch
também não tem o que processar aqui — não existe um fluxo recorrente de
laudos acumulados para processamento offline, só relatórios individuais
chegando um a um. Serverless (Lambda) é descartado pelo mesmo motivo que
batch é descartado, mas ao contrário: cold start por invocação é
incompatível com latência baixa e consistente em uma ferramenta clínica,
mesmo se encaixando no formato "uma requisição por vez". A desvantagem do
padrão tempo real — custo constante em vez de proporcional ao uso, já que o
modelo precisa ficar sempre carregado — é um trade-off aceitável e
deliberado quando a latência tem consequência clínica direta.

Dentro das opções de tempo real da AWS:

- **App Runner** (alvo documentado) — container sempre ativo, sem instância
  para provisionar ou corrigir, aponta para uma tag de imagem no ECR e roda.
  O encaixe mais direto para "inferência em tempo real com operação mínima".
- **EC2 puro** — mesmo comportamento sempre ativo, mas a carga operacional
  (provisionamento, patches, health checks) que o App Runner existe para
  remover volta para nós.
- **Lambda** — descartado acima (cold start).
- **AWS Batch** — formato errado; feito para jobs agendados/em lote, não
  para um endpoint de inferência permanente.
- **SageMaker** — plataforma de ML gerenciada completa (model registry,
  endpoints gerenciados, governança) — capacidade real, mas configuração
  desproporcional para um único classificador leve.

**Deployado de fato em Elastic Beanstalk**, plataforma Docker de instância
única, não App Runner — uma restrição real descoberta tarde, não uma
mudança de design: App Runner não está no Free Tier da AWS, descoberto só
na hora do deploy real. Beanstalk em uma instância EC2 única do free tier é
o equivalente mais próximo elegível ao Free Tier da intenção do App
Runner — mesma propriedade de sempre-ativo sem cold start, e o Beanstalk
ainda absorve a maior parte da carga operacional manual de EC2
(provisionamento, health checks, deploy) que o App Runner cuidaria
diretamente. Especificamente o tipo de ambiente de instância única, não o
com balanceamento de carga/auto-scaling — esse nível provisiona um Elastic
Load Balancer, cobrado separadamente e fora do Free Tier. O raciocínio
acima segue sendo a decisão *documentada*; o serviço deployado difere dela
só por essa razão orçamentária, registrada explicitamente em vez de trocada
em silêncio. História completa do pivô em `docs/technical-decisions.md`.

Uma versão pronta para produção dessa mesma arquitetura de tempo real
também precisaria de rate limiting na API (controle de abuso e de custo — o
formato de custo constante do tempo real torna tráfego não limitado um
vazamento de custo direto, não só uma questão de segurança) e criptografia
em trânsito/repouso para o texto dos laudos, já que é dado de paciente.

### Fluxo de dados e retrain

Como um dataset de laudos vira um modelo deployado ao vivo hoje, no Airflow
local deste projeto (**modo standalone, disparado manualmente** — demonstra
a orquestração de retrain, não está conectado a deploy automático):

```
[CSVs do Kaggle] (uma vez)
      |
      v
data/ (local)  --dvc add + dvc push-->  S3 (remoto DVC)
                                              |
   ==== fronteira da DAG do Airflow (disparo manual) ====
   |                                          |
   |  tarefa ingest: dvc pull  <--------------+
   |       |
   |       v
   |  load_raw() / split_test()  -->  pool (85%) + teste (15%, held out)
   |       |
   |       v
   |  tarefa train: treina Pipeline(TF-IDF + ComplementNB) no pool
   |       |
   |       v
   |  avalia no teste  -->  loga métricas/params no MLflow
   |       |
   |       v
   |  tarefa save_model: escreve pipeline.joblib + vocabulary.json (disco local)
   |
   ==== DAG termina aqui ====
                |
                v  (passo manual, fora da DAG)
     optimization/export_and_benchmark.py
                |
                v
     pipeline_fp32.onnx + números de benchmark de latência
                |
     dvc add + dvc push (manual) --> S3
                |
     commit dos ponteiros .dvc + git push (manual) --> GitHub main
                |
                v  (automático a partir daqui -- CI/CD já conectado)
     ci-cd.yml: lint -> test -> dvc pull (modelo) -> docker build
                |
                v
     push para o ECR --> atualiza Dockerrun.aws.json --> deploy no Elastic Beanstalk
                |
                v
     API ao vivo (/predict, /metrics) --coletada por--> Prometheus --> Grafana
```

A saída da própria DAG (`pipeline.joblib`) não é o que a API ao vivo serve
(`pipeline_fp32.onnx`) — conectar as duas hoje é uma corrente manual
(export, `dvc push`, `git push`), não automática. Em um cenário de produção
real, hospedado (Airflow no MWAA, por exemplo), essa corrente se fecharia
sozinha, com um gate de qualidade comparando o novo modelo contra o de
produção antes de promovê-lo — raciocínio completo de custo/escopo para o
que não foi construído de fato aqui em `docs/technical-decisions.md`.

### Monitoramento na AWS

Além da API, a stack de monitoramento (API + Prometheus + Grafana) roda em
uma instância EC2 simples, provisionada especificamente para a janela de
avaliação/demo e depois desligada — diferente da API (sempre ativa,
deployada automaticamente pela CI a cada push). Redeploy automático a cada
push que toque `docker-compose.yml`, `monitoring/**`, `app/**`, `Dockerfile`
ou `models/*.dvc`, via **AWS Systems Manager Run Command** (sem SSH, sem
chaves — a mesma role OIDC já usada pela API, com uma policy adicional de
`ssm:SendCommand` restrita a essa instância).

O que uma versão de produção real teria a mais: Amazon Managed Service for
Prometheus + Amazon Managed Grafana (alta disponibilidade, sem ponto único
de falha) em vez de uma instância EC2 rodando `docker-compose`; alerting
conectado a on-call de verdade (PagerDuty/Opsgenie); acesso autenticado com
RBAC em vez de visualização anônima; monitoramento de qualidade do modelo
(detecção de drift) além das métricas operacionais de request/latência/erro
já cobertas aqui; e infraestrutura permanente em vez do padrão de
subir/derrubar usado nesta demo.

## CI/CD

Dois workflows independentes no GitHub Actions — independentes porque
observam paths de disparo diferentes e miram recursos diferentes, rodando
em paralelo quando um push toca os dois, não em sequência:

| Workflow | Dispara em | Faz |
|---|---|---|
| `.github/workflows/ci-cd.yml` ("CI/CD") | Todo push em `main` (e PRs, exceto o job de deploy) | lint → test → build → **smoke test** → push para ECR → deploy no Beanstalk → health check (**rollback automático** em caso de falha) |
| `.github/workflows/deploy-monitoring.yml` ("Deploy monitoring stack") | Pushes tocando `docker-compose.yml`, `monitoring/**`, `app/**`, `Dockerfile` ou `models/*.dvc` | Redeploya a stack de monitoramento na EC2 via SSM (ver [§ Monitoramento na AWS](#monitoramento-na-aws)) |

**Verificar status**: aba **Actions** do GitHub lista cada execução, mais
recente primeiro, por nome do workflow. Uma falha em `ci-cd.yml` significa
ou um problema real de código (lint/test) ou um deploy rejeitado — o
rollback automático do passo de deploy garante que a produção continue
saudável mesmo com o job em vermelho.

**O que é validado antes de chegar em produção**:

- `lint`/`test` precisam passar antes de `build-and-push` sequer começar.
- O **smoke test** roda a imagem construída de verdade e faz uma chamada
  real a `/predict`, checando se a resposta tem o formato esperado (schema
  correto, valores válidos de categoria/urgência) — antes da imagem ser
  publicada no ECR. Isso pega "o container está fundamentalmente quebrado"
  (arquivo de modelo faltando, código quebrando); **não** pega "as previsões
  do modelo pioraram" — isso exigiria ground truth, um problema maior,
  considerado e deliberadamente não implementado (ver
  `docs/technical-decisions.md`).
- Depois do deploy, uma falha no health check dispara um **rollback
  automático** para a versão anteriormente no ar, então um deploy ruim se
  autocorrige dentro da mesma execução, sem esperar alguém perceber.

**Todo push retreina o modelo?** Não — `ci-cd.yml` só faz `dvc pull` do
artefato de modelo que os ponteiros `.dvc` já commitados apontam; ele nunca
chama o código de treino. O modelo em produção só muda quando alguém
atualiza esses ponteiros deliberadamente (treinar → exportar para ONNX →
`dvc push` → commitar os novos `.dvc` → push) — ver
[§ Fluxo de dados e retrain](#fluxo-de-dados-e-retrain) para a corrente
completa e por que hoje é manual, não automática.
