# Relatórios de auditoria de questões de múltipla escolha

Dados do Trabalho de Conclusão de Curso **"Proposta e Avaliação de Agentes de LLM Independentes para a Detecção de Falhas de Redação em Questões de Múltipla Escolha"** (Nicole Alves Rodrigues, Bacharelado em Ciência da Computação, Instituto Federal Goiano – Campus Iporá, 2026).

O pipeline auditou 240 questões de múltipla escolha de três bases (80 de cada), em duas execuções por questão, segundo as diretrizes de redação de itens de Haladyna, Downing e Rodriguez (2002). Este repositório reúne os 480 relatórios gerados, as tabelas de saída do pipeline e o script das análises apresentadas no Capítulo 4 do TCC.

## Estrutura

```
OBI/  QCONCURSOS/  WAPLA/
├── relatorio_auditoria_unificado.csv   uma linha por questão e por execução (160 linhas)
└── laudos/                             um relatório .txt por questão e por execução (160 arquivos)
analise.py                              script que gera as tabelas e os gráficos das análises a partir dos CSVs
```

Os laudos seguem o padrão `relatorio_auditoria_q<N>_run0<E>_<data>_<hora>.txt`, em que `N` é o número da questão no conjunto e `E` é a execução (1 ou 2). Cada laudo tem quatro seções: nível cognitivo, qualidade do enunciado, resolução cega e distratores, e consolidação das violações, com o agente de origem de cada uma.

## Colunas de `relatorio_auditoria_unificado.csv`

| Coluna | Conteúdo |
|---|---|
| `id_questao` | Identificador da questão no conjunto (Q1 a Q80) |
| `execucao` | 1 ou 2 |
| `nivel_bloom` | Nível da Taxonomia de Bloom revisada, como devolvido pelo agente (ver observação abaixo) |
| `plausibilidade_media` | Média das notas de plausibilidade (1 a 5) dos distratores |
| `alerta_gabarito` | Texto do alerta quando a resolução cega diverge do gabarito oficial |
| `gabarito_oficial` | Alternativa correta segundo a fonte |
| `gabarito_agente` | Alternativa estimada pela resolução cega; `MULTIPLOS` quando o agente considerou mais de uma correta; **vazio quando o agente respondeu NENHUM** |
| `nenhuma_alternativa_correta` | 1 quando a resolução cega respondeu NENHUM |
| `confere_gabarito` | 1 quando a resolução cega coincidiu com o gabarito oficial |
| `diretrizes_violadas` | Lista das diretrizes violadas no relatório consolidado |
| `g1` … `g31` | 1 quando a diretriz foi apontada nessa execução, 0 caso contrário (G28 dividida em `g28a` a `g28f`) |

## Observações sobre os dados

- **Diretriz violada em uma questão:** nas comparações entre conjuntos, considera-se violada a diretriz apontada nas duas execuções (consenso). A união das execuções é usada como análise de sensibilidade.
- **G19 e G29:** são calculadas em código, não pelos agentes. A G19 é marcada quando a resolução cega não coincide com o gabarito oficial. A G29 é marcada quando a plausibilidade média é menor que 3.
- **Nível de Bloom:** em 12 das 480 saídas, o agente usou grafias fora dos seis níveis previstos ("ANÁLISE" e "APLICAR"). O CSV mantém a saída original. O `analise.py` padroniza os rótulos antes das análises (função `normaliza_bloom`), e o nível Understand é rotulado como "Compreender", como no esquema de saída do agente.
- **Notas de plausibilidade por posição:** os números sobre as notas 1 e a posição do distrator (Seção 4.4 do TCC) não são gerados pelo `analise.py`. Eles podem ser recontados nos laudos, na lista de alternativas da seção 3 de cada relatório.
- **Avaliação humana:** as respostas dos avaliadores humanos (Seção 4.5 do TCC) não estão incluídas. O Termo de Consentimento previa a divulgação apenas de resultados agregados, que constam nas Tabelas 7 e 25 do TCC.

## Como reproduzir as análises

```
pip install numpy pandas scipy matplotlib
python analise.py
```

Na pasta do repositório, o script lê os três CSVs e gera, na pasta `resultados/`, as tabelas e os gráficos das análises. Os intervalos de confiança por *bootstrap* usam semente fixa.

## Fontes das questões

- **OBI:** Olimpíada Brasileira de Informática, modalidade Iniciação. <https://olimpiada.ic.unicamp.br/>
- **QConcursos:** questões de concursos da área de Computação. <https://www.qconcursos.com/>
- **WAPLA:** itens gerados por inteligência artificial e rejeitados por especialistas, disponibilizados pela competição do IV Workshop de Aplicações Práticas de Learning Analytics em Instituições de Ensino no Brasil (WAPLA), evento satélite do CBIE 2026.

As questões pertencem às respectivas fontes. Os laudos reproduzem trechos delas apenas para documentar a auditoria.

## Como citar

RODRIGUES, Nicole Alves. *Proposta e Avaliação de Agentes de LLM Independentes para a Detecção de Falhas de Redação em Questões de Múltipla Escolha*. 2026. Trabalho de Conclusão de Curso (Bacharelado em Ciência da Computação) – Instituto Federal Goiano, Campus Iporá, Iporá, 2026.
