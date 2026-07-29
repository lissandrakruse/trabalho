# Script separado do solver

O solver tambem pode ser executado sem abrir a interface. Ele nao sobe Flask,
nao usa o layout e nao depende de `app.py`; apenas le o CSV do dataset, resolve
a consulta por programacao linear e devolve o resultado.

## Execucao simples

Para executar uma consulta padrao e salvar o JSON automaticamente:

```bash
python solver.py
```

Esse comando calcula:

```text
P(label=rice | ph=acido, rainfall=alto)
```

e salva a saida completa em:

```text
reports/generated/solver_resultado_padrao.json
```

## Execucao personalizada

```bash
python scripts/solve_query.py --target label=rice --condition ph=acido --condition rainfall=alto
```

Para salvar a saida em JSON:

```bash
python scripts/solve_query.py --target label=rice --condition ph=acido --condition rainfall=alto --output reports/generated/solver_exemplo_rice.json
```

Para ver os atributos e valores validos do dataset:

```bash
python scripts/solve_query.py --show-domains
```

Para usar outro CSV:

```bash
python scripts/solve_query.py --dataset data/Crop_recommendation.csv --target label=rice --condition ph=acido
```

Para escolher manualmente um dos tres metodos HiGHS:

```bash
python scripts/solve_query.py --target label=rice --condition ph=acido --condition rainfall=alto --solver-method highs
python scripts/solve_query.py --target label=rice --condition ph=acido --condition rainfall=alto --solver-method highs-ds
python scripts/solve_query.py --target label=rice --condition ph=acido --condition rainfall=alto --solver-method highs-ipm
```

O script retorna:

- probabilidades `P(A)`, `P(B)` e `P(A e B)`
- suporte
- confianca/precisao
- lift
- intervalo linear minimo e maximo
- numero de variaveis e restricoes
- tempo de processamento
- programa linear
- conclusao automatica

## Por que usar intervalos

As probabilidades foram representadas por intervalos para reduzir efeitos de
arredondamento e permitir modelagem consistente das restricoes lineares.

## Topicos usados para construir o codigo

O codigo foi organizado a partir de uma cadeia de raciocinio probabilistico:

1. Carregamento do dataset

O sistema le o arquivo `data/Crop_recommendation.csv`, identifica os atributos
disponiveis e transforma os valores numericos em faixas categoricas. Isso cria
uma base adequada para consultas do tipo:

```text
P(label=rice | ph=acido, rainfall=alto)
```

2. Representacao dos mundos possiveis

Cada combinacao categorica observada no dataset representa um mundo possivel
`w`. Para cada mundo, o programa cria uma variavel:

```text
x_w >= 0
```

Essa variavel representa a massa de probabilidade atribuida ao mundo `w`.
Todas as variaveis juntas formam uma distribuicao de probabilidade.

3. Normalizacao probabilistica

Para garantir que as variaveis formem uma distribuicao valida, o programa
inclui a restricao:

```text
soma(x_w) = 1
```

Assim, a soma de todas as probabilidades dos mundos possiveis e igual a 1.

4. Probabilidades extraidas da base

O sistema calcula frequencias empiricas do dataset:

```text
P(A)       = count(A) / N
P(B)       = count(B) / N
P(A e B)   = count(A e B) / N
P(A | B)   = P(A e B) / P(B)
```

Essas probabilidades sao usadas tanto para mostrar as metricas na interface
quanto para montar as restricoes do programa linear.

5. Regras de associacao

A consulta escolhida pelo usuario e interpretada como uma regra:

```text
B -> A
```

No projeto, isso significa:

```text
se condicoes do solo ocorrem, entao determinada cultura e provavel
```

O sistema calcula:

```text
suporte     = P(A e B)
confianca   = P(A | B)
lift        = P(A | B) / P(A)
```

Uma regra com suporte conjunto zero nao e considerada uma regra aprendida. Ela
continua aparecendo como consulta do usuario, mas nao entra como restricao ativa
de confianca no programa linear, porque nao ha evidencia empirica para sustenta-la.

6. Restricoes lineares

As probabilidades extraidas viram restricoes do tipo:

```text
limite_inferior <= soma(x_w onde evento ocorre) <= limite_superior
```

O modelo inclui:

- restricoes marginais para valores dos atributos;
- restricoes conjuntas por pares de valores;
- restricoes especificas da consulta selecionada;
- restricao da regra de associacao selecionada quando ela possui suporte e
  confianca positivos.

7. Consulta condicional

O objetivo final e responder:

```text
P(A | B) = P(A e B) / P(B)
```

Como essa expressao e uma razao, ela nao pode ser enviada diretamente como uma
funcao objetivo linear simples. Por isso, o projeto usa a transformacao de
Charnes-Cooper.

8. Transformacao de Charnes-Cooper

A transformacao converte a razao condicional em um problema linear equivalente.
O codigo troca as variaveis originais `x_w` por variaveis transformadas `y_w`
e fixa a massa de `B` como 1:

```text
x_w = y_w / t
soma(y_w onde B) = 1
```

Depois disso, o solver consegue minimizar e maximizar:

```text
soma(y_w onde A e B)
```

O resultado e um intervalo:

```text
limite inferior <= P(A | B) <= limite superior
```

9. Solver usado

O solver padrao usado e o HiGHS, chamado pela funcao:

```text
scipy.optimize.linprog(method="highs")
```

Ele resolve dois problemas lineares para cada consulta:

- minimizacao de `P(A | B)`;
- maximizacao de `P(A | B)`.

No botao de comparacao, o projeto executa tres metodos do HiGHS:

```text
scipy.optimize.linprog(method="highs")
scipy.optimize.linprog(method="highs-ds")
scipy.optimize.linprog(method="highs-ipm")
```

## Solvers considerados e comparacao

Nesta versao, a comparacao executavel do projeto e:

```text
Projeto principal Flask/app.py  x  Script separado scripts/solve_query.py
```

O botao envia para o script separado exatamente os mesmos parametros que o
usuario escolheu na interface. O script entao resolve a mesma formulacao linear
com tres metodos:

- SciPy HiGHS;
- HiGHS Dual Simplex;
- HiGHS Interior Point.

A comparacao verifica se os resultados batem em:

- `P(A)`;
- `P(B)`;
- `P(A e B)`;
- confianca;
- lift;
- limite inferior do programa linear;
- limite superior do programa linear;
- quantidade de variaveis;
- quantidade de restricoes;
- tempo de execucao.

Tabela dos solvers citados no projeto:

| Solver | Status no projeto | Comparacao |
| --- | --- | --- |
| SciPy HiGHS | Executado no projeto principal e no script separado | Comparacao numerica ativa entre API Flask e `scripts/solve_query.py` |
| HiGHS Dual Simplex | Executado no script separado | Comparacao de metricas com a mesma consulta da interface |
| HiGHS Interior Point | Executado no script separado | Comparacao de metricas com a mesma consulta da interface |
| Gurobi | Comparacao documental | Nao executado no Render por depender de instalacao/licenca |
| lp_solve | Comparacao documental | Nao executado no Render nesta versao |
| cuPDLP-C | Comparacao documental | Nao executado no Render nesta versao |

Assim, quando o relatorio fala em "solver separado", ele esta falando do script
Python independente que reproduz a resolucao com os tres metodos HiGHS. Quando
cita Gurobi, lp_solve ou cuPDLP-C, o projeto esta registrando solvers
alternativos que podem ser usados em uma etapa futura de benchmark.

10. Comparacao independente

O projeto tambem possui um script separado do solver. A interface chama esse
script com os mesmos dados informados pelo usuario e compara o resultado com o
calculo principal do backend. Isso serve como validacao independente da
implementacao.

## Fundamentacao academica do modelo

O projeto foi construido sobre a ideia de que conhecimento probabilistico pode
ser representado por restricoes lineares. Em vez de apenas calcular uma
probabilidade pontual a partir da base, o sistema transforma as evidencias do
dataset em um conjunto de desigualdades. O solver entao procura os menores e
maiores valores de `P(A | B)` que ainda respeitam todas essas restricoes.

Essa abordagem aproxima o codigo dos modelos de raciocinio sob incerteza:

- a base de dados fornece as evidencias empiricas;
- as evidencias viram probabilidades marginais e conjuntas;
- as probabilidades sao convertidas em restricoes intervalares;
- a consulta condicional e formulada como problema de otimizacao;
- o solver calcula os limites compativeis com o conhecimento disponivel.

As probabilidades intervalares entram porque dados reais podem sofrer
arredondamento, discretizacao e pequenas variacoes numericas. Ao usar limites
inferior e superior, o sistema evita tratar uma frequencia observada como um
valor absolutamente rigido.

As regras de associacao entram como interpretacao estatistica da consulta
`B -> A`. Suporte mede a frequencia conjunta, confianca mede a probabilidade
condicional e lift compara a regra com a ocorrencia geral do consequente. Quando
uma regra tem suporte zero, ela nao e inserida como restricao de confianca,
porque uma restricao desse tipo representaria conhecimento que o dataset nao
aprendeu.

Por fim, a transformacao de Charnes-Cooper e usada porque `P(A | B)` e uma
fracao. Como `linprog` resolve programacao linear, a razao precisa ser
reescrita em uma forma linear antes da chamada ao solver.

## Referencias bibliograficas

- Nilsson, N. J. Probabilistic Logic. Artificial Intelligence, 1986.
- Charnes, A.; Cooper, W. W. Programming with linear fractional functionals. Naval Research Logistics Quarterly, 1962.
- Tessem, B. Interval probability propagation. International Journal of Approximate Reasoning, 1992.
- Artigos e materiais disponibilizados pelo professor no Classroom.
