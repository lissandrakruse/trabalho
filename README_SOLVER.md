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

- probabilidades empiricas `P(A)`, `P(B)` e `P(A e B)` usadas no PL
- suporte da regra liberada, quando a extracao gerar a regra consultada
- confianca/precisao da regra liberada, quando existir
- lift da regra liberada, quando existir
- intervalo linear minimo e maximo
- numero de variaveis e restricoes
- tempo de processamento
- programa linear
- conclusao automatica

## Por que usar intervalos

As probabilidades foram representadas por intervalos para reduzir efeitos de
arredondamento e permitir modelagem consistente das restricoes lineares.

## Visao geral matematica do projeto

O projeto combina tres camadas:

1. Evidencia empirica do dataset

A base fornece frequencias observadas, como `P(A)`, `P(B)` e `P(A e B)`.
Esses valores sao transformados em restricoes intervalares do programa linear.

2. Regras de associacao liberadas pela extracao

O minerador de regras gera regras do tipo `R -> S` e, para cada regra aceita,
retorna suporte, confianca e lift. O programa principal nao inventa esses
valores para qualquer consulta; ele apenas consome os valores das regras que
foram liberadas.

3. Programacao linear para a consulta condicional

A pergunta do usuario continua sendo `P(A | B)`. Como essa probabilidade e uma
razao, o projeto usa Charnes-Cooper para transformar a consulta em dois
problemas lineares: um de minimo e outro de maximo. O resultado e um intervalo
de valores possiveis para `P(A | B)`.

## Alinhamento com os comentarios da turma

O resumo passado no quadro pode ser lido como uma lista de requisitos tecnicos.
No projeto, eles ficam organizados assim:

1. Extrair conhecimento probabilistico da base

O sistema transforma a base em eventos categoricos e extrai:

- probabilidades marginais de cada valor de cada variavel, como `P(N=alto)`;
- probabilidades conjuntas por pares de valores, como `P(N=alto, ph=acido)`;
- evidencia especifica da consulta, como `P(A)`, `P(B)` e `P(A e B)`;
- regras de associacao liberadas pela extracao, com suporte, confianca e lift;
- metricas de classificacao quando existe um atributo-alvo de classe.

2. Representar probabilidades como intervalos

Cada probabilidade pontual observada e arredondada e convertida em uma faixa:

```text
valor_observado = 0.97666
0.976 <= P(evento) <= 0.978
```

No codigo atual a largura padrao e `0.001` em torno do valor arredondado para
tres casas. A ideia e evitar que pequenos erros de ponto flutuante tornem o
programa linear artificialmente inviavel.

3. Escrever conhecimento como restricoes lineares

Uma probabilidade marginal ou conjunta vira:

```text
L <= soma(x_w onde evento ocorre) <= U
```

Uma regra de associacao liberada `R -> S`, quando usada pela confianca, vira:

```text
L <= P(R e S) / P(R) <= U
P(R e S) - U.P(R) <= 0
-P(R e S) + L.P(R) <= 0
```

4. Permitir pergunta condicional do usuario

O usuario escolhe um evento alvo `A` e um conjunto de afirmacoes `B`. O sistema
responde:

```text
P(A | B) = P(A e B) / P(B)
```

Se o usuario repetir o proprio alvo dentro das afirmacoes, o sistema remove essa
repeticao de `B`, porque `P(A | A,B)` seria uma tautologia e nao uma pergunta
informativa.

5. Transformar a pergunta em objetivo do PL

Como `P(A | B)` e uma fracao, ela e reescrita por Charnes-Cooper. Depois da
transformacao, o solver resolve dois objetivos lineares:

```text
min P(A | B)
max P(A | B)
```

O resultado final e um intervalo inferior/superior para a consulta.

6. Resolver e comparar solvers

O projeto executa o SciPy HiGHS no backend principal e, na comparacao, executa
tambem `highs-ds` e `highs-ipm` pelo script separado. A tela compara valores,
tempo de processamento, quantidade de variaveis e quantidade de restricoes.

Gurobi, lp_solve e cuPDLP-C ficam documentados como alternativas para benchmark
futuro.

7. Pontos relacionados, mas opcionais nesta versao

Laplace smoothing, verossimilhanca e tecnicas como annihilation/reinforcement
foram registrados como extensoes possiveis. Eles nao sao necessarios para a
versao atual, porque o foco entregue e: evidencia empirica, intervalos,
restricoes lineares, pergunta condicional e comparacao de solvers.

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

Essas probabilidades sao evidencias da base. Elas montam as restricoes do
programa linear e ajudam a responder a consulta condicional, mas nao substituem
a saida da ferramenta de extracao de regras.

No resultado do sistema, `P(A e B)` fica separado como probabilidade empirica.
Os campos `suporte`, `confianca` e `lift` so recebem valor quando a regra
`B -> A` foi efetivamente gerada e liberada pelo minerador de regras.

5. Regras de associacao

A consulta escolhida pelo usuario e interpretada como uma regra:

```text
B -> A
```

No projeto, isso significa:

```text
se condicoes do solo ocorrem, entao determinada cultura e provavel
```

Na teoria de regras de associacao, as medidas sao:

```text
suporte(B -> A)   = P(B e A)
confianca(B -> A) = P(A | B)
lift(B -> A)      = confianca(B -> A) / P(A)
```

No projeto, essas tres medidas nao sao recalculadas pela tela como se toda
consulta fosse uma regra valida. Elas pertencem a saida do minerador de regras.
Se a regra consultada `B -> A` nao estiver entre as regras liberadas, a
interface mostra suporte, confianca e lift como `-`. Mesmo assim, a pergunta
probabilistica continua sendo resolvida pelo PL usando `P(A)`, `P(B)` e
`P(A e B)` como evidencias empiricas.

5.1. Como as regras sao aprendidas

O projeto tambem aprende regras diretamente do dataset, antes de considerar a
consulta escolhida pelo usuario. Essas regras nao precisam ter relacao com o
evento A nem com as condicoes B da interface.

O processo usado e:

```text
1. gerar candidatos do tipo atributo=valor -> outro_atributo=outro_valor
2. calcular suporte = P(antecedente e consequente)
3. calcular confianca = P(consequente | antecedente)
4. calcular lift = confianca / P(consequente)
5. manter apenas regras com suporte, confianca e lift acima dos limiares
6. ordenar por lift, depois confianca, depois suporte
7. incorporar as 3 melhores regras ao programa linear
```

Depois que a regra e aprendida, o programa linear nao precisa recalcular suporte
e lift como se fossem uma nova consulta. Ele consome a propria saida do
algoritmo de aprendizagem:

```text
regra = {
  antecedente,
  consequente,
  suporte,
  confianca,
  lift
}
```

Ou seja, as probabilidades ligadas a suporte, confianca e lift sao tratadas
como metricas retornadas pelo minerador de regras. O PL usa principalmente a
confianca da regra aprendida para criar a restricao linear; suporte e lift
ficam registrados como evidencia de que a regra foi aceita pelo algoritmo.

Limiar atual:

```text
suporte >= 0.010
confianca >= 0.200
lift >= 1.050
```

Exemplo de regra aprendida:

```text
label=mothbeans -> ph=alcalino
suporte=0.020
confianca=0.450
lift=5.351
```

Essa regra vira uma restricao de confianca no PL. Se a regra aprendida for
R -> S, entao:

```text
confianca(R -> S) = P(R e S) / P(R)
```

Como o PL precisa de restricoes lineares, a confianca e escrita como:

```text
L <= P(R e S) / P(R) <= U
```

e depois convertida em:

```text
P(R e S) - U.P(R) <= 0
-P(R e S) + L.P(R) <= 0
```

Assim, o modelo incorpora conhecimento aprendido por regras de associacao sem
depender de uma regra de consulta que possa ter suporte zero.

6. Restricoes lineares

As probabilidades extraidas viram restricoes do tipo:

```text
limite_inferior <= soma(x_w onde evento ocorre) <= limite_superior
```

O modelo inclui:

- restricoes marginais para valores dos atributos;
- restricoes conjuntas por pares de valores;
- restricoes especificas da consulta selecionada;
- restricoes das 3 melhores regras de associacao aprendidas do dataset;
- restricao da regra de associacao selecionada somente quando `B -> A` foi
  liberada pela extracao de regras.

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
- suporte da regra liberada;
- confianca da regra liberada;
- lift da regra liberada;
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

As regras de associacao entram como conhecimento aprendido pelo minerador.
Suporte mede a frequencia conjunta da regra liberada, confianca mede a
probabilidade condicional da regra e lift compara essa confianca com a ocorrencia
geral do consequente. Quando a consulta `B -> A` nao aparece entre as regras
liberadas, o sistema nao recalcula suporte, confianca e lift para preencher os
cards; ele mostra esses campos sem valor e resolve apenas a consulta
probabilistica pelo PL.

Por fim, a transformacao de Charnes-Cooper e usada porque `P(A | B)` e uma
fracao. Como `linprog` resolve programacao linear, a razao precisa ser
reescrita em uma forma linear antes da chamada ao solver.

## Referencias bibliograficas

- Nilsson, N. J. Probabilistic Logic. Artificial Intelligence, 1986.
- Charnes, A.; Cooper, W. W. Programming with linear fractional functionals. Naval Research Logistics Quarterly, 1962.
- Tessem, B. Interval probability propagation. International Journal of Approximate Reasoning, 1992.
- Hooker, J. N. Mathematical programming models for reasoning under uncertainty. Operations Research Proceedings, 1992.
- Hooker, J. N. Mathematical Programming Methods for Reasoning under Uncertainty, 1995.
- SciPy documentation: `scipy.optimize.linprog`.
- Gurobi documentation and introductory material on linear programming.
- lp_solve project documentation.
- cuPDLP-C: implementation for solving linear programming problems with first-order methods on GPU.
- Artigos e materiais disponibilizados pelo professor no Classroom.
