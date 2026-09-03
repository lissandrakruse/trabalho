# Probabilidades do Solo — Ω, Apriori e Programação Linear

Este projeto responde consultas condicionais `P(A | B)` sobre o dataset de
recomendação de culturas. A implementação separa corretamente três papéis:

1. o dataset categorizado forma os mundos possíveis observados `Ω`;
2. o Apriori extrai itemsets frequentes e gera regras de associação;
3. suporte e confiança são transformados em restrições probabilísticas do
   programa linear.

O projeto agora inclui uma camada de seleção ativa inspirada no artigo *Value
of Information in Probabilistic Logic Programs* (Ghosh e Ramakrishnan, 2019).
Ela seleciona restrições Apriori orientadas pela consulta, usa a redução da
largura `U-L` como utilidade e reotimiza somente os extremos `p_L` ou `p_U` que
forem violados. A formulação completa está em
[`README_VOI.md`](README_VOI.md).

`Lift` não é tratado como acurácia, qualidade de classificação, filtro ou
coeficiente do programa linear. Ele permanece somente como medida descritiva da
associação retornada pelo Apriori.

## Fluxo implementado

### 1. Preparação da base

Os atributos numéricos são convertidos em categorias. Para os atributos gerais,
os tercis produzem `baixo`, `medio` e `alto`. O pH usa `acido`, `neutro` e
`alcalino`. O atributo `label` mantém a cultura original.

### 2. Construção de Ω

Cada combinação categórica distinta observada é um mundo `w`:

```text
Ω = {w1, w2, ..., wn}
```

Mundos repetidos são armazenados uma vez, com sua contagem. Essa contagem é
preservada quando Ω é fornecido ao Apriori, de modo que o suporte continua sendo
calculado sobre as 2.200 linhas da base.

No programa linear, cada mundo recebe uma variável:

```text
x_w >= 0
sum_w x_w = 1
```

### 3. Mineração Apriori

O arquivo `apriori_rules.py` implementa o Apriori para transações categóricas
ponderadas. A configuração atual é:

```text
suporte mínimo = 0,010
confiança mínima = 0,000
tamanho máximo do itemset = 3
```

A confiança mínima da mineração é zero de propósito: nenhuma regra com suporte
frequente é escondida da consulta ou da auditoria. Para o programa linear, há
um segundo critério explícito: somente regras fortes, com confiança maior ou
igual a 0,70, acrescentam desigualdades de confiança.

Para cada itemset frequente de tamanho 2 ou 3, são geradas regras com consequente
unitário. Os suportes de todas as regras geradas são incorporados ao modelo;
1.205 das 5.312 regras também fornecem as desigualdades de confiança por
atingirem o limiar de 0,70. A ordenação da lista é apenas determinística e não
representa ranking.

### 4. Regras viram restrições lineares

Para uma regra `R -> S`, o Apriori fornece:

```text
suporte s   = P(R e S)
confiança c = P(S | R) = P(R e S) / P(R)
lift        = confiança / P(S)
```

O suporte produz uma restrição intervalar direta:

```text
L_s <= P(R e S) <= U_s
```

A confiança das regras fortes produz duas desigualdades lineares:

```text
P(R e S) - U_c P(R) <= 0
-P(R e S) + L_c P(R) <= 0
```

O lift não entra na formulação porque não é uma probabilidade linear. Ele é
exibido apenas para descrever a associação.

### 5. Restrições marginais e conjuntas

Além das regras Apriori, o modelo inclui:

- `P(X=x)` para todos os valores de todos os atributos;
- `P(X=x, Y=y)` para todos os pares de atributos e valores;
- normalização e não negatividade das massas dos mundos.

As restrições usam intervalos estreitos ao redor das frequências empíricas para
absorver arredondamento numérico.

### 6. A consulta não injeta a própria resposta

A interface mostra `P(A)`, `P(B)` e `P(A e B)` empíricos para auditoria. Porém,
o código não adiciona uma restrição específica fixando `P(A e B)` da consulta.
Se fizesse isso, o solver apenas reproduziria a resposta observada.

O intervalo de `P(A | B)` é inferido a partir das marginais, conjuntas por pares
e regras Apriori globais. Os cards de suporte, confiança e lift só recebem valor
quando a regra consultada `B -> A` realmente pertence à saída do Apriori.

Se `P(A e B)=0` na amostra, esse valor continua aparecendo somente como
auditoria. Para não confundir ausência na amostra com impossibilidade lógica, o
modelo acrescenta as combinações completas de `A e B` que não foram observadas,
sempre com contagem empírica zero. Essas novas colunas recebem as mesmas
restrições globais e permitem ao solver calcular um intervalo como `0` até um
valor pequeno, em vez de criar um vetor objetivo vazio e devolver `0 a 0`.

### 7. Charnes–Cooper e HiGHS

A consulta é fracionária:

```text
P(A | B) = P(A e B) / P(B)
```

O projeto aplica Charnes–Cooper, transforma o problema em programação linear e
executa dois objetivos:

```text
min P(A | B)
max P(A | B)
```

O solver principal é `scipy.optimize.linprog` com o método `highs-ipm` do
HiGHS. A matriz de restrições é esparsa para comportar os suportes Apriori e as
confianças fortes sem consumo excessivo de memória. As máscaras de eventos são
reutilizadas e o
sistema global de restrições fica em cache por processo, pois ele depende do
dataset e não da consulta escolhida. As 5.312 regras permanecem disponíveis;
o PL evita apenas as 4.107 desigualdades de confiança fracas e redundantes.
Essa seleção reduz a formulação para 6.804 restrições e mantém as rotas dentro
do tempo de resposta do Render. A comparação também executa `highs`,
`highs-ds` e `highs-ipm`.

### 8. Resumo didático e TXT numérico auditável

O painel da interface é identificado como **formulação matemática resumida**.
Expressões como `sum_w x_w` explicam o modelo, mas não são apresentadas como se
fossem a entrada literal do solver.

O botão de exportação gera um TXT diferente, construído pelo mesmo objeto usado
na chamada de `scipy.optimize.linprog`. O arquivo contém:

- mapeamento das `n` variáveis de mundos `y_w` e da variável de escala `t`;
- os vetores `c_lower` e `c_upper_as_min` realmente minimizados;
- todas as entradas não nulas de `A_ub`, `b_ub`, `A_eq` e `b_eq`;
- limites de cada variável e a origem de cada linha de desigualdade;
- SHA-256 canônico do modelo, repetido no resultado do solver e no TXT.

Nas consultas já observadas, os 466 mundos geram 467 variáveis após
Charnes–Cooper (`466 y_w + t`). Quando `A e B` não foi observado, o número de
variáveis cresce apenas para aquela consulta, de acordo com os mundos de
contagem zero que a completam. O TXT identifica cada mundo como observado ou
completado. O digest impede que uma formulação diferente seja exportada
silenciosamente: se os modelos divergirem, a geração falha.

### 9. Seleção ativa de restrições

O modelo-base usa marginais e conjuntas por pares. Suportes e confianças
Apriori relevantes para os literais da consulta são tratados como candidatas.
Em cada passo, o algoritmo escolhe a restrição mais violada pelos extremos
atuais.

Se `p_L` satisfaz a candidata, o limite inferior permanece exatamente igual e
não é resolvido novamente. O mesmo vale para `p_U`. Assim, a checagem de todas
as candidatas usa apenas multiplicação matriz-vetor e o HiGHS é chamado no
máximo duas vezes por restrição efetivamente escolhida.

Na consulta de referência, 25 de 52 candidatas relevantes reduzem a largura de
`0,023265` para `0,019084`, superam os baselines sob o mesmo orçamento e evitam
97,85% das chamadas candidatas no total. O experimento em dez consultas mostra
redução relativa média de 41,31%, taxa média de poda exata de 66,29% e economia
total média de 97,28% das chamadas candidatas.

## Interface do usuário

A página permite:

- escolher o evento alvo `A`;
- adicionar ou remover condições de `B`;
- consultar o intervalo linear;
- ver o diagnóstico empírico sem confundi-lo com a formulação;
- identificar se `B -> A` foi realmente gerada pelo Apriori;
- visualizar uma prévia das regras usadas no programa linear;
- consultar o resumo didático e exportar o modelo numérico auditável em TXT;
- gerar PDFs e comparação de solvers;
- executar a seleção ativa, comparar baselines e auditar cada restrição escolhida.

Não há painel de acurácia, precisão, recall ou F1, porque o exercício modela
restrições probabilísticas, não avaliação de um classificador.

## Execução

```bash
python -m pip install -r requirements.txt
python app.py
```

Solver independente:

```bash
python scripts/solve_query.py \
  --target label=rice \
  --condition ph=acido \
  --condition rainfall=alto
```

Planejador de VoI independente da interface e dos solvers lineares:

```bash
python scripts/plan_voi.py --crop rice --budget 2
```

Seleção ativa de restrições:

```bash
python scripts/select_constraints.py \
  --target label=rice \
  --condition ph=acido \
  --condition rainfall=alto \
  --budget 25
```

Os três métodos HiGHS resolvem o programa linear intervalar. A seleção ativa
usa o mesmo HiGHS apenas para atualizar extremos violados. O planejador de
medições que reproduz o artigo usa enumeração dos mundos, inferência
condicional, entropia e busca gulosa; ele não chama `linprog`.

Consulta padrão:

```bash
python solver.py
```

Testes:

```bash
python -m unittest discover -s tests -v
```

## Arquivos principais

- `app.py`: API Flask, formulação, solver, relatórios e interface;
- `apriori_rules.py`: Ω ponderado, itemsets frequentes e regras Apriori;
- `scripts/solve_query.py`: solver independente com a mesma formulação;
- `active_selection.py`: seleção gulosa e poda exata dos extremos;
- `scripts/select_constraints.py`: executor independente da seleção ativa;
- `voi.py`: definições de utilidade, VoI, seleção de subconjunto e plano
  condicional guloso;
- `scripts/plan_voi.py`: planejador de VoI executável fora da interface;
- `experiments/run_active_selection_experiment.py`: comparação em dez consultas;
- `experiments/run_voi_experiment.py`: comparação controlada em 22 culturas;
- `README_VOI.md`: modelo matemático, resultados e limites científicos;
- `index.html`, `script.js`, `styles.css`: interface;
- `data/Crop_recommendation.csv`: dataset.

## Referências matemáticas

- lógica probabilística e representação por mundos possíveis;
- algoritmo Apriori para itemsets frequentes e regras de associação;
- transformação de Charnes–Cooper para programação linear fracionária;
- `scipy.optimize.linprog` e métodos HiGHS.
