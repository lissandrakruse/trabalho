# Probabilidades do Solo — Ω, Apriori e Programação Linear

Este projeto responde consultas condicionais `P(A | B)` sobre o dataset de
recomendação de culturas. A implementação separa corretamente três papéis:

1. o dataset categorizado forma os mundos possíveis observados `Ω`;
2. o Apriori extrai itemsets frequentes e gera regras de associação;
3. suporte e confiança são transformados em restrições probabilísticas do
   programa linear.

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

A confiança mínima é zero de propósito: confiança e lift não são usados para
filtrar regras como se medissem qualidade. O único corte é o suporte mínimo,
necessário para definir os itemsets frequentes do Apriori.

Para cada itemset frequente de tamanho 2 ou 3, são geradas regras com consequente
unitário. Todas as regras geradas são incorporadas ao modelo. A ordenação da
lista é apenas determinística e não representa ranking.

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

A confiança produz duas desigualdades lineares:

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
HiGHS. A matriz de restrições é esparsa para comportar todas as regras Apriori
sem consumo excessivo de memória. As máscaras de eventos são reutilizadas e o
sistema global de restrições fica em cache por processo, pois ele depende do
dataset e não da consulta escolhida. Essa otimização reduz o tempo de resposta
no Render sem remover nenhuma das 5.312 regras. A comparação também executa
`highs`, `highs-ds` e `highs-ipm`.

## Interface do usuário

A página permite:

- escolher o evento alvo `A`;
- adicionar ou remover condições de `B`;
- consultar o intervalo linear;
- ver o diagnóstico empírico sem confundi-lo com a formulação;
- identificar se `B -> A` foi realmente gerada pelo Apriori;
- visualizar uma prévia das regras usadas no programa linear;
- gerar o LP completo, PDFs e comparação de solvers.

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
- `index.html`, `script.js`, `styles.css`: interface;
- `data/Crop_recommendation.csv`: dataset.

## Referências matemáticas

- lógica probabilística e representação por mundos possíveis;
- algoritmo Apriori para itemsets frequentes e regras de associação;
- transformação de Charnes–Cooper para programação linear fracionária;
- `scipy.optimize.linprog` e métodos HiGHS.
