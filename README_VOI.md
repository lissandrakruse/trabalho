# Seleção ativa de informação em lógica probabilística intervalar

Esta extensão do projeto aplica a ideia de **Valor da Informação (VoI)** ao
problema definido pelo professor: decidir quais regras ou restrições
probabilísticas devem entrar no programa linear para estreitar a resposta de
uma consulta `P(A | B)`.

A técnica é inspirada em S. Ghosh e C. R. Ramakrishnan, *Value of Information
in Probabilistic Logic Programs*, EPTCS 306, 2019, p. 71–84,
[DOI 10.4204/EPTCS.306.14](https://doi.org/10.4204/EPTCS.306.14). O trabalho
não reivindica VoI ou programação linear probabilística como técnicas novas; a
contribuição é a adaptação ao modelo intervalar sem rede probabilística
predefinida.

## Pergunta e hipótese

> Uma seleção ativa orientada pela consulta consegue identificar quais regras
> ou restrições reduzem mais o intervalo, usando menos informações do que a
> seleção aleatória ou baseada apenas em suporte e confiança?

A hipótese é que a seleção pelos extremos `p_L` e `p_U` produz menor largura
`U - L` sob o mesmo orçamento e evita resolver programas lineares que não podem
alterar o respectivo limite.

## Adaptação do artigo

| Conceito | Artigo-base | Projeto intervalar |
| --- | --- | --- |
| informação | observável e sua realização | restrição Apriori candidata |
| consulta | proposição ground `q` | consulta intervalar `P(A | B)` |
| incerteza | entropia da consulta | largura `W = U - L` |
| utilidade | menos entropia esperada | menor intervalo da resposta |
| custo | custo da observação | uma unidade por restrição incluída |
| cenário | observações já realizadas | conjunto de restrições já selecionadas |
| política | maior VoI que cabe no orçamento | maior violação dos extremos atuais |

O plano de medições por entropia da Figura 3 também foi implementado, mas fica
como **reprodução do artigo-base**. A contribuição principal é a seleção de
restrições do programa linear.

## Modelo matemático

Depois da transformação de Charnes–Cooper, seja `F` a região factível e `q(z)`
o objetivo linear da consulta:

```text
L(F) = min_{z em F} q(z)
U(F) = max_{z em F} q(z)
W(F) = U(F) - L(F)
ganho(C | F) = W(F) - W(F interseção C)
```

O modelo-base `F_0` contém marginais e conjuntas por pares. As candidatas são
restrições de suporte e confiança extraídas pelo Apriori e filtradas pela
sobreposição de literais com a consulta atual.

### Poda exata dos extremos

Se `z_L` minimiza a consulta em `F` e satisfaz a candidata `C`, então `z_L`
continua factível em `F interseção C`. Como o novo conjunto é subconjunto de
`F`, o limite inferior permanece exatamente `L(F)`. O mesmo argumento vale
para `z_U` e o limite superior.

```text
se A_C z_L <= 0: não resolver novamente o limite inferior
se A_C z_U <= 0: não resolver novamente o limite superior
```

Somente o extremo violado é reotimizado com HiGHS. A poda é exata; a ordem de
seleção é uma heurística gulosa.

### Seleção gulosa

Em cada passo, a pontuação é a maior violação positiva das desigualdades da
candidata nos dois extremos atuais:

```text
score(C) = max(0, max(A_C z_L), max(A_C z_U))
C_k = argmax_C score(C)
F_{k+1} = F_k interseção C_k
```

A verificação usa multiplicação matriz-vetor, sem chamar o solver. Depois da
inclusão, no máximo dois PLs são resolvidos, um para cada extremo violado. A
busca para quando termina o orçamento ou quando todos os extremos satisfazem
todas as candidatas restantes.

Para `K > 1`, a escolha gulosa não garante o subconjunto globalmente ótimo.
Ela evita a busca combinatória de `combinação(|C|, K)` subconjuntos.

## Resultado de referência

Consulta:

```text
P(label=rice | ph=acido, rainfall=alto)
```

Com orçamento 25 e filtro de pelo menos dois literais em comum:

| Indicador | Resultado |
| --- | ---: |
| restrições Apriori disponíveis | 2.707 |
| candidatas relevantes | 52 |
| restrições selecionadas | 25 |
| largura do modelo-base | 0,023265 |
| largura após seleção ativa | 0,019084 |
| redução relativa | 17,97% |
| largura por suporte/confiança | 0,023265 |
| largura aleatória média | 0,022122 |
| redução do modelo completo recuperada | 89,90% |
| verificações algébricas de extremos | 2.000 |
| reotimizações seletivas executadas | 43 |
| podas exatas por factibilidade | 1.446 (72,30%) |
| chamadas candidatas evitadas no total | 1.957 (97,85%) |
| subconjuntos que uma busca completa examinaria | 477.551.179.875.952 |

Algumas inclusões iniciais não alteram imediatamente a largura porque pode
existir outra solução ótima na mesma face. Restrições sucessivas eliminam essa
face; no caso de arroz, a redução aparece no passo 24.

## Avaliação em dez consultas

O experimento `experiments/run_active_selection_experiment.py` usa o mesmo
orçamento e as condições `ph=acido, rainfall=alto` para dez culturas, incluindo
três consultas sem ocorrência conjunta na amostra.

| Medida agregada | Resultado |
| --- | ---: |
| largura média do modelo-base | 0,066857 |
| largura média da seleção ativa | 0,014312 |
| redução relativa média | 41,31% |
| largura média por suporte/confiança | 0,040575 |
| largura aleatória média | 0,022285 |
| taxa média de poda exata | 66,29% |
| taxa média total de chamadas evitadas | 97,28% |
| vitórias sobre suporte/confiança | 6 de 10 |
| vitórias sobre a média aleatória | 5 de 10 |

A seleção ativa foi melhor **em média**, não em todas as consultas. Essa
distinção é importante: o experimento sustenta a hipótese agregada nesta
demonstração controlada, mas não prova dominância universal.

## Reprodução direta do artigo

O módulo `voi.py` também implementa observáveis, realizações, custos, cenários,
utilidade por entropia e o plano condicional da Figura 3. No teste com arroz,
chuva é a primeira medição e o VoI do plano com orçamento 2 é `0,158183` bits.

Esse planejador enumera os mundos e não chama `linprog`. Já a seleção ativa de
restrições chama o HiGHS apenas para atualizar os extremos violados.

## Como executar

Interface e API:

```bash
python app.py
```

Seleção ativa independente:

```bash
python scripts/select_constraints.py \
  --target label=rice \
  --condition ph=acido \
  --condition rainfall=alto \
  --budget 25
```

Experimento em dez consultas:

```bash
python experiments/run_active_selection_experiment.py
```

Reprodução do plano de observações do artigo:

```bash
python scripts/plan_voi.py --crop rice --budget 2
```

Relatório científico versionado:

```bash
python reports/generate_scientific_report.py
```

## Limites científicos

- a poda dos extremos é exata, mas a escolha pela maior violação é gulosa;
- o filtro de relevância pode excluir restrições indiretamente úteis;
- suporte, confiança e intervalos foram estimados na mesma base;
- a discretização em três faixas reduz a resolução agronômica;
- o experimento é uma demonstração controlada, não validação agronômica;
- publicação exige validação externa, análise de sensibilidade e mais consultas.
