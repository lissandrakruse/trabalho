# Projeto científico: seleção ativa em lógica probabilística intervalar

## Título provisório

**Seleção ativa de restrições por redução de incerteza em programas
lógico-probabilísticos com respostas intervalares**

## Autoria

Lissandra Kruse Fuganti

## Motivação

Modelos em lógica probabilística podem representar conhecimento probabilístico
não estruturado sem exigir uma rede bayesiana previamente construída. Essa
flexibilidade gera muitas fórmulas e restrições admissíveis. Como a resposta de
uma consulta é um intervalo, surge a questão de quais informações são úteis
para reduzir sua largura.

## Pergunta de pesquisa

Uma técnica de seleção ativa inspirada em Valor da Informação consegue
identificar quais regras ou restrições probabilísticas reduzem mais os
intervalos das consultas usando menos informação que a seleção aleatória ou
baseada somente em suporte e confiança?

## Hipótese

A seleção orientada pelos extremos `p_L` e `p_U`, com poda exata das
reotimizações desnecessárias, produz intervalos menores em média sob o mesmo
orçamento e custo computacional inferior à avaliação exaustiva de candidatas.

## Objetivo geral

Desenvolver e avaliar uma adaptação da seleção ativa por redução de incerteza
para programas lógico-probabilísticos intervalares sem estrutura gráfica
probabilística predefinida.

## Objetivos específicos

1. Representar regras Apriori de suporte e confiança como restrições candidatas.
2. Definir largura `U - L` como medida de incerteza da consulta intervalar.
3. Provar e implementar a poda exata baseada na factibilidade de `p_L` e `p_U`.
4. Selecionar restrições de forma gulosa e orientada pela consulta.
5. Comparar a seleção com sorteio e ranking por suporte/confiança.
6. Medir redução do intervalo, quantidade de informações e chamadas ao HiGHS.
7. Manter uma reprodução separada do algoritmo de VoI do artigo-base.

## Método

O conjunto-base contém restrições marginais e conjuntas por pares. As regras
Apriori relevantes para pelo menos dois literais de `A` e `B` formam o conjunto
de candidatas. Dois PLs produzem os extremos atuais. Cada candidata é testada
por multiplicação matriz-vetor; somente extremos violados são reotimizados.

A política escolhe a candidata com maior violação positiva, atualiza a região
factível e repete até consumir o orçamento. A poda é exata, enquanto a política
é gulosa e não promete ótimo global para lotes.

As frequências empíricas entram no programa linear sem arredondamento. Para uma
frequência completa `p`, a evidência é representada por `max(0,p-0,001) <=
P(E) <= min(1,p+0,001)`. A margem é intervalar; nenhuma redução para três casas
é feita nos coeficientes do solver.

## Desenho experimental

- base Crop Recommendation com 2.200 registros e 466 mundos observados;
- dez consultas de culturas;
- condições fixas `ph=acido, rainfall=alto`;
- orçamento de 25 restrições;
- três consultas sem ocorrência conjunta para testar completamento de mundos;
- baselines: suporte × confiança e média de cinco amostras aleatórias;
- desfechos: largura final, redução relativa e número de reotimizações.

## Resultados

A largura média caiu de `0,067066` para `0,015597`, redução relativa média de
`36,45%`. Os baselines terminaram com largura média `0,043087`
(suporte/confiança) e `0,026353` (aleatório). A seleção venceu o primeiro em 4
de 10 consultas e a média aleatória em 4 de 10. A taxa média de poda foi
`66,13%`; considerando também os extremos violados de candidatas não escolhidas,
a estratégia evitou `97,23%` das chamadas candidatas no total.

Na consulta de maçã, 23 de 46 candidatas relevantes reduziram a largura de
`0,217015` para `0,023056`, recuperando `100%` da redução do modelo completo.
Foram executadas 35 reotimizações; 1.200 das 1.656 avaliações de extremos foram
podadas exatamente por factibilidade, e 1.621 chamadas candidatas foram
evitadas no total em relação a uma pontuação gulosa ingênua por reotimização.

## Contribuição delimitada

A contribuição é a adaptação aplicada da aquisição de informação para seleção
de restrições em respostas intervalares, com prova de poda dos extremos,
implementação auditável e avaliação contra baselines. Não se reivindica a
criação de VoI, lógica probabilística, Apriori, Charnes–Cooper ou HiGHS.

## Ameaças à validade

- avaliação in-sample;
- política gulosa sem garantia de ótimo global;
- filtro de relevância pode perder relações indiretas;
- dez consultas não representam todos os programas probabilísticos;
- dados balanceados e discretizados não refletem toda a variabilidade de campo;
- ausência de validação externa e análise estatística inferencial.

## Próximas etapas

1. Separar dados de construção e validação.
2. Avaliar outros escores baratos para escolher a próxima restrição.
3. Comparar com o ótimo exaustivo apenas em instâncias pequenas.
4. Variar orçamento, limiar de confiança e filtro de relevância.
5. Medir tempo e memória em bases maiores.
6. Replicar em outro domínio lógico-probabilístico.

## Referência central

GHOSH, S.; RAMAKRISHNAN, C. R. Value of Information in Probabilistic Logic
Programs. *Electronic Proceedings in Theoretical Computer Science*, v. 306,
p. 71–84, 2019. DOI: 10.4204/EPTCS.306.14. arXiv:1909.08234.
