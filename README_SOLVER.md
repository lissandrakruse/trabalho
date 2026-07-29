# Script separado do solver

O solver tambem existe como script separado. Ele nao sobe Flask, nao usa a
interface e nao importa `app.py`; apenas le o CSV do dataset e devolve o
resultado em JSON.

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
