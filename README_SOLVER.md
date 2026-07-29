# Script do solver

O projeto tem um script para executar o solver sem abrir a interface web:

```bash
python scripts/solve_query.py --target label=rice --condition ph=acido --condition rainfall=alto
```

Para salvar a saida em JSON:

```bash
python scripts/solve_query.py --target label=rice --condition ph=acido --condition rainfall=alto --output reports/generated/solver_exemplo_rice.json
```

Para ver os atributos e valores validos do dataset:

```bash
python scripts/solve_query.py --show-domains --target label=rice
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
