# Pacote separado do solver

Esta pasta pode ser enviada separadamente para os colegas. Ela contem:

- `solver.py`: script autonomo do solver
- `data/Crop_recommendation.csv`: dataset usado nos calculos
- `requirements.txt`: dependencia necessaria para o solver linear
- `results/`: pasta onde a saida JSON sera salva

## Como executar

Instale a dependencia:

```bash
python -m pip install -r requirements.txt
```

Execute a consulta padrao:

```bash
python solver.py
```

No Windows, tambem pode executar:

```bash
executar_solver.bat
```

A consulta padrao calcula:

```text
P(label=rice | ph=acido, rainfall=alto)
```

O resultado completo fica em:

```text
results/solver_resultado_padrao.json
```

## Consultas personalizadas

Exemplo:

```bash
python solver.py --target label=rice --condition ph=acido
```

Com duas condicoes:

```bash
python solver.py --target label=rice --condition ph=acido --condition rainfall=alto
```

Para listar atributos e valores validos:

```bash
python solver.py --show-domains
```

Para salvar a saida em outro arquivo:

```bash
python solver.py --target label=rice --condition ph=acido --output results/minha_consulta.json
```

## Observacao sobre intervalos

As probabilidades foram representadas por intervalos para reduzir efeitos de
arredondamento e permitir modelagem consistente das restricoes lineares.

## Referencias

- Nilsson, N. J. Probabilistic Logic. Artificial Intelligence, 1986.
- Charnes, A.; Cooper, W. W. Programming with linear fractional functionals. Naval Research Logistics Quarterly, 1962.
- Tessem, B. Interval probability propagation. International Journal of Approximate Reasoning, 1992.
- Artigos e materiais disponibilizados pelo professor no Classroom.
