# Deploy no Render

Este projeto agora roda como aplicacao Python Flask. O frontend chama a API Python para calcular suporte, confianca, lift e o intervalo por programacao linear.

## Pelo render.yaml

1. No Render, clique em **New +**.
2. Escolha **Blueprint**.
3. Conecte o repositorio `lissandrakruse/trabalho`.
4. Confirme o servico `trabalho-probabilidades-solo`.

## Manualmente

Se criar como **Web Service**:

```text
Runtime: Python
Build Command: python -m pip install --upgrade pip setuptools wheel && python -m pip install -r requirements.txt
Start Command: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120
```

O projeto fixa Python em `.python-version`:

```text
3.11.9
```

Se o Render ainda mostrar `Using Python version 3.14.3 (default)`, adicione em **Environment**:

```text
PYTHON_VERSION=3.11.9
```

Nao precisa configurar variaveis de ambiente.

## Robo de teste funcional

O arquivo `scripts/site_robot.py` testa o site como um cliente externo. Ele verifica:

- saude do servico e textos principais da interface;
- carregamento dos 2.200 registros, 466 mundos e 5.312 regras Apriori;
- consulta de referencia `P(label=rice | ph=acido, rainfall=alto)`;
- intervalo do solver, 467 variaveis, 15.018 restricoes e SHA-256 do modelo;
- correspondencia entre o modelo resolvido e o TXT auditavel;
- geracao e cabecalho do relatorio PDF.

Execucao completa contra o Render:

```bash
python scripts/site_robot.py \
  --base-url https://trabalho-bh30.onrender.com/ \
  --output robot-test-report.json
```

Use `--quick` para testar sem gerar o TXT e o PDF. O processo termina com codigo 1 se alguma verificacao falhar e sempre grava as evidencias em JSON.

No GitHub, a acao **Robo de teste do site** pode ser iniciada manualmente na aba **Actions**. Ao final, ela disponibiliza o arquivo `robot-test-report.json` como artefato da execucao.
