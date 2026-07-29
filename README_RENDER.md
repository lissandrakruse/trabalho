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

Nao precisa configurar variaveis de ambiente.
