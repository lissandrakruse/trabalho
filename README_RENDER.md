# Deploy no Render

Este projeto e um site estatico. Ele usa `index.html`, `styles.css`, `script.js` e o dataset `data/Crop_recommendation.csv`.

## Pelo render.yaml

1. No Render, clique em **New +**.
2. Escolha **Blueprint**.
3. Conecte o repositorio `lissandrakruse/trabalho`.
4. Confirme o servico `trabalho-probabilidades-solo`.

## Manualmente

Se criar como **Static Site**:

```text
Build Command: echo "Static site ready"
Publish Directory: .
```

Nao precisa configurar variaveis de ambiente.
