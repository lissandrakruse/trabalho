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
Start Command: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 360
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
- consulta sem regra Apriori, com medidas vazias, explicacao objetiva e limite
  superior pequeno quando `P(A e B)` empirico e zero;
- comparacao real entre `highs`, `highs-ds` e `highs-ipm`;
- intervalo do solver, 467 variaveis, 6.804 restricoes e SHA-256 do modelo;
- correspondencia entre o modelo resolvido e o TXT auditavel;
- geracao e cabecalho dos relatorios PDF da consulta e dos solvers.

Execucao completa contra o Render:

```bash
python scripts/site_robot.py \
  --base-url https://trabalho-bh30.onrender.com/ \
  --output robot-test-report.json
```

Use `--quick` para testar sem gerar o TXT e o PDF. O processo termina com codigo 1 se alguma verificacao falhar e sempre grava as evidencias em JSON.

No GitHub, a acao **Robo de teste do site** e executada automaticamente em cada
novo commit da `main` e tambem pode ser iniciada manualmente na aba **Actions**.
Nos testes automaticos, o robo aguarda o Render confirmar o SHA implantado antes
de comecar. Ao final, ele disponibiliza o arquivo `robot-test-report.json` como
artefato da execucao.

## Desempenho da comparacao

A primeira comparacao executa os tres metodos reais do HiGHS. O resultado do
HiGHS-IPM, ja produzido pelo projeto principal, e reaproveitado na tabela, e os
outros dois metodos sao executados pelo script separado. A comparacao fica em
cache por consulta para que a geracao posterior do PDF nao resolva novamente os
mesmos programas lineares. O resultado do botao **Consultar** tambem fica em
cache e e reutilizado quando a comparacao recebe a mesma consulta. O limite de
360 segundos do Gunicorn contempla a primeira execucao em instancias gratuitas
do Render e a reutilizacao mantem o caminho normal abaixo de limites menores
configurados diretamente no painel do servico.

Na interface, a comparacao e dividida em tres requisicoes: primeiro o
HiGHS-IPM do projeto principal, depois `highs` e `highs-ds` pelo script separado.
Cada metodo termina dentro do limite individual do servico; a rota comparativa e
o PDF apenas reunem os resultados reais ja calculados.
