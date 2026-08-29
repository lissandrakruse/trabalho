const datasetStatus = document.querySelector("#datasetStatus");
const conditionList = document.querySelector("#conditionList");
const addConditionButton = document.querySelector("#addCondition");
const runQueryButton = document.querySelector("#runQuery");
const targetAttribute = document.querySelector("#targetAttribute");
const targetValue = document.querySelector("#targetValue");
const generateReportButton = document.querySelector("#generateReport");
const compareSolverButton = document.querySelector("#compareSolver");
const generateSolverReportButton = document.querySelector("#generateSolverReport");
const generateFullLinearProgramButton = document.querySelector("#generateFullLinearProgram");
const downloadReportLink = document.querySelector("#downloadReport");
const downloadSolverReportLink = document.querySelector("#downloadSolverReport");
const downloadFullLinearProgramLink = document.querySelector("#downloadFullLinearProgram");
const conditionalValue = document.querySelector("#conditionalValue");
const supportValue = document.querySelector("#supportValue");
const confidenceValue = document.querySelector("#confidenceValue");
const liftValue = document.querySelector("#liftValue");
const countValue = document.querySelector("#countValue");
const probabilityText = document.querySelector("#probabilityText");
const solverCompareStatus = document.querySelector("#solverCompareStatus");
const solverDemo = document.querySelector("#solverDemo");
const solverComparison = document.querySelector("#solverComparison");
const linearProgram = document.querySelector("#linearProgram");
const mathModel = document.querySelector("#mathModel");
const tabButtons = document.querySelectorAll(".tab-button");
const tabPanels = document.querySelectorAll(".tab-panel");

// FRONTEND DO EXERCICIO
//
// Controla a tela: monta os selects de A e B, envia P(A | B) para o Flask,
// recebe probabilidades/regras/intervalo linear e atualiza cards, explicacoes,
// PDFs e comparacao com solver separado.
let domains = {};
let labels = {};
let attributes = [];

function firstAttributeExcept(attribute) {
  return attributes.find((item) => item !== attribute) || attributes[0];
}

function fillSelect(select, values, labeler = (value) => value) {
  select.innerHTML = "";
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = labeler(value);
    select.appendChild(option);
  });
}

function fillValueSelect(attributeSelect, valueSelect) {
  fillSelect(valueSelect, domains[attributeSelect.value] || []);
}

function createConditionRow(attribute = "ph", value = "acido") {
  const row = document.createElement("div");
  row.className = "condition-row";

  const attributeLabel = document.createElement("label");
  attributeLabel.textContent = "Atributo";
  const attributeSelect = document.createElement("select");
  fillSelect(attributeSelect, Object.keys(domains), (item) => labels[item] || item);
  attributeSelect.value = attribute;
  attributeLabel.appendChild(attributeSelect);

  const valueLabel = document.createElement("label");
  valueLabel.textContent = "Valor";
  const valueSelect = document.createElement("select");
  valueLabel.appendChild(valueSelect);

  const removeButton = document.createElement("button");
  removeButton.className = "remove-condition";
  removeButton.type = "button";
  removeButton.title = "Remover condição";
  removeButton.textContent = "x";

  attributeSelect.addEventListener("change", () => fillValueSelect(attributeSelect, valueSelect));
  removeButton.addEventListener("click", () => row.remove());

  row.append(attributeLabel, valueLabel, removeButton);
  conditionList.appendChild(row);
  fillValueSelect(attributeSelect, valueSelect);
  if ((domains[attribute] || []).includes(value)) valueSelect.value = value;
}

function readConditions() {
  return [...conditionList.querySelectorAll(".condition-row")].map((row) => {
    const selects = row.querySelectorAll("select");
    return { attribute: selects[0].value, value: selects[1].value };
  });
}

function fmt(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(3);
}

function fmtDetailed(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(6);
}

function fmtCompare(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  const number = Number(value);
  return Number.isInteger(number) ? String(number) : number.toFixed(3);
}

function ruleFromResult(result) {
  // A interface nunca fabrica uma regra para preencher os cards. As medidas
  // aparecem apenas quando B -> A pertence a saida real do Apriori.
  return result.releasedAssociationRule || null;
}

function eventLabel(conditions) {
  if (!conditions.length) return "verdadeiro";
  return conditions.map((condition) => `${condition.attribute}=${condition.value}`).join(", ");
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  })[character]);
}

async function readJsonResponse(response) {
  const text = await response.text();
  try {
    return JSON.parse(text);
  } catch (error) {
    const preview = text.trim().slice(0, 160);
    throw new Error(
      `Resposta invalida do servidor (${response.status}). A API retornou texto/HTML em vez de JSON: ${preview || "resposta vazia"}`
    );
  }
}

function intervalText(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  const rounded = Number(value).toFixed(3);
  const lower = Math.max(0, Number(rounded) - 0.001).toFixed(3);
  const upper = Math.min(1, Number(rounded) + 0.001).toFixed(3);
  return `${lower} <= ${rounded} <= ${upper}`;
}

function buildPayload() {
  // Item 1c: transforma a escolha do usuario em uma pergunta condicional.
  // target = evento A; conditions = evento B.
  const target = {
    attribute: targetAttribute.value,
    value: targetValue.value,
  };
  const seen = new Set();
  const conditions = [];
  [...conditionList.querySelectorAll(".condition-row")].forEach((row) => {
    const selects = row.querySelectorAll("select");
    const condition = { attribute: selects[0].value, value: selects[1].value };
    const key = `${condition.attribute}\u0000${condition.value}`;
    const isTarget = condition.attribute === target.attribute && condition.value === target.value;
    if (isTarget || seen.has(key)) {
      row.remove();
      return;
    }
    seen.add(key);
    conditions.push(condition);
  });
  return {
    conditions,
    target,
  };
}

function setActiveTab(activeButton) {
  tabButtons.forEach((button) => {
    const isActive = button === activeButton;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-selected", String(isActive));
  });

  tabPanels.forEach((panel) => {
    panel.classList.toggle("is-active", panel.id === activeButton.getAttribute("aria-controls"));
  });
}

function renderSolverComparison(result) {
  const solverResult = result.standaloneSolver;
  const linear = solverResult.linear || {};
  const timing = result.timing || {};
  const targetText = escapeHtml(eventLabel([solverResult.target]));
  const conditionsText = escapeHtml(eventLabel(solverResult.conditions));
  const intervalTextValue = linear.ok
    ? `${fmt(linear.lower)} <= P(A | B) <= ${fmt(linear.upper)}`
    : `Nao resolvido: ${escapeHtml(linear.error || "motivo nao informado")}`;
  const labels = {
    pA: "P(A)",
    pB: "P(B)",
    pAB: "P(A e B) empirico (auditoria)",
    support: "Suporte da regra Apriori",
    confidence: "Confianca da regra Apriori",
    lift: "Lift descritivo",
    countBoth: "Casos A e B",
    countBase: "Casos B",
    linearLower: "Limite inferior",
    linearUpper: "Limite superior",
    variables: "Variaveis",
    constraints: "Restricoes",
    durationSeconds: "Tempo de resolucao (s)",
  };
  const rows = Object.entries(result.comparison.metrics).map(([key, item]) => {
    const difference = item.difference === null ? "-" : Number(item.difference).toExponential(2);
    let status = item.match ? "Igual" : "Diferente";
    if (key === "durationSeconds") {
      status = timing.faster === "standaloneSolver" ? "Solver mais rapido" : timing.faster === "main" ? "Projeto mais rapido" : "Tempo equivalente";
    }
    return `<tr><td>${labels[key] || key}</td><td>${fmtCompare(item.main)}</td><td>${fmtCompare(item.solver)}</td><td>${difference}</td><td>${status}</td></tr>`;
  });
  const solverCatalog = result.solverCatalog || [];
  const catalogRows = solverCatalog.map((solver) => {
    return `<tr><td>${escapeHtml(solver.name)}</td><td>${escapeHtml(solver.status)}</td><td>${escapeHtml(solver.comparison)}</td></tr>`;
  });
  const engineRows = (result.solverEngineResults || []).map((engine) => {
    const interval = engine.status === "ok"
      ? `${fmtCompare(engine.lower)} - ${fmtCompare(engine.upper)}`
      : escapeHtml(engine.error || "Erro");
    return [
      `<tr>`,
      `<td>${escapeHtml(engine.name)}</td>`,
      `<td>${escapeHtml(engine.method)}</td>`,
      `<td>${fmtCompare(engine.support)}</td>`,
      `<td>${fmtCompare(engine.confidence)}</td>`,
      `<td>${fmtCompare(engine.lift)}</td>`,
      `<td>${interval}</td>`,
      `<td>${fmtCompare(engine.variables)}</td>`,
      `<td>${fmtCompare(engine.constraints)}</td>`,
      `<td>${fmtCompare(engine.durationSeconds)}</td>`,
      `<td>${engine.allMatch ? "Igual" : "Diferente"}</td>`,
      `</tr>`,
    ].join("");
  });

  solverCompareStatus.textContent = result.comparison.allMatch ? "Resultados iguais" : "Diferencas encontradas";
  solverDemo.innerHTML = [
    `<h3>Solver executado com sucesso</h3>`,
    `<p>Consulta enviada ao solver separado: <strong>P(${targetText} | ${conditionsText})</strong>.</p>`,
    `<div class="solver-demo-grid">`,
    `<div><span>Solver</span><strong>HiGHS</strong><small>via scipy.optimize.linprog</small></div>`,
    `<div><span>Transformacao</span><strong>Charnes-Cooper</strong><small>razao condicional para LP</small></div>`,
    `<div><span>Modelo</span><strong>${fmtCompare(linear.variables)}</strong><small>variaveis x_w</small></div>`,
    `<div><span>Restricoes</span><strong>${fmtCompare(linear.constraints)}</strong><small>marginais, pares e regras Apriori</small></div>`,
    `</div>`,
    `<p>Resultado do solver separado padrao: <strong>${intervalTextValue}</strong>. O painel abaixo compara esse resultado com o calculo principal do projeto e com os 3 metodos executados.</p>`,
    `<p><strong>Solvers executados:</strong> SciPy HiGHS, HiGHS Dual Simplex e HiGHS Interior Point. Gurobi, lp_solve e cuPDLP-C ficam documentados como comparacao tecnica.</p>`,
    `<p><strong>Tempo de execucao:</strong> ${escapeHtml(timing.message || "Tempo de execucao indisponivel para comparacao.")}</p>`,
  ].join("");
  solverComparison.innerHTML = [
    `<p>${escapeHtml(result.message)}</p>`,
    engineRows.length
      ? `<h3>Comparacao entre 3 metodos de solver executados</h3><p>Os tres metodos abaixo foram executados de verdade com exatamente o mesmo A e o mesmo B escolhidos na interface. O HiGHS-IPM vem do projeto principal; HiGHS automatico e Dual Simplex sao executados pelo script separado <strong>scripts/solve_query.py</strong>.</p><table><thead><tr><th>Solver</th><th>Metodo SciPy</th><th>Suporte</th><th>Confianca</th><th>Lift</th><th>Intervalo</th><th>Variaveis</th><th>Restricoes</th><th>Tempo (s)</th><th>Status</th></tr></thead><tbody>${engineRows.join("")}</tbody></table>`
      : "",
    catalogRows.length
      ? `<h3>Solvers considerados</h3><table><thead><tr><th>Solver</th><th>Status</th><th>Comparacao</th></tr></thead><tbody>${catalogRows.join("")}</tbody></table>`
      : "",
    `<h3>Comparacao numerica executada</h3>`,
    `<table><thead><tr><th>Metrica</th><th>Projeto</th><th>Solver separado</th><th>Diferenca</th><th>Status</th></tr></thead><tbody>`,
    rows.join(""),
    `</tbody></table>`,
  ].join("");
}

function renderMathModel(result) {
  const aLabel = escapeHtml(eventLabel([result.target]));
  const bLabel = escapeHtml(eventLabel(result.conditions));
  const abLabel = escapeHtml(eventLabel([...result.conditions, result.target]));
  const variableCount = result.linear?.variables || "n";
  const constraintCount = result.linear?.constraints || "-";
  const interval = result.linear?.ok
    ? `${fmt(result.linear.lower)} <= P(A | B) <= ${fmt(result.linear.upper)}`
    : `Nao resolvido: ${escapeHtml(result.linear?.error || "motivo nao informado")}`;
  const learnedRules = result.learnedAssociationRules || [];
  const mining = result.aprioriMining || {};
  const releasedRule = ruleFromResult(result);
  const releasedRuleText = releasedRule
    ? `<code>${eventLabel(releasedRule.antecedent)} -> ${eventLabel(releasedRule.consequent)}; suporte=${fmt(releasedRule.support)}, confianca=${fmt(releasedRule.confidence)}, lift=${fmt(releasedRule.lift)}</code>`
    : `<code>A consulta ${eventLabel(result.conditions)} -> ${eventLabel([result.target])} nao foi gerada pelo Apriori; as medidas de regra ficam vazias.</code>`;
  const learnedRuleItems = learnedRules.length
    ? learnedRules
        .map((rule) => `<code>${eventLabel(rule.antecedent)} -> ${eventLabel(rule.consequent)}; sup=${fmt(rule.support)}, conf=${fmt(rule.confidence)}, lift=${fmt(rule.lift)}</code>`)
        .join("")
    : "<code>Nenhuma regra Apriori foi gerada.</code>";

  mathModel.innerHTML = [
    `<section class="math-block"><h3>1. Preparacao da base</h3><p>O projeto le o dataset de recomendacao de culturas, transforma atributos numericos em faixas categoricas e deixa cada registro pronto para perguntas do tipo <strong>P(A | B)</strong>. Nesta consulta, A = <strong>${aLabel}</strong> e B = <strong>${bLabel}</strong>.</p><div class="constraint-list"><code>A = ${aLabel}</code><code>B = ${bLabel}</code><code>A e B = ${abLabel}</code></div></section>`,
    `<section class="math-block"><h3>2. Mundos possiveis Omega</h3><p>Cada combinacao categorica observada forma um mundo <strong>w</strong> de Ω, preservando sua contagem. O programa cria uma variavel <strong>x<sub>w</sub></strong> para a massa de probabilidade desse mundo.</p><div class="constraint-list"><code>x<sub>w</sub> >= 0, para todo w em Omega</code><code>|Omega| = ${mining.omegaWorlds ?? variableCount}</code><code>sum<sub>w em Omega</sub> x<sub>w</sub> = 1</code></div></section>`,
    `<section class="math-block"><h3>3. Apriori</h3><p>Ω e fornecido ao Apriori como conjunto de transacoes ponderadas. O algoritmo encontrou <strong>${mining.frequentItemsets ?? "-"}</strong> itemsets frequentes e gerou <strong>${mining.ruleCount ?? learnedRules.length}</strong> regras. O suporte minimo e apenas o criterio de frequencia do Apriori; confianca e lift nao filtram qualidade.</p><div class="constraint-list"><code>suporte minimo = ${fmt(mining.minSupport)}</code><code>confianca minima = ${fmt(mining.minConfidence)}</code><code>tamanho maximo do itemset = ${mining.maxItemsetSize ?? "-"}</code></div></section>`,
    `<section class="math-block"><h3>4. Medidas da regra</h3><p>Suporte e confianca sao probabilidades usadas para construir restricoes. Lift fica apenas como descricao da associacao; nao representa acuracia e nao e usado para filtrar regras.</p><div class="constraint-list"><code>suporte(R -> S) = P(R e S)</code><code>confianca(R -> S) = P(R e S) / P(R)</code><code>lift(R -> S) = confianca / P(S) [somente descritivo]</code>${releasedRuleText}</div></section>`,
    `<section class="math-block"><h3>5. Regras convertidas em restricoes</h3><p>Todas as regras geradas entram na formulacao. O suporte ancora a intersecao e a confianca vira duas desigualdades lineares. A ordem da lista e apenas deterministica, nao um ranking de qualidade.</p><div class="constraint-list"><code>Lsup <= P(R e S) <= Usup</code><code>P(R e S) - Uconf.P(R) <= 0</code><code>-P(R e S) + Lconf.P(R) <= 0</code>${learnedRuleItems}</div></section>`,
    `<section class="math-block"><h3>6. Marginais, pares e mundos da consulta</h3><p>As probabilidades marginais e conjuntas por pares sao somadas as restricoes Apriori. P(A), P(B) e P(A e B) abaixo servem somente para auditoria da base. Quando P(A e B) empirico e zero, o modelo acrescenta mundos possiveis com contagem zero para o programa linear calcular um limite superior pequeno, em vez de declarar o evento impossivel.</p><div class="constraint-list"><code>P(A) empirico = ${fmt(result.pA)}</code><code>P(B) empirico = ${fmt(result.pB)}</code><code>P(A e B) empirico = ${fmt(result.pAB)}</code><code>Mundos observados = ${result.linear?.observedWorldVariables ?? "-"}</code><code>Mundos completados para a consulta = ${result.linear?.queryCompletionWorlds ?? 0}</code><code>L <= sum(x<sub>w</sub> onde evento) <= U</code></div></section>`,
    `<section class="math-block"><h3>7. Consulta condicional</h3><p>A pergunta final continua sendo probabilistica: qual intervalo e possivel para A quando B ocorre, respeitando as evidencias da base e as regras liberadas?</p><code>P(A | B) = P(A e B) / P(B) = sum(x<sub>w</sub> onde ${abLabel}) / sum(x<sub>w</sub> onde ${bLabel})</code></section>`,
    `<section class="math-block"><h3>8. Charnes-Cooper</h3><p>Como P(A | B) e uma razao, o modelo aplica Charnes-Cooper para transformar o problema fracionario em programacao linear.</p><div class="constraint-list"><code>y<sub>w</sub> = x<sub>w</sub> / P(B)</code><code>t = 1 / P(B)</code><code>sum(y<sub>w</sub> onde B) = 1</code><code>A y - b t <= 0</code><code>sum<sub>w em W</sub> y<sub>w</sub> - t = 0</code></div></section>`,
    `<section class="math-block"><h3>9. Solver</h3><p>O HiGHS resolve dois programas lineares: um minimiza e outro maximiza a massa dos mundos que satisfazem A e B. A comparacao executa highs, highs-ds e highs-ipm para avaliar consistencia numerica e tempo.</p><div class="constraint-list"><code>min sum(y<sub>w</sub> onde ${abLabel})</code><code>max sum(y<sub>w</sub> onde ${abLabel})</code><code>restricoes lineares: ${constraintCount}</code></div><p><strong>Resultado:</strong> ${interval}.</p></section>`,
  ].join("");
}

async function runQuery() {
  // Botao "Consultar": envia A e B para /api/query e renderiza cards,
  // explicacao probabilistica, Apriori e formulacao matematica.
  runQueryButton.disabled = true;
  runQueryButton.textContent = "Resolvendo...";
  downloadReportLink.classList.add("is-hidden");
  supportValue.textContent = "-";
  confidenceValue.textContent = "-";
  liftValue.textContent = "-";
  countValue.textContent = "-";
  probabilityText.innerHTML = "<div>Calculando consulta...</div>";
  linearProgram.textContent = "";
  mathModel.innerHTML = "";
  const payload = buildPayload();

  try {
    const response = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await readJsonResponse(response);
    if (!response.ok || !result.ok) throw new Error(result.error || "Erro na consulta.");

    const queriedRule = result.queriedAssociationRule || null;
    const releasedRule = ruleFromResult(result);
    supportValue.textContent = releasedRule ? fmt(releasedRule.support) : "-";
    confidenceValue.textContent = releasedRule ? fmt(releasedRule.confidence) : "-";
    liftValue.textContent = releasedRule ? fmt(releasedRule.lift) : "-";
    countValue.textContent = `${result.countBoth}/${result.countBase}`;
    conditionalValue.textContent = result.linear?.ok
      ? `${fmtDetailed(result.linear.lower)} a ${fmtDetailed(result.linear.upper)}`
      : "não definido";

    let linearInterval = `<div><strong>Intervalo linear:</strong> solver indisponível.</div>`;
    if (result.linear?.ok) {
      linearInterval = `<div><strong>Resultado de P(A | B) calculado pelo programa linear:</strong> ${fmtDetailed(result.linear.lower)} <= P(A | B) <= ${fmtDetailed(result.linear.upper)}.</div>`;
    } else if (result.linear?.reason === "zero_denominator") {
      linearInterval = `<div><strong>Intervalo linear:</strong> não definido porque P(B)=0; nenhuma linha do dataset satisfaz todas as afirmações.</div>`;
    }

    const isZeroResult = Number(result.countBase) > 0 && Number(result.countBoth) === 0;
    const zeroResultNotice = isZeroResult
      ? `<div><strong>Auditoria empirica:</strong> existem ${result.countBase} registros que satisfazem B, mas nenhum deles tambem satisfaz A; portanto P(A e B) empirico = 0. Isso nao fixa P(A | B) em zero no programa linear. O solver acrescentou ${result.linear?.queryCompletionWorlds || 0} mundos possiveis de contagem zero e calculou o intervalo acima.</div>`
      : "";
    const releasedRuleNotice = releasedRule
      ? `<div><strong>Regra Apriori encontrada:</strong> os cards mostram as medidas retornadas pelo minerador.</div>`
      : `<div><strong>Consulta sem regra Apriori correspondente:</strong> as medidas ficam vazias, mas o solver continua usando as demais regras e restricoes globais.</div>`;
    const ruleMetricContent = releasedRule
      ? [
          `<div><strong>Suporte:</strong> ${fmt(releasedRule.support)} usado para restringir P(R e S).</div>`,
          `<div><strong>Confianca:</strong> ${fmt(releasedRule.confidence)} usada na restricao P(R e S) = c.P(R).</div>`,
          `<div><strong>Lift:</strong> ${fmt(releasedRule.lift)} apenas descritivo; nao mede acuracia e nao filtra regras.</div>`,
        ].join("")
      : `<div><strong>Suporte, confianca e lift nos cards:</strong> - (a consulta nao pertence a saida do Apriori).</div>`;
    const queriedRuleContent = queriedRule
      ? `<div><strong>Status da regra consultada:</strong> ${escapeHtml(queriedRule.reason || "-")}.</div>`
      : "";
    const conclusionContent = releasedRule
      ? result.conclusion
      : `A consulta ${eventLabel(result.conditions)} -> ${eventLabel([result.target])} nao foi gerada pelo Apriori. O intervalo e inferido com as restricoes globais, sem fabricar uma regra para preencher os cards.`;
    const ruleLabel = releasedRule ? "Regra Apriori" : "Consulta sem regra Apriori";
    const learnedRuleCount = result.aprioriMining?.ruleCount || 0;
    const learnedRuleRows = (result.learnedAssociationRules || []).map((rule, index) => {
      return [
        `<tr>`,
        `<td>${index + 1}</td>`,
        `<td>${escapeHtml(eventLabel(rule.antecedent))} -> ${escapeHtml(eventLabel(rule.consequent))}</td>`,
        `<td>${fmt(rule.support)}</td>`,
        `<td>${fmt(rule.confidence)}</td>`,
        `<td>${fmt(rule.lift)}</td>`,
        `</tr>`,
      ].join("");
    });
    const learnedRuleContent = learnedRuleCount
      ? [
          `<div>O Apriori gerou ${learnedRuleCount} regras e todas foram incorporadas ao programa linear. A tabela mostra apenas a pre-visualizacao devolvida pela API.</div>`,
          `<table class="learned-rules-table"><thead><tr><th>#</th><th>Regra Apriori</th><th>Suporte</th><th>Confianca</th><th>Lift descritivo</th></tr></thead><tbody>`,
          learnedRuleRows.join(""),
          `</tbody></table>`,
        ].join("")
      : "Nenhuma regra Apriori foi gerada com o suporte minimo configurado.";
    probabilityText.innerHTML = [
      zeroResultNotice,
      releasedRuleNotice,
      `<div><strong>Uso correto das medidas:</strong> suporte e confianca alimentam restricoes; lift fica somente como descricao da associacao.</div>`,
      `<div><strong>${ruleLabel}:</strong> se ${eventLabel(result.conditions)}, entao ${eventLabel([result.target])}.</div>`,
      ruleMetricContent,
      queriedRuleContent,
      `<div class="learned-rules-block"><strong>Regras Apriori no PL:</strong> ${learnedRuleContent}</div>`,
      `<div><strong>Valores somente para auditoria da base:</strong> P(A)=${fmt(result.pA)}, P(B)=${fmt(result.pB)} e P(A e B)=${fmt(result.pAB)}. Estes valores empiricos nao sao apresentados como resultado do programa linear e a consulta nao injeta sua resposta no objetivo.</div>`,
      linearInterval,
      `<div><strong>Conclusao:</strong> ${conclusionContent}</div>`,
    ].join("");
    linearProgram.textContent = result.linearProgram;
    renderMathModel(result);
  } catch (error) {
    probabilityText.innerHTML = `<div><strong>Erro:</strong> ${error.message}</div>`;
    linearProgram.textContent = "";
    mathModel.innerHTML = "";
  } finally {
    runQueryButton.disabled = false;
    runQueryButton.textContent = "Consultar";
  }
}

async function generateReport() {
  // Botao "Gerar PDF": usa a mesma consulta escolhida na tela para criar o
  // relatorio da consulta atual no backend.
  generateReportButton.disabled = true;
  generateReportButton.textContent = "Gerando...";
  const payload = buildPayload();
  const reportWindow = window.open("", "_blank", "noopener");

  try {
    const response = await fetch("/api/report/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await readJsonResponse(response);
    if (!response.ok || !result.ok) throw new Error(result.error || "Erro ao gerar relatório.");
    const reportUrl = `${result.reportUrl}?t=${Date.now()}`;
    downloadReportLink.href = reportUrl;
    downloadReportLink.classList.remove("is-hidden");
    if (reportWindow) {
      reportWindow.location = reportUrl;
    }
  } catch (error) {
    if (reportWindow) reportWindow.close();
    probabilityText.innerHTML += `<div><strong>Relatório:</strong> ${error.message}</div>`;
  } finally {
    generateReportButton.disabled = false;
    generateReportButton.textContent = "Gerar PDF";
  }
}

async function generateSolverReport() {
  // Botao "Resolver e Gerar Relatorio": roda a comparacao com o solver
  // separado e abre o PDF comparativo.
  generateSolverReportButton.disabled = true;
  generateSolverReportButton.textContent = "Gerando comparativo...";
  solverCompareStatus.textContent = "Resolvendo solver e gerando relatorio";
  solverComparison.innerHTML = "";
  downloadSolverReportLink.classList.add("is-hidden");
  const reportWindow = window.open("", "_blank", "noopener");
  const payload = buildPayload();

  try {
    await prepareSolverComparison(payload);
    const response = await fetch("/api/report/solver-comparison", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await readJsonResponse(response);
    if (!response.ok || !result.ok) throw new Error(result.error || "Erro ao gerar comparativo.");
    renderSolverComparison({
      ...result,
      message: "Solver separado resolvido com os mesmos dados da consulta. Relatorio comparativo gerado com sucesso.",
    });
    const reportUrl = `${result.reportUrl}?t=${Date.now()}`;
    downloadSolverReportLink.href = reportUrl;
    downloadSolverReportLink.classList.remove("is-hidden");
    if (reportWindow) {
      reportWindow.location = reportUrl;
    }
  } catch (error) {
    if (reportWindow) reportWindow.close();
    solverCompareStatus.textContent = "Erro no relatorio";
    solverComparison.innerHTML = `<p><strong>Relatorio comparativo:</strong> ${escapeHtml(error.message)}</p>`;
  } finally {
    generateSolverReportButton.disabled = false;
    generateSolverReportButton.textContent = "Resolver e Gerar Relatorio de Comparacao";
  }
}

async function postSolverStage(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const result = await readJsonResponse(response);
  if (!response.ok || !result.ok) throw new Error(result.error || "Erro ao executar etapa do solver.");
  return result;
}

async function prepareSolverComparison(payload) {
  // Cada metodo fica em uma requisicao propria. Assim os tres solvers sao
  // realmente executados sem que uma unica requisicao ultrapasse o limite do Render.
  solverCompareStatus.textContent = "Etapa 1/3: HiGHS-IPM principal";
  await postSolverStage("/api/query", payload);
  solverCompareStatus.textContent = "Etapa 2/3: HiGHS automatico separado";
  await postSolverStage("/api/solver/run", { ...payload, solverMethod: "highs" });
  solverCompareStatus.textContent = "Etapa 3/3: Dual Simplex separado";
  await postSolverStage("/api/solver/run", { ...payload, solverMethod: "highs-ds" });
}

async function compareSolver() {
  // Botao "Resolver Solver Separado": compara o resultado principal com o
  // script independente scripts/solve_query.py sem gerar PDF.
  compareSolverButton.disabled = true;
  compareSolverButton.textContent = "Resolvendo...";
  solverCompareStatus.textContent = "Executando solver";
  solverComparison.innerHTML = "";
  const payload = buildPayload();

  try {
    await prepareSolverComparison(payload);
    const response = await fetch("/api/solver/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await readJsonResponse(response);
    if (!response.ok || !result.ok) throw new Error(result.error || "Erro ao comparar solver.");
    renderSolverComparison(result);
  } catch (error) {
    solverCompareStatus.textContent = "Erro";
    solverComparison.innerHTML = `<p><strong>Erro:</strong> ${escapeHtml(error.message)}</p>`;
  } finally {
    compareSolverButton.disabled = false;
    compareSolverButton.textContent = "Resolver Solver Separado";
  }
}

async function generateFullLinearProgram(downloadAfter = false) {
  // Exporta o mesmo modelo numerico usado pelo linprog: objetivos, matrizes,
  // lados direitos, limites, mapeamento de variaveis e SHA-256 de auditoria.
  const originalGenerateText = generateFullLinearProgramButton.textContent;
  generateFullLinearProgramButton.disabled = true;
  generateFullLinearProgramButton.textContent = "Gerando modelo...";
  downloadFullLinearProgramLink.classList.add("is-disabled");
  downloadFullLinearProgramLink.setAttribute("aria-disabled", "true");
  const originalDownloadText = downloadFullLinearProgramLink.textContent;
  if (downloadAfter) downloadFullLinearProgramLink.textContent = "Gerando TXT...";

  try {
    const response = await fetch("/api/linear-program/full", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayload()),
    });
    const result = await readJsonResponse(response);
    if (!response.ok || !result.ok) throw new Error(result.error || "Erro ao gerar o modelo auditavel.");
    const downloadUrl = `${result.downloadUrl || result.fileUrl}?t=${Date.now()}`;
    downloadFullLinearProgramLink.href = downloadUrl;
    downloadFullLinearProgramLink.classList.remove("is-disabled");
    downloadFullLinearProgramLink.setAttribute("aria-disabled", "false");
    const auditSummary = [
      result.message || "Modelo numerico auditavel gerado.",
      `SHA-256: ${result.modelDigest || "-"}`,
      `Variaveis do solver: ${result.solverVariables ?? "-"}`,
      `Restricoes: ${result.constraints ?? "-"}`,
    ].join("\n");
    linearProgram.textContent = `${linearProgram.textContent}\n\n${auditSummary}\nTXT ${downloadAfter ? "baixado" : "pronto para baixar"}.`;
    if (downloadAfter) {
      const temporaryLink = document.createElement("a");
      temporaryLink.href = downloadUrl;
      temporaryLink.download = "programa_linear_completo.txt";
      document.body.appendChild(temporaryLink);
      temporaryLink.click();
      temporaryLink.remove();
    }
  } catch (error) {
    linearProgram.textContent = `${linearProgram.textContent}\n\nErro ao gerar o modelo auditavel: ${error.message}`;
  } finally {
    generateFullLinearProgramButton.disabled = false;
    generateFullLinearProgramButton.textContent = originalGenerateText;
    downloadFullLinearProgramLink.textContent = originalDownloadText;
    downloadFullLinearProgramLink.classList.remove("is-disabled");
    downloadFullLinearProgramLink.setAttribute("aria-disabled", "false");
  }
}

async function boot() {
  const response = await fetch("/api/metadata");
  if (!response.ok) throw new Error("API Python indisponivel.");
  const metadata = await readJsonResponse(response);
  domains = metadata.domains;
  labels = metadata.labels;
  attributes = metadata.attributes;

  fillSelect(targetAttribute, attributes, (item) => labels[item] || item);
  targetAttribute.value = attributes.includes("label") ? "label" : attributes[attributes.length - 1];
  fillValueSelect(targetAttribute, targetValue);

  targetAttribute.addEventListener("change", () => fillValueSelect(targetAttribute, targetValue));
  addConditionButton.addEventListener("click", () => createConditionRow());
  runQueryButton.addEventListener("click", runQuery);
  generateReportButton.addEventListener("click", generateReport);
  compareSolverButton.addEventListener("click", compareSolver);
  generateSolverReportButton.addEventListener("click", generateSolverReport);
  generateFullLinearProgramButton.addEventListener("click", () => generateFullLinearProgram(false));
  downloadFullLinearProgramLink.addEventListener("click", (event) => {
    event.preventDefault();
    if (downloadFullLinearProgramLink.classList.contains("is-disabled")) return;
    generateFullLinearProgram(true);
  });
  tabButtons.forEach((button) => {
    button.addEventListener("click", () => setActiveTab(button));
  });

  const firstCondition = firstAttributeExcept(targetAttribute.value);
  createConditionRow(firstCondition, domains[firstCondition]?.[0]);
  const secondCondition = attributes.find((item) => item !== targetAttribute.value && item !== firstCondition);
  if (secondCondition) createConditionRow(secondCondition, domains[secondCondition]?.[0]);
  datasetStatus.textContent = `${metadata.total} registros carregados`;
  runQuery();
}

boot().catch((error) => {
  datasetStatus.textContent = "Python nao conectado";
  probabilityText.innerHTML = [
    "<div><strong>Erro:</strong> a interface carregou, mas nao conseguiu falar com o backend Python.</div>",
    "<div>Inicie o servidor com <strong>python app.py</strong> e acesse <strong>http://localhost:1000</strong>.</div>",
  ].join("");
  linearProgram.textContent = error.message;
  if (mathModel) mathModel.innerHTML = "";
});
