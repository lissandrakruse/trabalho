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
const supportValue = document.querySelector("#supportValue");
const confidenceValue = document.querySelector("#confidenceValue");
const liftValue = document.querySelector("#liftValue");
const countValue = document.querySelector("#countValue");
const probabilityText = document.querySelector("#probabilityText");
const classificationMetrics = document.querySelector("#classificationMetrics");
const solverCompareStatus = document.querySelector("#solverCompareStatus");
const solverDemo = document.querySelector("#solverDemo");
const solverComparison = document.querySelector("#solverComparison");
const linearProgram = document.querySelector("#linearProgram");
const mathModel = document.querySelector("#mathModel");
const tabButtons = document.querySelectorAll(".tab-button");
const tabPanels = document.querySelectorAll(".tab-panel");

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

function fmtCompare(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  const number = Number(value);
  return Number.isInteger(number) ? String(number) : number.toFixed(3);
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

function intervalText(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  const rounded = Number(value).toFixed(3);
  const lower = Math.max(0, Number(rounded) - 0.001).toFixed(3);
  const upper = Math.min(1, Number(rounded) + 0.001).toFixed(3);
  return `${lower} <= ${rounded} <= ${upper}`;
}

function buildPayload() {
  return {
    conditions: readConditions(),
    target: {
      attribute: targetAttribute.value,
      value: targetValue.value,
    },
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
    : escapeHtml(linear.error || "Intervalo nao calculado");
  const labels = {
    pA: "P(A)",
    pB: "P(B)",
    support: "P(A e B)",
    confidence: "P(A | B)",
    lift: "Lift",
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
    `<div><span>Restricoes</span><strong>${fmtCompare(linear.constraints)}</strong><small>marginais, conjunta e normalizacao</small></div>`,
    `</div>`,
    `<p>Resultado do solver separado padrao: <strong>${intervalTextValue}</strong>. O painel abaixo compara esse resultado com o calculo principal do projeto e com os 3 metodos executados.</p>`,
    `<p><strong>Solvers executados:</strong> SciPy HiGHS, HiGHS Dual Simplex e HiGHS Interior Point. Gurobi, lp_solve e cuPDLP-C ficam documentados como comparacao tecnica.</p>`,
    `<p><strong>Tempo de execucao:</strong> ${escapeHtml(timing.message || "Tempo de execucao indisponivel para comparacao.")}</p>`,
  ].join("");
  solverComparison.innerHTML = [
    `<p>${escapeHtml(result.message)}</p>`,
    engineRows.length
      ? `<h3>3 solvers executados com os parametros da interface</h3><table><thead><tr><th>Solver</th><th>Suporte</th><th>Confianca</th><th>Lift</th><th>Intervalo</th><th>Variaveis</th><th>Restricoes</th><th>Tempo (s)</th><th>Status</th></tr></thead><tbody>${engineRows.join("")}</tbody></table>`
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

function renderClassificationMetrics(classification) {
  if (!classification) {
    classificationMetrics.innerHTML = "";
    return;
  }

  classificationMetrics.innerHTML = [
    `<article><span>Acuracia</span><strong>${fmt(classification.accuracy)}</strong></article>`,
    `<article><span>Precisao</span><strong>${fmt(classification.precision)}</strong></article>`,
    `<article><span>Recall</span><strong>${fmt(classification.recall)}</strong></article>`,
    `<article><span>F1-score</span><strong>${fmt(classification.f1)}</strong></article>`,
    `<p>${escapeHtml(classification.interpretation)}</p>`,
    `<p>Matriz binaria: VP=${classification.truePositive}, FP=${classification.falsePositive}, FN=${classification.falseNegative}, VN=${classification.trueNegative}.</p>`,
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
    : "Nao calculado para esta consulta.";

  mathModel.innerHTML = [
    `<section class="math-block"><h3>1. Mundos e variáveis</h3><p>Cada mundo possível <strong>w</strong> representa uma combinação categorizada dos atributos do dataset. O modelo usa <strong>x<sub>w</sub></strong> como a probabilidade atribuída a esse mundo.</p><code>x<sub>w</sub> >= 0, para todo w em W &nbsp; (|W| = ${variableCount})</code></section>`,
    `<section class="math-block"><h3>2. Normalização</h3><p>A distribuição reconstruída precisa somar 1.</p><code>sum<sub>w em W</sub> x<sub>w</sub> = 1</code></section>`,
    `<section class="math-block"><h3>3. Evidências da base</h3><p>As frequências observadas viram restrições intervalares. A consulta atual define A = <strong>${aLabel}</strong> e B = <strong>${bLabel}</strong>.</p><div class="constraint-list"><code>P(A): ${intervalText(result.pA)}</code><code>P(B): ${intervalText(result.pB)}</code><code>P(A e B): ${intervalText(result.support)}</code></div><p>O solver também incorpora as probabilidades marginais dos valores categorizados de cada atributo.</p></section>`,
    `<section class="math-block"><h3>4. Consulta condicional</h3><p>A pergunta é uma razão entre a probabilidade conjunta e a probabilidade do evento condicionante.</p><code>P(A | B) = P(A e B) / P(B) = sum(x<sub>w</sub> onde ${abLabel}) / sum(x<sub>w</sub> onde ${bLabel})</code></section>`,
    `<section class="math-block"><h3>5. Charnes-Cooper</h3><p>Para resolver a razão como programação linear, o modelo aplica a transformação de Charnes-Cooper.</p><div class="constraint-list"><code>y<sub>w</sub> = x<sub>w</sub> / P(B)</code><code>t = 1 / P(B)</code><code>sum(y<sub>w</sub> onde B) = 1</code><code>A y - b t <= 0</code><code>sum<sub>w em W</sub> y<sub>w</sub> - t = 0</code></div></section>`,
    `<section class="math-block"><h3>6. Objetivo</h3><p>O limite inferior minimiza a massa dos mundos que satisfazem A e B; o superior maximiza a mesma expressão.</p><div class="constraint-list"><code>min sum(y<sub>w</sub> onde ${abLabel})</code><code>max sum(y<sub>w</sub> onde ${abLabel})</code></div><p><strong>Resultado:</strong> ${interval}. Restrições lineares no solver: ${constraintCount}.</p></section>`,
  ].join("");
}

async function runQuery() {
  runQueryButton.disabled = true;
  runQueryButton.textContent = "Resolvendo...";
  downloadReportLink.classList.add("is-hidden");
  const payload = buildPayload();

  try {
    const response = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok || !result.ok) throw new Error(result.error || "Erro na consulta.");

    supportValue.textContent = fmt(result.support);
    confidenceValue.textContent = fmt(result.confidence);
    liftValue.textContent = fmt(result.lift);
    countValue.textContent = `${result.countBoth}/${result.countBase}`;

    let linearInterval = `<div><strong>Intervalo linear:</strong> solver indisponível.</div>`;
    if (result.linear?.ok) {
      linearInterval = `<div><strong>Intervalo linear:</strong> ${fmt(result.linear.lower)} <= P(A | B) <= ${fmt(result.linear.upper)}.</div>`;
    } else if (result.linear?.reason === "zero_denominator") {
      linearInterval = `<div><strong>Intervalo linear:</strong> não calculado porque P(B)=0; nenhuma linha do dataset satisfaz todas as afirmações.</div>`;
    }

    const ruleLabel = result.support > 0 && result.confidence > 0 ? "Regra candidata" : "Consulta sem regra aprendida";
    probabilityText.innerHTML = [
      `<div><strong>${ruleLabel}:</strong> se ${eventLabel(result.conditions)}, então ${eventLabel([result.target])}.</div>`,
      `<div><strong>Suporte:</strong> ${fmt(result.support)} representa P(A e B).</div>`,
      `<div><strong>Confiança/Precisão:</strong> ${fmt(result.confidence)} representa P(A | B).</div>`,
      `<div><strong>Lift:</strong> ${fmt(result.lift)} compara a regra com a probabilidade marginal de A.</div>`,
      `<div><strong>Marginais:</strong> P(A)=${fmt(result.pA)} e P(B)=${fmt(result.pB)}.</div>`,
      linearInterval,
      `<div><strong>Conclusão:</strong> ${result.conclusion}</div>`,
    ].join("");
    linearProgram.textContent = result.linearProgram;
    renderClassificationMetrics(result.classification);
    renderMathModel(result);
  } catch (error) {
    probabilityText.innerHTML = `<div><strong>Erro:</strong> ${error.message}</div>`;
    classificationMetrics.innerHTML = "";
    linearProgram.textContent = "";
    mathModel.innerHTML = "";
  } finally {
    runQueryButton.disabled = false;
    runQueryButton.textContent = "Consultar";
  }
}

async function generateReport() {
  generateReportButton.disabled = true;
  generateReportButton.textContent = "Gerando...";
  const payload = buildPayload();

  try {
    const response = await fetch("/api/report/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok || !result.ok) throw new Error(result.error || "Erro ao gerar relatório.");
    const reportUrl = `${result.reportUrl}?t=${Date.now()}`;
    downloadReportLink.href = reportUrl;
    downloadReportLink.classList.remove("is-hidden");
    window.open(reportUrl, "_blank", "noopener");
  } catch (error) {
    probabilityText.innerHTML += `<div><strong>Relatório:</strong> ${error.message}</div>`;
  } finally {
    generateReportButton.disabled = false;
    generateReportButton.textContent = "Gerar PDF";
  }
}

async function generateSolverReport() {
  generateSolverReportButton.disabled = true;
  generateSolverReportButton.textContent = "Gerando comparativo...";
  solverCompareStatus.textContent = "Resolvendo solver e gerando relatorio";
  solverComparison.innerHTML = "";
  downloadSolverReportLink.classList.add("is-hidden");

  try {
    const response = await fetch("/api/report/solver-comparison", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayload()),
    });
    const result = await response.json();
    if (!response.ok || !result.ok) throw new Error(result.error || "Erro ao gerar comparativo.");
    renderSolverComparison({
      ...result,
      message: "Solver separado resolvido com os mesmos dados da consulta. Relatorio comparativo gerado com sucesso.",
    });
    const reportUrl = `${result.reportUrl}?t=${Date.now()}`;
    downloadSolverReportLink.href = reportUrl;
    downloadSolverReportLink.classList.remove("is-hidden");
    window.open(reportUrl, "_blank", "noopener");
  } catch (error) {
    solverCompareStatus.textContent = "Erro no relatorio";
    solverComparison.innerHTML = `<p><strong>Relatorio comparativo:</strong> ${escapeHtml(error.message)}</p>`;
  } finally {
    generateSolverReportButton.disabled = false;
    generateSolverReportButton.textContent = "Resolver e Gerar Relatorio de Comparacao";
  }
}

async function compareSolver() {
  compareSolverButton.disabled = true;
  compareSolverButton.textContent = "Resolvendo...";
  solverCompareStatus.textContent = "Executando solver";
  solverComparison.innerHTML = "";

  try {
    const response = await fetch("/api/solver/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayload()),
    });
    const result = await response.json();
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

async function generateFullLinearProgram() {
  generateFullLinearProgramButton.disabled = true;
  generateFullLinearProgramButton.textContent = "Gerando LP...";
  downloadFullLinearProgramLink.classList.add("is-disabled");
  downloadFullLinearProgramLink.setAttribute("aria-disabled", "true");

  try {
    const response = await fetch("/api/linear-program/full", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayload()),
    });
    const result = await response.json();
    if (!response.ok || !result.ok) throw new Error(result.error || "Erro ao gerar LP completo.");
    downloadFullLinearProgramLink.href = `${result.downloadUrl || result.fileUrl}?t=${Date.now()}`;
    downloadFullLinearProgramLink.classList.remove("is-disabled");
    downloadFullLinearProgramLink.setAttribute("aria-disabled", "false");
    linearProgram.textContent = `${linearProgram.textContent}\n\nArquivo TXT completo gerado e pronto para baixar.`;
  } catch (error) {
    linearProgram.textContent = `${linearProgram.textContent}\n\nErro ao gerar LP completo: ${error.message}`;
  } finally {
    generateFullLinearProgramButton.disabled = false;
    generateFullLinearProgramButton.textContent = "Gerar LP Completo";
  }
}

async function boot() {
  const response = await fetch("/api/metadata");
  if (!response.ok) throw new Error("API Python indisponivel.");
  const metadata = await response.json();
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
  generateFullLinearProgramButton.addEventListener("click", generateFullLinearProgram);
  downloadFullLinearProgramLink.addEventListener("click", (event) => {
    if (downloadFullLinearProgramLink.classList.contains("is-disabled")) {
      event.preventDefault();
    }
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
