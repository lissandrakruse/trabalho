const datasetStatus = document.querySelector("#datasetStatus");
const conditionList = document.querySelector("#conditionList");
const addConditionButton = document.querySelector("#addCondition");
const runQueryButton = document.querySelector("#runQuery");
const targetAttribute = document.querySelector("#targetAttribute");
const targetValue = document.querySelector("#targetValue");
const generateReportButton = document.querySelector("#generateReport");
const downloadReportLink = document.querySelector("#downloadReport");
const supportValue = document.querySelector("#supportValue");
const confidenceValue = document.querySelector("#confidenceValue");
const liftValue = document.querySelector("#liftValue");
const countValue = document.querySelector("#countValue");
const probabilityText = document.querySelector("#probabilityText");
const linearProgram = document.querySelector("#linearProgram");

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

function eventLabel(conditions) {
  if (!conditions.length) return "verdadeiro";
  return conditions.map((condition) => `${condition.attribute}=${condition.value}`).join(", ");
}

async function runQuery() {
  runQueryButton.disabled = true;
  runQueryButton.textContent = "Resolvendo...";
  const payload = {
    conditions: readConditions(),
    target: {
      attribute: targetAttribute.value,
      value: targetValue.value,
    },
  };

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

    probabilityText.innerHTML = [
      `<div><strong>Regra:</strong> se ${eventLabel(result.conditions)}, então ${eventLabel([result.target])}.</div>`,
      `<div><strong>Suporte:</strong> ${fmt(result.support)} representa P(A e B).</div>`,
      `<div><strong>Confiança/Precisão:</strong> ${fmt(result.confidence)} representa P(A | B).</div>`,
      `<div><strong>Lift:</strong> ${fmt(result.lift)} compara a regra com a probabilidade marginal de A.</div>`,
      `<div><strong>Marginais:</strong> P(A)=${fmt(result.pA)} e P(B)=${fmt(result.pB)}.</div>`,
      linearInterval,
      `<div><strong>Conclusão:</strong> ${result.conclusion}</div>`,
    ].join("");
    linearProgram.textContent = result.linearProgram;
  } catch (error) {
    probabilityText.innerHTML = `<div><strong>Erro:</strong> ${error.message}</div>`;
    linearProgram.textContent = "";
  } finally {
    runQueryButton.disabled = false;
    runQueryButton.textContent = "Consultar";
  }
}

async function generateReport() {
  generateReportButton.disabled = true;
  generateReportButton.textContent = "Gerando...";
  const payload = {
    conditions: readConditions(),
    target: {
      attribute: targetAttribute.value,
      value: targetValue.value,
    },
  };

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
    window.open(reportUrl, "_blank", "noopener");
  } catch (error) {
    probabilityText.innerHTML += `<div><strong>Relatório:</strong> ${error.message}</div>`;
  } finally {
    generateReportButton.disabled = false;
    generateReportButton.textContent = "Gerar PDF";
  }
}

async function boot() {
  const response = await fetch("/api/metadata");
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

  const firstCondition = firstAttributeExcept(targetAttribute.value);
  createConditionRow(firstCondition, domains[firstCondition]?.[0]);
  const secondCondition = attributes.find((item) => item !== targetAttribute.value && item !== firstCondition);
  if (secondCondition) createConditionRow(secondCondition, domains[secondCondition]?.[0]);
  datasetStatus.textContent = `${metadata.total} registros carregados`;
  runQuery();
}

boot().catch((error) => {
  datasetStatus.textContent = "Erro ao carregar dataset";
  linearProgram.textContent = error.message;
});
