const datasetStatus = document.querySelector("#datasetStatus");
const conditionList = document.querySelector("#conditionList");
const addConditionButton = document.querySelector("#addCondition");
const runQueryButton = document.querySelector("#runQuery");
const targetAttribute = document.querySelector("#targetAttribute");
const targetValue = document.querySelector("#targetValue");
const supportValue = document.querySelector("#supportValue");
const confidenceValue = document.querySelector("#confidenceValue");
const liftValue = document.querySelector("#liftValue");
const countValue = document.querySelector("#countValue");
const probabilityText = document.querySelector("#probabilityText");
const linearProgram = document.querySelector("#linearProgram");

const numericAttributes = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"];
const labelMap = {
  N: "Nitrogenio",
  P: "Fosforo",
  K: "Potassio",
  temperature: "Temperatura",
  humidity: "Umidade",
  ph: "pH",
  rainfall: "Chuva",
  label: "Cultura",
};

let rows = [];
let categoricalRows = [];
let domains = {};
let thresholds = {};

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines.shift().split(",");
  return lines.map((line) => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((header, index) => [header, values[index]]));
  });
}

function quantile(values, q) {
  const sorted = [...values].sort((a, b) => a - b);
  const position = (sorted.length - 1) * q;
  const base = Math.floor(position);
  const rest = position - base;
  if (sorted[base + 1] === undefined) return sorted[base];
  return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
}

function buildThresholds(rawRows) {
  numericAttributes.forEach((attribute) => {
    const values = rawRows.map((row) => Number(row[attribute])).filter(Number.isFinite);
    thresholds[attribute] = {
      low: quantile(values, 1 / 3),
      high: quantile(values, 2 / 3),
    };
  });
}

function categoryFor(attribute, value) {
  const number = Number(value);
  if (attribute === "ph") {
    if (number < 6) return "acido";
    if (number <= 7.5) return "neutro";
    return "alcalino";
  }
  const threshold = thresholds[attribute];
  if (number <= threshold.low) return "baixo";
  if (number <= threshold.high) return "medio";
  return "alto";
}

function categorizeRows(rawRows) {
  return rawRows.map((row) => {
    const categorized = {};
    numericAttributes.forEach((attribute) => {
      categorized[attribute] = categoryFor(attribute, row[attribute]);
    });
    categorized.label = row.label;
    return categorized;
  });
}

function buildDomains() {
  domains = {};
  Object.keys(categoricalRows[0]).forEach((attribute) => {
    domains[attribute] = [...new Set(categoricalRows.map((row) => row[attribute]))].sort();
  });
}

function fillSelect(select, values) {
  select.innerHTML = "";
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = labelMap[value] || value;
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
  fillSelect(attributeSelect, Object.keys(domains));
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

function matches(row, conditions) {
  return conditions.every((condition) => row[condition.attribute] === condition.value);
}

function probability(conditions) {
  if (!conditions.length) return 1;
  return categoricalRows.filter((row) => matches(row, conditions)).length / categoricalRows.length;
}

function count(conditions) {
  if (!conditions.length) return categoricalRows.length;
  return categoricalRows.filter((row) => matches(row, conditions)).length;
}

function fmt(value) {
  if (!Number.isFinite(value)) return "-";
  return value.toFixed(3);
}

function interval(value) {
  const rounded = Math.round(value * 1000) / 1000;
  return {
    lower: Math.max(0, rounded - 0.001),
    upper: Math.min(1, rounded + 0.001),
  };
}

function eventLabel(conditions) {
  if (!conditions.length) return "verdadeiro";
  return conditions.map((condition) => `${condition.attribute}=${condition.value}`).join(", ");
}

function linearExpression(conditions) {
  if (!conditions.length) return "1";
  return `soma(x_w para w onde ${eventLabel(conditions)})`;
}

function renderLinearProgram(target, base, pA, pB, pAB) {
  const iA = interval(pA);
  const iB = interval(pB);
  const iAB = interval(pAB);
  const numerator = linearExpression([...base, target]);
  const denominator = linearExpression(base);

  return [
    "Variaveis:",
    "  x_w >= 0 para cada mundo possivel w da base categorizada",
    "",
    "Restricao de normalizacao:",
    "  soma(x_w) = 1",
    "",
    "Restricoes extraidas da base:",
    `  ${fmt(iA.lower)} <= P(A) = ${linearExpression([target])} <= ${fmt(iA.upper)}`,
    `  ${fmt(iB.lower)} <= P(B) = ${denominator} <= ${fmt(iB.upper)}`,
    `  ${fmt(iAB.lower)} <= P(A e B) = ${numerator} <= ${fmt(iAB.upper)}`,
    "",
    "Funcao objetivo da consulta:",
    `  P(A | B) = P(A e B) / P(B)`,
    "",
    "Forma linear usada no relatorio:",
    `  minimizar ${numerator}, fixando P(B) dentro do intervalo observado`,
    `  maximizar ${numerator}, fixando P(B) dentro do intervalo observado`,
    "",
    "Resultado empirico:",
    `  A = ${eventLabel([target])}`,
    `  B = ${eventLabel(base)}`,
  ].join("\n");
}

function runQuery() {
  const base = readConditions();
  const target = { attribute: targetAttribute.value, value: targetValue.value };
  const both = [...base, target];
  const pA = probability([target]);
  const pB = probability(base);
  const pAB = probability(both);
  const confidence = pB > 0 ? pAB / pB : Number.NaN;
  const lift = pA > 0 && pB > 0 ? confidence / pA : Number.NaN;
  const baseCount = count(base);
  const bothCount = count(both);

  supportValue.textContent = fmt(pAB);
  confidenceValue.textContent = fmt(confidence);
  liftValue.textContent = fmt(lift);
  countValue.textContent = `${bothCount}/${baseCount}`;

  probabilityText.innerHTML = [
    `<div><strong>Regra:</strong> se ${eventLabel(base)}, então ${eventLabel([target])}.</div>`,
    `<div><strong>Suporte:</strong> ${fmt(pAB)} representa P(A e B), a proporção da base que satisfaz pergunta e afirmações.</div>`,
    `<div><strong>Confiança/Precisão:</strong> ${fmt(confidence)} representa P(A | B), isto é, entre os registros que satisfazem B, quantos também satisfazem A.</div>`,
    `<div><strong>Probabilidades marginais:</strong> P(A)=${fmt(pA)} e P(B)=${fmt(pB)}.</div>`,
  ].join("");
  linearProgram.textContent = renderLinearProgram(target, base, pA, pB, pAB);
}

async function boot() {
  const response = await fetch("data/Crop_recommendation.csv");
  const text = await response.text();
  rows = parseCsv(text);
  buildThresholds(rows);
  categoricalRows = categorizeRows(rows);
  buildDomains();

  fillSelect(targetAttribute, Object.keys(domains));
  targetAttribute.value = "label";
  fillValueSelect(targetAttribute, targetValue);
  targetValue.value = "rice";

  targetAttribute.addEventListener("change", () => fillValueSelect(targetAttribute, targetValue));
  addConditionButton.addEventListener("click", () => createConditionRow());
  runQueryButton.addEventListener("click", runQuery);

  createConditionRow("ph", "acido");
  createConditionRow("rainfall", "alto");
  datasetStatus.textContent = `${rows.length} registros carregados`;
  runQuery();
}

boot().catch((error) => {
  datasetStatus.textContent = "Erro ao carregar dataset";
  linearProgram.textContent = error.message;
});
