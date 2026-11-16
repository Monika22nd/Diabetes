const form = document.querySelector("#assessment-form");
const statusText = document.querySelector("#model-status");
const gauge = document.querySelector("#gauge");
const scoreValue = document.querySelector("#score-value");
const scoreLabel = document.querySelector("#score-label");
const thresholdText = document.querySelector("#threshold-text");
const flagsList = document.querySelector("#rule-flags");
const evidenceList = document.querySelector("#evidence-list");
const traceSteps = [...document.querySelectorAll("#trace span")];

let modelBundle;
let knowledgeBase = [];

const presets = {
  low: {
    age: 28, gender: "Female", bmi: 22.4, HbA1c_level: 4.9,
    blood_glucose_level: 95, smoking_history: "never",
    hypertension: 0, heart_disease: 0,
  },
  high: {
    age: 68, gender: "Male", bmi: 36.2, HbA1c_level: 7.1,
    blood_glucose_level: 220, smoking_history: "former",
    hypertension: 1, heart_disease: 1,
  },
};

function relu(values) {
  return values.map((value) => Math.max(0, value));
}

function sigmoid(value) {
  if (value >= 0) return 1 / (1 + Math.exp(-value));
  const exp = Math.exp(value);
  return exp / (1 + exp);
}

function dense(input, layer) {
  return layer.weights.map((row, index) => {
    let total = layer.bias[index];
    for (let i = 0; i < row.length; i += 1) total += row[i] * input[i];
    return total;
  });
}

function transform(record) {
  const prep = modelBundle.preprocessing;
  const vector = [];
  prep.numeric_features.forEach((name, index) => {
    vector.push((Number(record[name]) - prep.means[index]) / prep.scales[index]);
  });
  prep.categorical_features.forEach((name, index) => {
    prep.categories[index].forEach((category) => {
      vector.push(String(record[name]) === String(category) ? 1 : 0);
    });
  });
  return vector;
}

function predict(record) {
  let output = transform(record);
  modelBundle.network.layers.forEach((layer, index) => {
    output = dense(output, layer);
    if (index < modelBundle.network.layers.length - 1) output = relu(output);
  });
  return sigmoid(output[0]);
}

function ruleFlags(record) {
  const flags = [];
  if (record.bmi >= 30) flags.push("BMI is in a high range in this educational rule set.");
  if (record.HbA1c_level >= 6.5) flags.push("HbA1c requires interpretation by a qualified health professional.");
  if (record.blood_glucose_level >= 200) flags.push("Blood glucose requires interpretation by a qualified health professional.");
  if (record.hypertension === 1) flags.push("Hypertension is present in the submitted record.");
  if (record.heart_disease === 1) flags.push("Heart disease is present in the submitted record.");
  return flags;
}

function retrieveEvidence(question) {
  const words = question.toLowerCase().split(/\W+/).filter((word) => word.length > 2);
  return [...knowledgeBase]
    .map((item) => {
      const text = `${item.topic} ${item.title} ${item.text}`.toLowerCase();
      const score = words.reduce((total, word) => total + Number(text.includes(word)), 0);
      return { ...item, score };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, 3);
}

function readRecord() {
  const data = new FormData(form);
  return {
    age: Number(data.get("age")),
    gender: data.get("gender"),
    bmi: Number(data.get("bmi")),
    HbA1c_level: Number(data.get("HbA1c_level")),
    blood_glucose_level: Number(data.get("blood_glucose_level")),
    smoking_history: data.get("smoking_history"),
    hypertension: Number(data.get("hypertension")),
    heart_disease: Number(data.get("heart_disease")),
    question: data.get("question"),
  };
}

function validate(record) {
  const ranges = {
    age: [0, 120], bmi: [5, 100], HbA1c_level: [2, 20], blood_glucose_level: [40, 500],
  };
  for (const [field, [min, max]] of Object.entries(ranges)) {
    if (!Number.isFinite(record[field]) || record[field] < min || record[field] > max) {
      throw new Error(`${field} must be between ${min} and ${max}.`);
    }
  }
}

function render(record, probability) {
  const threshold = modelBundle.threshold;
  const score = probability * 100;
  gauge.style.setProperty("--score", score.toFixed(1));
  scoreValue.textContent = `${score.toFixed(1)}%`;
  scoreLabel.textContent = probability >= threshold ? "Above the demo threshold" : "Below the demo threshold";
  thresholdText.textContent = `Browser model threshold: ${(threshold * 100).toFixed(1)}%. This is not a clinical threshold.`;

  const flags = ruleFlags(record);
  flagsList.innerHTML = flags.length
    ? flags.map((flag) => `<li>${flag}</li>`).join("")
    : "<li>No deterministic review flags were raised.</li>";

  const evidence = retrieveEvidence(record.question);
  evidenceList.innerHTML = evidence.map((item) => `
    <article class="evidence-card">
      <a href="${item.url}" target="_blank" rel="noreferrer">${item.title}</a>
      <p>${item.text}</p>
    </article>
  `).join("");

  traceSteps.forEach((step, index) => {
    setTimeout(() => step.classList.add("done"), index * 90);
  });
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  try {
    if (!modelBundle) throw new Error("The model has not finished loading.");
    traceSteps.forEach((step) => step.classList.remove("done"));
    const record = readRecord();
    validate(record);
    render(record, predict(record));
  } catch (error) {
    statusText.textContent = error.message;
  }
});

document.querySelectorAll("[data-preset]").forEach((button) => {
  button.addEventListener("click", () => {
    const preset = presets[button.dataset.preset];
    Object.entries(preset).forEach(([name, value]) => {
      form.elements[name].value = value;
    });
    form.requestSubmit();
  });
});

Promise.all([
  fetch("assets/model_bundle.json").then((response) => response.json()),
  fetch("assets/knowledge_base.json").then((response) => response.json()),
]).then(([bundle, knowledge]) => {
  modelBundle = bundle;
  knowledgeBase = knowledge;
  statusText.textContent = "Browser model loaded. Input data stays on this page.";
  document.querySelector("#metric-auc").textContent = bundle.metrics.roc_auc.toFixed(3);
  document.querySelector("#metric-recall").textContent = bundle.metrics.recall.toFixed(3);
  document.querySelector("#metric-f1").textContent = bundle.metrics.f1.toFixed(3);
  form.requestSubmit();
}).catch((error) => {
  statusText.textContent = `Could not load demo assets: ${error.message}`;
});
