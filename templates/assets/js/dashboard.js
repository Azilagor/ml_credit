document.addEventListener("DOMContentLoaded", async () => {
  const user = requireAuth();
  if (!user) return;

  const form = document.getElementById("predictionForm");
  const resultBox = document.getElementById("resultBox");
  const historyBody = document.getElementById("historyBody");

  await loadHistory();

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideError("pageError");
    const payload = Object.fromEntries(new FormData(form).entries());
    payload.age = Number(payload.age);
    payload.income = Number(payload.income);
    payload.loan_amount = Number(payload.loan_amount);

    try {
      const data = await apiRequest("/applications/predict", {
        method: "POST",
        body: JSON.stringify(payload)
      });
      renderResult(data);
      await loadHistory();
    } catch (err) {
      if (USE_MOCK_FALLBACK) renderResult(mock.predictionResponse(payload));
      else showError("pageError", err.message);
    }
  });

  async function loadHistory() {
    let rows;
    try {
      rows = await apiRequest("/applications/my");
    } catch {
      rows = mock.predictions;
    }
    historyBody.innerHTML = rows.map(row => `
      <tr>
        <td>#${row.id || row.application_id}</td>
        <td>${row.full_name || "Current user"}</td>
        <td>${row.created_at || "-"}</td>
        <td>${decisionBadge(row.final_decision || row.decision)}</td>
        <td>${formatPercent(row.ensemble_probability || row.probability_of_default || row.pd)}</td>
        <td><span class="row-action" onclick="showExplanation(${row.id || row.application_id})">View</span></td>
      </tr>
    `).join("");
  }

  function renderResult(data) {
    resultBox.style.display = "block";
    document.getElementById("finalDecision").innerHTML = decisionBadge(data.final_decision);
    document.getElementById("ensembleProbability").textContent = formatPercent(data.ensemble_probability);
    document.getElementById("explanationText").textContent = data.explanation || "No explanation provided.";
    document.getElementById("modelList").innerHTML = (data.models || []).map(m => `
      <div class="model-item">
        <strong>${m.model_name}</strong>
        <span>${formatPercent(m.probability)} — ${m.decision}</span>
      </div>
    `).join("");
    document.getElementById("featureList").innerHTML = (data.top_features || []).map(f => `
      <li>${f.feature}: impact ${Number(f.impact).toFixed(2)}</li>
    `).join("");
    renderModelChart(data.models || []);
  }
});

async function showExplanation(id) {
  let data;
  try {
    data = await apiRequest(`/applications/${id}/explanation`);
  } catch {
    data = { explanation: "Rejected mainly due to high loan amount and low income stability.", top_features: mock.predictionResponse({loan_amount: 1000, income: 500, employment_status: "unemployed"}).top_features };
  }
  alert(`${data.explanation}\n\nTop features:\n${(data.top_features || []).map(f => `- ${f.feature}: ${f.impact}`).join("\n")}`);
}

function renderModelChart(models) {
  const canvas = document.getElementById("modelChart");
  if (!canvas || !window.Chart) return;
  if (window.modelChartInstance) window.modelChartInstance.destroy();
  window.modelChartInstance = new Chart(canvas, {
    type: "bar",
    data: {
      labels: models.map(m => m.model_name),
      datasets: [{ label: "Default probability", data: models.map(m => Number(m.probability || 0)) }]
    },
    options: { responsive: true, scales: { y: { beginAtZero: true, max: 1 } } }
  });
}
