document.addEventListener("DOMContentLoaded", () => {
  const user = requireAuth();
  if (!user) return;

  const form = document.getElementById("batchForm");
  const result = document.getElementById("batchResult");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideError("pageError");
    const fileInput = document.getElementById("csvFile");
    if (!fileInput.files.length) {
      showError("pageError", "Please select a CSV file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    document.getElementById("processingInfo").style.display = "block";

    try {
      const data = await apiRequest("/batch/upload", { method: "POST", body: formData });
      renderBatch(data);
    } catch (err) {
      if (USE_MOCK_FALLBACK) {
        renderBatch({ batch_id: 501, total_records: 500, approved: 312, rejected: 188, average_pd: 0.41 });
      } else {
        showError("pageError", err.message);
      }
    } finally {
      document.getElementById("processingInfo").style.display = "none";
    }
  });

  function renderBatch(data) {
    result.style.display = "block";
    document.getElementById("batchId").textContent = `#${data.batch_id}`;
    document.getElementById("batchTotal").textContent = data.total_records;
    document.getElementById("batchApproved").textContent = data.approved;
    document.getElementById("batchRejected").textContent = data.rejected;
    document.getElementById("batchAvgPd").textContent = formatPercent(data.average_pd);
    document.getElementById("downloadLink").href = `${API_BASE_URL}/batch/${data.batch_id}/download`;
    renderBatchChart(data);
  }
});

function renderBatchChart(data) {
  const canvas = document.getElementById("batchChart");
  if (!canvas || !window.Chart) return;
  if (window.batchChartInstance) window.batchChartInstance.destroy();
  window.batchChartInstance = new Chart(canvas, {
    type: "bar",
    data: {
      labels: ["Approved", "Rejected"],
      datasets: [{ label: "Records", data: [data.approved, data.rejected] }]
    },
    options: { responsive: true, scales: { y: { beginAtZero: true } } }
  });
}
