document.addEventListener("DOMContentLoaded", async () => {
  const user = requireAuth("admin");
  if (!user) return;
  await loadAdminData();
});

async function loadAdminData() {
  let stats, users, apps;
  try { stats = await apiRequest("/admin/statistics"); } catch { stats = mock.adminStats; }
  try { users = await apiRequest("/admin/users"); } catch { users = mock.users; }
  try { apps = await apiRequest("/admin/applications"); } catch { apps = mock.predictions; }

  document.getElementById("totalUsers").textContent = stats.total_users;
  document.getElementById("totalRequests").textContent = stats.total_requests;
  document.getElementById("approvalRate").textContent = formatPercent(stats.approval_rate);
  document.getElementById("avgProbability").textContent = formatPercent(stats.average_probability);

  renderUsers(users);
  renderApplications(apps);
  renderAdminCharts(apps);
}

function renderUsers(users) {
  const body = document.getElementById("usersBody");
  body.innerHTML = users.map(u => `
    <tr>
      <td>${u.id}</td>
      <td>${u.full_name || "-"}</td>
      <td>${u.email}</td>
      <td>${u.role}</td>
    </tr>
  `).join("");
}

function renderApplications(apps) {
  const body = document.getElementById("adminApplicationsBody");
  const filter = document.getElementById("decisionFilter").value;
  const rows = filter ? apps.filter(a => String(a.final_decision || a.decision).toLowerCase() === filter) : apps;
  body.innerHTML = rows.map(a => `
    <tr>
      <td>#${a.id || a.application_id}</td>
      <td>${a.full_name || a.name || "-"}</td>
      <td>${a.created_at || "-"}</td>
      <td>${decisionBadge(a.final_decision || a.decision)}</td>
      <td>${formatPercent(a.ensemble_probability || a.pd || a.probability_of_default)}</td>
      <td>${a.model_version || "v1.0"}</td>
    </tr>
  `).join("");
}

function renderAdminCharts(apps) {
  const approved = apps.filter(a => String(a.final_decision || a.decision).toLowerCase().includes("approve")).length;
  const rejected = apps.length - approved;
  const canvas = document.getElementById("decisionChart");
  if (!canvas || !window.Chart) return;
  new Chart(canvas, {
    type: "doughnut",
    data: {
      labels: ["Approved", "Rejected"],
      datasets: [{ data: [approved, rejected] }]
    },
    options: { responsive: true }
  });
}
