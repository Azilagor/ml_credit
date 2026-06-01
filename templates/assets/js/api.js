const API_BASE_URL = "http://127.0.0.1:8000/api/v1";
const USE_MOCK_FALLBACK = true;

function getToken() {
  return localStorage.getItem("access_token");
}

function setToken(token) {
  localStorage.setItem("access_token", token);
}

function clearToken() {
  localStorage.removeItem("access_token");
  localStorage.removeItem("current_user");
}

function setCurrentUser(user) {
  localStorage.setItem("current_user", JSON.stringify(user));
}

function getCurrentUser() {
  try { return JSON.parse(localStorage.getItem("current_user") || "null"); }
  catch { return null; }
}

async function apiRequest(path, options = {}) {
  const token = getToken();
  const headers = options.headers || {};

  if (!(options.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const response = await fetch(`${API_BASE_URL}${path}`, { ...options, headers });
  const contentType = response.headers.get("content-type") || "";
  const data = contentType.includes("application/json") ? await response.json() : await response.text();

  if (!response.ok) {
    const message = typeof data === "object" ? (data.detail || data.message || "Request failed") : data;
    throw new Error(message);
  }
  return data;
}

function showError(id, message) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = message;
  el.style.display = "block";
}

function hideError(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = "";
  el.style.display = "none";
}

function formatPercent(value) {
  const number = Number(value || 0);
  return `${(number * 100).toFixed(1)}%`;
}

function decisionBadge(decision) {
  const value = String(decision || "pending").toLowerCase();
  const cls = value.includes("approve") ? "badge-approved" : value.includes("reject") ? "badge-rejected" : "badge-pending";
  return `<span class="badge ${cls}">${decision || "pending"}</span>`;
}

const mock = {
  user: { id: 1, email: "user@example.com", full_name: "Demo User", role: "user" },
  admin: { id: 2, email: "admin@example.com", full_name: "Admin User", role: "admin" },
  predictions: [
    { id: 101, full_name: "Aruzhan S.", created_at: "2026-02-01", final_decision: "approved", ensemble_probability: 0.24 },
    { id: 102, full_name: "Daniyar K.", created_at: "2026-02-02", final_decision: "rejected", ensemble_probability: 0.71 },
    { id: 103, full_name: "Madina T.", created_at: "2026-02-03", final_decision: "approved", ensemble_probability: 0.33 }
  ],
  predictionResponse(payload) {
    const risk = Math.min(0.95, Math.max(0.05, (Number(payload.loan_amount) / Math.max(Number(payload.income) * 8, 1)) + (payload.employment_status === "unemployed" ? 0.25 : 0.05)));
    const lr = Math.min(0.98, risk + 0.04);
    const rf = Math.max(0.02, risk - 0.03);
    const ensemble = (lr + rf) / 2;
    const decision = ensemble >= 0.5 ? "rejected" : "approved";
    return {
      application_id: Math.floor(Math.random() * 10000),
      models: [
        { model_name: "Logistic Regression", probability: lr, decision: lr >= 0.5 ? "rejected" : "approved" },
        { model_name: "Random Forest", probability: rf, decision: rf >= 0.5 ? "rejected" : "approved" }
      ],
      ensemble_probability: ensemble,
      final_decision: decision,
      explanation: decision === "rejected" ? "The request was rejected mainly due to high loan amount compared with income and employment risk." : "The request was approved because income and loan amount show acceptable risk level.",
      top_features: [
        { feature: "loan_amount", impact: 0.34 },
        { feature: "income", impact: -0.21 },
        { feature: "employment_status", impact: payload.employment_status === "unemployed" ? 0.28 : 0.08 }
      ]
    };
  },
  adminStats: { total_users: 120, total_requests: 540, approval_rate: 0.62, average_probability: 0.41 },
  users: [
    { id: 1, full_name: "Demo User", email: "user@example.com", role: "user" },
    { id: 2, full_name: "Admin User", email: "admin@example.com", role: "admin" },
    { id: 3, full_name: "Bank Analyst", email: "analyst@example.com", role: "analyst" }
  ]
};
