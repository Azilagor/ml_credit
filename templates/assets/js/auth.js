function requireAuth(role = null) {
  const token = getToken();
  const user = getCurrentUser();
  if (!token || !user) {
    window.location.href = "login.html";
    return null;
  }
  if (role && user.role !== role) {
    window.location.href = "dashboard.html";
    return null;
  }
  const badge = document.getElementById("userBadge");
  if (badge) badge.textContent = `${user.full_name || user.email} (${user.role})`;
  return user;
}

function logout() {
  clearToken();
  window.location.href = "login.html";
}

document.addEventListener("DOMContentLoaded", () => {
  const loginForm = document.getElementById("loginForm");
  const registerForm = document.getElementById("registerForm");
  const logoutBtn = document.getElementById("logoutBtn");

  if (logoutBtn) logoutBtn.addEventListener("click", logout);

  if (loginForm) {
    loginForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      hideError("authError");
      const payload = Object.fromEntries(new FormData(loginForm).entries());
      try {
        const data = await apiRequest("/auth/login", {
          method: "POST",
          body: JSON.stringify(payload)
        });
        setToken(data.access_token || data.token);
        setCurrentUser(data.user || { email: payload.email, full_name: payload.email, role: payload.email.includes("admin") ? "admin" : "user" });
        window.location.href = (data.user && data.user.role === "admin") ? "admin.html" : "dashboard.html";
      } catch (err) {
        if (USE_MOCK_FALLBACK) {
          const user = payload.email.includes("admin") ? mock.admin : mock.user;
          setToken("mock-token");
          setCurrentUser(user);
          window.location.href = user.role === "admin" ? "admin.html" : "dashboard.html";
        } else {
          showError("authError", err.message);
        }
      }
    });
  }

  if (registerForm) {
    registerForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      hideError("authError");
      const payload = Object.fromEntries(new FormData(registerForm).entries());
      try {
        const data = await apiRequest("/auth/register", {
          method: "POST",
          body: JSON.stringify(payload)
        });
        setToken(data.access_token || data.token);
        setCurrentUser(data.user || { email: payload.email, full_name: payload.full_name, role: "user" });
        window.location.href = "dashboard.html";
      } catch (err) {
        if (USE_MOCK_FALLBACK) {
          setToken("mock-token");
          setCurrentUser({ id: 10, email: payload.email, full_name: payload.full_name, role: "user" });
          window.location.href = "dashboard.html";
        } else {
          showError("authError", err.message);
        }
      }
    });
  }
});
