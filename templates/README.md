# Credit Scoring Frontend

Static frontend for a machine-learning credit scoring system.

Stack:
- HTML
- CSS
- Vanilla JavaScript
- Fetch API
- Chart.js

Pages:
- `index.html` - entry page
- `login.html` - user login
- `register.html` - user registration
- `dashboard.html` - user credit request dashboard
- `admin.html` - admin dashboard
- `batch.html` - CSV batch analysis

Backend API base URL is configured in:

```js
assets/js/api.js
```

Default:

```js
const API_BASE_URL = "http://127.0.0.1:8000/api/v1";
```

Run locally:

```bash
python3 -m http.server 5173
```

Open:

```text
http://127.0.0.1:5173
```

The frontend includes mock fallback data when backend is unavailable.
