"""
Credit Scoring System — FastAPI single-file application
========================================================
Запуск:
    pip install -r requirements.txt
    python main.py seed          # создать БД + demo данные + обучить модели
    uvicorn main:app --reload

Swagger: http://127.0.0.1:8000/docs
"""


# IMPORTS

import csv
import hashlib
import io
import json
import os
import sys
import traceback
import warnings
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import (Cookie, Depends, FastAPI, File, Form, HTTPException,
                     Request, UploadFile, status)
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                                RedirectResponse, StreamingResponse)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey, Integer,
                         String, Text, create_engine, func)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker

warnings.filterwarnings("ignore")


# PATHS & CONFIG

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
DB_PATH = BASE_DIR / "credit_scoring.db"

for d in [TEMPLATES_DIR, STATIC_DIR, MODELS_DIR, DATA_DIR,
          STATIC_DIR / "css", STATIC_DIR / "js"]:
    d.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_PATH}"

SECRET_KEY = "credit_scoring_secret_2024"
SESSION_EXPIRE_HOURS = 24


# DATABASE

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(200), nullable=False)
    email = Column(String(200), unique=True, index=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    role = Column(String(20), default="USER")  # USER / ADMIN
    created_at = Column(DateTime, default=datetime.utcnow)
    applications = relationship("CreditApplication", back_populates="user")


class CreditApplication(Base):
    __tablename__ = "credit_applications"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    # questionnaire fields
    monthly_income = Column(Float)
    existing_debt = Column(Float)
    loan_amount = Column(Float)
    loan_term = Column(Integer)       # months
    employment_status = Column(String(50))
    employment_years = Column(Float)
    age = Column(Integer)
    credit_history_score = Column(Integer)  # 300-850
    previous_delays = Column(Integer)        # number of delays
    loan_purpose = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(30), default="pending")
    user = relationship("User", back_populates="applications")
    predictions = relationship("PredictionResult", back_populates="application")


class PredictionResult(Base):
    __tablename__ = "prediction_results"
    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(Integer, ForeignKey("credit_applications.id"))
    model_name = Column(String(100))
    probability_default = Column(Float)
    probability_delinquency = Column(Float)
    risk_level = Column(String(20))
    expected_credit_loss = Column(Float)
    decision = Column(String(30))
    created_at = Column(DateTime, default=datetime.utcnow)
    application = relationship("CreditApplication", back_populates="predictions")
    explanations = relationship("FeatureExplanation", back_populates="prediction")


class FeatureExplanation(Base):
    __tablename__ = "feature_explanations"
    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("prediction_results.id"))
    feature_name = Column(String(100))
    feature_value = Column(Float)
    effect_direction = Column(String(30))  # increases_risk / decreases_risk
    importance_value = Column(Float)
    explanation_text = Column(Text)
    prediction = relationship("PredictionResult", back_populates="explanations")


class BatchAnalysis(Base):
    __tablename__ = "batch_analyses"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(300))
    total_records = Column(Integer)
    approved_count = Column(Integer, default=0)
    rejected_count = Column(Integer, default=0)
    manual_review_count = Column(Integer, default=0)
    avg_pd = Column(Float, default=0.0)
    model_stats_json = Column(Text, default="")   # JSON: per-model aggregated stats
    created_at = Column(DateTime, default=datetime.utcnow)


class UserSession(Base):
    __tablename__ = "user_sessions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    token = Column(String(256), unique=True, index=True)
    expires_at = Column(DateTime)


Base.metadata.create_all(bind=engine)


def _run_migrations():
    """Add missing columns to existing databases (safe ALTER TABLE)."""
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(batch_analyses)")
    existing = {row[1] for row in cur.fetchall()}
    if "model_stats_json" not in existing:
        cur.execute("ALTER TABLE batch_analyses ADD COLUMN model_stats_json TEXT DEFAULT ''")
        conn.commit()
        print("[DB] Migration: added model_stats_json to batch_analyses")
    conn.close()

_run_migrations()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



# AUTH HELPERS

def hash_password(password: str) -> str:
    return hashlib.sha256((password + SECRET_KEY).encode()).hexdigest()


def make_session_token(user_id: int) -> str:
    raw = f"{user_id}-{datetime.utcnow().isoformat()}-{SECRET_KEY}"
    return hashlib.sha256(raw.encode()).hexdigest()


def create_session(db: Session, user_id: int) -> str:
    token = make_session_token(user_id)
    expires = datetime.utcnow() + timedelta(hours=SESSION_EXPIRE_HOURS)
    sess = UserSession(user_id=user_id, token=token, expires_at=expires)
    db.add(sess)
    db.commit()
    return token


def get_current_user(request: Request, db: Session = Depends(get_db)) -> Optional[User]:
    token = request.cookies.get("session_token")
    if not token:
        return None
    sess = db.query(UserSession).filter(
        UserSession.token == token,
        UserSession.expires_at > datetime.utcnow()
    ).first()
    if not sess:
        return None
    return db.query(User).filter(User.id == sess.user_id).first()


def require_login(request: Request, db: Session = Depends(get_db)) -> User:
    user = get_current_user(request, db)
    if not user:
        raise HTTPException(status_code=302, headers={"Location": "/login"})
    return user


def require_admin(request: Request, db: Session = Depends(get_db)) -> User:
    user = require_login(request, db)
    if user.role != "ADMIN":
        raise HTTPException(status_code=403, detail="Admin only")
    return user



# ML: FEATURE ENGINEERING

FEATURE_NAMES = [
    "monthly_income", "existing_debt", "loan_amount", "loan_term",
    "employment_years", "age", "credit_history_score", "previous_delays",
    "debt_to_income", "loan_to_income", "employment_status_enc",
    "loan_purpose_enc",
]

EMPLOYMENT_MAP = {
    "employed": 0, "self-employed": 1, "unemployed": 2,
    "retired": 3, "student": 4, "part-time": 5,
}

PURPOSE_MAP = {
    "personal": 0, "business": 1, "education": 2,
    "mortgage": 3, "auto": 4, "medical": 5, "other": 6,
}

FEATURE_HUMAN = {
    "monthly_income": "Monthly Income",
    "existing_debt": "Existing Debt",
    "loan_amount": "Requested Loan Amount",
    "loan_term": "Loan Term (months)",
    "employment_years": "Years Employed",
    "age": "Age",
    "credit_history_score": "Credit History Score",
    "previous_delays": "Previous Payment Delays",
    "debt_to_income": "Debt-to-Income Ratio",
    "loan_to_income": "Loan-to-Income Ratio",
    "employment_status_enc": "Employment Status",
    "loan_purpose_enc": "Loan Purpose",
}

FEATURE_EXPLANATIONS_NEGATIVE = {
    "debt_to_income": "A high debt-to-income ratio means a large portion of your income is already committed to debt payments, which significantly increases default risk.",
    "previous_delays": "Previous payment delays are one of the strongest predictors of future defaults. Each delay recorded reduces approval chances.",
    "loan_to_income": "Your requested loan amount is large relative to your monthly income, making repayment more difficult.",
    "existing_debt": "A high existing debt burden leaves less room for additional loan obligations.",
    "employment_status_enc": "Your employment status (unemployed or unstable) increases the risk of income disruption.",
    "credit_history_score": "A low credit history score reflects past credit problems, which is a key risk indicator.",
    "monthly_income": "Lower income limits your capacity to service additional debt.",
    "employment_years": "Short employment history suggests income instability.",
    "loan_amount": "The loan amount requested is high, increasing the bank's exposure.",
    "loan_term": "Loan term affects total interest and repayment burden.",
    "age": "Age is a factor in credit lifecycle assessment.",
    "loan_purpose_enc": "The loan purpose affects expected repayment discipline.",
}

FEATURE_EXPLANATIONS_POSITIVE = {
    "debt_to_income": "Your debt-to-income ratio is low, meaning most of your income is free for new obligations.",
    "previous_delays": "You have no (or very few) previous payment delays, which is a strong positive signal.",
    "loan_to_income": "The loan amount is manageable relative to your income.",
    "existing_debt": "Your current debt burden is low, leaving room for a new loan.",
    "employment_status_enc": "Stable employment is a strong positive factor in credit assessment.",
    "credit_history_score": "A good credit history score reflects reliable past repayment behaviour.",
    "monthly_income": "Your income level is sufficient to service the requested loan.",
    "employment_years": "Long and stable employment history is a positive creditworthiness signal.",
    "loan_amount": "The requested loan amount is proportionate to your profile.",
    "loan_term": "The loan term is structured in a way that balances repayment comfort.",
    "age": "Your age profile is within the standard creditworthiness range.",
    "loan_purpose_enc": "The stated loan purpose is associated with disciplined repayment patterns.",
}


def extract_features(app: CreditApplication) -> np.ndarray:
    income = max(app.monthly_income or 1, 1)
    dti = (app.existing_debt or 0) / income
    lti = (app.loan_amount or 0) / income
    emp_enc = EMPLOYMENT_MAP.get(
        (app.employment_status or "employed").lower().strip(), 0)
    pur_enc = PURPOSE_MAP.get(
        (app.loan_purpose or "personal").lower().strip(), 0)
    return np.array([[
        app.monthly_income or 0,
        app.existing_debt or 0,
        app.loan_amount or 0,
        app.loan_term or 12,
        app.employment_years or 0,
        app.age or 30,
        app.credit_history_score or 600,
        app.previous_delays or 0,
        dti,
        lti,
        emp_enc,
        pur_enc,
    ]])


def features_from_dict(d: dict) -> np.ndarray:
    income = max(float(d.get("monthly_income", 1) or 1), 1)
    existing_debt = float(d.get("existing_debt", 0) or 0)
    loan_amount = float(d.get("loan_amount", 0) or 0)
    dti = existing_debt / income
    lti = loan_amount / income
    emp_enc = EMPLOYMENT_MAP.get(str(d.get("employment_status", "employed")).lower().strip(), 0)
    pur_enc = PURPOSE_MAP.get(str(d.get("loan_purpose", "personal")).lower().strip(), 0)
    return np.array([[
        income,
        existing_debt,
        loan_amount,
        float(d.get("loan_term", 12) or 12),
        float(d.get("employment_years", 0) or 0),
        float(d.get("age", 30) or 30),
        float(d.get("credit_history_score", 600) or 600),
        float(d.get("previous_delays", 0) or 0),
        dti,
        lti,
        emp_enc,
        pur_enc,
    ]])



# ML: TRAINING

def generate_synthetic_dataset(n: int = 2000) -> pd.DataFrame:
    """Generates a realistic synthetic credit dataset for training."""
    rng = np.random.RandomState(42)
    monthly_income = rng.lognormal(mean=10.2, sigma=0.5, size=n).clip(20000, 500000)
    existing_debt = monthly_income * rng.uniform(0, 0.8, size=n)
    loan_amount = monthly_income * rng.uniform(1, 15, size=n)
    loan_term = rng.choice([6, 12, 24, 36, 48, 60], size=n)
    employment_status = rng.choice(
        list(EMPLOYMENT_MAP.keys()),
        p=[0.55, 0.20, 0.10, 0.05, 0.05, 0.05],
        size=n,
    )
    employment_years = rng.exponential(scale=5, size=n).clip(0, 40)
    age = rng.randint(18, 70, size=n)
    credit_history_score = rng.randint(300, 851, size=n)
    previous_delays = rng.choice([0, 1, 2, 3, 4, 5], p=[0.55, 0.20, 0.12, 0.07, 0.04, 0.02], size=n)
    loan_purpose = rng.choice(list(PURPOSE_MAP.keys()), size=n)

    dti = existing_debt / monthly_income
    lti = loan_amount / monthly_income
    emp_enc = np.array([EMPLOYMENT_MAP[e] for e in employment_status])
    pur_enc = np.array([PURPOSE_MAP[p] for p in loan_purpose])

    # Logistic model for default
    log_odds = (
        -2.5
        + 3.0 * dti
        + 2.5 * lti
        + 1.8 * (previous_delays / 5.0)
        - 1.5 * ((credit_history_score - 300) / 550)
        - 0.8 * (employment_years / 40.0)
        + 0.5 * (emp_enc / 5.0)
        + rng.normal(0, 0.5, size=n)
    )
    prob_default = 1 / (1 + np.exp(-log_odds))
    default = (rng.uniform(size=n) < prob_default).astype(int)

    df = pd.DataFrame({
        "monthly_income": monthly_income,
        "existing_debt": existing_debt,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "employment_years": employment_years,
        "age": age,
        "credit_history_score": credit_history_score,
        "previous_delays": previous_delays,
        "debt_to_income": dti,
        "loan_to_income": lti,
        "employment_status_enc": emp_enc,
        "loan_purpose_enc": pur_enc,
        "default": default,
    })
    return df


def train_and_save_models():
    print("[ML] Generating synthetic training dataset...")
    df = generate_synthetic_dataset(3000)
    X = df[FEATURE_NAMES].values
    y = df["default"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models_def = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "neural_network": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=42,
            early_stopping=True, validation_fraction=0.1,
        ),
    }

    meta = {}
    for name, clf in models_def.items():
        print(f"[ML] Training {name}...")
        if name in ("logistic_regression", "neural_network"):
            clf.fit(X_train_sc, y_train)
            y_pred_prob = clf.predict_proba(X_test_sc)[:, 1]
        else:
            clf.fit(X_train, y_train)
            y_pred_prob = clf.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_pred_prob)
        acc = accuracy_score(y_test, (y_pred_prob >= 0.5).astype(int))
        meta[name] = {"auc": round(auc, 4), "accuracy": round(acc, 4)}
        print(f"  AUC={auc:.4f}  ACC={acc:.4f}")
        joblib.dump(clf, MODELS_DIR / f"{name}.pkl")

    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    with open(MODELS_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("[ML] All models saved.")
    return meta


def load_models():
    """Load all trained models from disk. Returns dict of models + scaler."""
    required = ["logistic_regression", "random_forest",
                "gradient_boosting", "neural_network", "scaler"]
    for name in required:
        suffix = ".pkl"
        path = MODELS_DIR / f"{name}{suffix}"
        if not path.exists():
            return None
    result = {}
    for name in required[:-1]:
        result[name] = joblib.load(MODELS_DIR / f"{name}.pkl")
    result["scaler"] = joblib.load(MODELS_DIR / "scaler.pkl")
    return result



# ML: RISK CALCULATION

LGD = 0.45  # Loss Given Default (Basel standard)

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "neural_network": "Neural Network (MLP)",
}


def calc_delinquency_risk(pd_val: float, dti: float, delays: int) -> float:
    base = pd_val * 0.7 + dti * 0.2 + min(delays / 10, 0.3) * 0.1
    return round(min(base, 1.0), 4)


def calc_risk_level(pd_val: float) -> str:
    if pd_val < 0.30:
        return "Low"
    elif pd_val < 0.60:
        return "Medium"
    return "High"


def calc_decision(pd_val: float) -> str:
    if pd_val < 0.30:
        return "Approved"
    elif pd_val < 0.60:
        return "Manual Review"
    return "Rejected"


def calc_ecl(pd_val: float, loan_amount: float) -> float:
    return round(pd_val * LGD * loan_amount, 2)


def get_feature_importances(models: dict, model_name: str) -> np.ndarray:
    """Get feature importances for tree models; zeros for others."""
    clf = models.get(model_name)
    if clf is None:
        return np.zeros(len(FEATURE_NAMES))
    if hasattr(clf, "feature_importances_"):
        return clf.feature_importances_
    if hasattr(clf, "coef_"):
        return np.abs(clf.coef_[0])
    return np.zeros(len(FEATURE_NAMES))


def predict_all_models(models: dict, features: np.ndarray, loan_amount: float,
                       dti: float, delays: int) -> dict:
    """Run all models and compute risk metrics."""
    scaler = models["scaler"]
    features_sc = scaler.transform(features)

    results = {}
    pds = []

    for name in ["logistic_regression", "random_forest",
                  "gradient_boosting", "neural_network"]:
        clf = models[name]
        if name in ("logistic_regression", "neural_network"):
            X_in = features_sc
        else:
            X_in = features
        pd_val = float(clf.predict_proba(X_in)[0, 1])
        delinq = calc_delinquency_risk(pd_val, dti, delays)
        level = calc_risk_level(pd_val)
        ecl = calc_ecl(pd_val, loan_amount)
        decision = calc_decision(pd_val)
        results[name] = {
            "display_name": MODEL_DISPLAY_NAMES[name],
            "probability_default": round(pd_val, 4),
            "probability_delinquency": round(delinq, 4),
            "risk_level": level,
            "expected_credit_loss": ecl,
            "decision": decision,
        }
        pds.append(pd_val)

    avg_pd = float(np.mean(pds))
    results["ensemble"] = {
        "display_name": "Ensemble (Average)",
        "probability_default": round(avg_pd, 4),
        "probability_delinquency": round(calc_delinquency_risk(avg_pd, dti, delays), 4),
        "risk_level": calc_risk_level(avg_pd),
        "expected_credit_loss": calc_ecl(avg_pd, loan_amount),
        "decision": calc_decision(avg_pd),
    }
    return results


def build_explanations(models: dict, features: np.ndarray,
                        feature_values: list) -> list:
    """
    Returns top factors as list of dicts with human-readable explanation.
    Uses average importance across tree models.
    """
    imp_rf = get_feature_importances(models, "random_forest")
    imp_gb = get_feature_importances(models, "gradient_boosting")
    avg_imp = (imp_rf + imp_gb) / 2.0

    # Normalise
    total = avg_imp.sum()
    if total > 0:
        avg_imp = avg_imp / total

    # Threshold values for high/low judgement
    HIGH_RISK_THRESHOLDS = {
        "debt_to_income": 0.40,
        "loan_to_income": 5.0,
        "previous_delays": 1,
        "credit_history_score": 600,  # below = bad
        "employment_status_enc": 1,    # >= 1 could be unstable
    }

    explanations = []
    for i, feat in enumerate(FEATURE_NAMES):
        val = feature_values[i]
        imp = float(avg_imp[i])

        # Determine direction
        if feat == "credit_history_score":
            direction = "decreases_risk" if val >= HIGH_RISK_THRESHOLDS.get(feat, 600) else "increases_risk"
        elif feat in ("employment_years", "monthly_income"):
            direction = "decreases_risk" if val >= 3 else "increases_risk"
        elif feat == "previous_delays":
            direction = "increases_risk" if val > 0 else "decreases_risk"
        elif feat in ("debt_to_income", "loan_to_income", "existing_debt"):
            thr = HIGH_RISK_THRESHOLDS.get(feat, 0.4)
            direction = "increases_risk" if val >= thr else "decreases_risk"
        else:
            direction = "neutral"

        if direction == "increases_risk":
            explanation_text = FEATURE_EXPLANATIONS_NEGATIVE.get(feat, "")
        else:
            explanation_text = FEATURE_EXPLANATIONS_POSITIVE.get(feat, "")

        explanations.append({
            "feature_name": feat,
            "feature_human": FEATURE_HUMAN.get(feat, feat),
            "feature_value": round(val, 4),
            "effect_direction": direction,
            "importance_value": round(imp, 4),
            "explanation_text": explanation_text,
        })

    # Sort by importance descending
    explanations.sort(key=lambda x: x["importance_value"], reverse=True)
    return explanations[:8]


def build_recommendations(explanations: list, decision: str) -> list:
    recs = []
    for ex in explanations:
        if ex["effect_direction"] == "increases_risk":
            feat = ex["feature_name"]
            if feat == "debt_to_income":
                recs.append("Pay down existing debts to reduce your debt-to-income ratio below 40%.")
            elif feat == "previous_delays":
                recs.append("Ensure all upcoming payments are made on time to build a clean payment record.")
            elif feat == "loan_to_income":
                recs.append("Consider requesting a smaller loan amount or extending the repayment term.")
            elif feat == "credit_history_score":
                recs.append("Improve your credit score by paying bills on time and avoiding new credit applications.")
            elif feat == "employment_status_enc":
                recs.append("Moving to stable full-time employment would significantly improve your profile.")
            elif feat == "employment_years":
                recs.append("Accumulate more time in your current job to demonstrate income stability.")
            elif feat == "monthly_income":
                recs.append("Increasing your monthly income (e.g., through a raise or additional income source) would strengthen your application.")
            elif feat == "existing_debt":
                recs.append("Reduce your existing outstanding debt before applying for a new loan.")
    if not recs:
        if decision == "Approved":
            recs.append("Your profile is strong. Maintain your current financial habits.")
        else:
            recs.append("Work on improving your credit score and reducing your debt burden before reapplying.")
    return recs[:5]



# SEED

def seed():
    db = SessionLocal()
    try:
        # Admin
        if not db.query(User).filter(User.email == "admin@example.com").first():
            db.add(User(
                full_name="System Admin",
                email="admin@example.com",
                password_hash=hash_password("admin123"),
                role="ADMIN",
            ))
        # Demo user
        if not db.query(User).filter(User.email == "user@example.com").first():
            db.add(User(
                full_name="Demo User",
                email="user@example.com",
                password_hash=hash_password("user123"),
                role="USER",
            ))
        db.commit()
        print("[Seed] Demo accounts created.")
    finally:
        db.close()

    if not (MODELS_DIR / "random_forest.pkl").exists():
        train_and_save_models()
    else:
        print("[Seed] Models already exist, skipping training.")



# TEMPLATES (embedded HTML — written to disk on startup)

CSS = r"""
:root {
  --bg: #f5f7fb;
  --panel: #ffffff;
  --text: #182230;
  --muted: #667085;
  --border: #e4e7ec;
  --primary: #175cd3;
  --primary-dark: #1249a8;
  --success: #027a48;
  --danger: #b42318;
  --warning: #b54708;
  --shadow: 0 10px 30px rgba(16,24,40,.08);
  --radius: 14px;
}
* { box-sizing: border-box; }
body { margin: 0; font-family: Inter, Arial, sans-serif; background: var(--bg); color: var(--text); }
a { color: inherit; text-decoration: none; }
button, input, select, textarea { font: inherit; }

/* AUTH */
.auth-page { min-height:100vh; display:grid; place-items:center; padding:24px; }
.auth-card { background:var(--panel); border:1px solid var(--border); border-radius:var(--radius);
             box-shadow:var(--shadow); width:100%; max-width:420px; padding:32px; }
.auth-brand { margin-bottom:24px; }
.auth-brand h1 { margin:0 0 8px; font-size:26px; }
.auth-brand p { margin:0; color:var(--muted); }

/* FORM */
.form { display:grid; gap:16px; }
.form-row { display:grid; gap:7px; }
label { font-size:14px; font-weight:600; color:#344054; }
input, select, textarea {
  width:100%; border:1px solid #d0d5dd; background:#fff;
  border-radius:10px; padding:11px 12px; outline:none;
}
input:focus, select:focus { border-color:var(--primary); box-shadow:0 0 0 3px rgba(23,92,211,.12); }
.form-section-title { font-size:13px; font-weight:700; color:var(--muted); text-transform:uppercase;
                      letter-spacing:.06em; border-top:1px solid var(--border); padding-top:16px; margin-top:4px; }

/* BUTTONS */
.btn { border:0; border-radius:10px; padding:11px 18px; font-weight:700; cursor:pointer;
       display:inline-flex; align-items:center; justify-content:center; gap:8px; transition:.15s; }
.btn-primary { background:var(--primary); color:#fff; }
.btn-primary:hover { background:var(--primary-dark); }
.btn-light { background:#eef4ff; color:var(--primary); }
.btn-light:hover { background:#dce8ff; }
.btn-danger { background:#fee4e2; color:var(--danger); }
.btn-block { width:100%; }
.btn-sm { padding:6px 12px; font-size:13px; }

/* LAYOUT */
.app-shell { min-height:100vh; display:grid; grid-template-columns:260px 1fr; }
.sidebar { background:#fff; border-right:1px solid var(--border); padding:22px;
           position:sticky; top:0; height:100vh; overflow-y:auto; }
.brand { margin-bottom:28px; }
.brand h2 { margin:0 0 4px; font-size:20px; }
.brand span { color:var(--muted); font-size:13px; }
.nav { display:grid; gap:6px; }
.nav a, .nav button { border:0; background:transparent; text-align:left;
  padding:10px 12px; border-radius:10px; color:#344054; cursor:pointer; font-size:14px; }
.nav a.active, .nav a:hover, .nav button:hover { background:#eef4ff; color:var(--primary); }
.main { padding:28px; }
.topbar { display:flex; justify-content:space-between; align-items:flex-start;
          margin-bottom:24px; gap:16px; }
.topbar h1 { margin:0; font-size:26px; }
.topbar p { margin:6px 0 0; color:var(--muted); }
.user-badge { color:var(--muted); font-size:14px; white-space:nowrap; }

/* GRID */
.grid { display:grid; gap:18px; }
.grid-2 { grid-template-columns:repeat(2,minmax(0,1fr)); }
.grid-3 { grid-template-columns:repeat(3,minmax(0,1fr)); }
.grid-4 { grid-template-columns:repeat(4,minmax(0,1fr)); }

/* CARD */
.card { background:var(--panel); border:1px solid var(--border); border-radius:var(--radius);
        padding:22px; box-shadow:var(--shadow); }
.card h3 { margin:0 0 14px; font-size:18px; }
.card-subtitle { color:var(--muted); margin:-6px 0 16px; font-size:14px; }

/* STAT */
.stat { display:grid; gap:8px; }
.stat .label { color:var(--muted); font-size:13px; }
.stat .value { font-size:26px; font-weight:800; }

/* BADGES */
.badge { display:inline-flex; padding:4px 10px; border-radius:999px; font-size:12px; font-weight:700; }
.badge-Approved { background:#dcfae6; color:var(--success); }
.badge-Rejected { background:#fee4e2; color:var(--danger); }
.badge-Manual { background:#fef0c7; color:var(--warning); }
.badge-Low { background:#dcfae6; color:var(--success); }
.badge-Medium { background:#fef0c7; color:var(--warning); }
.badge-High { background:#fee4e2; color:var(--danger); }

/* TABLE */
.table-wrap { overflow-x:auto; border:1px solid var(--border); border-radius:12px; }
table { width:100%; border-collapse:collapse; background:#fff; }
th, td { padding:12px 14px; border-bottom:1px solid var(--border); text-align:left; font-size:14px; }
th { background:#f9fafb; color:#475467; font-weight:700; }
tr:last-child td { border-bottom:0; }

/* ALERTS */
.alert { padding:12px 14px; border-radius:10px; margin:12px 0; }
.alert-error { background:#fee4e2; color:var(--danger); }
.alert-success { background:#dcfae6; color:var(--success); }
.alert-info { background:#eef4ff; color:var(--primary); }

/* UPLOAD */
.upload-zone { border:2px dashed #b2ccff; border-radius:14px; padding:32px;
               text-align:center; background:#f5f8ff; cursor:pointer; }

/* MODEL COMPARISON TABLE */
.model-table-wrap { overflow-x:auto; }
.model-table { width:100%; border-collapse:collapse; font-size:14px; }
.model-table th { background:#f9fafb; padding:10px 14px; text-align:left; color:var(--muted); font-weight:700; border-bottom:2px solid var(--border); }
.model-table td { padding:10px 14px; border-bottom:1px solid var(--border); }
.model-table tr:last-child td { border-bottom:0; }
.model-table tr.ensemble-row td { background:#f0f5ff; font-weight:700; }

/* FEATURE BAR */
.feature-bar-wrap { display:grid; gap:10px; }
.feature-bar-item { display:grid; gap:4px; }
.feature-bar-label { display:flex; justify-content:space-between; font-size:13px; }
.feature-bar-label .fn { font-weight:600; }
.feature-bar-label .fv { color:var(--muted); }
.feature-bar-track { height:8px; background:#e4e7ec; border-radius:999px; overflow:hidden; }
.feature-bar-fill { height:100%; border-radius:999px; transition:width .4s; }
.fill-risk { background:#f97066; }
.fill-safe { background:#47cd89; }

/* RISK INDICATOR CARD */
.risk-indicator-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:12px; }
.risk-card { padding:16px; border-radius:12px; border:1px solid var(--border); background:#fff; }
.risk-card .risk-label { font-size:12px; color:var(--muted); font-weight:600; text-transform:uppercase; letter-spacing:.05em; }
.risk-card .risk-value { font-size:22px; font-weight:800; margin-top:4px; }

/* DECISION BANNER */
.decision-banner { padding:20px 24px; border-radius:14px; margin-bottom:20px; }
.decision-banner.approved { background:#d1fadf; border:2px solid #6ce9a6; }
.decision-banner.rejected { background:#fee4e2; border:2px solid #fda29b; }
.decision-banner.manual { background:#fef0c7; border:2px solid #fec84b; }
.decision-banner h2 { margin:0 0 6px; font-size:22px; }
.decision-banner p { margin:0; color:#344054; }

/* RECOMMENDATIONS */
.rec-list { list-style:none; padding:0; margin:0; display:grid; gap:10px; }
.rec-list li { display:flex; gap:10px; align-items:flex-start; padding:12px;
               background:#f9fafb; border-radius:10px; font-size:14px; }
.rec-list li::before { content:"💡"; flex-shrink:0; }

/* RESPONSIVE */
@media (max-width:900px) {
  .app-shell { grid-template-columns:1fr; }
  .sidebar { position:static; height:auto; }
  .grid-2, .grid-3, .grid-4 { grid-template-columns:1fr; }
  .topbar { flex-direction:column; }
  .risk-indicator-grid { grid-template-columns:1fr 1fr; }
}
@media (max-width:500px) {
  .risk-indicator-grid { grid-template-columns:1fr; }
}
"""

# ── Base layout macro ──────────────────────────────────────────────────────
def _sidebar(active: str, role: str) -> str:
    admin_link = '<a href="/admin">Admin Panel</a>' if role == "ADMIN" else ""
    pages = {
        "dashboard": "My Applications",
        "batch": "Batch Analysis",
    }
    links = ""
    for key, label in pages.items():
        cls = "active" if active == key else ""
        links += f'<a class="{cls}" href="/{key}">{label}</a>\n'
    return f"""
<aside class="sidebar">
  <div class="brand"><h2>Credit Scoring</h2><span>ML Assessment System</span></div>
  <nav class="nav">
    {links}
    {admin_link}
    <form method="post" action="/logout" style="margin-top:12px;">
      <button type="submit" class="btn btn-danger btn-block" style="width:100%;text-align:left;">Logout</button>
    </form>
  </nav>
</aside>"""


TMPL_LOGIN = """<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Login — Credit Scoring</title>
  <link rel="stylesheet" href="/static/css/styles.css"/>
</head>
<body class="auth-page">
<main class="auth-card">
  <div class="auth-brand">
    <h1>Credit Scoring</h1>
    <p>ML-powered creditworthiness assessment system</p>
  </div>
  {% if error %}
  <div class="alert alert-error">{{ error }}</div>
  {% endif %}
  <form method="post" action="/login" class="form">
    <div class="form-row"><label>Email</label>
      <input type="email" name="email" required placeholder="user@example.com" value="{{ email or '' }}"/></div>
    <div class="form-row"><label>Password</label>
      <input type="password" name="password" required placeholder="Password"/></div>
    <button class="btn btn-primary btn-block" type="submit">Login</button>
    <a class="btn btn-light btn-block" href="/register">Create account</a>
  </form>
  <p style="margin-top:20px;font-size:13px;color:var(--muted);text-align:center;">
    Demo: admin@example.com / admin123 &nbsp;|&nbsp; user@example.com / user123
  </p>
</main>
</body></html>"""

TMPL_REGISTER = """<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Register — Credit Scoring</title>
  <link rel="stylesheet" href="/static/css/styles.css"/>
</head>
<body class="auth-page">
<main class="auth-card">
  <div class="auth-brand">
    <h1>Create Account</h1>
    <p>Register to submit credit applications.</p>
  </div>
  {% if error %}
  <div class="alert alert-error">{{ error }}</div>
  {% endif %}
  <form method="post" action="/register" class="form">
    <div class="form-row"><label>Full Name</label>
      <input name="full_name" required placeholder="Your full name" value="{{ full_name or '' }}"/></div>
    <div class="form-row"><label>Email</label>
      <input type="email" name="email" required placeholder="user@example.com" value="{{ email or '' }}"/></div>
    <div class="form-row"><label>Password</label>
      <input type="password" name="password" required placeholder="Min. 6 characters"/></div>
    <button class="btn btn-primary btn-block" type="submit">Register</button>
    <a class="btn btn-light btn-block" href="/login">Already have an account</a>
  </form>
</main>
</body></html>"""

TMPL_DASHBOARD = """<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Dashboard — Credit Scoring</title>
  <link rel="stylesheet" href="/static/css/styles.css"/>
</head>
<body>
<div class="app-shell">
  {{ sidebar | safe }}
  <main class="main">
    <div class="topbar">
      <div>
        <h1>Credit Application</h1>
        <p>Fill in the questionnaire below to receive an ML-powered creditworthiness assessment.</p>
      </div>
      <div class="user-badge">{{ user.full_name }} ({{ user.role }})</div>
    </div>

    {% if error %}<div class="alert alert-error">{{ error }}</div>{% endif %}
    {% if success %}<div class="alert alert-success">{{ success }}</div>{% endif %}

    <div class="grid grid-2">
      <!-- APPLICATION FORM -->
      <div class="card">
        <h3>New Credit Application</h3>
        <p class="card-subtitle">Please answer all questions honestly for the most accurate assessment.</p>
        <form method="post" action="/applications" class="form">

          <div class="form-section-title">Personal Information</div>
          <div class="form-row"><label>Your Age</label>
            <input type="number" name="age" min="18" max="80" required placeholder="e.g. 32"/></div>
          <div class="form-row"><label>Monthly Income (₸)</label>
            <input type="number" name="monthly_income" min="0" required placeholder="e.g. 250000"/></div>

          <div class="form-section-title">Employment</div>
          <div class="form-row"><label>Employment Status</label>
            <select name="employment_status" required>
              <option value="">Select...</option>
              <option value="employed">Employed (full-time)</option>
              <option value="self-employed">Self-employed</option>
              <option value="part-time">Part-time employment</option>
              <option value="unemployed">Unemployed</option>
              <option value="retired">Retired</option>
              <option value="student">Student</option>
            </select></div>
          <div class="form-row"><label>Years at Current Job</label>
            <input type="number" name="employment_years" min="0" max="50" step="0.5" required placeholder="e.g. 3.5"/></div>

          <div class="form-section-title">Credit History</div>
          <div class="form-row"><label>Credit History Score (300–850)</label>
            <input type="number" name="credit_history_score" min="300" max="850" required placeholder="e.g. 680"/></div>
          <div class="form-row"><label>Number of Previous Payment Delays</label>
            <input type="number" name="previous_delays" min="0" max="20" required placeholder="e.g. 0"/></div>
          <div class="form-row"><label>Existing Monthly Debt Payments (₸)</label>
            <input type="number" name="existing_debt" min="0" required placeholder="e.g. 50000"/></div>

          <div class="form-section-title">Loan Details</div>
          <div class="form-row"><label>Loan Amount Requested (₸)</label>
            <input type="number" name="loan_amount" min="0" required placeholder="e.g. 1000000"/></div>
          <div class="form-row"><label>Loan Term (months)</label>
            <select name="loan_term" required>
              <option value="6">6 months</option>
              <option value="12" selected>12 months</option>
              <option value="24">24 months</option>
              <option value="36">36 months</option>
              <option value="48">48 months</option>
              <option value="60">60 months</option>
            </select></div>
          <div class="form-row"><label>Loan Purpose</label>
            <select name="loan_purpose" required>
              <option value="personal">Personal needs</option>
              <option value="business">Business development</option>
              <option value="education">Education</option>
              <option value="mortgage">Mortgage / Real estate</option>
              <option value="auto">Auto loan</option>
              <option value="medical">Medical expenses</option>
              <option value="other">Other</option>
            </select></div>

          <button class="btn btn-primary btn-block" type="submit" style="margin-top:8px;">
            🔍 Evaluate Creditworthiness
          </button>
        </form>
      </div>

      <!-- RESULT -->
      <div>
        {% if result %}
        {% set ens = result.ensemble %}
        {% set dec = ens.decision %}
        {% set dec_class = "approved" if dec == "Approved" else ("rejected" if dec == "Rejected" else "manual") %}
        <div class="decision-banner {{ dec_class }}">
          <h2>
            {% if dec == "Approved" %}✅ Application Approved{% elif dec == "Rejected" %}❌ Application Rejected{% else %}⚠️ Manual Review Required{% endif %}
          </h2>
          <p>Based on ensemble analysis of 4 ML models. Ensemble probability of default: <strong>{{ "%.1f"|format(ens.probability_default * 100) }}%</strong></p>
        </div>

        <!-- RISK INDICATORS -->
        <div class="card" style="margin-bottom:18px;">
          <h3>Risk Indicators</h3>
          <div class="risk-indicator-grid">
            <div class="risk-card">
              <div class="risk-label">Probability of Default</div>
              <div class="risk-value" style="color:{% if ens.probability_default < 0.3 %}var(--success){% elif ens.probability_default < 0.6 %}var(--warning){% else %}var(--danger){% endif %};">
                {{ "%.1f"|format(ens.probability_default * 100) }}%</div>
            </div>
            <div class="risk-card">
              <div class="risk-label">Delinquency Risk</div>
              <div class="risk-value" style="color:var(--warning);">
                {{ "%.1f"|format(ens.probability_delinquency * 100) }}%</div>
            </div>
            <div class="risk-card">
              <div class="risk-label">Risk Level</div>
              <div class="risk-value"><span class="badge badge-{{ ens.risk_level }}">{{ ens.risk_level }}</span></div>
            </div>
            <div class="risk-card">
              <div class="risk-label">Expected Credit Loss</div>
              <div class="risk-value" style="font-size:16px;">₸ {{ "{:,.0f}".format(ens.expected_credit_loss) }}</div>
            </div>
          </div>
        </div>

        <!-- MODEL COMPARISON -->
        <div class="card" style="margin-bottom:18px;">
          <h3>Model Comparison</h3>
          <p class="card-subtitle">Results from all 4 ML models + ensemble.</p>
          <div class="model-table-wrap">
            <table class="model-table">
              <thead><tr>
                <th>Model</th><th>PD %</th><th>Risk Level</th><th>ECL (₸)</th><th>Decision</th>
              </tr></thead>
              <tbody>
              {% for key, m in result.items() %}
              <tr {% if key == "ensemble" %}class="ensemble-row"{% endif %}>
                <td>{{ m.display_name }}</td>
                <td>{{ "%.1f"|format(m.probability_default * 100) }}%</td>
                <td><span class="badge badge-{{ m.risk_level }}">{{ m.risk_level }}</span></td>
                <td>{{ "{:,.0f}".format(m.expected_credit_loss) }}</td>
                <td><span class="badge badge-{{ m.decision.split()[0] }}">{{ m.decision }}</span></td>
              </tr>
              {% endfor %}
              </tbody>
            </table>
          </div>
        </div>

        <!-- FEATURE IMPORTANCE -->
        {% if explanations %}
        <div class="card" style="margin-bottom:18px;">
          <h3>Key Factors</h3>
          <p class="card-subtitle">Top factors influencing the decision.</p>
          <div class="feature-bar-wrap">
            {% for ex in explanations %}
            <div class="feature-bar-item">
              <div class="feature-bar-label">
                <span class="fn">{{ ex.feature_human }}</span>
                <span class="fv">
                  {% if ex.effect_direction == "increases_risk" %}⬆ Risk{% else %}✓ Safe{% endif %}
                </span>
              </div>
              <div class="feature-bar-track">
                <div class="feature-bar-fill {% if ex.effect_direction == 'increases_risk' %}fill-risk{% else %}fill-safe{% endif %}"
                     style="width:{{ [ex.importance_value * 400, 100]|min }}%;"></div>
              </div>
              <div style="font-size:12px;color:var(--muted);">{{ ex.explanation_text }}</div>
            </div>
            {% endfor %}
          </div>
        </div>
        {% endif %}

        <!-- RECOMMENDATIONS -->
        {% if recommendations %}
        <div class="card">
          <h3>Recommendations</h3>
          <p class="card-subtitle">Steps to improve your creditworthiness:</p>
          <ul class="rec-list">
            {% for r in recommendations %}<li>{{ r }}</li>{% endfor %}
          </ul>
        </div>
        {% endif %}

        {% else %}
        <div class="card" style="text-align:center;padding:40px;">
          <p style="font-size:32px;margin:0;">📋</p>
          <h3 style="margin:12px 0 8px;">Submit Your Application</h3>
          <p style="color:var(--muted);">Fill in the form on the left and click <strong>Evaluate Creditworthiness</strong> to receive your full ML assessment with model comparison, risk indicators, and personalised recommendations.</p>
        </div>
        {% endif %}
      </div>
    </div>

    <!-- HISTORY -->
    <div class="card" style="margin-top:20px;">
      <h3>Application History</h3>
      <div class="table-wrap">
        <table>
          <thead><tr><th>#</th><th>Date</th><th>Loan Amount</th><th>Loan Term</th><th>Decision</th><th>Avg PD</th><th>Details</th></tr></thead>
          <tbody>
          {% if applications %}
          {% for app in applications %}
          <tr>
            <td>{{ app.id }}</td>
            <td>{{ app.created_at.strftime("%d.%m.%Y %H:%M") }}</td>
            <td>₸ {{ "{:,.0f}".format(app.loan_amount) }}</td>
            <td>{{ app.loan_term }} mo.</td>
            <td>
              {% if app.predictions %}
              {% set avg_pd = app.predictions|map(attribute='probability_default')|list|sum / app.predictions|length %}
              <span class="badge badge-{{ app.predictions[0].decision.split()[0] }}">{{ app.predictions[0].decision }}</span>
              {% else %}<span style="color:var(--muted);">—</span>{% endif %}
            </td>
            <td>
              {% if app.predictions %}
              {{ "%.1f"|format(avg_pd * 100) }}%
              {% else %}—{% endif %}
            </td>
            <td><a href="/applications/{{ app.id }}" class="btn btn-sm btn-light">View</a></td>
          </tr>
          {% endfor %}
          {% else %}
          <tr><td colspan="7" style="text-align:center;color:var(--muted);padding:24px;">No applications yet.</td></tr>
          {% endif %}
          </tbody>
        </table>
      </div>
    </div>
  </main>
</div>
</body></html>"""

TMPL_APP_DETAIL = """<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Application #{{ app.id }} — Credit Scoring</title>
  <link rel="stylesheet" href="/static/css/styles.css"/>
</head>
<body>
<div class="app-shell">
  {{ sidebar | safe }}
  <main class="main">
    <div class="topbar">
      <div><h1>Application #{{ app.id }}</h1>
           <p>Submitted {{ app.created_at.strftime("%d %B %Y, %H:%M") }}</p></div>
      <div class="user-badge">{{ user.full_name }}</div>
    </div>

    <div class="grid grid-2" style="margin-bottom:18px;">
      <div class="card">
        <h3>Application Details</h3>
        <table style="width:100%;font-size:14px;">
          <tr><td style="color:var(--muted);padding:6px 0;">Age</td><td><strong>{{ app.age }}</strong></td></tr>
          <tr><td style="color:var(--muted);padding:6px 0;">Monthly Income</td><td><strong>₸ {{ "{:,.0f}".format(app.monthly_income) }}</strong></td></tr>
          <tr><td style="color:var(--muted);padding:6px 0;">Existing Debt</td><td><strong>₸ {{ "{:,.0f}".format(app.existing_debt) }}</strong></td></tr>
          <tr><td style="color:var(--muted);padding:6px 0;">Loan Amount</td><td><strong>₸ {{ "{:,.0f}".format(app.loan_amount) }}</strong></td></tr>
          <tr><td style="color:var(--muted);padding:6px 0;">Loan Term</td><td><strong>{{ app.loan_term }} months</strong></td></tr>
          <tr><td style="color:var(--muted);padding:6px 0;">Employment</td><td><strong>{{ app.employment_status }}</strong></td></tr>
          <tr><td style="color:var(--muted);padding:6px 0;">Employment Years</td><td><strong>{{ app.employment_years }}</strong></td></tr>
          <tr><td style="color:var(--muted);padding:6px 0;">Credit Score</td><td><strong>{{ app.credit_history_score }}</strong></td></tr>
          <tr><td style="color:var(--muted);padding:6px 0;">Payment Delays</td><td><strong>{{ app.previous_delays }}</strong></td></tr>
          <tr><td style="color:var(--muted);padding:6px 0;">Loan Purpose</td><td><strong>{{ app.loan_purpose }}</strong></td></tr>
        </table>
      </div>
      {% if predictions %}
      {% set ens = predictions|selectattr("model_name","eq","ensemble")|list %}
      {% set ens_p = ens[0] if ens else predictions[0] %}
      <div>
        {% set dec = ens_p.decision %}
        {% set dec_class = "approved" if dec == "Approved" else ("rejected" if dec == "Rejected" else "manual") %}
        <div class="decision-banner {{ dec_class }}" style="margin-bottom:14px;">
          <h2>{% if dec == "Approved" %}✅ Approved{% elif dec == "Rejected" %}❌ Rejected{% else %}⚠️ Manual Review{% endif %}</h2>
          <p>Ensemble PD: <strong>{{ "%.1f"|format(ens_p.probability_default * 100) }}%</strong></p>
        </div>
        <div class="risk-indicator-grid">
          <div class="risk-card"><div class="risk-label">Prob. Default</div>
            <div class="risk-value" style="color:{% if ens_p.probability_default < 0.3 %}var(--success){% elif ens_p.probability_default < 0.6 %}var(--warning){% else %}var(--danger){% endif %};">
              {{ "%.1f"|format(ens_p.probability_default*100) }}%</div></div>
          <div class="risk-card"><div class="risk-label">Delinquency Risk</div>
            <div class="risk-value">{{ "%.1f"|format(ens_p.probability_delinquency*100) }}%</div></div>
          <div class="risk-card"><div class="risk-label">Risk Level</div>
            <div class="risk-value"><span class="badge badge-{{ ens_p.risk_level }}">{{ ens_p.risk_level }}</span></div></div>
          <div class="risk-card"><div class="risk-label">Expected Credit Loss</div>
            <div class="risk-value" style="font-size:16px;">₸ {{ "{:,.0f}".format(ens_p.expected_credit_loss) }}</div></div>
        </div>
      </div>
      {% endif %}
    </div>

    {% if predictions %}
    <div class="card" style="margin-bottom:18px;">
      <h3>Model Comparison</h3>
      <div class="model-table-wrap">
        <table class="model-table">
          <thead><tr><th>Model</th><th>PD %</th><th>Delinquency %</th><th>Risk</th><th>ECL (₸)</th><th>Decision</th></tr></thead>
          <tbody>
          {% for p in predictions %}
          <tr {% if p.model_name == "ensemble" %}class="ensemble-row"{% endif %}>
            <td>{{ p.model_name | replace("_"," ") | title }}{% if p.model_name == "ensemble" %} ★{% endif %}</td>
            <td>{{ "%.1f"|format(p.probability_default*100) }}%</td>
            <td>{{ "%.1f"|format(p.probability_delinquency*100) }}%</td>
            <td><span class="badge badge-{{ p.risk_level }}">{{ p.risk_level }}</span></td>
            <td>{{ "{:,.0f}".format(p.expected_credit_loss) }}</td>
            <td><span class="badge badge-{{ p.decision.split()[0] }}">{{ p.decision }}</span></td>
          </tr>
          {% endfor %}
          </tbody>
        </table>
      </div>
    </div>

    <!-- EXPLANATIONS -->
    {% if explanations %}
    <div class="card" style="margin-bottom:18px;">
      <h3>Key Risk Factors</h3>
      <div class="feature-bar-wrap">
        {% for ex in explanations %}
        <div class="feature-bar-item">
          <div class="feature-bar-label">
            <span class="fn">{{ ex.feature_human }}</span>
            <span class="fv">{% if ex.effect_direction == "increases_risk" %}⬆ Increases risk{% else %}✓ Reduces risk{% endif %}</span>
          </div>
          <div class="feature-bar-track">
            <div class="feature-bar-fill {% if ex.effect_direction == 'increases_risk' %}fill-risk{% else %}fill-safe{% endif %}"
                 style="width:{{ [ex.importance_value * 400, 100]|min }}%;"></div>
          </div>
          <div style="font-size:12px;color:var(--muted);">{{ ex.explanation_text }}</div>
        </div>
        {% endfor %}
      </div>
    </div>
    {% endif %}

    <!-- RECOMMENDATIONS -->
    {% if recommendations %}
    <div class="card">
      <h3>Recommendations for Improvement</h3>
      <ul class="rec-list">{% for r in recommendations %}<li>{{ r }}</li>{% endfor %}</ul>
    </div>
    {% endif %}
    {% endif %}

    <a href="/dashboard" class="btn btn-light" style="margin-top:16px;">← Back to Dashboard</a>
  </main>
</div>
</body></html>"""

TMPL_BATCH = """<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Batch Analysis — Credit Scoring</title>
  <link rel="stylesheet" href="/static/css/styles.css"/>
  <style>
    .model-cmp-grid { display:grid; gap:14px; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); margin-bottom:20px; }
    .model-cmp-card { background:#fff; border:1px solid var(--border); border-radius:var(--radius);
                      padding:18px; box-shadow:var(--shadow); position:relative; }
    .model-cmp-card.ensemble-card { border:2px solid var(--primary); background:#f0f5ff; }
    .model-cmp-card .mc-name { font-size:13px; font-weight:700; color:var(--muted);
                                text-transform:uppercase; letter-spacing:.05em; margin-bottom:10px; }
    .model-cmp-card .mc-pd  { font-size:30px; font-weight:900; margin:0; }
    .model-cmp-card .mc-sub { font-size:12px; color:var(--muted); margin-top:2px; }
    .model-cmp-card .mc-decisions { display:flex; gap:6px; margin-top:12px; flex-wrap:wrap; }
    .mc-pill { font-size:11px; font-weight:700; padding:3px 9px;
               border-radius:999px; white-space:nowrap; }
    .mc-pill-green { background:#dcfae6; color:var(--success); }
    .mc-pill-red   { background:#fee4e2; color:var(--danger); }
    .mc-pill-yellow{ background:#fef0c7; color:var(--warning); }
    .mc-star { position:absolute; top:12px; right:14px; font-size:18px; }

    /* comparison table */
    .cmp-table th, .cmp-table td { padding:11px 14px; }
    .cmp-table .bar-cell { min-width:180px; }
    .bar-wrap { display:flex; align-items:center; gap:8px; }
    .bar-track { flex:1; height:10px; background:#e4e7ec; border-radius:999px; overflow:hidden; }
    .bar-fill  { height:100%; border-radius:999px; }
    .bar-blue  { background:var(--primary); }
    .bar-green { background:#47cd89; }
    .bar-red   { background:#f97066; }
    .bar-yellow{ background:#fec84b; }
    .bar-label { font-size:12px; font-weight:700; min-width:38px; text-align:right; }

    /* risk distribution */
    .risk-dist { display:flex; gap:0; border-radius:8px; overflow:hidden; height:28px; margin-top:8px; }
    .risk-dist-seg { display:flex; align-items:center; justify-content:center;
                     font-size:11px; font-weight:700; color:#fff; transition:width .4s; }
    .rds-low    { background:#47cd89; }
    .rds-medium { background:#fec84b; color:#344054; }
    .rds-high   { background:#f97066; }

    /* history table improvement */
    .hist-model-pills { display:flex; gap:4px; flex-wrap:wrap; }
  </style>
</head>
<body>
<div class="app-shell">
  {{ sidebar | safe }}
  <main class="main">
    <div class="topbar">
      <div><h1>Batch CSV Analysis</h1>
           <p>Upload a dataset — all 4 ML models score every record. Compare their decisions side by side.</p></div>
      <div class="user-badge">{{ user.full_name }}</div>
    </div>

    {% if error %}<div class="alert alert-error">{{ error }}</div>{% endif %}

    <!-- UPLOAD FORM -->
    <div class="card" style="margin-bottom:20px;">
      <h3>Upload Dataset</h3>
      <div class="grid grid-2" style="gap:20px;align-items:start;">
        <form method="post" action="/batch/upload" enctype="multipart/form-data" class="form">
          <div class="upload-zone">
            <p style="font-size:36px;margin:0;">📂</p>
            <p style="margin:8px 0 4px;"><strong>Select CSV file</strong></p>
            <p style="color:var(--muted);font-size:12px;margin:0 0 12px;">
              Required columns: monthly_income, existing_debt, loan_amount, loan_term,
              employment_status, employment_years, age, credit_history_score,
              previous_delays, loan_purpose
            </p>
            <input type="file" name="file" accept=".csv" required/>
          </div>
          <button class="btn btn-primary btn-block" type="submit">🚀 Start Batch Analysis</button>
        </form>
        <div>
          <p class="card-subtitle">How it works:</p>
          <ul style="color:var(--text);font-size:14px;line-height:1.9;padding-left:20px;margin:0 0 16px;">
            <li>Each row in your CSV = one credit application</li>
            <li>All <strong>4 ML models</strong> score every record independently</li>
            <li>Results are aggregated per model: avg PD, approval rate, risk distribution, ECL</li>
            <li>Ensemble = average PD across all models (most reliable)</li>
          </ul>
          <a href="/batch/template" class="btn btn-light btn-sm">⬇ Download CSV Template</a>
        </div>
      </div>
    </div>

    {% if batch_result and model_stats %}
    {% set total = batch_result.total_records %}

    <!-- OVERVIEW STATS -->
    <div class="grid grid-4" style="margin-bottom:20px;">
      <div class="card stat">
        <span class="label">Total Records</span>
        <span class="value">{{ total }}</span>
      </div>
      <div class="card stat">
        <span class="label">Approved (Ensemble)</span>
        <span class="value" style="color:var(--success);">{{ batch_result.approved_count }}</span>
      </div>
      <div class="card stat">
        <span class="label">Rejected (Ensemble)</span>
        <span class="value" style="color:var(--danger);">{{ batch_result.rejected_count }}</span>
      </div>
      <div class="card stat">
        <span class="label">Manual Review</span>
        <span class="value" style="color:var(--warning);">{{ batch_result.manual_review_count }}</span>
      </div>
    </div>

    <!-- MODEL CARDS -->
    <div class="card" style="margin-bottom:20px;">
      <h3>Model-by-Model Overview</h3>
      <p class="card-subtitle">Each card shows how a particular model assessed the entire dataset of {{ total }} records.</p>
      <div class="model-cmp-grid">
        {% for mkey, m in model_stats.items() %}
        <div class="model-cmp-card {% if mkey == 'ensemble' %}ensemble-card{% endif %}">
          {% if mkey == 'ensemble' %}<span class="mc-star">⭐</span>{% endif %}
          <div class="mc-name">{{ m.display_name }}</div>
          <p class="mc-pd" style="color:{% if m.avg_pd < 30 %}var(--success){% elif m.avg_pd < 60 %}var(--warning){% else %}var(--danger){% endif %};">
            {{ m.avg_pd }}%
          </p>
          <div class="mc-sub">Average probability of default</div>
          <div class="mc-decisions">
            <span class="mc-pill mc-pill-green">✓ {{ m.approved }} approved</span>
            <span class="mc-pill mc-pill-red">✗ {{ m.rejected }} rejected</span>
            <span class="mc-pill mc-pill-yellow">⚠ {{ m.manual }} review</span>
          </div>
          <div style="margin-top:12px;">
            <div style="font-size:11px;color:var(--muted);margin-bottom:4px;">Risk distribution</div>
            <div class="risk-dist">
              {% set lp = (m.low / total * 100)|round(0)|int %}
              {% set mp = (m.medium / total * 100)|round(0)|int %}
              {% set hp = (m.high / total * 100)|round(0)|int %}
              {% if lp > 0 %}<div class="risk-dist-seg rds-low" style="width:{{ lp }}%;" title="Low: {{ m.low }}">{% if lp > 8 %}{{ lp }}%{% endif %}</div>{% endif %}
              {% if mp > 0 %}<div class="risk-dist-seg rds-medium" style="width:{{ mp }}%;" title="Medium: {{ m.medium }}">{% if mp > 8 %}{{ mp }}%{% endif %}</div>{% endif %}
              {% if hp > 0 %}<div class="risk-dist-seg rds-high" style="width:{{ hp }}%;" title="High: {{ m.high }}">{% if hp > 8 %}{{ hp }}%{% endif %}</div>{% endif %}
            </div>
            <div style="display:flex;gap:10px;margin-top:4px;font-size:11px;color:var(--muted);">
              <span>🟢 Low: {{ m.low }}</span>
              <span>🟡 Med: {{ m.medium }}</span>
              <span>🔴 High: {{ m.high }}</span>
            </div>
          </div>
        </div>
        {% endfor %}
      </div>
    </div>

    <!-- DETAILED COMPARISON TABLE -->
    <div class="card" style="margin-bottom:20px;">
      <h3>Detailed Model Comparison</h3>
      <p class="card-subtitle">Side-by-side metrics for all models on the uploaded dataset ({{ total }} records).</p>
      <div class="model-table-wrap">
        <table class="model-table cmp-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Avg PD</th>
              <th class="bar-cell">Approval Rate</th>
              <th class="bar-cell">Rejection Rate</th>
              <th>Avg ECL (₸)</th>
              <th>Low Risk</th>
              <th>Med Risk</th>
              <th>High Risk</th>
            </tr>
          </thead>
          <tbody>
          {% for mkey, m in model_stats.items() %}
          <tr {% if mkey == 'ensemble' %}class="ensemble-row"{% endif %}>
            <td>
              {{ m.display_name }}
              {% if mkey == 'ensemble' %}<span style="font-size:11px;color:var(--primary);"> ★ Ensemble</span>{% endif %}
            </td>
            <td>
              <strong style="color:{% if m.avg_pd < 30 %}var(--success){% elif m.avg_pd < 60 %}var(--warning){% else %}var(--danger){% endif %};">
                {{ m.avg_pd }}%
              </strong>
            </td>
            <td class="bar-cell">
              <div class="bar-wrap">
                <div class="bar-track">
                  <div class="bar-fill bar-green" style="width:{{ m.approval_rate }}%;"></div>
                </div>
                <span class="bar-label" style="color:var(--success);">{{ m.approval_rate }}%</span>
              </div>
            </td>
            <td class="bar-cell">
              <div class="bar-wrap">
                <div class="bar-track">
                  <div class="bar-fill bar-red" style="width:{{ m.rejection_rate }}%;"></div>
                </div>
                <span class="bar-label" style="color:var(--danger);">{{ m.rejection_rate }}%</span>
              </div>
            </td>
            <td>₸ {{ "{:,.0f}".format(m.avg_ecl) }}</td>
            <td style="color:var(--success);">{{ m.low }}</td>
            <td style="color:var(--warning);">{{ m.medium }}</td>
            <td style="color:var(--danger);">{{ m.high }}</td>
          </tr>
          {% endfor %}
          </tbody>
        </table>
      </div>

      <!-- VISUAL BAR COMPARISON: Avg PD per model -->
      <div style="margin-top:24px;">
        <h4 style="margin:0 0 14px;font-size:15px;">Average PD by Model</h4>
        <div style="display:grid;gap:10px;">
          {% for mkey, m in model_stats.items() %}
          <div>
            <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px;">
              <span style="font-weight:{% if mkey == 'ensemble' %}800{% else %}500{% endif %};">{{ m.display_name }}</span>
              <span style="font-weight:700;color:{% if m.avg_pd < 30 %}var(--success){% elif m.avg_pd < 60 %}var(--warning){% else %}var(--danger){% endif %};">{{ m.avg_pd }}%</span>
            </div>
            <div style="height:12px;background:#e4e7ec;border-radius:999px;overflow:hidden;">
              <div style="height:100%;width:{{ m.avg_pd }}%;border-radius:999px;
                background:{% if m.avg_pd < 30 %}#47cd89{% elif m.avg_pd < 60 %}#fec84b{% else %}#f97066{% endif %};transition:width .5s;"></div>
            </div>
          </div>
          {% endfor %}
        </div>
      </div>
    </div>

    {% endif %}

    <!-- HISTORY -->
    <div class="card">
      <h3>Batch Analysis History</h3>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th><th>File</th><th>Date</th><th>Records</th>
              <th>Approved</th><th>Rejected</th><th>Manual</th><th>Avg PD</th>
            </tr>
          </thead>
          <tbody>
          {% if history %}
          {% for b in history %}
          <tr>
            <td>{{ b.id }}</td>
            <td style="font-size:13px;">{{ b.filename }}</td>
            <td style="font-size:13px;">{{ b.created_at.strftime("%d.%m.%Y %H:%M") }}</td>
            <td><strong>{{ b.total_records }}</strong></td>
            <td style="color:var(--success);">{{ b.approved_count }}</td>
            <td style="color:var(--danger);">{{ b.rejected_count }}</td>
            <td style="color:var(--warning);">{{ b.manual_review_count }}</td>
            <td>
              <span style="font-weight:700;color:{% if b.avg_pd < 0.3 %}var(--success){% elif b.avg_pd < 0.6 %}var(--warning){% else %}var(--danger){% endif %};">
                {{ "%.1f"|format(b.avg_pd * 100) }}%
              </span>
            </td>
          </tr>
          {% endfor %}
          {% else %}
          <tr><td colspan="8" style="text-align:center;color:var(--muted);padding:28px;">No batch analyses yet.</td></tr>
          {% endif %}
          </tbody>
        </table>
      </div>
    </div>
  </main>
</div>
</body></html>"""

TMPL_ADMIN = """<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Admin — Credit Scoring</title>
  <link rel="stylesheet" href="/static/css/styles.css"/>
</head>
<body>
<div class="app-shell">
  {{ sidebar | safe }}
  <main class="main">
    <div class="topbar">
      <div><h1>Admin Dashboard</h1><p>System statistics, all applications, and user management.</p></div>
      <div class="user-badge">{{ user.full_name }} (ADMIN)</div>
    </div>

    <!-- STATS -->
    <div class="grid grid-4" style="margin-bottom:18px;">
      <div class="card stat"><span class="label">Total Users</span><span class="value">{{ stats.total_users }}</span></div>
      <div class="card stat"><span class="label">Total Applications</span><span class="value">{{ stats.total_apps }}</span></div>
      <div class="card stat"><span class="label">Approved</span><span class="value" style="color:var(--success);">{{ stats.approved }}</span></div>
      <div class="card stat"><span class="label">Rejected</span><span class="value" style="color:var(--danger);">{{ stats.rejected }}</span></div>
      <div class="card stat"><span class="label">Manual Review</span><span class="value" style="color:var(--warning);">{{ stats.manual }}</span></div>
      <div class="card stat"><span class="label">Avg PD</span><span class="value">{{ "%.1f"|format(stats.avg_pd * 100) if stats.avg_pd else "0.0" }}%</span></div>
      <div class="card stat"><span class="label">Batch Analyses</span><span class="value">{{ stats.total_batches }}</span></div>
      <div class="card stat"><span class="label">Approval Rate</span><span class="value">{{ "%.0f"|format(stats.approval_rate) }}%</span></div>
    </div>

    <!-- MODEL META -->
    {% if model_meta %}
    <div class="card" style="margin-bottom:18px;">
      <h3>Trained Model Performance</h3>
      <div class="model-table-wrap">
        <table class="model-table">
          <thead><tr><th>Model</th><th>AUC-ROC</th><th>Accuracy</th><th>Status</th></tr></thead>
          <tbody>
          {% for name, m in model_meta.items() %}
          <tr>
            <td>{{ name | replace("_"," ") | title }}</td>
            <td><strong>{{ m.auc }}</strong></td>
            <td>{{ m.accuracy }}</td>
            <td><span class="badge badge-Approved">Active</span></td>
          </tr>
          {% endfor %}
          </tbody>
        </table>
      </div>
    </div>
    {% endif %}

    <!-- ALL APPLICATIONS -->
    <div class="card" style="margin-bottom:18px;">
      <h3>All Applications</h3>
      <div class="table-wrap">
        <table>
          <thead><tr><th>#</th><th>User</th><th>Date</th><th>Loan Amount</th><th>Decision</th><th>Avg PD</th><th>View</th></tr></thead>
          <tbody>
          {% for app in applications %}
          <tr>
            <td>{{ app.id }}</td>
            <td>{{ app.user.full_name if app.user else "—" }}</td>
            <td>{{ app.created_at.strftime("%d.%m.%Y %H:%M") }}</td>
            <td>₸ {{ "{:,.0f}".format(app.loan_amount) }}</td>
            <td>
              {% if app.predictions %}
              {% set ens = app.predictions|selectattr("model_name","eq","ensemble")|list %}
              {% set p = ens[0] if ens else app.predictions[0] %}
              <span class="badge badge-{{ p.decision.split()[0] }}">{{ p.decision }}</span>
              {% else %}—{% endif %}
            </td>
            <td>
              {% if app.predictions %}
              {% set ens = app.predictions|selectattr("model_name","eq","ensemble")|list %}
              {% set p = ens[0] if ens else app.predictions[0] %}
              {{ "%.1f"|format(p.probability_default*100) }}%
              {% else %}—{% endif %}
            </td>
            <td><a href="/applications/{{ app.id }}" class="btn btn-sm btn-light">View</a></td>
          </tr>
          {% endfor %}
          {% if not applications %}
          <tr><td colspan="7" style="text-align:center;color:var(--muted);padding:24px;">No applications yet.</td></tr>
          {% endif %}
          </tbody>
        </table>
      </div>
    </div>

    <!-- USERS -->
    <div class="card">
      <h3>Users</h3>
      <div class="table-wrap">
        <table>
          <thead><tr><th>#</th><th>Name</th><th>Email</th><th>Role</th><th>Registered</th><th>Applications</th></tr></thead>
          <tbody>
          {% for u in users %}
          <tr>
            <td>{{ u.id }}</td>
            <td>{{ u.full_name }}</td>
            <td>{{ u.email }}</td>
            <td><span class="badge {% if u.role == 'ADMIN' %}badge-Manual{% else %}badge-Approved{% endif %}">{{ u.role }}</span></td>
            <td>{{ u.created_at.strftime("%d.%m.%Y") }}</td>
            <td>{{ u.applications|length }}</td>
          </tr>
          {% endfor %}
          </tbody>
        </table>
      </div>
    </div>

    <!-- RETRAIN -->
    <form method="post" action="/admin/retrain" style="margin-top:18px;">
      <button class="btn btn-primary" type="submit">🔄 Retrain All Models</button>
    </form>
  </main>
</div>
</body></html>"""

TMPL_RETRAIN = """<!doctype html>
<html lang="en">
<head><meta charset="UTF-8"/><title>Retrain — Credit Scoring</title>
<link rel="stylesheet" href="/static/css/styles.css"/></head>
<body class="auth-page">
<div class="auth-card">
  <div class="auth-brand"><h1>✅ Models Retrained</h1>
  <p>All 4 ML models have been retrained successfully.</p></div>
  {% for name, m in meta.items() %}
  <div style="margin:8px 0;font-size:14px;"><strong>{{ name }}</strong> — AUC: {{ m.auc }} | Acc: {{ m.accuracy }}</div>
  {% endfor %}
  <a href="/admin" class="btn btn-primary btn-block" style="margin-top:20px;">Back to Admin</a>
</div>
</body></html>"""


def write_templates():
    """Write all embedded templates and static CSS to disk."""
    (STATIC_DIR / "css" / "styles.css").write_text(CSS)

    templates_map = {
        "login.html": TMPL_LOGIN,
        "register.html": TMPL_REGISTER,
        "dashboard.html": TMPL_DASHBOARD,
        "app_detail.html": TMPL_APP_DETAIL,
        "batch.html": TMPL_BATCH,
        "admin.html": TMPL_ADMIN,
        "retrain.html": TMPL_RETRAIN,
    }
    for name, content in templates_map.items():
        (TEMPLATES_DIR / name).write_text(content)



# APP INIT

write_templates()

app = FastAPI(
    title="Credit Scoring System",
    description="ML-powered creditworthiness assessment with multi-model comparison and SHAP-style explanations.",
    version="2.0.0",
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Lazy-load models
_models_cache: Optional[dict] = None


def get_models() -> dict:
    global _models_cache
    if _models_cache is None:
        _models_cache = load_models()
        if _models_cache is None:
            print("[APP] Models not found — training now...")
            train_and_save_models()
            _models_cache = load_models()
    return _models_cache



# AUTH ROUTES

@app.get("/", response_class=HTMLResponse)
def root():
    return RedirectResponse("/login")


@app.get("/login", response_class=HTMLResponse)
def login_get(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
def login_post(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.email == email).first()
    if not user or user.password_hash != hash_password(password):
        return templates.TemplateResponse("login.html", {
            "request": request, "error": "Invalid email or password.", "email": email
        })
    token = create_session(db, user.id)
    resp = RedirectResponse("/dashboard", status_code=302)
    resp.set_cookie("session_token", token, httponly=True,
                    max_age=SESSION_EXPIRE_HOURS * 3600)
    return resp


@app.get("/register", response_class=HTMLResponse)
def register_get(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@app.post("/register")
def register_post(
    request: Request,
    full_name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    if len(password) < 6:
        return templates.TemplateResponse("register.html", {
            "request": request, "error": "Password must be at least 6 characters.",
            "full_name": full_name, "email": email,
        })
    if db.query(User).filter(User.email == email).first():
        return templates.TemplateResponse("register.html", {
            "request": request, "error": "Email already registered.",
            "full_name": full_name, "email": email,
        })
    user = User(full_name=full_name, email=email,
                password_hash=hash_password(password), role="USER")
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_session(db, user.id)
    resp = RedirectResponse("/dashboard", status_code=302)
    resp.set_cookie("session_token", token, httponly=True,
                    max_age=SESSION_EXPIRE_HOURS * 3600)
    return resp


@app.post("/logout")
def logout(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get("session_token")
    if token:
        db.query(UserSession).filter(UserSession.token == token).delete()
        db.commit()
    resp = RedirectResponse("/login", status_code=302)
    resp.delete_cookie("session_token")
    return resp



# DASHBOARD

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=302)
    apps = (db.query(CreditApplication)
            .filter(CreditApplication.user_id == user.id)
            .order_by(CreditApplication.created_at.desc())
            .all())
    return templates.TemplateResponse("dashboard.html", {
        "request": request, "user": user,
        "sidebar": _sidebar("dashboard", user.role),
        "applications": apps, "result": None,
        "explanations": None, "recommendations": None,
    })



# APPLICATIONS

@app.post("/applications")
def create_application(
    request: Request,
    age: int = Form(...),
    monthly_income: float = Form(...),
    existing_debt: float = Form(...),
    loan_amount: float = Form(...),
    loan_term: int = Form(...),
    employment_status: str = Form(...),
    employment_years: float = Form(...),
    credit_history_score: int = Form(...),
    previous_delays: int = Form(...),
    loan_purpose: str = Form(...),
    db: Session = Depends(get_db),
):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=302)

    # Save application
    app_obj = CreditApplication(
        user_id=user.id,
        age=age,
        monthly_income=monthly_income,
        existing_debt=existing_debt,
        loan_amount=loan_amount,
        loan_term=loan_term,
        employment_status=employment_status,
        employment_years=employment_years,
        credit_history_score=credit_history_score,
        previous_delays=previous_delays,
        loan_purpose=loan_purpose,
        status="pending",
    )
    db.add(app_obj)
    db.commit()
    db.refresh(app_obj)

    # Run prediction
    models = get_models()
    features = extract_features(app_obj)
    income = max(monthly_income, 1)
    dti = existing_debt / income
    result = predict_all_models(models, features, loan_amount, dti, previous_delays)

    # Save predictions
    feature_vals = features[0].tolist()
    explanations = build_explanations(models, features, feature_vals)
    ensemble_decision = result["ensemble"]["decision"]
    recommendations = build_recommendations(explanations, ensemble_decision)

    for model_name, res in result.items():
        pred = PredictionResult(
            application_id=app_obj.id,
            model_name=model_name,
            probability_default=res["probability_default"],
            probability_delinquency=res["probability_delinquency"],
            risk_level=res["risk_level"],
            expected_credit_loss=res["expected_credit_loss"],
            decision=res["decision"],
        )
        db.add(pred)
        db.flush()
        # Save explanations for ensemble only
        if model_name == "ensemble":
            for ex in explanations:
                fe = FeatureExplanation(
                    prediction_id=pred.id,
                    feature_name=ex["feature_name"],
                    feature_value=ex["feature_value"],
                    effect_direction=ex["effect_direction"],
                    importance_value=ex["importance_value"],
                    explanation_text=ex["explanation_text"],
                )
                db.add(fe)

    app_obj.status = ensemble_decision.lower().replace(" ", "_")
    db.commit()

    # Re-fetch apps for history
    apps = (db.query(CreditApplication)
            .filter(CreditApplication.user_id == user.id)
            .order_by(CreditApplication.created_at.desc())
            .all())

    # Attach feature_human to explanations for template
    for ex in explanations:
        ex["feature_human"] = FEATURE_HUMAN.get(ex["feature_name"], ex["feature_name"])

    return templates.TemplateResponse("dashboard.html", {
        "request": request, "user": user,
        "sidebar": _sidebar("dashboard", user.role),
        "applications": apps,
        "result": result,
        "explanations": explanations,
        "recommendations": recommendations,
        "success": f"Application #{app_obj.id} submitted and scored successfully.",
    })


@app.get("/applications/{app_id}", response_class=HTMLResponse)
def application_detail(app_id: int, request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=302)

    app_obj = db.query(CreditApplication).filter(CreditApplication.id == app_id).first()
    if not app_obj:
        raise HTTPException(status_code=404, detail="Application not found")
    if app_obj.user_id != user.id and user.role != "ADMIN":
        raise HTTPException(status_code=403, detail="Access denied")

    predictions = (db.query(PredictionResult)
                   .filter(PredictionResult.application_id == app_id)
                   .all())

    # Get explanations for ensemble prediction
    ens_pred = next((p for p in predictions if p.model_name == "ensemble"), None)
    explanations = []
    if ens_pred:
        raw_exps = ens_pred.explanations
        explanations = [{
            "feature_human": FEATURE_HUMAN.get(e.feature_name, e.feature_name),
            "feature_name": e.feature_name,
            "feature_value": e.feature_value,
            "effect_direction": e.effect_direction,
            "importance_value": e.importance_value,
            "explanation_text": e.explanation_text,
        } for e in raw_exps]

    recommendations = []
    if ens_pred and explanations:
        recommendations = build_recommendations(explanations, ens_pred.decision)

    return templates.TemplateResponse("app_detail.html", {
        "request": request, "user": user,
        "sidebar": _sidebar("dashboard", user.role),
        "app": app_obj,
        "predictions": predictions,
        "explanations": explanations,
        "recommendations": recommendations,
    })



# BATCH

BATCH_RESULTS_DIR = BASE_DIR / "data" / "batch_results"
BATCH_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/batch", response_class=HTMLResponse)
def batch_get(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=302)
    history = db.query(BatchAnalysis).order_by(BatchAnalysis.created_at.desc()).all()
    return templates.TemplateResponse("batch.html", {
        "request": request, "user": user,
        "sidebar": _sidebar("batch", user.role),
        "history": history, "batch_result": None, "model_stats": None,
    })


@app.get("/batch/template")
def batch_template():
    columns = [
        "monthly_income", "existing_debt", "loan_amount", "loan_term",
        "employment_status", "employment_years", "age",
        "credit_history_score", "previous_delays", "loan_purpose",
    ]
    rows = [
        [250000, 50000, 1000000, 24, "employed", 3, 32, 680, 0, "personal"],
        [180000, 90000, 500000, 12, "self-employed", 1, 45, 550, 2, "business"],
        [120000, 0, 200000, 6, "part-time", 0.5, 25, 450, 0, "education"],
    ]
    content = ",".join(columns) + "\n"
    for row in rows:
        content += ",".join(str(v) for v in row) + "\n"
    return StreamingResponse(
        io.BytesIO(content.encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=batch_template.csv"},
    )


@app.post("/batch/upload")
async def batch_upload(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=302)

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        history = db.query(BatchAnalysis).order_by(BatchAnalysis.created_at.desc()).all()
        return templates.TemplateResponse("batch.html", {
            "request": request, "user": user,
            "sidebar": _sidebar("batch", user.role),
            "history": history, "batch_result": None, "model_stats": None,
            "error": f"Could not parse CSV: {e}",
        })

    models = get_models()
    approved = rejected = manual = 0

    # Per-model accumulators
    MODEL_KEYS = ["logistic_regression", "random_forest", "gradient_boosting", "neural_network", "ensemble"]
    acc: dict = {k: {"pds": [], "ecls": [], "approved": 0, "rejected": 0, "manual": 0,
                     "low": 0, "medium": 0, "high": 0} for k in MODEL_KEYS}

    for _, row in df.iterrows():
        try:
            feats = features_from_dict(row.to_dict())
            income = max(float(row.get("monthly_income", 1) or 1), 1)
            existing_debt = float(row.get("existing_debt", 0) or 0)
            loan_amount = float(row.get("loan_amount", 0) or 0)
            dti = existing_debt / income
            delays = int(row.get("previous_delays", 0) or 0)
            res = predict_all_models(models, feats, loan_amount, dti, delays)

            for mkey in MODEL_KEYS:
                m = res[mkey]
                acc[mkey]["pds"].append(m["probability_default"])
                acc[mkey]["ecls"].append(m["expected_credit_loss"])
                dec = m["decision"]
                if dec == "Approved":
                    acc[mkey]["approved"] += 1
                elif dec == "Rejected":
                    acc[mkey]["rejected"] += 1
                else:
                    acc[mkey]["manual"] += 1
                lvl = m["risk_level"]
                if lvl == "Low":
                    acc[mkey]["low"] += 1
                elif lvl == "Medium":
                    acc[mkey]["medium"] += 1
                else:
                    acc[mkey]["high"] += 1

            # Ensemble decision for overall counters
            ens_dec = res["ensemble"]["decision"]
            if ens_dec == "Approved":
                approved += 1
            elif ens_dec == "Rejected":
                rejected += 1
            else:
                manual += 1
        except Exception:
            pass

    total = len(df)
    avg_pd = float(np.mean(acc["ensemble"]["pds"])) if acc["ensemble"]["pds"] else 0.0

    # Build per-model summary
    model_stats = {}
    for mkey in MODEL_KEYS:
        a = acc[mkey]
        n = len(a["pds"]) or 1
        model_stats[mkey] = {
            "display_name": MODEL_DISPLAY_NAMES.get(mkey, mkey),
            "avg_pd": round(float(np.mean(a["pds"])) * 100, 1) if a["pds"] else 0,
            "avg_ecl": round(float(np.mean(a["ecls"])), 0) if a["ecls"] else 0,
            "approved": a["approved"],
            "rejected": a["rejected"],
            "manual": a["manual"],
            "approval_rate": round(a["approved"] / n * 100, 1),
            "rejection_rate": round(a["rejected"] / n * 100, 1),
            "low": a["low"],
            "medium": a["medium"],
            "high": a["high"],
        }

    batch = BatchAnalysis(
        filename=file.filename,
        total_records=total,
        approved_count=approved,
        rejected_count=rejected,
        manual_review_count=manual,
        avg_pd=avg_pd,
        model_stats_json=json.dumps(model_stats),
    )
    db.add(batch)
    db.commit()
    db.refresh(batch)

    history = db.query(BatchAnalysis).order_by(BatchAnalysis.created_at.desc()).all()
    return templates.TemplateResponse("batch.html", {
        "request": request, "user": user,
        "sidebar": _sidebar("batch", user.role),
        "history": history,
        "batch_result": batch,
        "model_stats": model_stats,
    })





# ADMIN

@app.get("/admin", response_class=HTMLResponse)
def admin_dashboard(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user:
        return RedirectResponse("/login", status_code=302)
    if user.role != "ADMIN":
        raise HTTPException(status_code=403, detail="Admin only")

    total_users = db.query(func.count(User.id)).scalar()
    total_apps = db.query(func.count(CreditApplication.id)).scalar()

    all_ens = (db.query(PredictionResult)
               .filter(PredictionResult.model_name == "ensemble")
               .all())
    approved = sum(1 for p in all_ens if p.decision == "Approved")
    rejected = sum(1 for p in all_ens if p.decision == "Rejected")
    manual = sum(1 for p in all_ens if p.decision == "Manual Review")
    avg_pd = float(np.mean([p.probability_default for p in all_ens])) if all_ens else 0.0
    total_batches = db.query(func.count(BatchAnalysis.id)).scalar()
    approval_rate = (approved / len(all_ens) * 100) if all_ens else 0.0

    apps = (db.query(CreditApplication)
            .order_by(CreditApplication.created_at.desc())
            .limit(50).all())
    users = db.query(User).order_by(User.created_at.desc()).all()

    model_meta = None
    meta_path = MODELS_DIR / "meta.json"
    if meta_path.exists():
        model_meta = json.loads(meta_path.read_text())

    return templates.TemplateResponse("admin.html", {
        "request": request, "user": user,
        "sidebar": _sidebar("admin", user.role),
        "stats": {
            "total_users": total_users,
            "total_apps": total_apps,
            "approved": approved,
            "rejected": rejected,
            "manual": manual,
            "avg_pd": avg_pd,
            "total_batches": total_batches,
            "approval_rate": approval_rate,
        },
        "applications": apps,
        "users": users,
        "model_meta": model_meta,
    })


@app.post("/admin/retrain", response_class=HTMLResponse)
def admin_retrain(request: Request, db: Session = Depends(get_db)):
    user = get_current_user(request, db)
    if not user or user.role != "ADMIN":
        raise HTTPException(status_code=403)
    global _models_cache
    meta = train_and_save_models()
    _models_cache = None  # Force reload
    return templates.TemplateResponse("retrain.html", {
        "request": request, "user": user, "meta": meta,
    })



# REST API ENDPOINTS (for Swagger / external use)

from pydantic import BaseModel


class ApplicationIn(BaseModel):
    monthly_income: float
    existing_debt: float
    loan_amount: float
    loan_term: int
    employment_status: str
    employment_years: float
    age: int
    credit_history_score: int
    previous_delays: int
    loan_purpose: str


class PredictionOut(BaseModel):
    application_id: int
    ensemble_decision: str
    ensemble_pd: float
    ensemble_risk_level: str
    ensemble_ecl: float
    models: dict


@app.post("/api/predict", response_model=PredictionOut, tags=["API"])
def api_predict(data: ApplicationIn, request: Request, db: Session = Depends(get_db)):
    """
    REST endpoint: submit an application and get predictions from all models.
    Requires session cookie (login first via /login).
    """
    user = get_current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    app_obj = CreditApplication(
        user_id=user.id, **data.dict(), status="pending"
    )
    db.add(app_obj)
    db.commit()
    db.refresh(app_obj)

    models = get_models()
    features = extract_features(app_obj)
    income = max(data.monthly_income, 1)
    dti = data.existing_debt / income
    result = predict_all_models(models, features, data.loan_amount, dti, data.previous_delays)

    ens = result["ensemble"]
    return PredictionOut(
        application_id=app_obj.id,
        ensemble_decision=ens["decision"],
        ensemble_pd=ens["probability_default"],
        ensemble_risk_level=ens["risk_level"],
        ensemble_ecl=ens["expected_credit_loss"],
        models=result,
    )


@app.get("/api/applications", tags=["API"])
def api_list_applications(request: Request, db: Session = Depends(get_db)):
    """List current user's applications."""
    user = get_current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    apps = (db.query(CreditApplication)
            .filter(CreditApplication.user_id == user.id)
            .order_by(CreditApplication.created_at.desc())
            .all())
    return [{"id": a.id, "loan_amount": a.loan_amount,
             "created_at": str(a.created_at), "status": a.status} for a in apps]


@app.get("/api/admin/statistics", tags=["API (Admin)"])
def api_statistics(request: Request, db: Session = Depends(get_db)):
    """Admin statistics endpoint."""
    user = get_current_user(request, db)
    if not user or user.role != "ADMIN":
        raise HTTPException(status_code=403)
    all_ens = db.query(PredictionResult).filter(PredictionResult.model_name == "ensemble").all()
    return {
        "total_users": db.query(func.count(User.id)).scalar(),
        "total_applications": db.query(func.count(CreditApplication.id)).scalar(),
        "approved": sum(1 for p in all_ens if p.decision == "Approved"),
        "rejected": sum(1 for p in all_ens if p.decision == "Rejected"),
        "manual_review": sum(1 for p in all_ens if p.decision == "Manual Review"),
        "avg_pd": float(np.mean([p.probability_default for p in all_ens])) if all_ens else 0,
    }



# ENTRY POINT

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "seed":
        print("[Seed] Initialising database and training models...")
        Base.metadata.create_all(bind=engine)
        seed()
        print("[Seed] Done. Run: uvicorn main:app --reload")
    else:
        import uvicorn
        seed()  # Always ensure demo data + models exist on startup
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)