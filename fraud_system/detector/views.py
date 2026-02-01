import joblib
import os
import numpy as np
from django.shortcuts import render
from .models import Transaction

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

rf_model = joblib.load(os.path.join(BASE_DIR, "rf_model.pkl"))
iso_model = joblib.load(os.path.join(BASE_DIR, "iso_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))


def predict_fraud(request):
    if request.method == "POST":
        amount = float(request.POST.get("amount"))

        data = np.array([[amount]])
        data_scaled = scaler.transform(data)

        rf_pred = rf_model.predict(data_scaled)[0]
        iso_pred = iso_model.predict(data_scaled)[0]

        risk_score = (rf_pred + (1 if iso_pred == -1 else 0)) / 2
        is_fraud = risk_score > 0.5
        txn_type = request.POST.get("txn_type")
        txn_time = request.POST.get("txn_time")


        # Risk category mapping
        if risk_score < 0.3:
            risk_level = "Low Risk"
        elif risk_score < 0.7:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"

        # Decision explanation (TRL-5 level)
        reason = "Normal transaction behavior"

        if rf_pred == 1 and iso_pred == -1:
            reason = "High-risk pattern and anomaly detected"
        elif rf_pred == 1:
            reason = "Matches known fraud patterns"
        elif iso_pred == -1:
            reason = "Abnormal transaction behavior detected"
        if txn_time == "Night":
            reason += " | Night-time transaction"
        if txn_type == "Online":
             reason += " | Online transaction"

        Transaction.objects.create(
            amount=amount,
            is_fraud=is_fraud,
            risk_score=risk_score
        )

        return render(request, "result.html", {
            "amount": amount,
            "is_fraud": is_fraud,
            "risk": round(risk_score * 100, 2),
            "risk_level": risk_level,
            "reason": reason,
            "model_info": "Hybrid ML (Random Forest + Isolation Forest)",
            "txn_type": txn_type,
            "txn_time": txn_time

        })


def home(request):
    return render(request, "home.html")


def transaction_history(request):
    transactions = Transaction.objects.all().order_by("-created_at")
    return render(request, "history.html", {
        "transactions": transactions
    })


def analytics_dashboard(request):
    total = Transaction.objects.count()
    frauds = Transaction.objects.filter(is_fraud=True).count()
    safe = Transaction.objects.filter(is_fraud=False).count()

    fraud_rate = 0
    if total > 0:
        fraud_rate = round((frauds / total) * 100, 2)

    high_risk = Transaction.objects.filter(risk_score__gte=0.7).count()

    return render(request, "analytics.html", {
        "total": total,
        "frauds": frauds,
        "safe": safe,
        "fraud_rate": fraud_rate,
        "high_risk": high_risk
    })
def architecture(request):
    return render(request, "architecture.html")

