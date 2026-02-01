from django.urls import path
from .views import (
    home,
    predict_fraud,
    transaction_history,
    analytics_dashboard,
    architecture   # ✅ THIS WAS MISSING
)

urlpatterns = [
    path("", home, name="home"),
    path("predict/", predict_fraud, name="predict"),
    path("history/", transaction_history, name="history"),
    path("analytics/", analytics_dashboard, name="analytics"),
    path("architecture/", architecture, name="architecture"),
]
