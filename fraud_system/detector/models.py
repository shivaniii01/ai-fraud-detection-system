from django.db import models

class Transaction(models.Model):
    amount = models.FloatField()
    is_fraud = models.BooleanField()
    risk_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Transaction {self.id} - Fraud: {self.is_fraud}"
