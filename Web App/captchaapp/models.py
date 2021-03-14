from django.db import models


# Create your models here.

class Inference(models.Model):
    image = models.ImageField(null=True)
    prediction = models.CharField(max_length=255)
