from django.db import models
from django.urls import reverse

# Create your models here.
class Session(models.Model):
    session_name = models.CharField(max_length=20)
    session_strength = models.IntegerField()

    def get_absolute_url(self):
        return reverse('faceRecognition:index')
