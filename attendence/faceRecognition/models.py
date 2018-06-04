from django.db import models
from django.urls import reverse
import jsonfield

# Create your models here.
class Session(models.Model):
    session_name = models.CharField(max_length=20)
    session_strength = models.IntegerField()

    def get_absolute_url(self):
        return reverse('faceRecognition:index')

class Attendence(models.Model):
	session_attendence = jsonfield.JSONField()
	date = models.DateTimeField()
	session_name = models.ForeignKey(Session, null=True, on_delete=models.CASCADE)