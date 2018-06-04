from django.contrib import admin

# Register your models here.
from faceRecognition.models import Session, Attendence

admin.site.register(Session)
admin.site.register(Attendence)