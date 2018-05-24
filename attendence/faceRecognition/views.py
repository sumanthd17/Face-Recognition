from django.shortcuts import render
from django.http import HttpResponse
from django.views import generic
from django.views.generic.edit import CreateView, UpdateView, DeleteView
from django.urls import reverse_lazy

# Create your views here.
from faceRecognition.models import Session

# view for the index page
class IndexView(generic.ListView):
    #return HttpResponse("You're at the face recognition index!!")
    context_object_name = 'Sessions_list'
    template_name = 'faceRecognition/index.html'

    def get_queryset(self):
        return Session.objects.all()

#view for session entry page
class SessionEntry(CreateView):
    model = Session
    # fields mentioned below become the entry rows in the generated form
    fields = ['session_name', 'session_strength']

class SessionUpdate(UpdateView):
    model = Session
    # the fields mentioned below become the entyr rows in the update form
    fields = ['session_name', 'session_strength']

class SessionDelete(DeleteView):
    model = Session
    # the delete button forwards to the url mentioned below.
    success_url = reverse_lazy('faceRecognition:index')
