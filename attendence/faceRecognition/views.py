from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.views import generic
from django.views.generic.edit import CreateView, UpdateView, DeleteView
from django.urls import reverse_lazy
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from django import forms
from .forms import UserRegistrationForm

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

def ActivateCamera(request, pk):
    return HttpResponse('You are at the activate camera page!!!')

def register(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            userObj = form.cleaned_data
            username = userObj['username']
            email =  userObj['email']
            password =  userObj['password']
            if not (User.objects.filter(username=username).exists() or User.objects.filter(email=email).exists()):
                User.objects.create_user(username, email, password)
                user = authenticate(username = username, password = password)
                login(request, user)
                return HttpResponseRedirect('/')
            else:
                raise forms.ValidationError('Looks like a username with that email or password already exists')

    else:
        form = UserRegistrationForm()

    return render(request, 'registration/register.html', {'form' : form})
