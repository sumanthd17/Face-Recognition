from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.views import generic
from django.views.generic.edit import CreateView, UpdateView, DeleteView
from django.urls import reverse_lazy
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from django import forms
from .forms import UserRegistrationForm
import os
from PIL import Image  # only for testing, not required in deployment
import json

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
    if not os.path.exists('./AvailableSessions/' + pk):
        os.mkdir('./AvailableSessions/' + pk)
        HttpResponse('directory created!!')
        # here after creating the directory the images take from the camera should 
        # be saved to the images directory
        # take attendence then takes these images and finds the faces and saves all the 
        # faces to the faces directory 
    return HttpResponse(pk)

def TakeAttendence(request, pk):
    import face_recognition
    known_face_encodings = []
    known_face_names = []
    students_attendence_data = {}
    pk = str(pk)    # pk is the primary key fro the session
    # knownImages is the ID card photos folder
    for image_name in os.listdir('./KnownImages/' + pk):
        image = face_recognition.load_image_file('./KnownImages/' + pk + '/' + image_name)              #loading each image from the folder
        image_face_encoding = face_recognition.face_encodings(image)[0]                                 #find the face encoding for each known photo
        known_face_encodings.append(image_face_encoding)                                                # saving it in known_face_encodings list
        known_face_names.append(image_name)                                                             # here known_face_name is the image name AKA roll number
        students_attendence_data[image_name] = 0                                                        # defining the status of students (initial state all are assumed absent)
        #print(known_face_names)

    face_locations = []
    face_encodings = []
    # available sessions is the folder containting the photos taken from the camera
    for image_name in os.listdir('./AvailableSessions/' + pk + '/Images'):
        image = face_recognition.load_image_file('./AvailableSessions/' + pk + '/Images/' + image_name) #loading the image
        face_locations = face_recognition.face_locations(image)                                         # finding the faces in the image
        # saving the face found to new directory
        #all the faces from the images are saves into the AvailableSessions/pk/faces_from_image directory
        counter = 1
        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_image = image[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)
            req_path = './AvailableSessions/' + pk + '/faces_from_Image'
            if not os.path.exists(req_path):
                os.mkdir(req_path)
            pil_image.save(req_path + '/' + str(counter) + '.jpg', 'JPEG', quality=80, optimize=True, progressive=True)
            counter += 1
            #pil_image.show()

        face_encodings = face_recognition.face_encodings(image, face_locations)                         # findin the face encodings for the extracted faces
        
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)               # each face_encoding is compares with all the known face encodings
            name = "Unknown"                                                                            # initially all faces are unknown

            if True in matches:
                first_match_index = matches.index(True)                                                 # if a match is found the student is marked present
                name = known_face_names[first_match_index]
                students_attendence_data[name] = 1
            face_names.append(name)

    import collections
    students_attendence_data = collections.OrderedDict(sorted(students_attendence_data.items()))        # structuring the data in JSON format
    return HttpResponse(json.dumps(students_attendence_data))                                           # displaying the data on the web page

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
                return HttpResponseRedirect('/faceRecognition')
            else:
                raise forms.ValidationError('Looks like a username with that email or password already exists')

    else:
        form = UserRegistrationForm()

    return render(request, 'registration/register.html', {'form' : form})
