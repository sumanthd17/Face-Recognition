from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
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
from faceRecognition.models import Session, Attendence
import datetime
from django.views.decorators.csrf import csrf_exempt
import logging
logging.basicConfig(filename='../LogFile.log', level=logging.DEBUG)

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

@csrf_exempt
def ActivateCamera(request, pk):
    if not os.path.exists('./AvailableSessions/' + pk):
        os.mkdir('./AvailableSessions/' + pk)
        HttpResponse('directory created!!')
        # here after creating the directory the images take from the camera should 
        # be saved to the images directory
        # take attendence then takes these images and finds the faces and saves all the 
        # faces to the faces directory 
    return HttpResponse(pk)

import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.475):
    """
    Recognizes faces in given image using a trained KNN classifier
    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


@csrf_exempt
def TakeAttendence(request):
    # data = request.body
    # try:
    #     data = json.loads(str(data, 'utf-8'))
    # except :
    #     logging.error(str(datetime.datetime.now())+"\tJSON Data not received in correct format.")   #Logging Error message if data not received in correct format.
    # print(data)
    data =   {
      "classRoom": "102",
      "courseNumber": "ICS200",
      "attendanceDate": "08/06/2018",
      "fromPeriod": "07:00",
      "toPeriod": "07:30",
      "status": "",
      "error": "",
      "studentlist": [
        {
          "DSC_0688": 0
        },
        {
          "DSC_0626": 0
        },
        {
          "DSC_0011": 0
        },
        {
          "DSC_0847": 0
        },
        {
          "DSC_0824": 0
        }
      ]
    }
    data1={}
    for i in range(0,len(data["studentlist"])):
      for key in data["studentlist"][i].keys():
       data1[key]=0

    data.update(studentlist=data1)
    PATH = './AvailableSessions/' + str(data['classRoom']) + '/' + str(data['courseNumber'])
    for image_file in os.listdir(PATH + '/Images'):
        full_file_path = os.path.join(PATH + '/Images', image_file)

        print("Looking for faces in {}".format(image_file))

        # Find all people in the image using a trained classifier model
        # Note: You can pass in either a classifier file name or a classifier model instance
        predictions = predict(full_file_path, model_path=PATH + "/trained_knn_model.clf")


        # Print results on the console
        for name, (top, right, bottom, left) in predictions:
            print("- Found {} at ({}, {})".format(name, left, top))
            if name in data['studentlist']:
                print('qwe')
                data['studentlist'][name] = 1

        print(predictions)
    return JsonResponse(data)


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
                logging.info(str(datetime.datetime.now())+"\tNew user Registered.")                     #Logging info that a new user has been registered.
                user = authenticate(username = username, password = password)
                login(request, user)
                return HttpResponseRedirect('/faceRecognition')
            else:
                raise forms.ValidationError('Looks like a username with that email or password already exists')

    else:
        form = UserRegistrationForm()

    return render(request, 'registration/register.html', {'form' : form})
