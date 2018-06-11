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
          "S20140010019": 0
        },
        {
          "S20140010002": 0
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

<<<<<<< HEAD
        print("Looking for faces in {}".format(image_file))
=======
    room = data['classRoom']
    if not os.path.exists('./AvailableSessions/' + str(room)):
        logging.error(str(datetime.datetime.now())+"\tClassroom not found and JSON data sent with error message.")  #logging error message if classroom number sent wrong
        data['error'] = 'path for pictures does not exist'
        data['status'] = 'error'
        # send response back
        data['error'] = 'PATH NOT FOUND'
        data['status'] = 'ERROR OCCURED'

    courseID = data['courseNumber']
    if not os.path.exists('./AvailableSessions/' + str(room) + '/' + str(courseID)):
        logging.error(str(datetime.datetime.now())+"\tCourse not registered for that classroom and JSON data sent with error message.") #logging data if course not registered in that specified room.
        data['error'] = 'path for classroom does not exist'
        data['status'] = 'error'
        # send response back
        data['error'] = 'PATH NOT FOUND'
        data['status'] = 'ERROR OCCURED'

    else:
        import glob
        import face_recognition

        face_locations = []
        face_encodings = []
        PATH = './AvailableSessions/' + str(room) + '/' + str(courseID)
        # encodings for the group photo
        list_of_files = glob.glob(PATH + '/*.jpg')
        latest_photo = max(list_of_files, key=os.path.getctime)
        image = face_recognition.load_image_file(str(latest_photo))
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        # saving extracted images to faces_from_Image folder
        counter = 1
        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_image = image[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)
            req_path = PATH + '/faces_from_Image/'
            if not os.path.exists(req_path):
                os.mkdir(req_path)
            pil_image.save(req_path + '/' + str(counter) + '.jpg', 'JPEG', quality=80, optimize=True, progressive=True)
            counter += 1

        # encodings for the id card photos
        known_face_encodings = []
        known_face_names = data['studentlist']
        names = []
        attendence = {}
        #for image_name in os.listdir(PATH + '/KnownImages'):
        for image_name in known_face_names:
            image = face_recognition.load_image_file(PATH + '/KnownImages/' + image_name + '.jpg') # add '.jpg' after wards
            image_face_encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(image_face_encoding)
            names.append(image_name)
            attendence[image_name] = 0
            print(image_name)

        # comparing faces
        dummy = []
        counter = 0
        for known_face_encoding in known_face_encodings:
            matches = face_recognition.compare_faces(face_encodings, known_face_encoding,tolerance = 0.45)
            #print(matches)
            ''''if True in matches:
                first_match_index = matches.index(True)
                name = names[first_match_index]
                data['studentlist'][name] = 1
                attendence[name] = 1'''
            if True in matches:
                attendence[names[counter]] = 1
            else:
                attendence[names[counter]] = 0
            counter += 1


        data['error'] = 'NO ERROR'
        data['status'] = 'SUCCESS'
        del data["studentlist"]
        data["studentlist"]=[]
        for key in data1.keys():
            p={}
            p[key]=attendence[key]
            data["studentlist"].append(p)
        
        logging.info(str(datetime.datetime.now())+"\t JsonResponse of attendance sent back.")               #Logging success message that JSON response with student attendance has been sent.
    print(dummy)
    return JsonResponse(attendence)
'''@csrf_exempt
def TakeAttendence(request, pk):
    import json
    import requests
    from django.http import StreamingHttpResponse
    url = "http://127.0.0.1:8000/faceRecognition/session/" + pk + "/ActivateCamera/"
    if request.method == 'POST':
        #return HttpResponse("qwe")
        students_attendence_data = requests.get(url).json()
        #return JsonResponse(students_attendence_data)

    import face_recognition
    known_face_encodings = []
    known_face_names = []
    pk = str(pk)    # pk is the primary key fro the session
    # knownImages is the ID card photos folder
    for image_name in students_attendence_data:
        image = face_recognition.load_image_file('./KnownImages/' + pk + '/' + image_name + '.jpg')              #loading each image from the folder
        image_face_encoding = face_recognition.face_encodings(image)[0]                                 #find the face encoding for each known photo
        known_face_encodings.append(image_face_encoding)                                                # saving it in known_face_encodings list
        known_face_names.append(image_name)                                                             # here known_face_name is the image name AKA roll number
        students_attendence_data[image_name] = 0                                                        # defining the status of students (initial state all are assumed absent)
        #print(known_face_names)
>>>>>>> 172e6d0845188c842ec185c8279c11dd78b77802

        # Find all people in the image using a trained classifier model
        # Note: You can pass in either a classifier file name or a classifier model instance
        predictions = predict(full_file_path, model_path=PATH + "/trained_knn_model.clf")


        # Print results on the console
        for name, (top, right, bottom, left) in predictions:
            print("- Found {} at ({}, {})".format(name, left, top))

        print(predictions)
    return HttpResponse('Attendence done')

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
