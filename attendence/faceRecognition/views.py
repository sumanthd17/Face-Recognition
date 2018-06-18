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

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    Trains a k-nearest neighbors classifier for face recognition.
    :param train_dir: directory that contains a sub-directory for each known person, with its name.
     (View in source code to see train_dir example tree structure)
     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...
    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Choose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf

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

def show_prediction_labels_on_image(img_path, predictions,data,counter):
    """
    Shows the face recognition results visually.
    :param img_path: path to image to be recognized
    :param predictions: results of the predict function
    :return:
    """
    PATH = './AvailableSessions/' + str(data['classRoom']) + '/' + str(data['courseNumber']+'/'+'FramePictures')
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    #pil_image.show()
    #os.mkdir('FramePictures')
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    print(os.path.abspath(PATH))
    pil_image.save(PATH + '/' + 'Frame'+str(counter)+ '.jpg', 'JPEG', quality=80, optimize=True, progressive=True)

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
      ],
      "SECURITY_KEY": "QWERTYUIOPASDFGH",
      "SECURITY_CODE": "ZXCVBNMASDFGHJKL",
      #"CIPHER": b':\xdd\n\x8b\xb5\xdf\xdfb\x07\xd8'
      "CIPHER": ':Ý\n\x8bµßßb\x07Ø',
      "MESSAGE": "Attendence"
    }
    data1={}
    for i in range(0,len(data["studentlist"])):
      for key in data["studentlist"][i].keys():
       data1[key]=0

<<<<<<< HEAD
    import cv2
    import math
    import json

    with open('config.json') as json_data:
        config = json.load(json_data)
        print(config)

    from Crypto.Cipher import AES
    obj = AES.new(data['SECURITY_KEY'], AES.MODE_CFB, data['SECURITY_CODE'])
    message = 'Attendence'
    cipher = data['CIPHER']
    cipher = cipher.encode('ISO-8859-1')
    print(cipher)
    obj2 = AES.new(config['SECURITY']['KEY'], AES.MODE_CFB, config['SECURITY']['CODE'])
    DECODED = obj2.decrypt(cipher).decode('utf-8')
    print(DECODED)

    if DECODED == config['SECURITY']['MESSAGE']:
        PATH = str(config['PATHS']['Sessions']) + str(data['classRoom']) + '/' + str(data['courseNumber'])
        vidcap = cv2.VideoCapture(PATH + '/AttendenceVideo.mp4')
        success,image = vidcap.read()
        success = True
        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        div = math.ceil(length / 5)
        count = 0

        while success:
            success,image = vidcap.read()

            if count%div == 0 :
                 cv2.imwrite(PATH + '/Images/frame%d.jpg'%count,image)
            count+=1


        data.update(studentlist=data1)
        for image_file in os.listdir(PATH + '/Images'):
            full_file_path = os.path.join(PATH + '/Images', image_file)
            if not os.path.exists(PATH + '/trained_knn_model.clf'):
                print("Training KNN classifier...")
                classifier = train(PATH + '/KnownImages', model_save_path=PATH + "/trained_knn_model.clf", n_neighbors=2)
                print("Training complete!")

            print("Looking for faces in {}".format(image_file))

            # Find all people in the image using a trained classifier model
            # Note: You can pass in either a classifier file name or a classifier model instance
            predictions = predict(full_file_path, model_path=PATH + "/trained_knn_model.clf")

            print(data)
            # Print results on the console
            for name, (top, right, bottom, left) in predictions:
                print("- Found {} at ({}, {})".format(name, left, top))
                if name in data['studentlist']:
                    print('qwe')
                    data1[name] += 1
                    print(data['studentlist'][name])
            show_prediction_labels_on_image(os.path.join(PATH + '/Images', image_file), predictions,data)
            print(predictions)
            print(data)
        data["studentlist"]=[]
        for key in data1.keys():
          p={}
          p[key]=data1[key]
          data["studentlist"].append(p)
        data["imagepaths"]=[]
        p={}
        p["Frame1"]='Frame1.jpg'
        p["Frame2"]='Frame2.jpg'
        p["Frame3"]='Frame3jpg'
        p["Frame4"]='Frame4.jpg'
        p["Frame5"]='Frame5.jpg'
        data["imagepaths"].append(p)
        return JsonResponse(data)
    else:
        data['status'] = 'error occured during validation'
        data['error'] = 'UNAUTHORISED ACCESS'
        return JsonResponse(data)
=======
    data.update(studentlist=data1)
    print(data["studentlist"])
    PATH = './AvailableSessions/' + str(data['classRoom']) + '/' + str(data['courseNumber'])
    counter=1
    for image_file in os.listdir(PATH + '/Images'):
        full_file_path = os.path.join(PATH + '/Images', image_file)
        if not os.path.exists(PATH + '/trained_knn_model.clf'):
            print("Training KNN classifier...")
            classifier = train(PATH + '/KnownImages', model_save_path=PATH + "/trained_knn_model.clf", n_neighbors=2)
            print("Training complete!")

        print("Looking for faces in {}".format(image_file))

        # Find all people in the image using a trained classifier model
        # Note: You can pass in either a classifier file name or a classifier model instance
        predictions = predict(full_file_path, model_path=PATH + "/trained_knn_model.clf")


        # Print results on the console
        
        for name, (top, right, bottom, left) in predictions:
            print("- Found {} at ({}, {})".format(name, left, top))
            if name in data['studentlist']:
                data1[name]+=1
                print(data['studentlist'][name])
        show_prediction_labels_on_image(os.path.join(PATH + '/Images', image_file), predictions,data,counter)
        print(predictions)
        counter+=1
    data["studentlist"]=[]
    for key in data1.keys():
      p={}
      p[key]=data1[key]
      data["studentlist"].append(p)
    data["imagepaths"]=[]
    p={}
    p["Frame1"]='Frame1.jpg'
    p["Frame2"]='Frame2.jpg'
    p["Frame3"]='Frame3jpg'
    p["Frame4"]='Frame4.jpg'
    p["Frame5"]='Frame5.jpg'
    data["imagepaths"].append(p)        
    return JsonResponse(data)
>>>>>>> 2c4d03c375ec7c22cc841c3168c5fc9931f9af7b


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
