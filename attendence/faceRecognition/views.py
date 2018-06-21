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
import cv2
import math
import json
import logging
import shutil
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
    if not os.path.exists(str(config['PATHS']['Sessions']) + pk):
        os.mkdir(str(config['PATHS']['Sessions']) + pk)
        #HttpResponse('directory created!!')
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

def show_prediction_labels_on_image(img_path, predictions, data, counter):
    """
    Shows the face recognition results visually.
    :param img_path: path to image to be recognized
    :param predictions: results of the predict function
    :return:
    """
    with open('config.json') as json_data:
        config = json.load(json_data)
        print(config)

    PATH = str(config['PATHS']['Sessions']) + str(data['classRoom']) + '/' + str(data['courseNumber']+'/'+'FramePictures')
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        if name == 'unknown':
            continue
        else:
            name = 'P'

        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height + 20), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height + 10), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    #pil_image.show()
    #os.mkdir('FramePictures')
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    print(os.path.abspath(PATH))

    #ImgSavePath = '../../../usr/local/apache-tomcat-8.5.8/webapps/Edu_Erp_IIITS/assets/studentAttendanceImages'
    #pil_image.save(ImgSavePath + '/' + 'recognisedFaces_Frame'+str(counter)+ '.jpg', 'JPEG', quality=80, optimize=True, progressive=True)
    pil_image.save('recognisedFaces_Frame'+str(counter)+ '.jpg', 'JPEG', quality=80, optimize=True, progressive=True)

    # for saving unrecognised Images
    pil_uk_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_uk_image)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        if name == 'unknown':
            name = 'UK'
        else:
            continue

        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height + 20), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height + 10), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs
    del draw
    #UKImgSavePath = '../../../usr/local/apache-tomcat-8.5.8/webapps/Edu_Erp_IIITS/assets/studentAttendanceImages/unrecognised'
    #pil_image.save(UKImgSavePath + '/' + 'unrecognisedFaces_Frame'+str(counter)+ '.jpg', 'JPEG', quality=80, optimize=True, progressive=True)
    pil_uk_image.save('unrecognisedFaces_Frame'+str(counter)+ '.jpg', 'JPEG', quality=80, optimize=True, progressive=True)

@csrf_exempt
def TakeAttendence(request):

    # Reading the config file
    with open('config.json') as json_data:
        config = json.load(json_data)
        print(config)

    if str(config['METHOD']['REQ_METHOD']) == 'XML':
        # accepting the data from request and storing it in data.xml and log the action
        # try:
        #     data = request.body
        # except:
        #     logging.error(str(datetime.datetime.now())+"\tXML Data not received in correct format.")   #Logging Error message if data not received in correct format.
        # logging.info(str(datetime.datetime.now())+"\tXML Data received in correct format.")   #Logging message that data received in correct format.
            
        # data = data.decode('utf-8')
        # with open('./data.xml', 'w') as file:
        #     file.write(data)

        # extracting all the values from the XML tree structure and creating a JSON structure
        from xml.dom.minidom import parse, Node
        xmlTree = parse("./data.xml")
        #get all departments
        data={}
        for node1 in xmlTree.getElementsByTagName("courseNumber") :
            for node2 in node1.childNodes:
                if(node2.nodeType == Node.TEXT_NODE) :
                    data['courseNumber']=node2.data
        for node1 in xmlTree.getElementsByTagName("classRoom") :
            for node2 in node1.childNodes:
                if(node2.nodeType == Node.TEXT_NODE) :
                    data['classRoom']=node2.data
        for node1 in xmlTree.getElementsByTagName("attendanceDate") :
            for node2 in node1.childNodes:
                if(node2.nodeType == Node.TEXT_NODE) :
                    data['attendanceDate']=node2.data
        for node1 in xmlTree.getElementsByTagName("toPeriod") :
            for node2 in node1.childNodes:
                if(node2.nodeType == Node.TEXT_NODE) :
                    data['toPeriod']=node2.data
        for node1 in xmlTree.getElementsByTagName("fromPeriod") :
            for node2 in node1.childNodes:
                if(node2.nodeType == Node.TEXT_NODE) :
                    data['fromPeriod']=node2.data
        for node1 in xmlTree.getElementsByTagName("error") :
            for node2 in node1.childNodes:
                if(node2.nodeType == Node.TEXT_NODE) :
                    data['error']=node2.data
        for node1 in xmlTree.getElementsByTagName("status") :
            for node2 in node1.childNodes:
                if(node2.nodeType == Node.TEXT_NODE) :
                    data['status']=node2.data
        for node1 in xmlTree.getElementsByTagName("SECURITY_KEY") :
            for node2 in node1.childNodes:
                if(node2.nodeType == Node.TEXT_NODE) :
                    data['SECURITY_KEY']=node2.data
        for node1 in xmlTree.getElementsByTagName("SECURITY_CODE") :
            for node2 in node1.childNodes:
                if(node2.nodeType == Node.TEXT_NODE) :
                    data['SECURITY_CODE']=node2.data
        for node1 in xmlTree.getElementsByTagName("CIPHER") :
            for node2 in node1.childNodes:
                if(node2.nodeType == Node.TEXT_NODE) :
                    data['CIPHER']=node2.data
        for node1 in xmlTree.getElementsByTagName("MESSAGE") :
            for node2 in node1.childNodes:
                if(node2.nodeType == Node.TEXT_NODE) :
                    data['MESSAGE']=node2.data
        studentlist = xmlTree.getElementsByTagName('studentlist')
        data['studentlist']=[]
        for i in studentlist:
            p={}
            p[str(i.attributes['rollNumber'].value)]=0
            data['studentlist'].append(p)



        print(data)

    elif str(config['METHOD']['REQ_METHOD']) == 'CSV':
        # accepting data from request and storing it in file.txt and log the action
        # data = request.body
        # try:
        #     data = data.decode('utf-8')
        # except:
        #     logging.error(str(datetime.datetime.now())+"\tCSV Data not received in correct format.")   #Logging Error message if data not received in correct format.
        # logging.info(str(datetime.datetime.now())+"\tCSV Data received in correct format.")   #Logging message that data received in correct format.
        # with open('file.txt', 'w') as file:
        #     file.write(data)

        # extracting the data from the csv data structure and stroing it in JSON data structures
        # Here we have the option of changing the delimiter
        import csv
        l=[]
        with open('file.txt','r') as csvfile:
          spamreader = csv.reader(csvfile,delimiter=config['METHOD']['DELIMITOR'],quotechar='|')
          for row in spamreader:
            l.append(row)
        count=0
        counter=0
        x=0
        y=0
        data={}
        for item in l:
            count+=1
            counter+=1
            if item[0]=='roll numbers start':
                x=count
            if item[0]=='roll numbers end':
                y=counter-1

            if item[0]=='classRoom':
                data[item[0]]=str(item[1])
            if item[0]=='courseNumber':
                data[item[0]]=str(item[1])
            if item[0]=='attendanceDate':
                data[item[0]]=str(item[1])
            if item[0]=='fromPeriod':
                data[item[0]]=str(item[1])
            if item[0]=='toPeriod':
                data[item[0]]=str(item[1])
            if item[0]=='status':
                data[item[0]]=""
            if item[0]=='error':
                data[item[0]]=""
            if item[0]=='SECURITY_KEY':
                data[item[0]]=str(item[1])
            if item[0]=='SECURITY_CODE':
                data[item[0]]=str(item[1])
            if item[0]=='MESSAGE':
                data[item[0]]=str(item[1])
            if item[0]=='CIPHER':
                data[item[0]]=str(item[1])

        data["studentlist"]=[]
        for i in range(x,y):
            p={}
            p[l[i][0]]=0
            data["studentlist"].append(p)
        print(data)

    elif str(config['METHOD']['REQ_METHOD']) == 'JSON':
        # accepting the data from request and storing it in a JSON data structure and log the action
        data = request.body
        try:
            data = json.loads(str(data, 'utf-8'))
        except :
            logging.error(str(datetime.datetime.now())+"\tJSON Data not received in correct format.")   #Logging Error message if data not received in correct format.
        logging.info(str(datetime.datetime.now())+"\tJSON Data received in correct format.")   #Logging message that data received in correct format.
        print(data)

        # data =   {
        #   "classRoom": "102",
        #   "courseNumber": "ICS200",
        #   "attendanceDate": "08/06/2018",
        #   "fromPeriod": "07:00",
        #   "toPeriod": "07:30",
        #   "status": "",
        #   "error": "",
        #   "studentlist": [
        #     {
        #       "DSC_0688": 0
        #     },
        #     {
        #       "DSC_0626": 0
        #     },
        #     {
        #       "DSC_0011": 0
        #     },
        #     {
        #       "DSC_0847": 0
        #     },
        #     {
        #       "DSC_0824": 0
        #     }
        #   ],
        #   "SECURITY_KEY": "QWERTYUIOPASDFGH",
        #   "SECURITY_CODE": "ZXCVBNMASDFGHJKL",
        #   #"CIPHER": b':\xdd\n\x8b\xb5\xdf\xdfb\x07\xd8'
        #   "CIPHER": ':Ý\n\x8bµßßb\x07Ø',
        #   "MESSAGE": "Attendence"
        # }
    data1={}
    for i in range(0,len(data["studentlist"])):
      for key in data["studentlist"][i].keys():
       data1[key]=0
    data.update(studentlist=data1)

    # encryption of cipher and checking it with config file data
    from Crypto.Cipher import AES
    obj = AES.new(data['SECURITY_KEY'], AES.MODE_CFB, data['SECURITY_CODE'])
    message = data['MESSAGE']
    cipher1 = obj.encrypt(message)
    obj2 = AES.new(config['SECURITY']['KEY'], AES.MODE_CFB, config['SECURITY']['CODE'])
    message2 = config['SECURITY']['MESSAGE']
    cipher2 = obj2.encrypt(message2)
    #DECODED = obj2.decrypt(cipher).decode('utf-8')
    #print(DECODED)

    # chech the working of cipher later
    if data['MESSAGE'] == 'Attendence':
        PATH = str(config['PATHS']['Sessions']) + str(data['classRoom']) + '/' + str(data['courseNumber'])
        # extracting the 5 frames from the video file (Equally spaced)
        '''vidcap = cv2.VideoCapture(PATH + '/AttendenceVideo.mp4')
        success,image = vidcap.read()
        success = True
        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        div = math.ceil(length / 5)
        count = 0

        while success:
            success,image = vidcap.read()

            if count%div == 0 :
                 cv2.imwrite(PATH + '/Images/frame%d.jpg'%count,image) # storing the images in PATH = str(config['PATHS']['Sessions']) + str(data['classRoom']) + '/' + str(data['courseNumber'])/Images folder
            count+=1
        count = 0'''

        # for all the images in the Images folder(group photos) face recognition is appilied 
        for image_file in os.listdir(PATH + '/Images'):
            full_file_path = os.path.join(PATH + '/Images', image_file)

            if config['USE']['DATABASE'] == 'YES':
                # connecting to the database 
                import MySQLdb
                conn = MySQLdb.connect(user=config['DATABASE']['USERNAME'], passwd=config['DATABASE']['PASSWORD'], db=config['DATABASE']['DB_NAME'])
                cursor = conn.cursor()

                # RAW mysql query for getting images and roll numbers
                cursor.execute("SELECT " + config['DATABASE']['PHOTO_CLM'] + ',' + config['DATABASE']['ROLL_CLM'] +  " FROM " + config['DATABASE']['TABLE_NAME'])
                row = cursor.fetchone()
                # accessing one row of the table at a time
                while row is not None:
                    from PIL import Image
                    import base64

                    img_str = row[0]
                    roll = row[1]
                    # converting the bas64 str to image and saving the photo to KnoenImages directory
                    imgdata = base64.b64decode(img_str)
                    if not os.path.exists(PATH + '/KnownImages'):
                        os.mkdir(PATH + '/KnownImages')
                    if not os.path.exists(PATH + '/KnownImages/' + str(roll)):
                        os.mkdir(PATH + '/KnownImages/' + str(roll))
                    filename = PATH + '/KnownImages/' + str(roll) + '/' + str(roll) + '.jpg'
                    with open(filename, 'wb') as f:
                        f.write(imgdata)
                    row = cursor.fetchone()


            # IF a trained classifier already exits for the that class training is skipped
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
                    data1[name] += 1
            count += 1
            show_prediction_labels_on_image(os.path.join(PATH + '/Images', image_file), predictions,data, count)

        # deleting the KnownImages folder after he attendence has been taken
        # optional - delete the classifer after he attendence has been taken
        if config['USE']['DATABASE'] == 'YES':
            shutil.rmtree(PATH + '/KnownImages')
            os.remove(PATH + '/trained_knn_model.clf')
        elif config['USE']['DATABASE'] == 'NO':
            #os.remove(PATH + '/trained_knn_model.clf')
            print('ggwp')

        # restructuring the data accorinnd to the need of ERP
        data["studentlist"]=[]
        for key in data1.keys():
          p={}
          p[key]=data1[key]
          data["studentlist"].append(p)
        data["imagepaths"]=[]
        p={}
        p["Frame1"]='Frame1.jpg'
        p["Frame2"]='Frame2.jpg'
        p["Frame3"]='Frame3.jpg'
        p["Frame4"]='Frame4.jpg'
        p["Frame5"]='Frame5.jpg'
        data["imagepaths"].append(p)

        # restructuring the data in XML format and rendering out XML response
        if config['METHOD']['RSP_METHOD'] == 'XML':
            import xml.etree.cElementTree as ET
            root = ET.Element("data")
            cr = ET.SubElement(root, "classRoom").text = data['classRoom']
            cn = ET.SubElement(root, "courseNumber").text = data['courseNumber']
            ad = ET.SubElement(root, "attendanceDate").text = data['attendanceDate']
            fp = ET.SubElement(root, "fromPeriod").text = data['fromPeriod']
            tp = ET.SubElement(root, "toPeriod").text = data['toPeriod']
            err = ET.SubElement(root, "error").text = data['error']
            sta = ET.SubElement(root, "status").text = data['status']
            sec_key = ET.SubElement(root, "SECURITY_KEY").text = data['SECURITY_KEY']
            sec_code = ET.SubElement(root, "SECURITY_CODE").text = data['SECURITY_CODE']
            #ci = ET.SubElement(root, "CIPHER").text = data['CIPHER']
            msg = ET.SubElement(root, "MESSAGE").text = data['MESSAGE']

            for i in data['studentlist']:
                for j in i.keys():
                    sl = ET.SubElement(root, "studentlist",rollNumber=j).text = str(i[j])

            for i in data['imagepaths']:
                for j in i.keys():
                    sl = ET.SubElement(root, "imagepaths",rollNumber=j).text = str(i[j])

            tree = ET.ElementTree(root)
            tree.write("output.xml")
            logging.info(str(datetime.datetime.now())+"\t"+str(len(data['studentlist']))+" students XML data sent successfully.")         #Logging info that data has been sent successfully.
            return HttpResponse(open('output.xml').read())

        # restructuring the data in CSV format and rendering out in plain text fromat
        elif config['METHOD']['RSP_METHOD'] == 'CSV':
            f = open('output.txt','w')
            for i in data:
                if(i=='studentlist'):
                    f.write('studentlist\nroll numbers start\n')
                    for j in data[i]:
                        for k in j.keys():
                            f.write(str(k)+str(config['METHOD']['DELIMITOR'])+str(j[k])+'\n')
                    f.write('roll numbers end\n')
                elif(i=='imagepaths'):
                    f.write('imagepaths\nimagepath start\n')
                    for j in data[i]:
                        for k in j.keys():
                            f.write(str(k)+str(config['METHOD']['DELIMITOR'])+str(j[k])+'\n')
                    f.write('imagepath end\n')
                else:
                    f.write(i+str(config['METHOD']['DELIMITOR'])+data[i]+'\n')

            logging.info(str(datetime.datetime.now())+"\t"+str(len(data['studentlist']))+" students CSV data sent successfully.")         #Logging info that data has been sent successfully.
            with open('output.txt', 'r') as f:
                data = f.read()
            return HttpResponse(data, content_type='text/plain')

        # rendering JSON response
        elif config['METHOD']['RSP_METHOD'] == 'JSON':
            logging.info(str(datetime.datetime.now())+"\t"+str(len(data['studentlist']))+" students JSON data sent successfully.")         #Logging info that data has been sent successfully.
            return JsonResponse(data)

    # if authorisation failed while comparing token then error is rendered
    else:
        data['status'] = 'error occured during validation'
        data['error'] = 'UNAUTHORISED ACCESS'
        logging.info(str(datetime.datetime.now())+"\tUnauthorized user trying to send and receive data.")         #Logging info that there was an unauthorized access

        data["studentlist"]=[]
        for key in data1.keys():
          p={}
          p[key]=data1[key]
          data["studentlist"].append(p)

        # restructuring the data in XML format and rendering out XML response
        if config['METHOD']['RSP_METHOD'] == 'XML':
            import xml.etree.cElementTree as ET
            root = ET.Element("data")
            cr = ET.SubElement(root, "classRoom").text = data['classRoom']
            cn = ET.SubElement(root, "courseNumber").text = data['courseNumber']
            ad = ET.SubElement(root, "attendanceDate").text = data['attendanceDate']
            fp = ET.SubElement(root, "fromPeriod").text = data['fromPeriod']
            tp = ET.SubElement(root, "toPeriod").text = data['toPeriod']
            err = ET.SubElement(root, "error").text = data['error']
            sta = ET.SubElement(root, "status").text = data['status']
            sec_key = ET.SubElement(root, "SECURITY_KEY").text = data['SECURITY_KEY']
            sec_code = ET.SubElement(root, "SECURITY_CODE").text = data['SECURITY_CODE']
            #ci = ET.SubElement(root, "CIPHER").text = data['CIPHER']
            msg = ET.SubElement(root, "MESSAGE").text = data['MESSAGE']

            for i in data['studentlist']:
                for j in i.keys():
                    sl = ET.SubElement(root, "studentlist",rollNumber=j).text = str(i[j])

            tree = ET.ElementTree(root)
            tree.write("output.xml")
            logging.info(str(datetime.datetime.now())+"\t"+str(len(data['studentlist']))+" students XML data sent successfully.")         #Logging info that data has been sent successfully.
            return HttpResponse(open('output.xml').read())

        # restructuring the data in CSV format and rendering out in plain text fromat
        elif config['METHOD']['RSP_METHOD'] == 'CSV':
            print(data)
            f = open('output.csv','w')
            for i in data:
                if(i=='studentlist'):
                    f.write('studentlist\nroll numbers start\n')
                    for j in data[i]:
                        for k in j.keys():
                            f.write(str(k)+str(config['METHOD']['DELIMITOR'])+str(j[k])+'\n')
                    f.write('roll numbers end\n')
                else:
                    f.write(i+str(config['METHOD']['DELIMITOR'])+data[i]+'\n')
            
            logging.info(str(datetime.datetime.now())+"\t"+str(len(data['studentlist']))+" students CSV data sent successfully.")         #Logging info that data has been sent successfully.
            with open('output.csv', 'r') as f:
                data = f.read()
            return HttpResponse(data, content_type='text/plain')

        # rendering JSON response
        elif config['METHOD']['RSP_METHOD'] == 'JSON':
            logging.info(str(datetime.datetime.now())+"\t"+str(len(data['studentlist']))+" students JSON data sent successfully.")         #Logging info that data has been sent successfully.
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
