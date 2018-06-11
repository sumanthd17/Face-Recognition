# all the different methods we tried fr recognition

@csrf_exempt
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
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)               # each face_encoding is compares with all the known face encodings
            name = "Unknown"                                                                            # initially all faces are unknown

            if True in matches:
                first_match_index = matches.index(True)                                                 # if a match is found the student is marked present
                name = known_face_names[first_match_index]
                students_attendence_data[name] = 1
            face_names.append(name)

    import collections
    students_attendence_data = collections.OrderedDict(sorted(students_attendence_data.items()))        # structuring the data in JSON format
    session_name = Session.objects.filter(pk=pk)
    print(session_name)
    latest_attendence = Attendence()
    latest_attendence.session_attendence = json.dumps(students_attendence_data)
    latest_attendence.date = datetime.datetime.now()
    latest_attendence.session_name = Session(pk)
    latest_attendence.save()
    return HttpResponse(json.dumps(students_attendence_data))



    ##################################
@csrf_exempt
def TakeAttendence(request):
    '''data = request.body
    try:
        data = json.loads(str(data, 'utf-8'))
    except :
        logging.error(str(datetime.datetime.now())+"\tJSON Data not received in correct format.")   #Logging Error message if data not received in correct format.
    print(data)'''

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
        logging.info(str(datetime.datetime.now())+"\t JsonResponse of attendance sent back.")               #Logging success message that JSON response with student attendance has been sent.
    print(dummy)
    return JsonResponse(attendence)
                                          # displaying the data on the web page