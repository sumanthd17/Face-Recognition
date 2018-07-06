# Attendance Using Face Recognition

**Face recognition** based attendance system implemented by using `face_recognition` module in python.
`face_recognition` is built using [dlib](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf)'s state of art-recognition built with deep learning.
The Web app is built using [django](https://www.djangoproject.com/)

### Features:
* The current model is very robust in recognising the face and the threshold hyperparameter can be changed based on the users need.
* The model accepts multiple image formats(eg *.JPG, *.JPEG etc) for both traning images and test images
* The model also works with video files too, given a video file random frames from the video are taken for face recognition task.
* The model accepts requests in one of the 3 different formats(JSON, XML and CSV) this can be configured in the config.JSON file. The response is also generated in the required format.
* Inorder to tackle privacy issue the model can also accept training images in blob format(SQL db)
* After the face recognition task is completed we generate annoted images and for cross-verfication. All the recognised faces are marked P with a bounding box around the face and all un-recognised faces are marked UK
* In-order to limit un-restricted access the model has a security measure in the form of tokens.

### Installation:
**Requirements**:
* Python 3.3+
* Linux 

**Installation instructions:**
* Create a virtualenv with python version >= 3.3
* Make sure you have dlib installed with Python bindings:
    * [Installation instructions for dlib](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf)
* Download or clone this repository.
* Go to the ERP directory.
* Install all the libraries and modules required from the same directory using pip
  `pip install -r requirements.txt`
* Now navigate to attendence dirctory in the ERP directory.
* Run the command:
  `python3 manage.py runserver 0.0.0.0:4000`
  The port number can be changed in the above command.

## config.JSON
Make sure to edit the config file according to your need before runnig the project
* Default request and response formats are JSON, but the model also supports XML and CSV
* Default db usgae is set to off, and if are planning to use db setup then make sure to edit the DATABASE field.
* Make sure to chnage the paths for the annoted images

## Instructions for usage:
* The images used for comaprison should be named same as the data of the peresons sent in the POST request.
* The images should be places as follows:
 ``` 
 AvailableSessions
   |------>KnownImages
      |---->NAME
         |----NAME.jpg
  ```
  
  eg.
  ```
  AvailableSessions
    |------>KnownImages
       |---->123456
          |----123456.jpg
``` 
* For changing the request and response methods the fields corresponding to REQ_METHOD and RSP_METHOD `config.JSON` file can be edited.There it can also be specied whether database has to be used or not by specifying 'YES' in DATABASE field of the USE field.
* Images with annotations will be saved in the specified path that can be given in the AnnotedImgPath in the PATHS field of the `config.JSON` file.
   
## Deployment
Deployment on pythonanywhere and AWS will be tricky due to the use of dlib.
So currently the program has been deployed on an Ubuntu server. The request(JSON, XML or CSV) is accepted at the server and after processing the request and response is generated in the format mentioned in the config.JSON file.
Currently the request is handled at `http://IP/faceRecognition/sessions/TakeAttendence`

## Built With
* [dlib](http://dlib.net/) - A C++ library containing machine learning algorithms and tools
* [django](https://www.djangoproject.com/) - The framework used for building the attendance system
* [face_recogniton](https://github.com/ageitgey/face_recognition) - Python library used for face recognition

## Contributing
Will be updated soon.

## Authors
* [D Sumanth](https://github.com/sumanthd17) - Student at IIIT SriCity
* [K Sai Suhas Tanmay](https://github.com/suhastanmay) - Student at IIIT SriCity

## License
This project is licensed under the MIT License.

## Acknowledgements
* Kasturi Rangan Rajagopalan for orignal idea and solution design
* [ageitgey](https://github.com/ageitgey/face_recognition) for creating an excellent module for face_recognition
* [MSCC](http://msconsortium.co.in/)(Mind Space Consulting Consortium) for giving us an oppurtunity do this project.
* Ravi Venkatraman Sir who helped and guided us in making the project successful.
* Hrishikesh Venkatraman Sir and IIIT SriCity for giving us a chance and promoting us for the project.
