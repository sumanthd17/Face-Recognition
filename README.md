# Attendance Using Face Recognition

**Face recognition** based attendance system implemented by using `face_recognition` module in python.
`face_recognition` is built using [dlib](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf)'s state of art-recognition built with deep learning.
This model of attendance system is implemented with django 

### Features:
* This model can recognize the detected faces by comapring the detected faces with images provided in any format(eg .jpg,.jpeg etc) or from images that are stored in MySql database.
* Data of whose attendance has to be verified can be sent in either JSON, XML or text seperated by a delimiter in specified format to given URL by POST request and the data of the attendance can be recieved as a response in either of the above three formats.
* Images with annotations around the recognised and unrecognised faces will also be available for verification of the attendance.
* This model also has a security measure which uses a token which authorizes only permitted users to have access to the attendance system and not allowing anyone without permission to make changes in the attendance.

### Installation:
**Requirements**:
* Python 3.3+
* Linux 

**Installation instructions:**
* Make sure you have dlib installed with Python bindings:
    * [Installation instructions for dlib](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf)
* Download or clone this repository.
* Go to the ERP directory.
* Install all the libraries and modules required from the same directory using pip3

  `pip3 install -r requirements.txt`
* Now navigate to attendence dirctory in the ERP directory.
* Run the command:
  
  `python3 manage.py runserver 0.0.0.0:4000`
  The port number can be changed in the above command.
 
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
After cloning the respository and installing all the requiremennts including dlib starting the server will enable the attendance system. If the images of persons to be recognized are set as mentioned above then requests(JSON,XML,text seperated by delimiter) for attendance can be sent to the url `http://IPADDRESS:PORTNUMBER/faceRecognition/session/TakeAttendence/` for eg: if the project is hosted on a server with ip address 182.223.547.112 and port number on which the project is running is 4000 then the url will be `http://182.223.547.112:4000/faceRecognition/session/TakeAttendence/`. The sample request structure for different formats are present in the following files :
   for JSON - jsoninput.txt
   for XML - xmlinput.txt
   for text with delimiter - textinput.txt
After the request is sent the return response is sent to the same url and attendance can be obtained there.
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
* [MSCC](http://msconsortium.co.in/)(Mind Space Consulting Consortium) for giving us an oppurtunity do this project.
* Ravi Venkatraman Sir who helped and guided us in making the project successful.
* Hrishikesh Venkatraman Sir and IIIT SriCity for giving us a chance and promoting us for the project.
