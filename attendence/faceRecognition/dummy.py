import face_recognition
import os




'''image = face_recognition.load_image_file('201601001.jpg')
image_face_encoding = face_recognition.face_encodings(image)[0]
known_face_encodings.append(image_face_encoding)
known_face_names.append(image)'''

def dummy(pk):
	known_face_encodings = []
	known_face_names = []

	pk = str(pk)
	for image_name in os.listdir('../KnownImages/' + pk):
	print(image_name)
	image = face_recognition.load_image_file('../KnownImages/' + pk + '/' + image_name)
	image_face_encoding = face_recognition.face_encodings(image)[0]
	known_face_encodings.append(image_face_encoding)
	known_face_names.append(image)