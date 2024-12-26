import streamlit as st 
import cv2
from PIL import Image
import numpy as np 
import pickle
import cvlib as cv


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")

@st.cache_resource
def load():
	with open('model_svc.pkl', 'rb') as file:
		genderModel = pickle.load(file)

	with open('model_svc2.pkl', 'rb') as file:
		ageModel = pickle.load(file)
	return genderModel,ageModel

def predict(image):
    gender_group = {
        1 : 'Homme',
        2 : 'Femme',
        0 : 'Enfant'
    }
    age_group = {
		1:'Enfant',
		2:'adolescence',
		3:'Jeune',
		4:'Adulte',
		5:'Senior'
    }
    models = load()
    genderModel = models[0]
    ageModel = models[1]
    gender = genderModel.predict([image])[0]
    age = ageModel.predict([image])[0]
    gender_label = gender_group[gender]
    age_label = age_group[age]
    return gender_label,age_label

def detect_faces(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	# Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
	# Extract face image from frame
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
		# Resize face image to match SVM input size
        face_img = cv2.resize(face_img, (65, 65))
		# Flatten face image to 1D array
        face_array = np.array(face_img).flatten()
        prediction = predict(face_array)
		# Get age and gender labels
        gender_label = prediction[0]
        age_label = prediction[1]
		# Draw rectangle around face
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
		# Draw age and gender label above rectangle
        st.write(gender_label+"-"+age_label)
        cv2.putText(img, gender_label+"-"+age_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    return img,faces 

def main():
		
	"""Gender and Age Detection App"""

	st.title("Application de détection de genre et âge")

	activities = ["Insérer une image","Video"]
	choice = st.sidebar.selectbox("Select Activty",activities)

	if choice == 'Insérer une image':
		st.subheader("Détection de genre et âge dans une image")

		image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])

		if image_file is not None:
			our_image = Image.open(image_file)
			result_img,result_faces = detect_faces(our_image)
			st.image(result_img)
			st.success("Found {} faces".format(len(result_faces)))
            
	elif choice == 'Video':
		st.subheader("Détection de genre et âge dans vidéo")
		st.write("Cliquez sur start pour utiliser la webcam et détecter votre genre et âge")
		run=st.checkbox("start")
		FRAME_WINDOW=st.image([])
		webcam = cv2.VideoCapture(0)
		while run:
			# read frame from webcam 
			status, frame = webcam.read()
			# apply face detection
			face, confidence = cv.detect_face(frame)
            # loop through detected faces
			for idx, f in enumerate(face):
				# get corner points of face rectangle        
				(startX, startY) = f[0], f[1]
				(endX, endY) = f[2], f[3]
				cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				face_img = gray[startY:endY, startX:endX]
				# Resize face image to match SVM input size
				face_img = cv2.resize(face_img, (65, 65))
				# Flatten face image to 1D array
				face_array = np.array(face_img).flatten()
				prediction = predict(face_array)
				gender = prediction[0]
				age = prediction[1]
				Y = startY - 10 if startY - 10 > 10 else startY + 10
				Y2 = startY - 25 if startY - 15 > 10 else startY + 25 
				cv2.putText(frame, f'{gender}--{age}', (startX, Y2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
				frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
				FRAME_WINDOW.image(frame,channels="RGB")



if __name__ == '__main__':
		main()