import cv2
import dlib
import numpy as np
import os

# 얼굴 인식 모델 초기화
detector = dlib.get_frontal_face_detector()#정면 얼굴 탐지하는 얼굴 탐지기 생성

#사전 학습된 얼굴 인식을 위한 모델 업로드. 이 모델은 주어진 얼굴 이미지로부터 특징 벡터를(인코딩) 계산하는데 사용.
recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

#얼굴 랜드마크를 추출하기 위한 모델을 로드. 68개의 얼굴 랜드마크를 예측하는데 사용.
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 얼굴 데이터 저장
face_encodings = []
face_names = []

# 얼굴 데이터 학습 함수
def learn_face(name, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #얼굴을 탐지기로 얼굴 찾고 faces에 넣기.
    faces = detector(gray)
    
    if len(faces) == 0:
        print("No face detected for learning.")
        return
        
    for face in faces:
        #탐지된 각 얼굴에 대해서 랜드마크를 추출. shape에 저장.
        shape = predictor(gray, face)
        #랜드마크와 image를 기반으로 얼굴 인코딩을 계산.
        encoding = np.array(recognizer.compute_face_descriptor(image, shape))
        #계산된 인코딩과 이름을 각각 face_encodings와 face_names 리스트에 추가.
        face_encodings.append(encoding)
        face_names.append(name)

# 얼굴 인식 및 캡처 함수
def recognize_faces():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            shape = predictor(gray, face)
            encoding = np.array(recognizer.compute_face_descriptor(frame, shape))

            if len(face_encodings) > 0:
                #face_encodings 배열과 현재 얼굴 encoding간의 유클리드 거리를 계산. 이 거리는 두 얼굴 인코딩 간의 유사성을 나타낸다.
                distances = np.linalg.norm(face_encodings - encoding, axis=1)
                min_distance_index = np.argmin(distances)

                if distances[min_distance_index] < 0.6:  # 임계값
                    name = face_names[min_distance_index]
                else:
                    name = "Unknown"
            else:
                name = "Can't recognize face"
            # 얼굴 박스와 이름 표시
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
            cv2.putText(frame, name, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Face Recognition", frame)

        # 'c' 키를 눌러 캡처
        if cv2.waitKey(1) & 0xFF == ord('c'):
            name = input("Enter name for the captured face: ")
            learn_face(name, frame)

        # 'q' 키를 눌러 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()