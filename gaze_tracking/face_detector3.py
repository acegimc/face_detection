import os
import cv2
import dlib
import numpy as np
import pickle

class FaceRecognizer:
    def __init__(self):

        self.detector = dlib.get_frontal_face_detector()
        
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "shape_predictor_68_face_landmarks.dat"))
        self.predictor = dlib.shape_predictor(model_path)

        cwd2 = os.path.abspath(os.path.dirname(__file__))
        model_path2 = os.path.abspath(os.path.join(cwd2, "dlib_face_recognition_resnet_model_v1.dat"))
        self.recognizer = dlib.face_recognition_model_v1(model_path2)
        
        
        self.face_encodings = []
        self.face_names = []
        self.load_faces()
        #self.face_names 리스트의 현재 길이에 1을 더하여 다음에 사용할 ID를 설정.
        #리스트에서 얼굴 이름이 제거 되었을 때, 다음 ID를 올바르게 설정.
        self.next_id = len(self.face_names) + 1

    # 이전에 저장된 얼굴 인코딩과 이름을 파일에서 불러옴.
    def load_faces(self):
        """Load previously saved face encodings and names."""
        if os.path.exists('face_data.pkl'):
            with open('face_data.pkl', 'rb') as f:
                self.face_encodings, self.face_names = pickle.load(f)
    # 현재의 얼굴 인코딩과 이름을 파일에 저장.
    def save_faces(self):
        """Save face encodings and names to a file."""
        with open('face_data.pkl', 'wb') as f:
            pickle.dump((self.face_encodings, self.face_names), f)

    def learn_face(self,image):

        """Learn a new face from an image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if len(faces) == 0:
            print("No face detected for learning.")
            return

        for face in faces:
            shape = self.predictor(gray, face)
            encoding = np.array(self.recognizer.compute_face_descriptor(image, shape))

            # 이미 학습된 얼굴인지 확인
            # if any(np.allclose(encoding, enc, atol=1e-4) for enc in self.face_encodings):
            #     print("이미 저장된 얼굴 데이터입니다.")
            #     return
            # for i, enc in enumerate(self.face_encodings):
            #     if np.allclose(encoding, enc, atol=1e-4):
            #         print()
            
            self.face_encodings.append(encoding)
            self.face_names.append(f"User_{self.next_id}")# User_1, User_2 allocating automatically
            self.next_id += 1 
            self.save_faces()

    def recognize_faces(self,frame = None):
        """Recognize faces from the webcam feed."""
        if frame is not None:
    
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)

            for face in faces:
                shape = self.predictor(gray, face)
                encoding = np.array(self.recognizer.compute_face_descriptor(frame, shape))

                if len(self.face_encodings) > 0:
                    distances = np.linalg.norm(self.face_encodings - encoding, axis=1)
                    min_distance_index = np.argmin(distances)
                        #거리 기준 0.6 -> 0.5 조정.
                    if distances[min_distance_index] < 0.6:
                        name = self.face_names[min_distance_index]
                        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                        cv2.putText(frame, name, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            
                        cv2.imshow("Face Recognition", frame)
                        cv2.waitKey(2000)
                        print("OPEN!")
                        exit()

                            #face_recognized = True
                    else:
                        print("You are thief!")
                        exit(1)
                            # print("Unknown 얼굴 발견. 학습할까요? (y/n)")
                            # user_input = input()
                            # if user_input.lower() == 'y':
                                # self.learn_faces(face)
                else:
                    name = "Can't recognize face"

                    
                
                # if face_recognized:
                #     cv2.putText(frame, "Open", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                #     cv2.imshow("Face Recognition", frame)
                #     cv2.waitKey(2000)  # 2초 동안 보여줌
                #     break
        cap = cv2.VideoCapture(0)
        #face_recognized = False

        while True:
            ret, frame = cap.read()#frame을 받아온다.
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)

            for face in faces:
                shape = self.predictor(gray, face)
                encoding = np.array(self.recognizer.compute_face_descriptor(frame, shape))



                if len(self.face_encodings) > 0:
                    distances = np.linalg.norm(self.face_encodings - encoding, axis=1)
                    min_distance_index = np.argmin(distances)
                    #거리 기준 0.6 -> 0.5 조정.
                    if distances[min_distance_index] < 0.6:
                        name = self.face_names[min_distance_index]
                        #face_recognized = True
                    else:
                        name = "Unknown"
                        # print("Unknown 얼굴 발견. 학습할까요? (y/n)")
                        # user_input = input()
                        # if user_input.lower() == 'y':
                            # self.learn_faces(face)
                else:
                    name = "Can't recognize face"

                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                cv2.putText(frame, name, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # if face_recognized:
            #     cv2.putText(frame, "Open", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #     cv2.imshow("Face Recognition", frame)
            #     cv2.waitKey(2000)  # 2초 동안 보여줌
            #     break

            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('c'):
                self.learn_face(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
