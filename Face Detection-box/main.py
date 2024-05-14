
import cv2
import mediapipe as mp
import time
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


#For webcam input:
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("1.mp4")
prevTime = 0
with mp_face_detection.FaceDetection(
    min_detection_confidence=0.5) as face_detection:
  while True:
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      break

    #Convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
      for detection in results.detections:
        mp_drawing.draw_detection(image, detection)

    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
    cv2.imshow('BlazeFace Face Detection', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()