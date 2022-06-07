import cv2
import mediapipe as mp
import time

capture = cv2.VideoCapture(0)
prevTime = 0

mp_draw = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2)
draw_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=1)

while True:
    success, img = capture.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if results.multi_face_landmarks:
        for mesh_face_id, face_landmarks in enumerate(results.multi_face_landmarks):
            mp_draw.draw_landmarks(img, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, draw_spec, draw_spec)
            for mesh_id, landmark in enumerate(face_landmarks.landmark):
                #print(landmark)
                h, w, ch = img.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                print(mesh_face_id, mesh_id, x, y)

    currTime = time.time()
    fps = 1/(currTime - prevTime)
    prevTime = currTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)