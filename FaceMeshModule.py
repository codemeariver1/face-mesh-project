import cv2
import mediapipe as mp
import time


class FaceMeshDetector():
    def __init__(self, static_mode=False, max_faces=2, refine=False, min_detection_conf=0.5, min_tracking_conf=0.5):
        self.static_mode = static_mode
        self.max_faces = max_faces
        self.refine = refine
        self.min_detection_conf = min_detection_conf
        self.min_tracking_conf = min_tracking_conf

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            self.static_mode, self.max_faces, self.refine, self.min_detection_conf, self.min_tracking_conf
        )
        self.draw_spec = self.mp_draw.DrawingSpec(thickness=1, circle_radius=1)

    def find_face_mesh(self, img, draw=True):
        self.img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(self.img_rgb)
        mesh_face = []
        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, face_landmarks, self.mp_face_mesh.FACEMESH_TESSELATION, self.draw_spec, self.draw_spec
                    )

                mesh = []
                for mesh_id, landmark in enumerate(face_landmarks.landmark):
                    # print(landmark)
                    h, w, ch = img.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.putText(img, str(mesh_id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 255, 0), 1)
                    # print(mesh_id, x, y)
                    mesh.append([x, y])
                mesh_face.append(mesh)
        return img, mesh_face


def main():
    capture = cv2.VideoCapture(0)
    prevTime = 0
    detector = FaceMeshDetector()

    while True:
        success, img = capture.read()
        img, mesh_face = detector.find_face_mesh(img)
        if len(mesh_face) != 0:
            print(mesh_face[0])
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()