import mediapipe as mp
import cv2


class PoseTracker:
    def __init__(self, landmarks_to_track=[]):
        self.landmarks_to_track = landmarks_to_track
        self.results = []
        self.landmarks_list = []

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5,
                                      smooth_landmarks=True)

    def find_position(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        self.mp_drawing.draw_landmarks(img,
                                       self.results.pose_landmarks,
                                       self.mp_pose.POSE_CONNECTIONS)

        return img

    def get_landmarks_positions(self):
        detected_landmarks = self.results.pose_world_landmarks
        if detected_landmarks:
            landmarks_list = []
            for index, lm in enumerate(detected_landmarks.landmark):
                if index in self.landmarks_to_track:
                    lm_x, lm_y = (-1) * round(lm.x, 2), (-1) * round(lm.y, 2)
                    landmarks_list.extend([lm_x, lm_y])

            first_lm_x = landmarks_list[2]
            if first_lm_x < 0:
                self.landmarks_list = landmarks_list

        return self.landmarks_list
