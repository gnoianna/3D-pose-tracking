import mediapipe as mp
import cv2


class PoseTracker:
    def __init__(self, indexes_to_track=[]):
        self.mode = False
        self.smooth = True
        self.detection_confidence = 0.5
        self.track_confidence = 0.5
        self.indexes_to_track = indexes_to_track
        self.landmarks_list = []
        self.results = []

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=self.mode,
                                      smooth_landmarks=self.smooth,
                                      min_detection_confidence=self.detection_confidence,
                                      min_tracking_confidence=self.track_confidence)

    def find_position(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks,
                                            self.mp_pose.POSE_CONNECTIONS)

        return img

    def get_landmarks_positions(self):
        self.landmarks_list = []
        if self.results.pose_world_landmarks:
            for index, lm in enumerate(self.results.pose_world_landmarks.landmark):
                if index in self.indexes_to_track:
                    cx, cy, cz = round(lm.x, 2), (-1) * round(lm.y, 2), round(lm.z, 2)
                    self.landmarks_list.extend([cx, cy, cz])

        return self.landmarks_list
