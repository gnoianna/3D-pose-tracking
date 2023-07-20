import cv2
import socket
from tracker import PoseTracker

"""
Tracked landmarks
2 - left eye
23 - left hip
24 - right hip
25 - left knee
26 - right knee
27 - left ankle
28 - right ankle
"""


def main():
    landmarks_to_track = [2, 23, 24, 25, 26, 27, 28]
    capture = cv2.VideoCapture(0)
    detector = PoseTracker(landmarks_to_track)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address_port = ("127.0.0.1", 5052)

    while True:
        success, img = capture.read()
        if success:
            img = detector.find_position(img)
            landmarks_list = detector.get_landmarks_positions()
            sock.sendto(str.encode(str(landmarks_list)), server_address_port)
            cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
