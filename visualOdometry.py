import cv2
import numpy as np

# Replace with your ESP32-CAM stream URL
url = "https://link/to/url"

class Map:
    def __init__(self):
        self.landmarks = {}

    def add_landmark(self, id, pos, des):
        self.landmarks[id] = (pos, des)
    
    def get_landmarks(self):
        return self.landmarks

def setup_orb():
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    return orb, bf

def estimate_pose(prev_kp, kp, matches, K):
    src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    E, mask = cv2.findEssentialMat(src_pts, dst_pts, K)
    _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts, K)
    
    return R, t

def process_frame(frame, orb, bf, prev_kp, prev_des, map, K):
    kp, des = orb.detectAndCompute(frame, None)
    if prev_kp is None:
        return kp, des, np.eye(3), np.zeros((3,1))  # Initial pose
    
    matches = bf.match(prev_des, des)
    matches = sorted(matches, key=lambda x: x.distance)
    
    R, t = estimate_pose(prev_kp, kp, matches, K)
    
    # Update map with new landmarks
    for i, m in enumerate(matches):
        map.add_landmark(m.trainIdx, kp[m.trainIdx].pt, des[m.trainIdx])
    
    return kp, des, R, t

# Camera intrinsic parameters (example values, you should calibrate your camera)
# Calibration can be completed with images of a checkerboard
# fx and fy --> focal lengths, cx and cy --> optical center coordinates
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

cap = cv2.VideoCapture(url)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

orb, bf = setup_orb()
prev_frame = None
prev_kp = None
prev_des = None
R_total = np.eye(3)
t_total = np.zeros((3,1))
map = Map()

while True:
    ret, frame = cap.read()
    if ret:
        if prev_frame is not None:
            kp, des, R, t = process_frame(frame, orb, bf, prev_kp, prev_des, map, K)
            prev_kp, prev_des = kp, des
            R_total = R @ R_total
            t_total = t_total + R_total @ t
        else:
            prev_kp, prev_des = orb.detectAndCompute(frame, None)

        prev_frame = frame

        # Display the current frame
        cv2.imshow("ESP32-CAM Stream", frame)
        
        # Display map (for example purposes, you might want to visualize this differently)
        map_img = np.zeros((480, 640, 3), dtype=np.uint8)
        for pos, des in map.get_landmarks().values():
            cv2.circle(map_img, (int(pos[0]), int(pos[1])), 3, (0, 255, 0), -1)
        cv2.imshow("Map", map_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Failed to retrieve frame from stream. Retrying...")
        cv2.waitKey(1000)

cap.release()
cv2.destroyAllWindows()
