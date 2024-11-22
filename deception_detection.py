import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import distance as dist
from fer import FER
import threading
import time

# Constants and global variables
MAX_FRAMES = 120
RECENT_FRAMES = int(MAX_FRAMES / 10)
EYE_BLINK_HEIGHT = .15
SIGNIFICANT_BPM_CHANGE = 8
LIP_COMPRESSION_RATIO = .35
TEXT_HEIGHT = 30
FACEMESH_FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
EPOCH = time.time()

# Global variables for detection
blinks = [False] * MAX_FRAMES
hand_on_face = [False] * MAX_FRAMES
face_area_size = 0
hr_times = list(range(0, MAX_FRAMES))
hr_values = [400] * MAX_FRAMES
avg_bpms = [0] * MAX_FRAMES
gaze_values = [0] * MAX_FRAMES
emotion_detector = FER(mtcnn=True)
calculating_mood = False
mood = ''
tells = dict()

def decrement_tells(tells):
    for key, tell in tells.copy().items():
        if 'ttl' in tell:
            tell['ttl'] -= 1
            if tell['ttl'] <= 0:
                del tells[key]
    return tells

def new_tell(result, ttl_for_tells):
    return {'text': result, 'ttl': ttl_for_tells}

def smooth(signal, window_size):
    window = np.ones(window_size) / window_size
    return np.convolve(signal, window, mode='same')

def calculate_bpm(signal, fps, min_bpm=50, max_bpm=150):
    signal = smooth(signal, window_size=5)
    peaks, _ = find_peaks(signal, distance=fps/2.5, height=0.05)
    if len(peaks) < 2:
        return None
    peak_intervals = np.diff(peaks) / fps * 60
    valid_peaks = peak_intervals[(peak_intervals >= min_bpm) & (peak_intervals <= max_bpm)]
    if len(valid_peaks) == 0:
        return None
    return np.mean(valid_peaks)

def draw_on_frame(image, face_landmarks, hands_landmarks):
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    if face_landmarks:
        mp_drawing.draw_landmarks(
            image,
            face_landmarks,
            mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image,
            face_landmarks,
            mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image,
            face_landmarks,
            mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
    if hands_landmarks:
        for hand_landmarks in hands_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

def add_text(image, tells, calibrated):
    global mood
    text_y = TEXT_HEIGHT
    if mood:
        write("Mood: {}".format(mood), image, int(.75 * image.shape[1]), TEXT_HEIGHT)
    if calibrated:
        for tell in tells.values():
            write(tell['text'], image, 10, text_y)
            text_y += TEXT_HEIGHT

def write(text, image, x, y):
    cv2.putText(img=image, text=text, org=(x, y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[0, 0, 0],
                lineType=cv2.LINE_AA, thickness=4)
    cv2.putText(img=image, text=text, org=(x, y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[255, 255, 255],
                lineType=cv2.LINE_AA, thickness=2)

def get_aspect_ratio(top, bottom, right, left):
    height = dist.euclidean([top.x, top.y], [bottom.x, bottom.y])
    width = dist.euclidean([right.x, right.y], [left.x, left.y])
    return height / width

def get_area(image, draw, topL, topR, bottomR, bottomL):
    topY = int((topR.y + topL.y) / 2 * image.shape[0])
    botY = int((bottomR.y + bottomL.y) / 2 * image.shape[0])
    leftX = int((topL.x + bottomL.x) / 2 * image.shape[1])
    rightX = int((topR.x + bottomR.x) / 2 * image.shape[1])
    return image[topY:botY, rightX:leftX]

def is_blinking(face):
    eyeR = [face[p] for p in [159, 145, 133, 33]]
    eyeR_ar = get_aspect_ratio(*eyeR)
    eyeL = [face[p] for p in [386, 374, 362, 263]]
    eyeL_ar = get_aspect_ratio(*eyeL)
    eyeA_ar = (eyeR_ar + eyeL_ar) / 2
    return eyeA_ar < EYE_BLINK_HEIGHT

def get_blink_tell(blinks):
    if sum(blinks[:RECENT_FRAMES]) < 3:
        return None
    recent_closed = 1.0 * sum(blinks[-RECENT_FRAMES:]) / RECENT_FRAMES
    avg_closed = 1.0 * sum(blinks) / MAX_FRAMES
    if recent_closed > (20 * avg_closed):
        return "Increased blinking"
    elif avg_closed > (20 * recent_closed):
        return "Decreased blinking"
    else:
        return None

def check_hand_on_face(hands_landmarks, face):
    if hands_landmarks:
        face_landmarks = [face[p] for p in FACEMESH_FACE_OVAL]
        face_points = [[[p.x, p.y] for p in face_landmarks]]
        face_contours = np.array(face_points).astype(np.single)
        for hand_landmarks in hands_landmarks:
            hand = []
            for point in hand_landmarks.landmark:
                hand.append((point.x, point.y))
            for finger in [4, 8, 20]:
                overlap = cv2.pointPolygonTest(face_contours, hand[finger], False)
                if overlap != -1:
                    return True
    return False

def get_avg_gaze(face):
    gaze_left = get_gaze(face, 476, 474, 263, 362)
    gaze_right = get_gaze(face, 471, 469, 33, 133)
    return round((gaze_left + gaze_right) / 2, 1)

def get_gaze(face, iris_L_side, iris_R_side, eye_L_corner, eye_R_corner):
    iris = (face[iris_L_side].x + face[iris_R_side].x, face[iris_L_side].y + face[iris_R_side].y)
    eye_center = (face[eye_L_corner].x + face[eye_R_corner].x, face[eye_L_corner].y + face[eye_R_corner].y)
    gaze_dist = dist.euclidean(iris, eye_center)
    eye_width = abs(face[eye_R_corner].x - face[eye_L_corner].x)
    gaze_relative = gaze_dist / eye_width
    if (eye_center[0] - iris[0]) < 0:
        gaze_relative *= -1
    return gaze_relative

def detect_gaze_change(avg_gaze):
    global gaze_values
    gaze_values = gaze_values[1:] + [avg_gaze]
    gaze_relative_matches = 1.0 * gaze_values.count(avg_gaze) / MAX_FRAMES
    if gaze_relative_matches < .01:
        return gaze_relative_matches
    return 0

def get_lip_ratio(face):
    return get_aspect_ratio(face[0], face[17], face[61], face[291])

def get_mood(image):
    global emotion_detector, calculating_mood, mood
    detected_mood, score = emotion_detector.top_emotion(image)
    calculating_mood = False
    if score and (score > .4 or detected_mood == 'neutral'):
        mood = detected_mood
        return mood

def get_emotions(image):
    global emotion_detector
    emotion_data = {
        "angry": 0,
        "disgust": 0,
        "fear": 0,
        "happy": 0,
        "sad": 0,
        "surprise": 0,
        "neutral": 0
    }
    emotions = emotion_detector.detect_emotions(image)
    if emotions:
        for emotion in emotions:
            for key in emotion["emotions"]:
                emotion_data[key] += emotion["emotions"][key]
    return emotion_data

def get_face_relative_area(face):
    face_width = abs(max(face[454].x, 0) - max(face[234].x, 0))
    face_height = abs(max(face[152].y, 0) - max(face[10].y, 0))
    return face_width * face_height

def find_face_and_hands(image_original, face_mesh, hands):
    image = image_original.copy()
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_mesh.process(image)
    hands_landmarks = hands.process(image).multi_hand_landmarks
    face_landmarks = None
    if faces.multi_face_landmarks and len(faces.multi_face_landmarks) > 0:
        face_landmarks = faces.multi_face_landmarks[0]
    return face_landmarks, hands_landmarks

def process_frame(image, face_landmarks, hands_landmarks, calibrated=False, fps=None, ttl_for_tells=30):
    global tells, calculating_mood
    global blinks, hand_on_face, face_area_size
    tells = decrement_tells(tells)
    if face_landmarks:
        face = face_landmarks.landmark
        face_area_size = get_face_relative_area(face)
        if not calculating_mood:
            emothread = threading.Thread(target=get_mood, args=(image,))
            emothread.start()
            calculating_mood = True
        cheekL = get_area(image, False, topL=face[449], topR=face[350], bottomR=face[429], bottomL=face[280])
        cheekR = get_area(image, False, topL=face[121], topR=face[229], bottomR=face[50], bottomL=face[209])
        bpm = get_bpm_change_value(image, False, face_landmarks, hands_landmarks, fps)
        bpm_display = f"BPM: {bpm:.2f}" if bpm else "BPM: ..."
        tells['avg_bpms'] = new_tell(bpm_display, ttl_for_tells)
        if bpm:
            bpm_delta = bpm - avg_bpms[-1]
            if abs(bpm_delta) > SIGNIFICANT_BPM_CHANGE:
                change_desc = "Heart rate increasing" if bpm_delta > 0 else "Heart rate decreasing"
                tells['bpm_change'] = new_tell(change_desc, ttl_for_tells)
        blinks = blinks[1:] + [is_blinking(face)]
        recent_blink_tell = get_blink_tell(blinks)
        if recent_blink_tell:
            tells['blinking'] = new_tell(recent_blink_tell, ttl_for_tells)
        recent_hand_on_face = check_hand_on_face(hands_landmarks, face)
        hand_on_face = hand_on_face[1:] + [recent_hand_on_face]
        if recent_hand_on_face:
            tells['hand'] = new_tell("Hand covering face", ttl_for_tells)
        avg_gaze = get_avg_gaze(face)
        if detect_gaze_change(avg_gaze):
            tells['gaze'] = new_tell("Change in gaze", ttl_for_tells)
        if get_lip_ratio(face) < LIP_COMPRESSION_RATIO:
            tells['lips'] = new_tell("Lip compression", ttl_for_tells)
    return tells

def get_bpm_change_value(image, draw, face_landmarks, hands_landmarks, fps):
    if face_landmarks:
        face = face_landmarks.landmark
        cheekL = get_area(image, draw, topL=face[449], topR=face[350], bottomR=face[429], bottomL=face[280])
        cheekR = get_area(image, draw, topL=face[121], topR=face[229], bottomR=face[50], bottomL=face[209])
        global hr_values
        cheekLwithoutBlue = np.average(cheekL[:, :, 1:3])
        cheekRwithoutBlue = np.average(cheekR[:, :, 1:3])
        hr_values = hr_values[1:] + [cheekLwithoutBlue + cheekRwithoutBlue]
    bpm = calculate_bpm(hr_values, fps)
    return bpm
