import cv2
import pygame
from ffpyplayer.player import MediaPlayer
from deception_detection import process_frame, find_face_and_hands, MAX_FRAMES
import mediapipe as mp
import numpy as np

# Global variables for screen dimensions
video_width = 640
video_height = 480
side_panel_width = 160

# Colors
COLOR_BACKGROUND = (20, 20, 20)
COLOR_BUTTON = (50, 50, 50)
COLOR_BUTTON_HOVER = (70, 70, 70)
COLOR_TEXT = (255, 255, 255)

def draw_fps(screen, fps, x, y):
    font = pygame.font.Font(None, 36)
    fps_text = font.render(f'FPS: {int(fps)}', True, (0, 255, 0))
    screen.blit(fps_text, (x, y))

def draw_tells_on_frame(screen, tells, x, y):
    font = pygame.font.Font(None, 36)
    for idx, (key, tell) in enumerate(tells.items()):
        tell_text = font.render(f'{tell["text"]} (TTL: {tell["ttl"]})', True, (255, 0, 0))
        screen.blit(tell_text, (x, y + idx * 30))

def draw_calibration_indicator(screen, x, y, remaining_frames):
    font = pygame.font.Font(None, 36)
    calib_text = font.render(f'Calibrating... {remaining_frames} frames remaining', True, (255, 255, 0))
    screen.blit(calib_text, (x, y))

def draw_landmarks_and_hands(image, face_landmarks, hands_landmarks):
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

def draw_button(screen, rect, text, font, is_hovered=False):
    color = COLOR_BUTTON_HOVER if is_hovered else COLOR_BUTTON
    pygame.draw.rect(screen, color, rect)
    text_surf = font.render(text, True, COLOR_TEXT)
    screen.blit(text_surf, (rect.x + (rect.width - text_surf.get_width()) // 2, rect.y + (rect.height - text_surf.get_height()) // 2))

def play_video(file_path, screen, draw_landmarks=False):
    pygame.display.set_caption('Video Playback')
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)

    cap = cv2.VideoCapture(file_path)
    player = MediaPlayer(file_path)
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    hands = mp.solutions.hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7)

    exit_button = pygame.Rect(10, 10, 80, 30)
    play_button = pygame.Rect(10, 50, 80, 30)
    pause_button = pygame.Rect(10, 90, 80, 30)
    stop_button = pygame.Rect(10, 130, 80, 30)
    recalibrate_button = pygame.Rect(10, 170, 140, 30)
    running = True
    is_paused = False
    calibrated = False
    calibration_frames = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if exit_button.collidepoint(event.pos):
                    running = False
                if play_button.collidepoint(event.pos):
                    is_paused = False
                if pause_button.collidepoint(event.pos):
                    is_paused = True
                if stop_button.collidepoint(event.pos):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    player.seek(0, relative=False)
                    is_paused = True
                if recalibrate_button.collidepoint(event.pos):
                    calibrated = False
                    calibration_frames = 0

        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                break
            audio_frame, val = player.get_frame(show=False)
            face_landmarks, hands_landmarks = find_face_and_hands(frame, face_mesh, hands)
            tells = process_frame(frame, face_landmarks, hands_landmarks, calibrated, fps=cap.get(cv2.CAP_PROP_FPS))
            calibration_frames += 1
            if calibration_frames >= MAX_FRAMES:
                calibrated = True

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (video_width, video_height))

            if draw_landmarks:
                draw_landmarks_and_hands(frame, face_landmarks, hands_landmarks)

            frame = np.rot90(frame)
            frame = pygame.surfarray.make_surface(frame)

            screen.fill((0, 0, 0))
            screen.blit(frame, (side_panel_width, 0))

            pygame.draw.rect(screen, (200, 0, 0), exit_button)
            exit_text = font.render('Exit', True, (255, 255, 255))
            screen.blit(exit_text, (20, 10))

            if not calibrated:
                draw_calibration_indicator(screen, side_panel_width + 10, 10, MAX_FRAMES - calibration_frames)
            else:
                draw_fps(screen, clock.get_fps(), side_panel_width + 10, 10)
                draw_tells_on_frame(screen, tells, side_panel_width + 10, 50)

            draw_button(screen, play_button, 'Play', font, play_button.collidepoint(pygame.mouse.get_pos()))
            draw_button(screen, pause_button, 'Pause', font, pause_button.collidepoint(pygame.mouse.get_pos()))
            draw_button(screen, stop_button, 'Stop', font, stop_button.collidepoint(pygame.mouse.get_pos()))
            draw_button(screen, recalibrate_button, 'Recalibrate', font, recalibrate_button.collidepoint(pygame.mouse.get_pos()))

            pygame.display.flip()
            clock.tick(30)

    cap.release()
    player.close_player()

def play_webcam(screen, draw_landmarks=False):
    pygame.display.set_caption('Webcam Feed')
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)

    cap = cv2.VideoCapture(0)
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    hands = mp.solutions.hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7)

    exit_button = pygame.Rect(10, 10, 80, 30)
    recalibrate_button = pygame.Rect(10, 50, 140, 30)
    running = True
    calibrated = False
    calibration_frames = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if exit_button.collidepoint(event.pos):
                    running = False
                if recalibrate_button.collidepoint(event.pos):
                    calibrated = False
                    calibration_frames = 0

        ret, frame = cap.read()
        if not ret:
            break

        face_landmarks, hands_landmarks = find_face_and_hands(frame, face_mesh, hands)
        tells = process_frame(frame, face_landmarks, hands_landmarks, calibrated, fps=cap.get(cv2.CAP_PROP_FPS))
        calibration_frames += 1
        if calibration_frames >= MAX_FRAMES:
            calibrated = True

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (video_width, video_height))

        if draw_landmarks:
            draw_landmarks_and_hands(frame, face_landmarks, hands_landmarks)

        frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)

        screen.fill((0, 0, 0))
        screen.blit(frame, (side_panel_width, 0))

        pygame.draw.rect(screen, (200, 0, 0), exit_button)
        exit_text = font.render('Exit', True, (255, 255, 255))
        screen.blit(exit_text, (20, 10))

        if not calibrated:
            draw_calibration_indicator(screen, side_panel_width + 10, 10, MAX_FRAMES - calibration_frames)
        else:
            draw_fps(screen, clock.get_fps(), side_panel_width + 10, 10)
            draw_tells_on_frame(screen, tells, side_panel_width + 10, 50)

        draw_button(screen, recalibrate_button, 'Recalibrate', font, recalibrate_button.collidepoint(pygame.mouse.get_pos()))

        pygame.display.flip()
        clock.tick(30)

    cap.release()
