# OpenCV: захват с камеры и рисование поверх кадра
import math
import cv2
import mediapipe as mp
import serial
from pathlib import Path

# --- MediaPipe Tasks: Hand Landmarker ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# --- Пороги управления (подстройте под камеру и расстояние) ---
# Левая рука: сколько пальцев считать «открытой ладонью» (газ) / «кулаком» (тормоз)
OPEN_PALM_MIN_FINGERS = 4
CLOSED_FIST_MAX_FINGERS = 1
# Правая рука: наклон ладони в градусах (мертвая зона по центру = «прямо»)
STEER_DEAD_ZONE_DEG = 14.0

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def _dist(a, b):
    return abs(a - b)


def count_extended_fingers(landmarks):
    """Сколько пальцев «выпрямлены» (та же геометрия, что в старом коде, координаты y нормализованные 0..1)."""
    p = [lm.y for lm in landmarks]
    distance_good = _dist(p[0], p[5]) + (_dist(p[0], p[5]) / 2)
    f1 = 1 if _dist(p[0], p[8]) > distance_good else 0
    f2 = 1 if _dist(p[0], p[12]) > distance_good else 0
    f3 = 1 if _dist(p[0], p[16]) > distance_good else 0
    f4 = 1 if _dist(p[0], p[20]) > distance_good else 0
    f0 = 1 if _dist(p[4], p[17]) > distance_good else 0
    return f0 + f1 + f2 + f3 + f4


def classify_hand_openness(landmarks):
    """Левая рука: открытая ладонь = газ, кулак = тормоз, между = нейтраль."""
    n = count_extended_fingers(landmarks)
    if n <= CLOSED_FIST_MAX_FINGERS:
        return "brake"
    if n >= OPEN_PALM_MIN_FINGERS:
        return "gas"
    return "neutral"


def palm_tilt_degrees(landmarks):
    """
    Угол наклона ладони в плоскости кадра: вектор запястье(0) -> основание среднего(9).
    0 градусов — «ладонь вверх» по кадру; отрицательный — наклон в одну сторону, положительный — в другую.
    """
    lm0, lm9 = landmarks[0], landmarks[9]
    dx = lm9.x - lm0.x
    dy = lm9.y - lm0.y
    return math.degrees(math.atan2(dx, -dy))


def classify_steering(landmarks):
    """Правая рука: влево / прямо / вправо по углу наклона."""
    ang = palm_tilt_degrees(landmarks)
    if ang < -STEER_DEAD_ZONE_DEG:
        return "left", ang
    if ang > STEER_DEAD_ZONE_DEG:
        return "right", ang
    return "center", ang


def handedness_label(handedness_list):
    """Из результата MediaPipe: 'Left' или 'Right' (первая категория с макс. score)."""
    if not handedness_list:
        return None
    return handedness_list[0].category_name


def draw_hand_landmarks_bgr(img, landmarks, h, w):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], (255, 255, 255), 2)
    for pt in pts:
        cv2.circle(img, pt, 3, (0, 255, 0), -1)


def draw_drive_hud(img, left_state, right_steer, steer_angle_deg, left_seen, right_seen):
    """
    Панель состояний (латиница — стандартный шрифт OpenCV не рисует кириллицу).
    left_state: 'gas' | 'brake' | 'neutral' | None
    right_steer: 'left' | 'center' | 'right' | None
    """
    h, w = img.shape[:2]
    overlay = img.copy()
    panel_h = 120
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 28
    cv2.putText(img, "LEFT hand (gas/brake):", (10, y), font, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    y += 26
    if left_seen and left_state == "gas":
        cv2.putText(img, "  GAZ", (10, y), font, 0.7, (0, 255, 100), 2, cv2.LINE_AA)
    elif left_seen and left_state == "brake":
        cv2.putText(img, "  TORMOZ", (10, y), font, 0.7, (0, 80, 255), 2, cv2.LINE_AA)
    elif left_seen and left_state == "neutral":
        cv2.putText(img, "  NEYTRAL", (10, y), font, 0.7, (200, 200, 100), 2, cv2.LINE_AA)
    else:
        cv2.putText(img, "  --- (net v kadre)", (10, y), font, 0.65, (120, 120, 120), 1, cv2.LINE_AA)

    y = 28
    x0 = max(280, w // 2 - 40)
    cv2.putText(img, "RIGHT hand (povorot):", (x0, y), font, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    y += 26
    if right_seen and right_steer == "left":
        cv2.putText(img, "  VLEVO", (x0, y), font, 0.7, (255, 180, 0), 2, cv2.LINE_AA)
    elif right_seen and right_steer == "center":
        cv2.putText(img, "  PRYAMO", (x0, y), font, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
    elif right_seen and right_steer == "right":
        cv2.putText(img, "  VPRAVO", (x0, y), font, 0.7, (255, 180, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(img, "  --- (net v kadre)", (x0, y), font, 0.65, (120, 120, 120), 1, cv2.LINE_AA)

    # Угол наклона правой ладони (если была в кадре)
    if right_seen and steer_angle_deg is not None:
        cv2.putText(
            img,
            f"tilt: {steer_angle_deg:+.1f} deg",
            (x0, y + 28),
            font,
            0.5,
            (160, 160, 255),
            1,
            cv2.LINE_AA,
        )

    # Мини-индикатор «руля»: стрелка внизу панели
    cx = w // 2
    base_y = panel_h - 12
    cv2.line(img, (cx - 60, base_y), (cx + 60, base_y), (80, 80, 80), 2)
    if right_seen and right_steer == "left":
        tip = (cx - 45, base_y)
    elif right_seen and right_steer == "right":
        tip = (cx + 45, base_y)
    else:
        tip = (cx, base_y - 18)
    cv2.line(img, (cx, base_y), tip, (0, 200, 255), 2)
    cv2.circle(img, tip, 5, (0, 200, 255), -1)


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "models" / "hand_landmarker.task"

camera = cv2.VideoCapture(0)

# portNo = "COM8"
# uart = serial.Serial(portNo, 9600)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
)

frame_timestamp_ms = 0

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        good, img = camera.read()
        if not good:
            break

        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        frame_timestamp_ms += 33
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        left_state = None
        right_steer = None
        steer_angle_deg = None
        left_seen = False
        right_seen = False

        if result.hand_landmarks:
            n = len(result.hand_landmarks)
            for i in range(n):
                hand_lms = result.hand_landmarks[i]
                draw_hand_landmarks_bgr(img, hand_lms, h, w)

                hs = result.handedness[i] if i < len(result.handedness) else []
                label = handedness_label(hs)

                if label == "Left":
                    left_seen = True
                    left_state = classify_hand_openness(hand_lms)
                elif label == "Right":
                    right_seen = True
                    right_steer, steer_angle_deg = classify_steering(hand_lms)

        draw_drive_hud(img, left_state, right_steer, steer_angle_deg, left_seen, right_seen)

        # Primer dlya Arduino (raskomentiruyte i zamenite COM):
        # cmd = f"L:{left_state or 'na'};R:{right_steer or 'na'};\n"
        # uart.write(cmd.encode("utf-8"))

        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
