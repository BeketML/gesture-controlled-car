# OpenCV: захват с камеры и рисование поверх кадра
import math
import time
import urllib.error
import urllib.parse
import urllib.request
import cv2
import mediapipe as mp
import serial
from pathlib import Path

# --- MediaPipe Tasks: Hand Landmarker ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# --- Пороги управления ---
# Правая рука: мертвая зона поворота в градусах
STEER_DEAD_ZONE_DEG = 10.0

# --- Wi-Fi: ESP8266 в режиме AP (firmware/esp32_car_drive_server.ino) ---
# Ноут подключается к точке (ssid в прошивке), обычно базовый URL: http://192.168.4.1
# Команды: GET /?State=F|G|I|S|... как в HTTP_handleRoot на машинке.
CAR_WIFI_ENABLED = True
CAR_WIFI_BASE_URL = "http://192.168.0.15"
CAR_WIFI_HEARTBEAT_SEC = 0.35
CAR_WIFI_TIMEOUT_SEC = 0.35

# ---------------------------------------------------------------------------
# Serial (опционально): та же буква состояния шлётся в COM-порт.
# ---------------------------------------------------------------------------
CAR_SERIAL_ENABLED = False
CAR_SERIAL_PORT = "COM4"
CAR_SERIAL_BAUD = 9600

_KEY_FWD        = {ord('w'), ord('W'), 2490368}
_KEY_REV        = {ord('s'), ord('S'), 2621440}
_KEY_LEFT       = {ord('a'), ord('A'), 2424832}
_KEY_RIGHT      = {ord('d'), ord('D'), 2555904}
_KEY_STOP       = {ord(' ')}
_KEY_QUIT       = ord('q')
_KEY_TIMEOUT_SEC = 0.1

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def fingers_up(landmarks):
    """
    Возвращает [index, middle, ring, pinky] — True если палец выпрямлен.
    Кончик выше (меньше y) среднего сустава в координатах MediaPipe.
    """
    p = [lm.y for lm in landmarks]
    return [
        p[8]  < p[6],
        p[12] < p[10],
        p[16] < p[14],
        p[20] < p[18],
    ]


def classify_left_gesture(landmarks):
    """
    Жесты левой руки (по 4 пальцам: указательный, средний, безымянный, мизинец):

      Кулак          [0,0,0,0] -> 'brake'   -> STOP
      Открытая ладонь[1,1,1,1] -> 'gas'     -> FORWARD
      Рога           [1,0,0,1] -> 'reverse' -> BACK
      Всё остальное            -> 'neutral'  -> STOP
    """
    idx, mid, rng, pin = fingers_up(landmarks)
    if not idx and not mid and not rng and not pin:
        return "brake"
    if idx and mid and rng and pin:
        return "gas"
    if idx and not mid and not rng and pin:
        return "reverse"
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


def compute_drive_mode(left_seen, left_state, right_seen, right_steer):
    """
    Логический режим. Жесты левой руки:
      Кулак          -> STOP
      Открытая ладонь -> FWD (+ поворот правой)
      Рога (index+pinky) -> REV
      Нейтраль       -> STOP
    """
    if not left_seen or left_state in ("brake", "neutral"):
        return "STOP"
    if left_state == "reverse":
        return "REV"
    if not right_seen:
        return "FWD"
    if right_steer == "left":
        return "FWD_L"
    if right_steer == "right":
        return "FWD_R"
    return "FWD"


def drive_mode_to_car_state(mode: str) -> str:
    """Соответствие логического режима прошивке L298N / ESP8266WebServer (параметр State)."""
    return {
        "STOP": "S",
        "FWD": "F",
        "FWD_L": "G",
        "FWD_R": "I",
        "REV": "B",
    }.get(mode, "S")


def send_drive_to_car(base_url: str, mode: str, timeout_sec: float) -> tuple[bool, str]:
    """GET {base}/?State=<буква> — как в esp32_car_drive_server.ino (HTTP_handleRoot)."""
    state = drive_mode_to_car_state(mode)
    url = base_url.rstrip("/") + "/?" + urllib.parse.urlencode({"State": state})
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            if resp.status == 200:
                return True, "ok"
            return False, f"http {resp.status}"
    except urllib.error.HTTPError as e:
        return False, f"http {e.code}"
    except urllib.error.URLError:
        return False, "net err"
    except TimeoutError:
        return False, "timeout"
    except Exception:
        return False, "err"


def draw_hand_landmarks_bgr(img, landmarks, h, w):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], (255, 255, 255), 2)
    for pt in pts:
        cv2.circle(img, pt, 3, (0, 255, 0), -1)


def draw_drive_hud(
    img,
    left_state,
    right_steer,
    steer_angle_deg,
    left_seen,
    right_seen,
    drive_mode=None,
    wifi_line=None,
    kb_mode=None,
):
    """
    Панель состояний (латиница — стандартный шрифт OpenCV не рисует кириллицу).
    left_state: 'gas' | 'brake' | 'reverse' | 'neutral' | None
    right_steer: 'left' | 'center' | 'right' | None
    """
    h, w = img.shape[:2]
    overlay = img.copy()
    panel_h = 158
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 28
    cv2.putText(img, "LEFT (roga=BACK palma=FWD kulak=STOP):", (10, y), font, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    y += 26
    if left_seen and left_state == "gas":
        cv2.putText(img, "  PALMA -> FORWARD", (10, y), font, 0.65, (0, 255, 100), 2, cv2.LINE_AA)
    elif left_seen and left_state == "reverse":
        cv2.putText(img, "  ROGA  -> BACK", (10, y), font, 0.65, (0, 140, 255), 2, cv2.LINE_AA)
    elif left_seen and left_state == "brake":
        cv2.putText(img, "  KULAK -> STOP", (10, y), font, 0.65, (0, 60, 255), 2, cv2.LINE_AA)
    elif left_seen and left_state == "neutral":
        cv2.putText(img, "  (neytral) -> STOP", (10, y), font, 0.6, (200, 200, 100), 1, cv2.LINE_AA)
    else:
        cv2.putText(img, "  --- (net v kadre)", (10, y), font, 0.6, (120, 120, 120), 1, cv2.LINE_AA)

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
            (x0, y + 48),
            font,
            0.5,
            (160, 160, 255),
            1,
            cv2.LINE_AA,
        )

    # Мини-индикатор «руля» или стрелка НАЗАД
    cx = w // 2
    base_y = panel_h - 32
    is_rev = isinstance(drive_mode, str) and "REV" in drive_mode
    if not is_rev:
        cv2.line(img, (cx - 60, base_y), (cx + 60, base_y), (80, 80, 80), 2)
    if is_rev:
        cv2.arrowedLine(img, (cx, base_y - 20), (cx, base_y + 8), (0, 140, 255), 3, tipLength=0.35)
        cv2.putText(
            img,
            "BACK",
            (cx - 28, base_y - 26),
            font,
            0.6,
            (0, 140, 255),
            2,
            cv2.LINE_AA,
        )
    elif right_seen and right_steer == "left":
        tip = (cx - 45, base_y)
        cv2.line(img, (cx, base_y), tip, (0, 200, 255), 2)
        cv2.circle(img, tip, 5, (0, 200, 255), -1)
    elif right_seen and right_steer == "right":
        tip = (cx + 45, base_y)
        cv2.line(img, (cx, base_y), tip, (0, 200, 255), 2)
        cv2.circle(img, tip, 5, (0, 200, 255), -1)
    elif right_seen:
        tip = (cx, base_y - 18)
        cv2.line(img, (cx, base_y), tip, (0, 200, 255), 2)
        cv2.circle(img, tip, 5, (0, 200, 255), -1)

    if drive_mode is not None:
        cmd_color = (0, 200, 255) if (isinstance(drive_mode, str) and "REV" in drive_mode) else (180, 255, 180)
        cv2.putText(
            img,
            f"CMD: {drive_mode}",
            (10, panel_h - 8),
            font,
            0.55,
            cmd_color,
            2 if (isinstance(drive_mode, str) and "REV" in drive_mode) else 1,
            cv2.LINE_AA,
        )
    if wifi_line:
        wx = max(220, w // 2 - 60)
        cv2.putText(
            img,
            wifi_line[:52],
            (wx, panel_h - 8),
            font,
            0.45,
            (200, 200, 255),
            1,
            cv2.LINE_AA,
        )

    if kb_mode is not None:
        kb_text = f"KB: {kb_mode}"
        kb_color = (0, 255, 255)
        kb_weight = 2
    else:
        kb_text = "WASD/arrows=KB override"
        kb_color = (80, 80, 80)
        kb_weight = 1
    cv2.putText(img, kb_text, (w - 230, panel_h - 8), font, 0.45, kb_color, kb_weight, cv2.LINE_AA)


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

_start_time_ms = int(time.monotonic() * 1000)
_last_drive_mode = None
_last_wifi_send_time = 0.0
_wifi_hud = ""
_key_drive_mode = None
_key_last_press = 0.0

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        good, img = camera.read()
        if not good:
            break

        img = cv2.flip(img, 1)
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        frame_timestamp_ms = int(time.monotonic() * 1000) - _start_time_ms
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
                    left_state = classify_left_gesture(hand_lms)
                elif label == "Right":
                    right_seen = True
                    right_steer, steer_angle_deg = classify_steering(hand_lms)

        gesture_mode = compute_drive_mode(
            left_seen, left_state, right_seen, right_steer
        )
        drive_mode = _key_drive_mode if _key_drive_mode is not None else gesture_mode

        if CAR_WIFI_ENABLED and CAR_WIFI_BASE_URL:
            now = time.monotonic()
            if (
                drive_mode != _last_drive_mode
                or (now - _last_wifi_send_time) >= CAR_WIFI_HEARTBEAT_SEC
            ):
                _, note = send_drive_to_car(
                    CAR_WIFI_BASE_URL, drive_mode, CAR_WIFI_TIMEOUT_SEC
                )
                _last_drive_mode = drive_mode
                _last_wifi_send_time = now
                _wifi_hud = f"WiFi: {note}"
        else:
            _wifi_hud = "WiFi: off"

        car_state = drive_mode_to_car_state(drive_mode)
        draw_drive_hud(
            img,
            left_state,
            right_steer,
            steer_angle_deg,
            left_seen,
            right_seen,
            drive_mode=f"{drive_mode} State={car_state}",
            wifi_line=_wifi_hud,
            kb_mode=_key_drive_mode,
        )

        cv2.imshow("Image", img)

        key = cv2.waitKeyEx(1)
        if key == _KEY_QUIT:
            break
        elif key in _KEY_FWD:
            _key_drive_mode = "FWD"
            _key_last_press = time.monotonic()
        elif key in _KEY_REV:
            _key_drive_mode = "REV"
            _key_last_press = time.monotonic()
        elif key in _KEY_LEFT:
            _key_drive_mode = "FWD_L"
            _key_last_press = time.monotonic()
        elif key in _KEY_RIGHT:
            _key_drive_mode = "FWD_R"
            _key_last_press = time.monotonic()
        elif key in _KEY_STOP:
            _key_drive_mode = "STOP"
            _key_last_press = time.monotonic()

        if _key_drive_mode is not None and (time.monotonic() - _key_last_press) > _KEY_TIMEOUT_SEC:
            _key_drive_mode = None

camera.release()
cv2.destroyAllWindows()
