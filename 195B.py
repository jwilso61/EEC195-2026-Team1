import sensor
import time
import math
import gc
import pyb
import micropython
from pyb import Pin, Timer, LED

micropython.alloc_emergency_exception_buf(100)

BLACK = 0
WHITE = 255
GREY = 120

# =========================================================
# TWO-LANE TRACKER WITH:
# - hardware pin assignments from your control file
# - side-loss recovery / search bias
# - stacked thin far-field slices
# - continuity-weighted far-field detection
# - TURN_COMMIT mode for wide turns when far inner line leaves frame
# - NEAR TWO-LINE ANCHOR: if nearest slice still sees both lines,
#   keep center anchored there
# =========================================================

# -------------------------- PINS / HARDWARE --------------------------
SERVO_PIN = "P7"
SERVO_TIMER_ID = 2
SERVO_FREQ_HZ = 100

MOTOR_PIN = "P8"
MOTOR_TIMER_ID = 4
MOTOR_TIMER_CH = 2
MOTOR_FREQ_HZ = 1500
INA_PIN = "P4"
INB_PIN = "P5"

# -------------------------- BASIC TUNING --------------------------
STEERING_REVERSE = False
MOTOR_REVERSE = False

ENABLE_RECOVERY_LOGIC = False

SERVO_CENTER_US = 1500
SERVO_LEFT_US = 1900
SERVO_RIGHT_US = 1100
SERVO_MIN_US = 1000
SERVO_MAX_US = 2000

MAX_STEER_DELTA_US = 800
SEARCH_BIAS_PX = 10

# PID steering uses the fitted lane-center line from the first three ROIs.
# KP is roughly equivalent to the old proportional steering gain.
STEER_PID_KP = 10.0
STEER_PID_KI = 0.0
STEER_PID_KD = 0
STEER_PID_INTEGRAL_LIMIT_PX_S = 120.0
STEER_PID_DERIVATIVE_ALPHA = 0.35
STEERING_CONTROL_Y = 119

MOTOR_DUTY_PERCENT = 25
MOTOR_SEARCH_DUTY_PERCENT = 9
MOTOR_START_BOOST_PERCENT = 24
MOTOR_START_BOOST_MS = 250

PREEMPTIVE_BRAKE_ENABLE = True
LOOKAHEAD_BRAKE_START_DEG = 8.0
LOOKAHEAD_BRAKE_FULL_DEG = 24.0
LOOKAHEAD_MIN_DUTY_PERCENT = 7

# Far-vs-near lookahead response.
# Difference is measured in pixels:
#   far lane center - near fitted-line prediction at the far-field y.
LOOKAHEAD_STEER_DIFF_SENSITIVITY = 0.4  # 0.05   # demand per px of far/near difference
LOOKAHEAD_STEER_MAX_US = 90  # 180              # max steering added from far-field change
LOOKAHEAD_SPEED_DIFF_SENSITIVITY = 0.2  # 0.04   # demand per px of far/near difference
LOOKAHEAD_SPEED_MAX_DUTY_DROP = 2         # max duty removed from far-field change

MID_HEADING_FALLBACK_ENABLE = True
MID_HEADING_FALLBACK_MIN_DEG = 3.0
MID_HEADING_FALLBACK_SCALE = .9  # more = more turning

# -------------------------- TURN COMMIT TUNING --------------------------
TURN_COMMIT_MID_IDXS = (1, 2)      # near-mid and mid slices dominate during commit
TURN_COMMIT_MID_GAIN = 2.4
TURN_COMMIT_NEAR_GAIN = 0.60

TURN_COMMIT_SEARCH_BIAS_PX = 26
TURN_COMMIT_MIN_STEER_US = 220
TURN_COMMIT_MIN_HEADING_DEG = 4.0
TURN_COMMIT_DUTY_PERCENT = 8

# -------------------------- NEAR ANCHOR TUNING --------------------------
NEAR_ANCHOR_IDX = 0
NEAR_ANCHOR_BLEND = 0.85
NEAR_ANCHOR_MIN_QUALITY = 0.30
TURN_COMMIT_REQUIRES_NEAR_LOSS = True

# -------------------------- VISION TUNING --------------------------
WHITE_THRESHOLD = [(155, 255)]
# QQVGA= 160W X 120L

MAX_BLOB_WIDTH = 150  # 80
MAX_BLOB_HEIGHT = 45
MAX_BLOB_PIXELS = 3000

# lower  = accept dashed, angled, blurry, or partially visible tape
# higher = rejects weak/noisy blobs, but may accidentally reject real tape
MIN_BLOB_DENSITY = 0.10  # 0.18

EXPECTED_LANE_WIDTH = 130
MIN_LANE_WIDTH = 25
MAX_LANE_WIDTH = 160
ONE_LINE_OFFSET = 68

WIDTH_SCORE_GAIN = 1.0
CENTER_SCORE_GAIN = 0.8  # 2.2
EDGE_SCORE_GAIN = 0.3  # 1.1
SINGLE_LINE_SCORE_PENALTY = 26
MIN_ACCEPT_QUALITY = 0.00  # .12

FAR_SLICE_START_IDX = 4
FAR_MIN_PAIR_QUALITY = 0.45

MIN_VALID_CENTER_POINTS = 2
MAX_MISSING_FRAMES = 8

# STEERING CONTROL
# --------- in basic tuning
# STEER_PID_KP/KI/KD    # PID steering gains for fitted lane-center line
# MAX_STEER_DELTA_US    # steering authority limit
# --------- in vision tuning
# ALPHA_CENTER          # smoothing of fitted-line lateral error
# ALPHA_HEADING         # smoothing of heading debug / recovery estimate

ALPHA_CENTER = 0.5
ALPHA_HEADING = 0.85
ALPHA_WIDTH = 0.35

PRINT_EVERY = 10
START_DELAY_MS = 1500

# -------------------------- ROI LAYOUT --------------------------
# LANE_SLICES format:
#   (x, y, w, h, weight, allow_one_line)
#
# x, y:
#   Top-left corner of the ROI.
#   Larger y values are closer to the bottom of the image,
#   which corresponds to the road closer to the car.
#
# w, h:
#   Width and height of the ROI.
#   All regions span the full image width.
#   Lower/near regions are taller.
#   Upper/far regions are thinner to improve continuity checking.
#
# weight:
#   How much that ROI contributes to the estimated lane center.
#   Higher weight = stronger influence on steering.
#
# allow_one_line:
#   True  = if only one lane marking is visible, estimate the lane center
#           using the remembered/expected lane width.
#   False = require both left and right lane markings.
#           This is safer for far-field regions, where false detections
#           are more likely.
#
# Near slices: Used for stable lane centering.
# Mid slices: Used for early turn detection.
# Far slices: Used for lookahead only if detections are continuous/plausible.

LANE_SLICES = [  # x, y, w, h, weight, allow_one_line,
                 # px_thresh, area_thresh, expected_width, min_width, max_width
    (0, 88, 160, 26, 0.22, True,  6, 6, 145, 70, 160),  # near
    (0, 76, 160, 16, 0.35, True,  6, 6, 130, 60, 155),  # near-mid
    (0, 64, 160, 12, 0.18, True,  6, 6, 110, 50, 145),  # mid
    (0, 54, 160,  9, 0.11, False, 5, 5,  90, 40, 130),  # far
    (0, 45, 160,  8, 0.08, False, 4, 5,  75, 30, 115),
    (0, 37, 160,  7, 0.05, False, 4, 4,  60, 25, 100),
    (0, 30, 160,  6, 0.03, False, 3, 4,  50, 20,  90),
]

# QQVGA= 160W X 120L
IMG_W = 160
IMG_H = 120
IMG_CENTER_X = IMG_W // 2


# -------------------------- UTILITIES --------------------------
def clamp(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def lowpass(prev, new, alpha):
    return alpha * new + (1.0 - alpha) * prev


def score_to_quality(score, ceiling=1.0):
    q = 1.0 - (0.012 * score)
    return clamp(q, 0.0, ceiling)


def predict_x(points, current_y, fallback_x):
    n = len(points)

    if n >= 2:
        x0, y0 = points[-2]
        x1, y1 = points[-1]
        dy = y1 - y0
        if dy != 0:
            slope = (x1 - x0) / float(dy)
            pred = x1 + slope * (current_y - y1)
        else:
            pred = x1
        return clamp(pred, 0, IMG_W - 1)

    if n == 1:
        return clamp(points[-1][0], 0, IMG_W - 1)

    return clamp(fallback_x, 0, IMG_W - 1)


def fit_center_line_x(points, target_y, fallback_x):
    n = len(points)

    if n <= 0:
        return fallback_x

    if n == 1:
        return points[0][0]

    sum_w = 0.0
    sum_y = 0.0
    sum_x = 0.0
    for p in points:
        x = p[0]
        y = p[1]
        w = p[2] if len(p) >= 3 else 1.0
        sum_w += w
        sum_y += y * w
        sum_x += x * w

    if sum_w <= 0.0:
        return fallback_x

    mean_y = sum_y / sum_w
    mean_x = sum_x / sum_w

    var_y = 0.0
    cov_xy = 0.0
    for p in points:
        x = p[0]
        y = p[1]
        w = p[2] if len(p) >= 3 else 1.0
        dy = y - mean_y
        var_y += w * dy * dy
        cov_xy += w * dy * (x - mean_x)

    if var_y <= 0.0001:
        return mean_x

    slope = cov_xy / var_y
    line_x = mean_x + slope * (target_y - mean_y)
    return clamp(line_x, 0, IMG_W - 1)


def line_heading_deg(points):
    if len(points) < 2:
        return 0.0

    lower_point = max(points, key=lambda p: p[1])
    upper_point = min(points, key=lambda p: p[1])

    dx = upper_point[0] - lower_point[0]
    dy = lower_point[1] - upper_point[1]

    if dy != 0:
        return math.degrees(math.atan(dx / dy))

    return 0.0


def weighted_center_and_y(points, fallback_x, fallback_y):
    sum_w = 0.0
    sum_x = 0.0
    sum_y = 0.0

    for p in points:
        x = p[0]
        y = p[1]
        w = p[2] if len(p) >= 3 else 1.0
        sum_w += w
        sum_x += x * w
        sum_y += y * w

    if sum_w <= 0.0:
        return fallback_x, fallback_y

    return sum_x / sum_w, sum_y / sum_w


def update_steering_pid(error_px, now_ms):
    global steer_pid_integral, steer_pid_prev_error, steer_pid_prev_ms
    global steer_pid_filtered_derivative

    if steer_pid_prev_ms is None:
        steer_pid_prev_ms = now_ms
        steer_pid_prev_error = error_px
        steer_pid_filtered_derivative = 0.0

    dt_ms = time.ticks_diff(now_ms, steer_pid_prev_ms)
    if dt_ms <= 0:
        dt = 0.001
    else:
        dt = dt_ms / 1000.0

    steer_pid_integral += error_px * dt
    steer_pid_integral = clamp(
        steer_pid_integral,
        -STEER_PID_INTEGRAL_LIMIT_PX_S,
        STEER_PID_INTEGRAL_LIMIT_PX_S
    )

    derivative = (error_px - steer_pid_prev_error) / dt
    steer_pid_filtered_derivative = lowpass(
        steer_pid_filtered_derivative,
        derivative,
        STEER_PID_DERIVATIVE_ALPHA
    )

    steer_pid_prev_error = error_px
    steer_pid_prev_ms = now_ms

    return (STEER_PID_KP * error_px) + \
           (STEER_PID_KI * steer_pid_integral) + \
           (STEER_PID_KD * steer_pid_filtered_derivative)


def reset_steering_pid():
    global steer_pid_integral, steer_pid_prev_error, steer_pid_prev_ms
    global steer_pid_filtered_derivative

    steer_pid_integral = 0.0
    steer_pid_prev_error = 0.0
    steer_pid_prev_ms = None
    steer_pid_filtered_derivative = 0.0


# -------------------------- LEDS --------------------------
red_led = LED(1)
green_led = LED(2)
blue_led = LED(3)


def leds(red=False, green=False, blue=False):
    red_led.on() if red else red_led.off()
    green_led.on() if green else green_led.off()
    blue_led.on() if blue else blue_led.off()


# -------------------------- MOTOR / SERVO HARDWARE --------------------------
ina = Pin(INA_PIN, Pin.OUT_PP)
inb = Pin(INB_PIN, Pin.OUT_PP)

servo_pin = Pin(SERVO_PIN, Pin.OUT_PP)
servo_pin.low()
servo_pulse_us = SERVO_CENTER_US

motor_timer = Timer(MOTOR_TIMER_ID, freq=MOTOR_FREQ_HZ)
motor_ch = motor_timer.channel(MOTOR_TIMER_CH, Timer.PWM, pin=Pin(MOTOR_PIN))


def set_servo_us(us):
    global servo_pulse_us
    us = int(clamp(us, SERVO_MIN_US, SERVO_MAX_US))
    servo_pulse_us = us
    return us


def servo_cb(timer):
    servo_pin.high()
    pyb.udelay(servo_pulse_us)
    servo_pin.low()


servo_timer = Timer(SERVO_TIMER_ID, freq=SERVO_FREQ_HZ)
servo_timer.callback(servo_cb)

motor_enabled = False
motor_duty = 0
motor_started_ms = time.ticks_ms()


def motor_forward(duty_percent):
    global motor_enabled, motor_duty, motor_started_ms

    if MOTOR_REVERSE:
        ina.low()
        inb.high()
    else:
        ina.high()
        inb.low()

    if not motor_enabled:
        motor_started_ms = time.ticks_ms()

    motor_enabled = True
    duty_percent = int(clamp(duty_percent, 0, 100))

    if time.ticks_diff(time.ticks_ms(), motor_started_ms) < MOTOR_START_BOOST_MS:
        duty_percent = max(duty_percent, MOTOR_START_BOOST_PERCENT)

    motor_duty = duty_percent
    motor_ch.pulse_width_percent(motor_duty)


def motor_stop():
    global motor_enabled, motor_duty
    motor_enabled = False
    motor_duty = 0
    motor_ch.pulse_width_percent(0)
    ina.low()
    inb.low()


# -------------------------- CAMERA --------------------------
sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.QQVGA)  # QQVGA= 160W X 120L
sensor.skip_frames(time=2000)
sensor.set_auto_gain(False)
sensor.set_auto_whitebal(False)

clock = time.clock()


# -------------------------- BLOB / CANDIDATE SELECTION --------------------------
def get_good_blobs(img, roi, px_thresh, area_thresh):
    blobs = img.find_blobs(
        WHITE_THRESHOLD,
        roi=roi,
        pixels_threshold=px_thresh,
        area_threshold=area_thresh,
        merge=True,
        margin=5
    )

    good = []

    for b in blobs:
        # Draw every raw blob before filtering
        img.draw_rectangle(b.rect(), color=GREY)
        img.draw_cross(b.cx(), b.cy(), color=GREY)

        # Temporarily accept everything for debug
        good.append(b)

    return good


def choose_lane_candidate(blobs, pred_center, expected_width, pred_left, pred_right,
                          allow_one_line, min_width, max_width):
    if not blobs:
        return None

    blobs = sorted(blobs, key=lambda b: b.cx())

    best = None
    best_score = 1e9

    # --------------------------------------------------
    # 1. Prefer true left/right lane pairs
    # --------------------------------------------------
    for i in range(len(blobs)):
        for j in range(i + 1, len(blobs)):
            left_blob = blobs[i]
            right_blob = blobs[j]

            lx = left_blob.cx()
            rx = right_blob.cx()
            width = rx - lx

            if width < min_width or width > max_width:
                continue

            center_x = 0.5 * (lx + rx)

            width_err = abs(width - expected_width)
            center_err = abs(center_x - pred_center)

            edge_err = 0.0
            if pred_left is not None:
                edge_err += abs(lx - pred_left)
            if pred_right is not None:
                edge_err += abs(rx - pred_right)

            score = (WIDTH_SCORE_GAIN * width_err) + \
                    (CENTER_SCORE_GAIN * center_err) + \
                    (EDGE_SCORE_GAIN * edge_err)

            if score < best_score:
                best_score = score
                best = {
                    "pair": True,
                    "single": False,
                    "left_seen": True,
                    "right_seen": True,
                    "missing_side": 0,
                    "left_blob": left_blob,
                    "right_blob": right_blob,
                    "left_x": lx,
                    "right_x": rx,
                    "center_x": center_x,
                    "lane_width": width,
                    "quality": score_to_quality(score, 1.0)
                }

    if best is not None:
        return best

    # --------------------------------------------------
    # 2. Single-line fallback
    # --------------------------------------------------
    # Estimate the lane center from one visible marking.
    # This keeps the tracker centered between lanes instead of steering
    # directly onto a single stripe.
    # --------------------------------------------------
    if not allow_one_line:
        return None

    best_single = None
    best_single_score = 1e9

    for b in blobs:
        bx = b.cx()

        # Decide whether this visible line is probably left or right
        # relative to the predicted lane center.
        if bx < pred_center:
            # Visible line is probably the left lane marking.
            center_x = bx + (expected_width * 0.5)
            left_seen = True
            right_seen = False
            missing_side = 1      # +1 means search right later if desired
            left_x = bx
            right_x = -1

            edge_err = abs(bx - pred_left) if pred_left is not None else 0.0

        else:
            # Visible line is probably the right lane marking.
            center_x = bx - (expected_width * 0.5)
            left_seen = False
            right_seen = True
            missing_side = -1     # -1 means search left later if desired
            left_x = -1
            right_x = bx

            edge_err = abs(bx - pred_right) if pred_right is not None else 0.0

        center_x = clamp(center_x, 0, IMG_W - 1)
        center_err = abs(center_x - pred_center)

        score = SINGLE_LINE_SCORE_PENALTY + \
                (CENTER_SCORE_GAIN * center_err) + \
                (EDGE_SCORE_GAIN * edge_err)

        if score < best_single_score:
            best_single_score = score
            best_single = {
                "pair": False,
                "single": True,
                "left_seen": left_seen,
                "right_seen": right_seen,
                "missing_side": missing_side,
                "left_blob": b if left_seen else None,
                "right_blob": b if right_seen else None,
                "left_x": left_x,
                "right_x": right_x,
                "center_x": center_x,
                "lane_width": expected_width,
                "quality": score_to_quality(score, 0.65)
            }

    return best_single

# -------------------------- STATE --------------------------
filtered_lateral_error = 0.0
filtered_heading_error = 0.0
filtered_lane_width = EXPECTED_LANE_WIDTH

last_valid_lateral = 0.0
last_valid_heading = 0.0
last_valid_lane_width = EXPECTED_LANE_WIDTH

steer_pid_integral = 0.0
steer_pid_prev_error = 0.0
steer_pid_prev_ms = None
steer_pid_filtered_derivative = 0.0

last_left_x = IMG_CENTER_X - (EXPECTED_LANE_WIDTH * 0.5)
last_right_x = IMG_CENTER_X + (EXPECTED_LANE_WIDTH * 0.5)
last_center = IMG_CENTER_X

missing_frames = 0
frame_counter = 0

# -------------------------- STARTUP --------------------------
set_servo_us(SERVO_CENTER_US)
motor_stop()
leds(blue=True)
time.sleep_ms(START_DELAY_MS)
leds()

# -------------------------- MAIN LOOP --------------------------
while True:
    clock.tick()
    frame_counter += 1
    now_ms = time.ticks_ms()
    img = sensor.snapshot()

    center_points = []
    steering_center_points = []
    far_center_points = []
    center_history = []
    left_history = []
    right_history = []

    accepted = []
    mid_points = []

    near_pair_seen = False
    near_pair_center = last_center
    near_pair_quality = 0.0

    lane_widths = []
    total_left_seen = 0
    total_right_seen = 0

    far_left_seen = 0
    far_right_seen = 0

    effective_weight_sum = 0.0
    base_weight_sum = 0.0

    far_pair_streak = 0
    best_far_pair_streak = 0
    lookahead_heading_deg = 0.0
    lookahead_brake_demand = 0.0
    lookahead_diff_px = 0.0
    lookahead_steer_us = 0.0
    lookahead_speed_demand = 0.0

    expected_width = last_valid_lane_width if last_valid_lane_width > 0 else EXPECTED_LANE_WIDTH

    for idx, s in enumerate(LANE_SLICES):
        x, y, w, h, base_weight, allow_one_line, px_thresh, area_thresh, exp_width, min_width, max_width = s
        y_mid = y + (h//2)

        img.draw_rectangle((x, y, w, h), color=GREY)  # draw rois

        pred_center = predict_x(center_history, y_mid, last_center)

        if last_left_x is not None:
            fallback_left = last_left_x
        else:
            fallback_left = pred_center - (expected_width * 0.5)

        if last_right_x is not None:
            fallback_right = last_right_x
        else:
            fallback_right = pred_center + (expected_width * 0.5)

        pred_left = predict_x(left_history, y_mid, fallback_left)
        pred_right = predict_x(right_history, y_mid, fallback_right)

        blobs = get_good_blobs(img, (x, y, w, h), px_thresh, area_thresh)
        cand = choose_lane_candidate(
            blobs,
            pred_center,
            exp_width,
            pred_left,
            pred_right,
            allow_one_line,
            min_width,
            max_width
        )

        if cand is None or cand["quality"] < MIN_ACCEPT_QUALITY:
            if idx >= FAR_SLICE_START_IDX:
                far_pair_streak = 0
            continue

        eff_weight = base_weight * cand["quality"]
        base_weight_sum += base_weight
        effective_weight_sum += eff_weight

        cx = cand["center_x"]
        center_points.append((cx, y_mid, eff_weight))
        if idx < FAR_SLICE_START_IDX:
            steering_center_points.append((cx, y_mid, eff_weight))
        else:
            far_center_points.append((cx, y_mid, eff_weight))
        center_history.append((cx, y_mid))

        accepted.append({
            "idx": idx,
            "center_x": cx,
            "y_mid": y_mid,
            "eff_weight": eff_weight,
            "pair": cand["pair"],
            "quality": cand["quality"]
        })

        if idx == NEAR_ANCHOR_IDX and cand["pair"] and cand["quality"] >= NEAR_ANCHOR_MIN_QUALITY:
            near_pair_seen = True
            near_pair_center = cx
            near_pair_quality = cand["quality"]

        if idx in TURN_COMMIT_MID_IDXS:
            mid_points.append((cx, y_mid))

        if cand["left_seen"]:
            total_left_seen += 1
            left_history.append((cand["left_x"], y_mid))
            last_left_x = cand["left_x"]

        if cand["right_seen"]:
            total_right_seen += 1
            right_history.append((cand["right_x"], y_mid))
            last_right_x = cand["right_x"]

        if idx >= FAR_SLICE_START_IDX:
            if cand["left_seen"]:
                far_left_seen += 1
            if cand["right_seen"]:
                far_right_seen += 1

        if cand["pair"] and idx < FAR_SLICE_START_IDX:
            lane_widths.append(cand["lane_width"])

        if cand["left_seen"] and cand["left_blob"] is not None:
            img.draw_rectangle(cand["left_blob"].rect(), color=BLACK)
            img.draw_cross(cand["left_blob"].cx(), cand["left_blob"].cy(), color=BLACK)

        if cand["right_seen"] and cand["right_blob"] is not None:
            img.draw_rectangle(cand["right_blob"].rect(), color=BLACK)
            img.draw_cross(cand["right_blob"].cx(), cand["right_blob"].cy(), color=BLACK)

        img.draw_cross(int(cx), y_mid, color=WHITE)

        if idx >= FAR_SLICE_START_IDX:
            if cand["pair"] and cand["quality"] >= FAR_MIN_PAIR_QUALITY:
                far_pair_streak += 1
                if far_pair_streak > best_far_pair_streak:
                    best_far_pair_streak = far_pair_streak
            else:
                far_pair_streak = 0
        if cand["pair"]:
            img.draw_line(int(cand["left_x"]), y_mid,
                          int(cand["right_x"]), y_mid,
                          color=WHITE)
            img.draw_cross(int(cand["center_x"]), y_mid, color=WHITE)

    # ---------------- Side-loss condition ----------------
    if ENABLE_RECOVERY_LOGIC:
        missing_left = (total_left_seen == 0 and total_right_seen > 0)
        missing_right = (total_right_seen == 0 and total_left_seen > 0)
        both_missing = (total_left_seen == 0 and total_right_seen == 0)
        if missing_left:
            search_direction = -1
        elif missing_right:
            search_direction = 1
        else:
            search_direction = 0
    else:  # Temporarily disabled while tuning basic lane detection and steering.
        missing_left = False
        missing_right = False
        both_missing = False
        search_direction = 0
        far_missing_left = False
        far_missing_right = False
        mid_heading_deg = 0.0
        if len(mid_points) >= 2:
            lower_mid = max(mid_points, key=lambda p: p[1])
            upper_mid = min(mid_points, key=lambda p: p[1])
            dx_mid = upper_mid[0] - lower_mid[0]
            dy_mid = lower_mid[1] - upper_mid[1]
            if dy_mid != 0:
                mid_heading_deg = math.degrees(math.atan(dx_mid / dy_mid))
        turn_commit = False
        commit_direction = 0

    # ---------------- Turn-commit detection ----------------
    far_missing_left = (far_left_seen == 0 and far_right_seen > 0)
    far_missing_right = (far_right_seen == 0 and far_left_seen > 0)

    mid_heading_deg = 0.0
    if len(mid_points) >= 2:
        lower_mid = max(mid_points, key=lambda p: p[1])
        upper_mid = min(mid_points, key=lambda p: p[1])

        dx_mid = upper_mid[0] - lower_mid[0]
        dy_mid = lower_mid[1] - upper_mid[1]

        if dy_mid != 0:
            mid_heading_deg = math.degrees(math.atan(dx_mid / dy_mid))

    turn_commit = False
    commit_direction = 0

    if abs(mid_heading_deg) >= TURN_COMMIT_MIN_HEADING_DEG:
        if far_missing_left:
            turn_commit = True
            commit_direction = -1
        elif far_missing_right:
            turn_commit = True
            commit_direction = 1

    if TURN_COMMIT_REQUIRES_NEAR_LOSS and near_pair_seen:
        turn_commit = False
        commit_direction = 0

    # ---------------- Lane estimate ----------------
    lane_detected = (len(steering_center_points) >= MIN_VALID_CENTER_POINTS)

    if lane_detected:
        missing_frames = 0

        if turn_commit:
            weighted_center_sum = 0.0
            weight_sum = 0.0

            for a in accepted:
                if a["idx"] >= FAR_SLICE_START_IDX:
                    continue

                w = a["eff_weight"]

                if a["idx"] in TURN_COMMIT_MID_IDXS:
                    w *= TURN_COMMIT_MID_GAIN
                elif a["idx"] == 0:
                    w *= TURN_COMMIT_NEAR_GAIN

                weighted_center_sum += a["center_x"] * w
                weight_sum += w
        else:
            weighted_center_sum = 0.0
            weight_sum = 0.0
            for cx, cy, w_eff in steering_center_points:
                weighted_center_sum += cx * w_eff
                weight_sum += w_eff

        if weight_sum > 0:
            lane_center_x = weighted_center_sum / weight_sum
        else:
            lane_center_x = last_center

        if near_pair_seen:
            lane_center_x = (NEAR_ANCHOR_BLEND * near_pair_center) + \
                            ((1.0 - NEAR_ANCHOR_BLEND) * lane_center_x)

        lateral_error_px = lane_center_x - IMG_CENTER_X
        last_center = lane_center_x

        if len(lane_widths) > 0:
            measured_width = sum(lane_widths) / len(lane_widths)
            filtered_lane_width = lowpass(filtered_lane_width, measured_width, ALPHA_WIDTH)
            last_valid_lane_width = filtered_lane_width

        pts_sorted = sorted(center_points, key=lambda p: p[1])
        for i in range(len(pts_sorted) - 1):
            x0, y0, _ = pts_sorted[i]
            x1, y1, _ = pts_sorted[i + 1]
            img.draw_line(int(x0), int(y0), int(x1), int(y1), color=WHITE)

        img.draw_line(IMG_CENTER_X, 0, IMG_CENTER_X, IMG_H, color=BLACK)

        raw_heading_deg = line_heading_deg(steering_center_points)
        lookahead_heading_deg = line_heading_deg(far_center_points)

        if near_pair_seen and abs(mid_heading_deg) >= TURN_COMMIT_MIN_HEADING_DEG:
            heading_error_deg = mid_heading_deg

        elif turn_commit:
            heading_error_deg = mid_heading_deg

        elif MID_HEADING_FALLBACK_ENABLE and abs(mid_heading_deg) >= MID_HEADING_FALLBACK_MIN_DEG:
            heading_error_deg = mid_heading_deg * MID_HEADING_FALLBACK_SCALE

        else:
            heading_error_deg = raw_heading_deg

        lane_line_x = fit_center_line_x(
            steering_center_points,
            STEERING_CONTROL_Y,
            lane_center_x
        )
        lane_line_error_px = lane_line_x - IMG_CENTER_X

        if len(far_center_points) > 0:
            far_center_x, far_center_y = weighted_center_and_y(
                far_center_points,
                lane_center_x,
                STEERING_CONTROL_Y
            )
            near_at_far_x = fit_center_line_x(
                steering_center_points,
                far_center_y,
                lane_center_x
            )
            lookahead_diff_px = far_center_x - near_at_far_x
            lookahead_steer_demand = clamp(
                abs(lookahead_diff_px) * LOOKAHEAD_STEER_DIFF_SENSITIVITY,
                0.0,
                1.0
            )
            if lookahead_diff_px < 0:
                lookahead_steer_us = -LOOKAHEAD_STEER_MAX_US * lookahead_steer_demand
            else:
                lookahead_steer_us = LOOKAHEAD_STEER_MAX_US * lookahead_steer_demand

            lookahead_speed_demand = clamp(
                abs(lookahead_diff_px) * LOOKAHEAD_SPEED_DIFF_SENSITIVITY,
                0.0,
                1.0
            )

        filtered_lateral_error = lowpass(filtered_lateral_error, lane_line_error_px, ALPHA_CENTER)
        filtered_heading_error = lowpass(filtered_heading_error, heading_error_deg, ALPHA_HEADING)

        last_valid_lateral = filtered_lateral_error
        last_valid_heading = filtered_heading_error

        if PREEMPTIVE_BRAKE_ENABLE and len(far_center_points) >= 2:
            lookahead_abs = abs(lookahead_heading_deg)
            lookahead_brake_demand = (lookahead_abs - LOOKAHEAD_BRAKE_START_DEG) / \
                                     (LOOKAHEAD_BRAKE_FULL_DEG - LOOKAHEAD_BRAKE_START_DEG)
            lookahead_brake_demand = clamp(lookahead_brake_demand, 0.0, 1.0)

        if base_weight_sum > 0:
            confidence = effective_weight_sum / base_weight_sum
        else:
            confidence = 0.0

        confidence = clamp(confidence, 0.0, 1.0)

    else:
        missing_frames += 1
        confidence = 0.0
        lookahead_heading_deg = 0.0
        lookahead_brake_demand = 0.0
        lookahead_diff_px = 0.0
        lookahead_steer_us = 0.0
        lookahead_speed_demand = 0.0

        if missing_frames <= MAX_MISSING_FRAMES:
            filtered_lateral_error = last_valid_lateral
            filtered_heading_error = last_valid_heading
        else:
            filtered_lateral_error *= 0.8
            filtered_heading_error *= 0.8
            reset_steering_pid()

    # ---------------- Search bias ----------------
    if ENABLE_RECOVERY_LOGIC:
        if turn_commit:
            search_bias = TURN_COMMIT_SEARCH_BIAS_PX * commit_direction
        else:
            search_bias = SEARCH_BIAS_PX * search_direction
        commanded_lateral_error = filtered_lateral_error + search_bias
    else:
        search_bias = 0
        commanded_lateral_error = filtered_lateral_error

    # ---------------- Mode ----------------
    if ENABLE_RECOVERY_LOGIC:
        if both_missing and missing_frames > MAX_MISSING_FRAMES:
            mode = "LOST"
        elif turn_commit:
            mode = "TURN_COMMIT"
        elif missing_left or missing_right:
            mode = "SEARCH"
        else:
            mode = "TRACK"
    else:
        if lane_detected:
            mode = "TRACK"
        else:
            mode = "LOST"

    # ---------------- Steering output ----------------
    if mode == "LOST":
        reset_steering_pid()
        steer_term = 0.0
    else:
        steer_term = update_steering_pid(commanded_lateral_error, now_ms)
        steer_term += lookahead_steer_us

    if ENABLE_RECOVERY_LOGIC:
        if turn_commit and commit_direction != 0:
            min_commit = TURN_COMMIT_MIN_STEER_US * commit_direction

            if (steer_term * min_commit) <= 0:
                steer_term = min_commit
            elif abs(steer_term) < abs(min_commit):
                steer_term = min_commit

    steer_term = clamp(steer_term, -MAX_STEER_DELTA_US, MAX_STEER_DELTA_US)

    if STEERING_REVERSE:
        servo_us = SERVO_CENTER_US + steer_term
    else:
        servo_us = SERVO_CENTER_US - steer_term

    last_servo = set_servo_us(servo_us)

    # ---------------- Motor output ----------------
    if mode == "LOST":
        motor_stop()
        leds(red=True)
    elif ENABLE_RECOVERY_LOGIC:
        steering_demand = abs(steer_term) / float(MAX_STEER_DELTA_US)
        steering_demand = clamp(steering_demand, 0.0, 1.0)
        duty = MOTOR_DUTY_PERCENT - int((MOTOR_DUTY_PERCENT - MOTOR_SEARCH_DUTY_PERCENT) * steering_demand)
        if PREEMPTIVE_BRAKE_ENABLE:
            duty -= int((duty - LOOKAHEAD_MIN_DUTY_PERCENT) * lookahead_brake_demand)
            duty = int(clamp(duty, LOOKAHEAD_MIN_DUTY_PERCENT, MOTOR_DUTY_PERCENT))
        duty -= int(LOOKAHEAD_SPEED_MAX_DUTY_DROP * lookahead_speed_demand)
        duty = int(clamp(duty, LOOKAHEAD_MIN_DUTY_PERCENT, MOTOR_DUTY_PERCENT))
        leds(green=True)
        motor_forward(duty)
    else:
        if mode == "TURN_COMMIT":
            duty = TURN_COMMIT_DUTY_PERCENT
            leds(red=True, green=True)
        elif mode == "SEARCH":
            duty = MOTOR_SEARCH_DUTY_PERCENT
            leds(red=True, blue=True)
        else:
            steering_demand = abs(steer_term) / float(MAX_STEER_DELTA_US)
            steering_demand = clamp(steering_demand, 0.0, 1.0)
            duty = MOTOR_DUTY_PERCENT - int((MOTOR_DUTY_PERCENT - MOTOR_SEARCH_DUTY_PERCENT) * steering_demand)
            if PREEMPTIVE_BRAKE_ENABLE:
                duty -= int((duty - LOOKAHEAD_MIN_DUTY_PERCENT) * lookahead_brake_demand)
                duty = int(clamp(duty, LOOKAHEAD_MIN_DUTY_PERCENT, MOTOR_DUTY_PERCENT))
            duty -= int(LOOKAHEAD_SPEED_MAX_DUTY_DROP * lookahead_speed_demand)
            duty = int(clamp(duty, LOOKAHEAD_MIN_DUTY_PERCENT, MOTOR_DUTY_PERCENT))
            leds(green=True)

        motor_forward(duty)

    # ---------------- Debug ----------------
    if frame_counter % PRINT_EVERY == 0:
        print("fps:%.1f mode:%s conf:%.2f pid_err:%.2f cmd_err:%.2f hdg:%.2f mid_hdg:%.2f look_hdg:%.2f look_diff:%.2f look_steer:%.1f brake:%.2f look_spd:%.2f nearPair:%d nearQ:%.2f width:%.2f servo:%d motor:%d far_streak:%d farMissL:%d farMissR:%d" %
              (clock.fps(),
               mode,
               confidence,
               filtered_lateral_error,
               commanded_lateral_error,
               filtered_heading_error,
               mid_heading_deg,
               lookahead_heading_deg,
               lookahead_diff_px,
               lookahead_steer_us,
               lookahead_brake_demand,
               lookahead_speed_demand,
               1 if near_pair_seen else 0,
               near_pair_quality,
               last_valid_lane_width,
               last_servo,
               motor_duty if motor_enabled else 0,
               best_far_pair_streak,
               1 if far_missing_left else 0,
               1 if far_missing_right else 0))

    if frame_counter % 100 == 0:
        gc.collect()
