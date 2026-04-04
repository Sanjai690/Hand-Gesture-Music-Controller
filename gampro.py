import cv2
import mediapipe as mp
import pygame
import random
import math
import sys
import time

# ---------------- Initialize ----------------
pygame.init()
WIDTH, HEIGHT = 900, 500
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Gesture Boxing Game")

clock = pygame.time.Clock()
font = pygame.font.Font(None, 40)

# Colors
WHITE = (255, 255, 255)
RED = (220, 50, 50)
BLUE = (50, 100, 220)
BLACK = (0, 0, 0)

# Fighter properties
player_health = 100
enemy_health = 100
last_punch_time = 0
camera_shake = 0

# Animation variables
move_phase = 0
enemy_move_phase = 0

# ---------------- Mediapipe Hands ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Track last wrist position for punch detection
last_wrist_x = None
last_wrist_time = None

# ---------------- Functions ----------------
def draw_health_bar(x, y, health, color):
    pygame.draw.rect(screen, BLACK, (x - 2, y - 2, 204, 24), 2)
    pygame.draw.rect(screen, color, (x, y, 2 * health, 20))

def draw_text(text, x, y, color=WHITE):
    label = font.render(text, True, color)
    screen.blit(label, (x, y))

def detect_punch(landmarks):
    global last_wrist_x, last_wrist_time

    wrist = landmarks[0]  # wrist landmark
    curr_x = wrist.x
    curr_time = time.time()

    punch_detected = False

    if last_wrist_x is not None and last_wrist_time is not None:
        dx = curr_x - last_wrist_x
        dt = curr_time - last_wrist_time

        if dt > 0:
            speed = dx / dt
            if speed > 1.5 and abs(dx) > 0.15:  # strong forward motion
                punch_detected = True

    last_wrist_x = curr_x
    last_wrist_time = curr_time

    return punch_detected

# ---------------- Game Loop ----------------
running = True
while running:
    screen.fill((30, 30, 30))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- Camera Input ---
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    punch = False
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if detect_punch(hand_landmarks.landmark):
                punch = True

    # --- Player Punch ---
    if punch and time.time() - last_punch_time > 1:  # cooldown
        last_punch_time = time.time()
        enemy_health -= 10
        camera_shake = 10

    # --- Enemy AI ---
    if random.random() < 0.01:
        player_health -= 5
        camera_shake = 10

    # --- Movement Animation ---
    move_phase += 0.2
    player_bounce = int(5 * math.sin(move_phase))

    enemy_move_phase += 0.25
    enemy_bounce = int(5 * math.sin(enemy_move_phase))

    # --- Camera Shake ---
    shake_x, shake_y = 0, 0
    if camera_shake > 0:
        shake_x = random.randint(-camera_shake, camera_shake)
        shake_y = random.randint(-camera_shake, camera_shake)
        camera_shake -= 1

    # --- Draw Fighters ---
    player_rect = pygame.Rect(150 + shake_x, 250 + player_bounce + shake_y, 80, 150)
    enemy_rect = pygame.Rect(650 + shake_x, 250 + enemy_bounce + shake_y, 80, 150)

    pygame.draw.rect(screen, BLUE, player_rect)
    pygame.draw.rect(screen, RED, enemy_rect)

    # --- Health Bars ---
    draw_health_bar(50, 50, player_health, BLUE)
    draw_health_bar(650, 50, enemy_health, RED)

    draw_text("YOU", 100, 20, BLUE)
    draw_text("ENEMY", 700, 20, RED)

    # --- Check Win Condition ---
    if player_health <= 0 or enemy_health <= 0:
        winner = "YOU WIN!" if enemy_health <= 0 else "YOU LOSE!"
        draw_text(winner, WIDTH // 2 - 100, HEIGHT // 2, WHITE)
        pygame.display.update()
        pygame.time.wait(3000)
        running = False

    pygame.display.update()
    clock.tick(30)

# ---------------- Cleanup ----------------
cap.release()
pygame.quit()
sys.exit()
