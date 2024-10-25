import cv2
import os
import time
import pygame
import mediapipe as mp
# Initialize Pygame and Pygame mixer
pygame.init()
try:
    pygame.mixer.init()
    print("Pygame mixer initialized successfully.")
except pygame.error as e:
    print(f"Failed to initialize Pygame mixer: {e}")
    exit(1)

# Load the sound file (ensure you have a sound file in the specified directory)
sound_path = r"C:\Users\azaan\OneDrive\Desktop\camera\suono.wav"
sound_path_mano=r"C:\Users\azaan\OneDrive\Desktop\camera\suono2.wav"
try:
    sound = pygame.mixer.Sound(sound_path)
    sound2 = pygame.mixer.Sound(sound_path_mano)
except pygame.error as e:
    print(f"Failed to load sound file: {e}")
    exit(1)

# HOG detector initialization
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Paths
capture_directory = r"C:\Users\azaan\OneDrive\Desktop\camera\bilder"

# Ensure the directory exists
if not os.path.exists(capture_directory):
    print("Creating directory:", capture_directory)
    os.makedirs(capture_directory, exist_ok=True)

# Variable to keep track of the last time a photo was taken
last_photo_time = time.time()

# Video capture initialization
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open video capture device.")
else:
    print("Video capture device successfully opened.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Could not read frame from video capture device.")
        break

    frame = cv2.resize(frame, (640, 480))

    # Detect persons
    (persons, _) = hog.detectMultiScale(frame, winStride=(9, 9), padding=(16, 16), scale=1.05)
    #hand dtection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    hand_detected = False
    if results.multi_hand_landmarks:
        hand_detected = True
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    current_time = time.time()

    if len(persons) > 0 and (current_time - last_photo_time) >= 5:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(capture_directory, f"person_{timestamp}.jpg")

        print(f"Detected persons: {len(persons)}")
        print(f"Saving image to {filename}")

        # Ensure that image writing was successful
        if cv2.imwrite(filename, frame):
            print(f"Image successfully saved to {filename}")

            # Play sound on person detection
            sound.play()

            # Update last photo time
            last_photo_time = time.time()
        else:
            print(f"Error: Could not save image to {filename}")
    
    #hand
    if hand_detected  and (current_time - last_photo_time) >= 5:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(capture_directory, f"hand_{timestamp}.jpg")

        print(f"hand detected")
        print(f"Saving image to {filename}")

        # Ensure that image writing was successful
        if cv2.imwrite(filename, frame):
            print(f"Image successfully saved to {filename}")

            # Play sound on person detection
            sound2.play()

            # Update last photo time
            last_photo_time = time.time()
        else:
            print(f"Error: Could not save image to {filename}")

    # Draw rectangles around detected persons
    for (x, y, w, h) in persons:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Person Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
pygame.quit()  # Cleanup pygame
