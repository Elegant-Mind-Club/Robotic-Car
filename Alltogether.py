import cv2
import numpy as np
import socket
import time
from ultralytics import YOLO
from multiprocessing import Process, Queue, Value

# Load YOLO
model = YOLO("yolov8n.pt")  # Load YOLOv8 weightsKrisha Basrur <krishabasrur@gmail.com>

# URL of the video stream
URL = "http://192.168.1.88/stream"  # Replace with your ESP32 camera stream URL

# Replace with your ESP32's IP address and port
esp32_ip = "192.168.1.88"
esp32_port = 8080

# Function to capture frames
def capture_frames(queue, running):
    cap = cv2.VideoCapture(URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    if not cap.isOpened():
        print(f"Error: Unable to open video stream at {URL}")
        running.value = 0
        return

    while running.value:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from video stream")
            running.value = 0
            break
        queue.put(frame)

    cap.release()

# Function to process frames
def process_frames(queue, running, person_detected, last_person_detected_time, person_position):
    while running.value:
        if not queue.empty():
            frame = queue.get()
            frame_resized = cv2.resize(frame, (640, 640))
            results = model(frame_resized)

            person_detected.value = False
            person_position.value = -1

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()  # Convert to int
                    center_x = (x1 + x2) // 2

                    if model.names[cls] == 'person':
                        person_detected.value = True
                        last_person_detected_time.value = time.time()
                        person_position.value = center_x
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, 'Person', (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow('Object Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running.value = 0
                break

    cv2.destroyAllWindows()

# Function to send commands
def send_commands(running, person_detected, last_person_detected_time, person_position):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((esp32_ip, esp32_port))
        client_socket.settimeout(10)  # Increase the timeout for socket operations
    except socket.error as e:
        print(f"Socket error: {e}")
        running.value = 0
        return

    def send_command(command):
        print(f"Sending command: {command}")
        client_socket.send((command + '\n').encode())
        try:
            ack = client_socket.recv(1024).decode().strip()
            if ack == "ACK":
                print("Acknowledgment received")
            else:
                print("No acknowledgment received")
        except socket.timeout:
            print("Acknowledgment timeout")

    while running.value:
        current_time = time.time()
        if person_detected.value:
            if person_position.value < 320:  # Assuming 640x640 frame, center is at 320
                turn_command = "TURN_LEFT -1000 -1000 1000 1000"  # Turn left
                send_command(turn_command)
                time.sleep(0.1)  # Turn for 0.1 seconds
            elif person_position.value > 320:
                turn_command = "TURN_RIGHT 1000 1000 -1000 -1000"  # Turn right
                send_command(turn_command)
                time.sleep(0.1) 

            command = "FORWARD -1000 -1000 -1000 -1000"  # Move forward
            send_command(command)
            time.sleep(0.1) 
        else:
            if current_time - last_person_detected_time.value > 20:
                # Turn for 0.1 seconds with speed 500
                turn_command = "TURN 500 500 -500 -500"
                send_command(turn_command)
                time.sleep(0.1)
                # Stop for 1 second
                stop_command = "STOP 0 0 0 0"
                send_command(stop_command)
                time.sleep(1)
            else:
                stop_command = "STOP 0 0 0 0"  # Stop the car
                send_command(stop_command)
                time.sleep(1)  # Stop for 1 second

    # Ensure the car stops when the process is ending
    stop_command = "STOP 0 0 0 0"
    send_command(stop_command)
    client_socket.close()

if __name__ == "__main__":
    frame_queue = Queue(maxsize=10)
    running = Value('i', 1)
    person_detected = Value('i', 0)
    last_person_detected_time = Value('d', time.time())
    person_position = Value('i', -1)

    capture_process = Process(target=capture_frames, args=(frame_queue, running))
    process_process = Process(target=process_frames, args=(frame_queue, running, person_detected, last_person_detected_time, person_position))
    command_process = Process(target=send_commands, args=(running, person_detected, last_person_detected_time, person_position))

    capture_process.start()
    process_process.start()
    command_process.start()

    capture_process.join()
    process_process.join()
    command_process.join()

