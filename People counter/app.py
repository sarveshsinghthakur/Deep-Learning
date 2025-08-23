import cv2
import time
import numpy as np
from ultralytics import YOLO
import colorsys
import os

model = YOLO("yolov8n.pt")
print("YOLO model loaded")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame from webcam")
    cap.release()
    exit()
frame_width, frame_height = frame.shape[1], frame.shape[0]
line_x = frame_width // 2
in_count, out_count = 0, 0
total_people = set()
track_memory = {}
speed_memory = {}
prev_frame_time = time.time()
paused = False
confidence_threshold = 0.4
last_crossing = ""
os.makedirs("crossing_frames", exist_ok=True)
crossing_log = open("crossing_log.txt", "a")

def calculate_fps(prev_time):
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if current_time > prev_time else 0
    return fps, current_time

def put_text_with_border(frame, text, position, font, font_scale, text_color, border_color=(0, 0, 0), thickness=1, border_thickness=2):
    cv2.putText(frame, text, position, font, font_scale, border_color, border_thickness)
    cv2.putText(frame, text, position, font, font_scale, text_color, thickness)

def get_unique_color(pid):
    hue = (pid % 10) / 10.0
    rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)

def draw_direction_arrow(frame, cx, cy, prev_x, pid):
    if pid in track_memory and abs(cx - track_memory[pid]) > 5:
        arrow_length = 20
        if cx > track_memory[pid]:
            cv2.arrowedLine(frame, (cx, cy), (cx + arrow_length, cy), (255, 255, 0), 2, tipLength=0.3)
        else:
            cv2.arrowedLine(frame, (cx, cy), (cx - arrow_length, cy), (255, 255, 0), 2, tipLength=0.3)

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        fps, prev_frame_time = calculate_fps(prev_frame_time)

        frame_with_roi = frame.copy()
        cv2.rectangle(frame_with_roi, (0, 0), (line_x, frame_height), (0, 0, 255, 50), -1)
        cv2.rectangle(frame_with_roi, (line_x, 0), (frame_width, frame_height), (0, 255, 0, 50), -1)
        frame = cv2.addWeighted(frame_with_roi, 0.3, frame, 0.7, 0)

        results = model.track(frame, persist=True, classes=[0], verbose=False, conf=confidence_threshold)

        current_people_count = 0

        if results[0].boxes is None or results[0].boxes.id is None:
            track_memory.clear()
            speed_memory.clear()
            print("No people detected in this frame")
        else:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            current_people_count = len(boxes)
            print(f"Detected {current_people_count} people")

            for box, pid, conf in zip(boxes, ids, confidences):
                if conf < confidence_threshold:
                    continue
                x1, y1, x2, y2 = map(int, box)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                total_people.add(pid)

                box_color = get_unique_color(pid)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                draw_direction_arrow(frame, cx, cy, track_memory.get(pid, cx), pid)

                put_text_with_border(frame, f"ID: {pid} Conf: {conf:.2f}", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), (0, 0, 0), 1, 2)

                if pid in speed_memory:
                    prev_cx, prev_cy, prev_time = speed_memory[pid]
                    dist = ((cx - prev_cx)**2 + (cy - prev_cy)**2)**0.5
                    time_diff = time.time() - prev_time
                    speed = dist / time_diff if time_diff > 0 else 0
                    put_text_with_border(frame, f"Speed: {speed:.1f} px/s", (x1, y1 - 25), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), (0, 0, 0), 1, 2)
                speed_memory[pid] = (cx, cy, time.time())

                if pid not in track_memory:
                    track_memory[pid] = cx
                    continue

                prev_x = track_memory[pid]
                buffer = 10
                if prev_x < line_x - buffer and cx >= line_x:
                    in_count += 1
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    last_crossing = f"IN at {timestamp}"
                    print(f"Person {pid} crossed vertical line (left to right): IN count = {in_count}, cx={cx}, prev_x={prev_x}")
                    crossing_log.write(f"{timestamp},Person {pid},IN\n")
                    cv2.imwrite(f"crossing_frames/in_{timestamp.replace(':', '-')}_{pid}.jpg", frame)
                elif prev_x > line_x + buffer and cx <= line_x:
                    out_count += 1
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    last_crossing = f"OUT at {timestamp}"
                    print(f"Person {pid} crossed vertical line (right to left): OUT count = {out_count}, cx={cx}, prev_x={prev_x}")
                    crossing_log.write(f"{timestamp},Person {pid},OUT\n")
                    cv2.imwrite(f"crossing_frames/out_{timestamp.replace(':', '-')}_{pid}.jpg", frame)

                track_memory[pid] = cx

        cv2.line(frame, (line_x, 0), (line_x, frame_height), (255, 0, 0), 2)
        put_text_with_border(frame, "<-- OUT", (line_x - 100, frame_height // 2 - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), (0, 0, 0), 1, 2)
        put_text_with_border(frame, "IN -->", (line_x + 20, frame_height // 2 - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), (0, 0, 0), 1, 2)

        put_text_with_border(frame, f'IN: {in_count}', (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), (0, 0, 0), 1, 2)
        put_text_with_border(frame, f'OUT: {out_count}', (30, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), (0, 0, 0), 1, 2)
        put_text_with_border(frame, f'Current People: {current_people_count}', (30, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), (0, 0, 0), 1, 2)
        put_text_with_border(frame, f'Total Unique People: {len(total_people)}', (30, 140), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), (0, 0, 0), 1, 2)
        put_text_with_border(frame, f'FPS: {fps:.1f}', (30, 170), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), (0, 0, 0), 1, 2)
        put_text_with_border(frame, last_crossing, (30, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), (0, 0, 0), 1, 2)
        put_text_with_border(frame, f"Line X: {line_x}", (30, 230), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), (0, 0, 0), 1, 2)
        put_text_with_border(frame, "Press: q=quit, r=reset, p=pause/resume, l=move line left, r=move line right", 
                            (30, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), (0, 0, 0), 1, 2)

        cv2.imshow("Interactive People Counter", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        in_count, out_count = 0, 0
        total_people.clear()
        track_memory.clear()
        speed_memory.clear()
        last_crossing = ""
        print("Counters reset")
    elif key == ord('p'):
        paused = not paused
        print("Paused" if paused else "Resumed")
    elif key == ord('l') and not paused:
        line_x = max(50, line_x - 10)
        print(f"Line moved left to x={line_x}")
    elif key == ord('r') and not paused:
        line_x = min(frame_width - 50, line_x + 10)
        print(f"Line moved right to x={line_x}")

crossing_log.close()
cap.release()
cv2.destroyAllWindows()