import streamlit as st
import cv2
import time
import numpy as np
from ultralytics import YOLO
import colorsys
import os
import base64
from datetime import datetime

model = YOLO("yolov8n.pt")
st.write("YOLO model loaded")

if "frame" not in st.session_state:
    st.session_state.frame = None
if "paused" not in st.session_state:
    st.session_state.paused = False
if "in_count" not in st.session_state:
    st.session_state.in_count = 0
if "out_count" not in st.session_state:
    st.session_state.out_count = 0
if "total_people" not in st.session_state:
    st.session_state.total_people = set()
if "track_memory" not in st.session_state:
    st.session_state.track_memory = {}
if "speed_memory" not in st.session_state:
    st.session_state.speed_memory = {}
if "prev_frame_time" not in st.session_state:
    st.session_state.prev_frame_time = time.time()
if "last_crossing" not in st.session_state:
    st.session_state.last_crossing = ""
if "line_x" not in st.session_state:
    st.session_state.line_x = None
if "frame_width" not in st.session_state:
    st.session_state.frame_width = None
if "frame_height" not in st.session_state:
    st.session_state.frame_height = None
if "crossing_log" not in st.session_state:
    st.session_state.crossing_log = []
if "frame_counter" not in st.session_state:
    st.session_state.frame_counter = 0

confidence_threshold = 0.3
os.makedirs("crossing_frames", exist_ok=True)

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
    if pid in st.session_state.track_memory and abs(cx - st.session_state.track_memory[pid]) > 5:
        arrow_length = 20
        if cx > st.session_state.track_memory[pid]:
            cv2.arrowedLine(frame, (cx, cy), (cx + arrow_length, cy), (255, 255, 0), 2, tipLength=0.3)
        else:
            cv2.arrowedLine(frame, (cx, cy), (cx - arrow_length, cy), (255, 255, 0), 2, tipLength=0.3)

def get_frame(data):
    try:
        data = data.split(',')[1]
        frame_data = np.frombuffer(base64.b64decode(data), np.uint8)
        st.session_state.frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
        if st.session_state.frame_width is None and st.session_state.frame is not None:
            st.session_state.frame_width = st.session_state.frame.shape[1]
            st.session_state.frame_height = st.session_state.frame.shape[0]
            st.session_state.line_x = st.session_state.frame_width // 2
        st.session_state.crossing_log.append(f"Frame {st.session_state.frame_counter}: Captured successfully")
    except Exception as e:
        st.session_state.crossing_log.append(f"Frame {st.session_state.frame_counter}: Error decoding frame: {e}")

st.components.v1.html("""
<script>
    async function setupCamera() {
        const video = document.createElement('video');
        video.style.display = 'none';
        document.body.appendChild(video);
        try {
            const stream = await navigator.mediaDevices.getUserMedia({video: true});
            video.srcObject = stream;
            await video.play();
            console.log("Webcam initialized successfully");
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            function captureFrame() {
                if (video.readyState === video.HAVE_ENOUGH_DATA) {
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    ctx.drawImage(video, 0, 0);
                    const data = canvas.toDataURL('image/jpeg', 0.8);
                    window.streamlit.get_frame(data);
                }
                requestAnimationFrame(captureFrame);
            }
            captureFrame();
        } catch (error) {
            console.error("Error accessing webcam:", error);
        }
    }
    setupCamera();
</script>
""", height=0)

st.session_state.crossing_log.append("Webcam started at " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

col1, col2 = st.columns([3, 1])
with col1:
    frame_placeholder = st.empty()
with col2:
    st.write("### People Counter")
    in_count_display = st.empty()
    out_count_display = st.empty()
    current_people_display = st.empty()
    total_people_display = st.empty()
    fps_display = st.empty()
    last_crossing_display = st.empty()
    line_x_display = st.empty()
    tracking_display = st.empty()
    log_display = st.empty()
    st.button("Pause/Resume", key="pause", on_click=lambda: st.session_state.__setitem__("paused", not st.session_state.paused))
    st.button("Reset", key="reset", on_click=lambda: st.session_state.__setitem__("in_count", 0) or st.session_state.__setitem__("out_count", 0) or st.session_state.total_people.clear() or st.session_state.track_memory.clear() or st.session_state.speed_memory.clear() or st.session_state.__setitem__("last_crossing", "") or st.session_state.__setitem__("frame_counter", 0))
    st.button("Move Line Left", key="move_left", on_click=lambda: st.session_state.__setitem__("line_x", max(50, st.session_state.line_x - 10)) if not st.session_state.paused else None)
    st.button("Move Line Right", key="move_right", on_click=lambda: st.session_state.__setitem__("line_x", min(st.session_state.frame_width - 50, st.session_state.line_x + 10)) if not st.session_state.paused else None)

while True:
    if st.session_state.frame is None:
        st.session_state.crossing_log.append(f"Frame {st.session_state.frame_counter}: No frame captured. Waiting...")
        time.sleep(0.5)
        continue

    if not st.session_state.paused:
        st.session_state.frame_counter += 1
        frame = st.session_state.frame.copy()
        fps, st.session_state.prev_frame_time = calculate_fps(st.session_state.prev_frame_time)

        frame_with_roi = frame.copy()
        cv2.rectangle(frame_with_roi, (0, 0), (st.session_state.line_x, st.session_state.frame_height), (0, 0, 255, 50), -1)
        cv2.rectangle(frame_with_roi, (st.session_state.line_x, 0), (st.session_state.frame_width, st.session_state.frame_height), (0, 255, 0, 50), -1)
        frame = cv2.addWeighted(frame_with_roi, 0.3, frame, 0.7, 0)

        results = model.track(frame, persist=True, classes=[0], verbose=False, conf=confidence_threshold)

        current_people_count = 0

        if results[0].boxes is None or results[0].boxes.id is None:
            st.session_state.track_memory.clear()
            st.session_state.speed_memory.clear()
            st.session_state.crossing_log.append(f"Frame {st.session_state.frame_counter}: No people detected")
        else:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            current_people_count = len(boxes)
            st.session_state.crossing_log.append(f"Frame {st.session_state.frame_counter}: Detected {current_people_count} people")

            for box, pid, conf in zip(boxes, ids, confidences):
                if conf < confidence_threshold:
                    continue
                x1, y1, x2, y2 = map(int, box)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                st.session_state.total_people.add(pid)
                st.session_state.crossing_log.append(f"Frame {st.session_state.frame_counter}: Person {pid} at cx={cx}, conf={conf:.2f}")

                box_color = get_unique_color(pid)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                draw_direction_arrow(frame, cx, cy, st.session_state.track_memory.get(pid, cx), pid)

                put_text_with_border(frame, f"ID: {pid} Conf: {conf:.2f}", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), (0, 0, 0), 1, 2)

                if pid in st.session_state.speed_memory:
                    prev_cx, prev_cy, prev_time = st.session_state.speed_memory[pid]
                    dist = ((cx - prev_cx)**2 + (cy - prev_cy)**2)**0.5
                    time_diff = time.time() - prev_time
                    speed = dist / time_diff if time_diff > 0 else 0
                    put_text_with_border(frame, f"Speed: {speed:.1f} px/s", (x1, y1 - 25), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), (0, 0, 0), 1, 2)
                st.session_state.speed_memory[pid] = (cx, cy, time.time())

                if pid not in st.session_state.track_memory:
                    st.session_state.track_memory[pid] = cx
                    continue

                prev_x = st.session_state.track_memory[pid]
                buffer = 20
                if prev_x < st.session_state.line_x - buffer and cx >= st.session_state.line_x:
                    st.session_state.in_count += 1
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.last_crossing = f"IN at {timestamp}"
                    st.session_state.crossing_log.append(f"Frame {st.session_state.frame_counter}: Person {pid} crossed vertical line (left to right): IN count = {st.session_state.in_count}, cx={cx}, prev_x={prev_x}")
                    cv2.imwrite(f"crossing_frames/in_{timestamp.replace(':', '-')}_{pid}.jpg", frame)
                    with open("crossing_log.txt", "a") as f:
                        f.write(f"{timestamp},Person {pid},IN\n")
                elif prev_x > st.session_state.line_x + buffer and cx <= st.session_state.line_x:
                    st.session_state.out_count += 1
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.last_crossing = f"OUT at {timestamp}"
                    st.session_state.crossing_log.append(f"Frame {st.session_state.frame_counter}: Person {pid} crossed vertical line (right to left): OUT count = {st.session_state.out_count}, cx={cx}, prev_x={prev_x}")
                    cv2.imwrite(f"crossing_frames/out_{timestamp.replace(':', '-')}_{pid}.jpg", frame)
                    with open("crossing_log.txt", "a") as f:
                        f.write(f"{timestamp},Person {pid},OUT\n")

                st.session_state.track_memory[pid] = cx

        cv2.line(frame, (st.session_state.line_x, 0), (st.session_state.line_x, st.session_state.frame_height), (255, 0, 0), 2)
        put_text_with_border(frame, "<-- OUT", (st.session_state.line_x - 100, st.session_state.frame_height // 2 - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), (0, 0, 0), 1, 2)
        put_text_with_border(frame, "IN -->", (st.session_state.line_x + 20, st.session_state.frame_height // 2 - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), (0, 0, 0), 1, 2)

        put_text_with_border(frame, f'IN: {st.session_state.in_count}', (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), (0, 0, 0), 1, 2)
        put_text_with_border(frame, f'OUT: {st.session_state.out_count}', (30, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), (0, 0, 0), 1, 2)
        put_text_with_border(frame, f'Current People: {current_people_count}', (30, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), (0, 0, 0), 1, 2)
        put_text_with_border(frame, f'Total Unique People: {len(st.session_state.total_people)}', (30, 140), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), (0, 0, 0), 1, 2)
        put_text_with_border(frame, f'FPS: {fps:.1f}', (30, 170), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), (0, 0, 0), 1, 2)
        put_text_with_border(frame, st.session_state.last_crossing, (30, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), (0, 0, 0), 1, 2)
        put_text_with_border(frame, f"Line X: {st.session_state.line_x}", (30, 230), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), (0, 0, 0), 1, 2)
        put_text_with_border(frame, f"Frame: {st.session_state.frame_counter}", (30, 260), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), (0, 0, 0), 1, 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        frame_placeholder.image(jpeg.tobytes(), channels="BGR", use_column_width=True)

        in_count_display.write(f"**IN Count**: {st.session_state.in_count}")
        out_count_display.write(f"**OUT Count**: {st.session_state.out_count}")
        current_people_display.write(f"**Current People**: {current_people_count}")
        total_people_display.write(f"**Total Unique People**: {len(st.session_state.total_people)}")
        fps_display.write(f"**FPS**: {fps:.1f}")
        last_crossing_display.write(f"**Last Crossing**: {st.session_state.last_crossing}")
        line_x_display.write(f"**Line X Position**: {st.session_state.line_x}")
        tracking_display.write(f"**Tracking**: {len(st.session_state.track_memory)} people")
        log_display.write("### Event Log\n" + "\n".join(st.session_state.crossing_log[-10:]))

    time.sleep(0.1)