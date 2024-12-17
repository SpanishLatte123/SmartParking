import os
import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from ultralytics import YOLO
import time

# Load YOLOv8 model
model = YOLO('yolov8s.pt')

# Load parking polygons from a NumPy file
def load_polygons(file_path):
    """Load parking space polygons from a NumPy file."""
    return np.load(file_path, allow_pickle=True)

polygons = load_polygons("parking_space_init_data/polygon_coordinates.npy")

# Initialize DataFrames for logging
parking_status_log = pd.DataFrame(columns=["Timestamp"] + [f"Space_{i}" for i in range(1, len(polygons) + 1)])
entry_exit_log = pd.DataFrame(columns=["Space_ID", "Car_Status", "Entry_Time", "Exit_Time"])

def initialize_excel_file(file_path):
    """Ensure the Excel file exists and initialize sheets if needed."""
    if not os.path.exists(file_path):
        with pd.ExcelWriter(file_path) as writer:
            parking_status_log.to_excel(writer, sheet_name="Status_Log", index=False)
            entry_exit_log.to_excel(writer, sheet_name="Entry_Exit_Log", index=False)
        print(f"Initialized {file_path} with empty logs.")

def is_parking_space_available(polygon, cars_bounding_boxes, threshold=0.5):
    """Check if a parking space is available based on overlapping bounding boxes."""
    parking_polygon = Polygon(polygon)
    poly_area = parking_polygon.area

    for box in cars_bounding_boxes:
        x1, y1, x2, y2 = box
        car_polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        intersection = parking_polygon.intersection(car_polygon)
        inter_area = intersection.area if intersection.is_valid else 0
        overlap_ratio = inter_area / poly_area if poly_area > 0 else 0
        if overlap_ratio > threshold:
            return False
    return True

def update_entry_exit_log(entry_exit_log, space_id, status, timestamp):
    """Track entry and exit events for cars."""
    global prev_statuses
    prev_status = prev_statuses.get(space_id, "Free")

    if prev_status == "Free" and status == "Occupied":
        # Log car entry
        new_entry = {"Space_ID": space_id, "Car_Status": "Occupied", "Entry_Time": timestamp, "Exit_Time": None}
        entry_exit_log = pd.concat([entry_exit_log, pd.DataFrame([new_entry])], ignore_index=True)
    elif prev_status == "Occupied" and status == "Free":
        # Log car exit
        entry_exit_log.loc[
            (entry_exit_log["Space_ID"] == space_id) & (entry_exit_log["Exit_Time"].isna()), "Exit_Time"
        ] = timestamp

    prev_statuses[space_id] = status
    return entry_exit_log

def create_abstracted_frame(polygons, occupancy_summary, frame_size):
    """Generate an abstracted view showing parking space availability."""
    abstracted_frame = np.ones((frame_size[1], frame_size[0], 3), dtype=np.uint8) * 255
    for i, polygon in enumerate(polygons, start=1):
        color = (0, 255, 0) if occupancy_summary[i] == "Free" else (0, 0, 255)
        cv2.fillPoly(abstracted_frame, [np.array(polygon, dtype=np.int32)], color=color)
        center_x = int(np.mean([pt[0] for pt in polygon]))
        center_y = int(np.mean([pt[1] for pt in polygon]))
        cv2.putText(abstracted_frame, str(i), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return abstracted_frame

def process_video(video_path, output_path, abstracted_output_path, excel_path, threshold=0.5):
    """Process video, detect parking space availability, and log data."""
    global parking_status_log, entry_exit_log, prev_statuses
    prev_statuses = {i: "Free" for i in range(1, len(polygons) + 1)}

    # Initialize Excel file
    initialize_excel_file(excel_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (width, height)

    processed_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size)
    abstracted_writer = cv2.VideoWriter(abstracted_output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size)

    cv2.namedWindow("Processed Video", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Abstracted View", cv2.WINDOW_NORMAL)

    last_log_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform car detection
        results = model(frame)
        cars_bounding_boxes = [
            (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            for box in results[0].boxes.data.cpu().numpy()
            if int(box[-1]) == 2  # Filter for cars
        ]

        # Draw bounding boxes for cars
        for (x1, y1, x2, y2) in cars_bounding_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Generate overlay and abstracted view
        overlay = frame.copy()
        occupancy_summary = {}
        for i, polygon in enumerate(polygons, start=1):
            is_free = is_parking_space_available(polygon, cars_bounding_boxes, threshold)
            status = "Free" if is_free else "Occupied"
            occupancy_summary[i] = status
            color = (0, 255, 0) if is_free else (0, 0, 255)
            cv2.fillPoly(overlay, [np.array(polygon, dtype=np.int32)], color=color)
            center_x = int(np.mean([pt[0] for pt in polygon]))
            center_y = int(np.mean([pt[1] for pt in polygon]))
            cv2.putText(overlay, str(i), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        abstracted_frame = create_abstracted_frame(polygons, occupancy_summary, frame_size)

        # Write frames to video
        processed_writer.write(overlay)
        abstracted_writer.write(abstracted_frame)

        # Log data every second
        current_time = time.time()
        if current_time - last_log_time >= 1:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            status_row = {"Timestamp": timestamp}
            for space_id, status in occupancy_summary.items():
                status_row[f"Space_{space_id}"] = status
                entry_exit_log = update_entry_exit_log(entry_exit_log, space_id, status, timestamp)
            parking_status_log = pd.concat([parking_status_log, pd.DataFrame([status_row])], ignore_index=True)

            # Write logs to Excel
            with pd.ExcelWriter(excel_path) as writer:
                parking_status_log.to_excel(writer, sheet_name="Status_Log", index=False)
                entry_exit_log.to_excel(writer, sheet_name="Entry_Exit_Log", index=False)
            last_log_time = current_time

        # Display videos
        cv2.imshow("Processed Video", overlay)
        cv2.imshow("Abstracted View", abstracted_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    processed_writer.release()
    abstracted_writer.release()
    cv2.destroyAllWindows()

# Run the processing
process_video(
    "sample_vids/stock.mp4",
    "outputs/processed_video.avi",
    "outputs/abstracted_video.avi",
    "outputs/parking_data.xlsx",
    threshold=0.5
)
