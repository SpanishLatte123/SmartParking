import cv2
import numpy as np

# Path to the video
video_path = "sample_vids/stock.mp4"

# Open the video and read the first frame
cap = cv2.VideoCapture(video_path)
ret, image = cap.read()
cap.release()

if not ret:
    print("Error: Unable to read video file.")
    exit()

# Initialization
overlay_image = image.copy()
points, polygons = [], []
alpha = 0.3  # Transparency for the overlay

# Create a named window
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

# Mouse callback function
def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(overlay_image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Image", overlay_image)

# Draw the current polygon
def draw_current_polygon():
    temp_overlay = overlay_image.copy()
    if len(points) > 2:
        cv2.fillPoly(temp_overlay, [np.array(points)], color=(0, 255, 0))
    cv2.imshow("Image", temp_overlay)

# Draw all finalized polygons
def draw_all_polygons():
    global overlay_image
    overlay = image.copy()
    for i, polygon in enumerate(polygons, start=1):
        cv2.fillPoly(overlay, [np.array(polygon)], color=(0, 255, 0))
        center = np.mean(polygon, axis=0).astype(int)
        cv2.putText(overlay, str(i), tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
    overlay_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

# Save abstracted polygons
def save_abstracted_image():
    abstracted = np.full_like(image, 255)
    for polygon in polygons:
        cv2.fillPoly(abstracted, [np.array(polygon)], color=(100, 100, 100))
    cv2.imwrite("outputs/abstracted_video.jpg", abstracted)

# Save polygon coordinates as a numpy file
def save_polygon_coordinates():
    np.save("parking_space_init_data/polygon_coordinates.npy", polygons)
    print("Polygon coordinates saved to 'outputs/polygon_coordinates.npy'")

# Set mouse callback
cv2.setMouseCallback("Image", select_points)

# Main loop
while True:
    draw_current_polygon()
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # Save and quit
        cv2.imwrite("outputs/overlay_video.jpg", overlay_image)
        save_abstracted_image()
        save_polygon_coordinates()
        break
    elif key == ord('r'):  # Reset current polygon
        points = []
        overlay_image = image.copy()
        draw_all_polygons()
    elif key == ord('d'):  # Finalize current polygon
        if len(points) > 2:
            polygons.append(points)
            points = []
            draw_all_polygons()
    elif key == ord('0'):  # Reset all polygons
        points, polygons = [], []
        overlay_image = image.copy()

cv2.destroyAllWindows()
