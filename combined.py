import cv2
import numpy as np
import csv
import time


def find_quadrant(shape, frame_width, frame_height):
    if len(shape) == 3:
        x, y, _ = shape
    elif len(shape) == 4:
        x, y, _, _ = shape
    else:
        raise ValueError("Invalid shape format")

    cx = x + _ // 2
    cy = y + _ // 2

    quadrant_x = 1 if cx < frame_width // 2 else 2
    quadrant_y = 1 if cy < frame_height // 2 else 2

    return quadrant_x + 2 * (quadrant_y - 1)


def dist(x1, y1, x2, y2):
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


videoCapture = cv2.VideoCapture("D:\AI Internship\AI Assignment video.mp4")
prevCircle = None
dist_threshold = 50
entry_exit_threshold = 20
quadrant_events = []
video_start_time = time.time()

while True:
    ret, frame = videoCapture.read()
    if not ret:
        break

    frame = cv2.resize(frame, (700, 500))
    frame_height, frame_width, _ = frame.shape

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    red_mask1 = cv2.inRange(hsv_frame, lower_red, upper_red)
    contours, _ = cv2.findContours(
        red_mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            if w > 20 and h > 20:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                quadrant_number = find_quadrant((x, y, w, h), frame_width, frame_height)
                if (
                    prevCircle is not None
                    and dist(x, y, prevCircle[0], prevCircle[1]) > dist_threshold
                ):
                    event_type = "Entry" if quadrant_number != prevCircle[2] else "Exit"
                    timestamp = time.time() - video_start_time
                    quadrant_events.append(
                        (timestamp, quadrant_number, "Red", event_type)
                    )
                cv2.putText(
                    frame,
                    f"Quadrant {quadrant_number}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurFrame = cv2.GaussianBlur(grayFrame, (17, 17), 0)
    circles = cv2.HoughCircles(
        blurFrame,
        cv2.HOUGH_GRADIENT,
        1.2,
        100,
        param1=100,
        param2=30,
        minRadius=120,
        maxRadius=400,
    )

    if circles is not None:
        circle = np.uint16(np.around(circles))
        chosen = None
        chosen = (50, 50, 10)
        for i in circles[0, :]:
            if chosen is None or isinstance(chosen, int):
                chosen = tuple(i)
            elif prevCircle is not None:
                if dist(chosen[0], chosen[1], prevCircle[0], prevCircle[1]) <= dist(
                    i[0], i[1], prevCircle[0], prevCircle[1]
                ):
                    chosen = tuple(i)

        if (
            prevCircle is not None
            and dist(chosen[0], chosen[1], prevCircle[0], prevCircle[1])
            > dist_threshold
        ):
            event_type = (
                "Entry" if chosen[2] < prevCircle[2] - entry_exit_threshold else "Exit"
            )
            timestamp = time.time() - video_start_time
            quadrant_events.append(
                (
                    timestamp,
                    find_quadrant(chosen, frame_width, frame_height),
                    "Red",
                    event_type,
                )
            )

        cv2.circle(frame, (int(chosen[0]), int(chosen[1])), 1, (0, 100, 100), 3)
        cv2.circle(
            frame, (int(chosen[0]), int(chosen[1])), int(chosen[2]), (255, 0, 255), 3
        )
        prevCircle = chosen

    cv2.imshow("Objects and Quadrants", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

csv_file_path = "/absolute/path/to/event_data.csv"

with open(csv_file_path, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Time", "Quadrant Number", "Ball Colour", "Type"])

    while True:
        ret, frame = videoCapture.read()
        if not ret:
            break

        if (
            prevCircle is not None
            and dist(chosen[0], chosen[1], prevCircle[0], prevCircle[1])
            > dist_threshold
        ):
            event_type = (
                "Entry" if chosen[2] < prevCircle[2] - entry_exit_threshold else "Exit"
            )
            timestamp = time.time() - video_start_time
            quadrant_number = find_quadrant(chosen, frame_width, frame_height)
            quadrant_events.append((timestamp, quadrant_number, "Red", event_type))
            csv_writer.writerow([timestamp, quadrant_number, "Red", event_type])

csvfile.close()

videoCapture.release()
cv2.destroyAllWindows()
