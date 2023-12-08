import cv2
import numpy as np

def find_quadrant(rect, frame_width, frame_height):
    x, y, w, h = rect
    cx = x + w // 2
    cy = y + h // 2

    quadrant_x = 1 if cx < frame_width // 2 else 2
    quadrant_y = 1 if cy < frame_height // 2 else 2

    return quadrant_x + 2 * (quadrant_y - 1)

videoCapture = cv2.VideoCapture('D:\AI Internship\AI Assignment video.mp4')

while True:
    ret, frame = videoCapture.read()

    if not ret:
        break

    frame = cv2.resize(frame, (700, 500))
    frame_height, frame_width, _ = frame.shape

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    edges = cv2.Canny(blurred_frame, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            quadrant_number = find_quadrant((x, y, w, h), frame_width, frame_height)
            
            cv2.putText(frame, f'Quadrant {quadrant_number}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('Rectangles and Quadrants', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()
