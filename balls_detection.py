import cv2
import numpy as np

videoCapture = cv2.VideoCapture('D:\AI Internship\AI Assignment video.mp4')
prevCircle = None
dist = lambda x1, y1, x2, y2: (x1 - x2) ** 2 + (y1 - y2) ** 2

while True:
    ret, frame = videoCapture.read()

    if not ret:
        break
    frame = cv2.resize(frame, (700, 500))

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

        cv2.circle(frame, (int(chosen[0]), int(chosen[1])), 1, (0, 100, 100), 3)
        cv2.circle(frame, (int(chosen[0]), int(chosen[1])), int(chosen[2]), (255, 0, 255), 3)
        prevCircle = chosen

    cv2.imshow("circles", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

videoCapture.release()
cv2.destroyAllWindows()