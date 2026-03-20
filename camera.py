import cv2

cap = cv2.VideoCapture(0)
print(type(cap))

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera not found!")
        break

    frame = cv2.flip(frame, 1)

    cv2.imshow("My Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()