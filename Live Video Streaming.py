import cv2

rtsp_url = 'rtsp://your_stream_url'
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Cannot connect to the RTSP stream. Check the URL or stream availability.")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame")
        break
    cv2.imshow('RTSP Stream', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

