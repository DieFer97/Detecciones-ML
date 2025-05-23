import cv2
from ultralytics import YOLO

modelo = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ No se pudo acceder a la cámara.")
    exit()

print("✅ Cámara iniciada. Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error al capturar el frame.")
        break

    results = modelo.predict(source=frame, conf=0.5, show=False)

    frame_renderizado = results[0].plot()

    cv2.imshow("YOLOv8 - Detección en Tiempo Real", frame_renderizado)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Saliendo...")
        break

cap.release()
cv2.destroyAllWindows()
