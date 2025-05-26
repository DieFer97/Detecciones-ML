import cv2
import time
import argparse
from ultralytics import YOLO

def main(model_path, conf, device, camera_index):
    modelo = YOLO(model_path)
    if device:
        modelo.to(device)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ No se pudo acceder a la cámara.")
        return
    
    print("✅ Cámara iniciada. Presiona 'q' para salir, espacio para pausar.")
    paused = False

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error al capturar el frame.")
                    break

                frame_small = cv2.resize(frame, (640, 360))
                start = time.time()
                results = modelo.predict(source=frame_small, conf=conf, show=False, task='segment', verbose=False)
                end = time.time()

                fps = 1 / (end - start)
                annotated = results[0].plot()
                cv2.putText(
                    annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
                )
                cv2.imshow("YOLOv8 Segmentation - Tiempo Real", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("👋 Saliendo...")
                break
            elif key == ord(' '):
                paused = not paused
                print("⏸️ Pausado." if paused else "▶️ Reanudado.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentación en tiempo real con YOLOv8.")
    parser.add_argument("--model", type=str, default="yolov8n-seg.pt", help="Ruta al modelo YOLOv8-seg")
    parser.add_argument("--conf",  type=float, default=0.5, help="Umbral de confianza")
    parser.add_argument("--device",type=str,   default="",  help="Dispositivo: 'cpu' o '0' para GPU")
    parser.add_argument("--camera",type=int,   default=0,   help="Índice de la cámara")
    args = parser.parse_args()

    main(args.model, args.conf, args.device, args.camera)