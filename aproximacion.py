import cv2
import time
import torch
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
                results = modelo.predict(source=frame_small, conf=conf, show=False, task='classify', verbose=False)
                end = time.time()

                fps = 1 / (end - start)
                res = results[0]

                probs_raw = getattr(res, 'probs', None)
                if probs_raw is None:
                    raise RuntimeError("No se encontró 'res.probs', revisa la salida del modelo.")
                raw = getattr(probs_raw, 'data', probs_raw)
                if isinstance(raw, torch.Tensor):
                    probs_list = raw.cpu().tolist()
                elif hasattr(raw, 'tolist'):
                    probs_list = raw.tolist()
                else:
                    probs_list = list(raw)

                idx   = probs_list.index(max(probs_list))
                score = probs_list[idx]
                name  = res.names[idx] if hasattr(res, 'names') else modelo.names[idx]

                label = f"{name}: {score*100:.1f}%"
                cv2.putText(
                    frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
                )
                cv2.putText(
                    frame, f"FPS: {fps:.1f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA
                )
                cv2.imshow("YOLOv8 Clasificación - Tiempo Real", frame)

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
    parser = argparse.ArgumentParser(description="Clasificación en tiempo real con YOLOv8.")
    parser.add_argument("--model", type=str, default="yolov8n-cls.pt", help="Ruta al modelo YOLOv8-cls")
    parser.add_argument("--conf",  type=float, default=0.5, help="Umbral de confianza")
    parser.add_argument("--device",type=str,   default="",  help="Dispositivo: 'cpu' o '0' para GPU")
    parser.add_argument("--camera",type=int,   default=0,   help="Índice de la cámara")
    args = parser.parse_args()

    main(args.model, args.conf, args.device, args.camera)