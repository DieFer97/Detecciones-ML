import cv2
import torch
from ultralytics import YOLO

modelo = YOLO('yolov8n-cls.pt')

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

    results = modelo.predict(source=frame, task='classify', show=False, verbose=False)
    res = results[0]

    probs_raw = getattr(res, 'probs', None)
    if probs_raw is None:
        raise RuntimeError("No se encontró 'res.probs', revisa la salida del modelo.")

    if hasattr(probs_raw, 'data'):
        raw = probs_raw.data
        if isinstance(raw, torch.Tensor):
            probs_list = raw.cpu().tolist()
        elif hasattr(raw, 'tolist'):
            probs_list = raw.tolist()
        else:
            probs_list = list(raw)
    elif isinstance(probs_raw, torch.Tensor):
        probs_list = probs_raw.cpu().tolist()
    elif hasattr(probs_raw, 'tolist'):
        probs_list = probs_raw.tolist()
    elif isinstance(probs_raw, (list, tuple)):
        probs_list = list(probs_raw)
    else:
        raise RuntimeError(f"Tipo inesperado para 'probs': {type(probs_raw)}")

    idx   = probs_list.index(max(probs_list))
    score = probs_list[idx]
    name  = res.names[idx]

    label = f"{name}: {score*100:.1f}%"
    cv2.putText(
        frame, label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.imshow("YOLOv8 Clasificación - Tiempo Real", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Saliendo...")
        break

cap.release()
cv2.destroyAllWindows()
