import cv2
import numpy as np 
import time
from tensorflow.keras.models import load_model

# Cargar el modelo
model = load_model("modelo2_mnist.h5")

# Iniciar cámara
cap = cv2.VideoCapture(0)

# Establecer resolución fija
WIDTH, HEIGHT = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# Leer unos frames para asegurar que la cámara se estabilice
for _ in range(5):
    ret, frame = cap.read()
    time.sleep(0.05)

print("Presiona c para capturar y predecir. Presiona q para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No se pudo acceder a la cámara.")
        break
    
    # Forzar tamaño real
    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    # Obtener dimensiones del frame
    box_size = 280
    cx, cy = WIDTH // 2, HEIGHT // 2

    # Calcular coordenadas del cuadro centrado
    x1 = cx - box_size // 2
    y1 = cy - box_size // 2
    x2 = cx + box_size // 2
    y2 = cy + box_size // 2
    
    # Dibujar el rectángulo guía
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "Presiona 'c' para predecir, 'q' para salir", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.imshow("Camara - Reconocimiento de digitos", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)

        resized = cv2.resize(thresh, (28, 28))
        normalized = resized.astype("float32") / 255.0
        input_img = normalized.reshape(1, 28, 28)

        pred = model.predict(input_img)
        digit = np.argmax(pred)
        confianza = np.max(pred)

        print(f"Predicción: {digit} (Confianza: {confianza:.2f})")

        cv2.imshow("Digito capturado", resized)
        cv2.waitKey(0)
    
    elif key == ord('q'):
        print("Saliendo ... ")
        break

cap.release()
cv2.destroyAllWindows()