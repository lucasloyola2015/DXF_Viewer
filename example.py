import cv2
import numpy as np

# Cargar imagen (en escala de grises)
img = cv2.imread('Junta_1.jpg', cv2.IMREAD_GRAYSCALE)

# Umbralizado para obtener binaria
_, thresh = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)

# Encontrar contornos (jerarquía completa)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# Crear imagen de salida toda blanca
result = np.ones_like(img) * 255

# Pintar sólo el contorno exterior de negro, y mantener huecos internos en blanco
for i, h in enumerate(hierarchy[0]):
    if h[3] == -1:  # Es un contorno externo
        cv2.drawContours(result, contours, i, 0, thickness=cv2.FILLED)  # Pintar negro

# Mostrar y guardar resultado
cv2.imshow('Resultado', result)
cv2.imwrite('resultado.png', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
