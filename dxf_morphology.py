"""
Módulo para análisis morfológico de imágenes DXF utilizando OpenCV.
Este módulo contiene funciones para procesar imágenes DXF convertidas a formato compatible con OpenCV.
"""

import cv2
import numpy as np
import os


def dxf_to_morphology(dxf_image, threshold_value=200):
    """
    Función principal que procesa una imagen DXF y realiza análisis morfológico completo.
    
    Args:
        dxf_image: Imagen DXF en formato OpenCV (numpy array)
        threshold_value: Valor de umbral para binarización (0-255)
        
    Returns:
        Diccionario con imágenes generadas, métricas y datos de contornos
    """
    try:
        # Verificar que la imagen de entrada sea válida
        if dxf_image is None or dxf_image.size == 0:
            raise ValueError("La imagen de entrada no es válida")
        
        print(f"Procesando imagen de tamaño: {dxf_image.shape}")
        
        # --- Inicio del código de process_dxf_image integrado ---
        # Verificar forma y tipo de la imagen
        print(f"Imagen de entrada: forma={dxf_image.shape}, tipo={dxf_image.dtype}")
        
        # Convertir a escala de grises si es necesario
        if len(dxf_image.shape) == 3:
            gray = cv2.cvtColor(dxf_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = dxf_image.copy()
        
        # Verificar imagen vacía
        if np.all(gray == 0) or np.all(gray == 255):
            print("ADVERTENCIA: La imagen está completamente negra o blanca")
        
        # Estadísticas de imagen
        print(f"Estadísticas de imagen: min={np.min(gray)}, max={np.max(gray)}, mean={np.mean(gray):.2f}")
        
        # Aplicar desenfoque gaussiano
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Probar diferentes métodos de umbralización
        _, thresh_inv = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)
        thresh_adapt = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)
        _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Buscar contornos con cada método
        contours_inv, hierarchy_inv = cv2.findContours(thresh_inv.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        contours_adapt, hierarchy_adapt = cv2.findContours(thresh_adapt.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        contours_otsu, hierarchy_otsu = cv2.findContours(thresh_otsu.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        # Seleccionar método con más contornos
        contours_counts = [
            (len(contours_inv) if contours_inv is not None else 0, thresh_inv, contours_inv, hierarchy_inv, "Umbral invertido"),
            (len(contours_adapt) if contours_adapt is not None else 0, thresh_adapt, contours_adapt, hierarchy_adapt, "Umbral adaptativo"),
            (len(contours_otsu) if contours_otsu is not None else 0, thresh_otsu, contours_otsu, hierarchy_otsu, "Umbral Otsu")
        ]
        contours_counts.sort(reverse=True, key=lambda x: x[0])
        count, thresh, contours, hierarchy, method_name = contours_counts[0]
        print(f"Método seleccionado: {method_name} con {count} contornos")
        
        # Manejar caso sin contornos
        if count == 0:
            print("No se encontraron contornos con ningún método de umbralización")
            contours = []
            hierarchy = None
        
        # Contar contornos externos e internos
        contornos_externos = 0
        contornos_internos = 0
        if hierarchy is not None:
            contornos_externos = sum(1 for h in hierarchy[0] if h[3] == -1)
            contornos_internos = len(contours) - contornos_externos
            print(f"Contornos externos: {contornos_externos}, Contornos internos: {contornos_internos}")
        
        # --- Fin del código de process_dxf_image integrado ---
        
        # Crear imagen binaria para visualización
        h, w = dxf_image.shape[:2]
        if thresh is None:
            thresh = np.ones((h, w), dtype=np.uint8) * 255
        thresh_display = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        # Inicializar métricas
        metrics = {
            'contornos_externos': {'cantidad': contornos_externos, 'perimetro_total': 0, 'area_total': 0, 'detalles': []},
            'contornos_internos': {'cantidad': contornos_internos, 'perimetro_total': 0, 'area_total': 0, 'detalles': []}
        }

        # Si no hay contornos, devolver imágenes vacías con mensaje
        if not contours or len(contours) == 0:
            print("No se encontraron contornos en la imagen")
            blank_img = np.ones((h, w, 3), dtype=np.uint8) * 255
            
            # Añadir mensaje a la imagen en blanco
            cv2.putText(blank_img, "NO SE ENCONTRARON CONTORNOS", (w//2-150, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            return {
                'images': {
                    'original': dxf_image,
                    'binaria': thresh,
                    'contornos_externos': blank_img,
                    'contornos_internos': blank_img,
                    'todos_contornos': blank_img,
                    'contornos_rellenos': thresh
                },
                'metrics': metrics
            }
        
        
        
        # Crear imágenes para contornos externos e internos
        external_contours_img = np.ones((h, w, 3), dtype=np.uint8) * 255
        internal_contours_img = np.ones((h, w, 3), dtype=np.uint8) * 255
        all_contours_img = np.ones((h, w, 3), dtype=np.uint8) * 255
        filled_contours_img = np.ones((h, w), dtype=np.uint8) * 255
        
        # Dibujar contornos externos en rojo
        for i, h in enumerate(hierarchy[0]):
            if h[3] == -1:  # Es un contorno externo
                # Dibujar contorno con línea muy gruesa para asegurar visibilidad
                cv2.drawContours(external_contours_img, contours, i, (0, 0, 255), thickness=5)
                cv2.drawContours(all_contours_img, contours, i, (0, 0, 255), thickness=5)
                
                # Dibujar contorno relleno en negro (como en Example.py)
                cv2.drawContours(filled_contours_img, contours, i, 0, thickness=cv2.FILLED)
                
                # Calcular métricas
                area = cv2.contourArea(contours[i])
                perimetro = cv2.arcLength(contours[i], True)
                
                # Calcular circularidad (4π*área/perímetro²)
                # Un círculo perfecto tiene circularidad = 1
                circularidad = 0
                if perimetro > 0:
                    circularidad = 4 * np.pi * area / (perimetro * perimetro)
                
                # Calcular centro usando momentos
                M = cv2.moments(contours[i])
                cx, cy = 0, 0
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Mostrar área en la imagen
                    cv2.putText(all_contours_img, f"A:{area:.0f}", (cx, cy),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                
                    # Dibujar un punto en el centro del contorno para mayor visibilidad
                    cv2.circle(external_contours_img, (cx, cy), 10, (0, 0, 255), -1)
                    cv2.circle(all_contours_img, (cx, cy), 10, (0, 0, 255), -1)
                
                # Guardar métricas del contorno
                metrics['contornos_externos']['detalles'].append({
                    'id': contornos_externos,
                    'area': area,
                    'perimetro': perimetro,
                    'circularidad': circularidad,
                    'centro_x': cx,
                    'centro_y': cy
                })
                
                # Actualizar totales
                metrics['contornos_externos']['cantidad'] += 1
                metrics['contornos_externos']['perimetro_total'] += perimetro
                metrics['contornos_externos']['area_total'] += area
                
                contornos_externos += 1
        
        # Dibujar contornos internos en verde
        for i, h in enumerate(hierarchy[0]):
            if h[3] != -1:  # Es un contorno interno
                # Dibujar contorno con línea muy gruesa para asegurar visibilidad
                cv2.drawContours(internal_contours_img, contours, i, (0, 255, 0), thickness=5)
                cv2.drawContours(all_contours_img, contours, i, (0, 255, 0), thickness=5)
                
                # Calcular métricas
                area = cv2.contourArea(contours[i])
                perimetro = cv2.arcLength(contours[i], True)
                
                # Calcular circularidad (4π*área/perímetro²)
                circularidad = 0
                if perimetro > 0:
                    circularidad = 4 * np.pi * area / (perimetro * perimetro)
                
                # Calcular centro usando momentos
                M = cv2.moments(contours[i])
                cx, cy = 0, 0
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Mostrar área en la imagen
                    cv2.putText(all_contours_img, f"A:{area:.0f}", (cx, cy),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                    # Dibujar un punto en el centro del contorno para mayor visibilidad
                    cv2.circle(internal_contours_img, (cx, cy), 10, (0, 255, 0), -1)
                    cv2.circle(all_contours_img, (cx, cy), 10, (0, 255, 0), -1)
                
                # Guardar métricas del contorno
                metrics['contornos_internos']['detalles'].append({
                    'id': contornos_internos,
                    'area': area,
                    'perimetro': perimetro,
                    'circularidad': circularidad,
                    'centro_x': cx,
                    'centro_y': cy
                })
                
                # Actualizar totales
                metrics['contornos_internos']['cantidad'] += 1
                metrics['contornos_internos']['perimetro_total'] += perimetro
                metrics['contornos_internos']['area_total'] += area
                
                contornos_internos += 1
        
        print(f"Dibujados {metrics['contornos_externos']['cantidad']} contornos externos y {metrics['contornos_internos']['cantidad']} contornos internos")
        
        # Añadir texto informativo a las imágenes
        cv2.putText(external_contours_img, "CONTORNOS EXTERNOS", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(internal_contours_img, "CONTORNOS INTERNOS", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(all_contours_img, "TODOS LOS CONTORNOS", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(filled_contours_img, "CONTORNOS RELLENOS", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Devolver todas las imágenes generadas y las métricas calculadas
        return {
            'images': {
                'original': dxf_image,
                'binaria': thresh,
                'contornos_externos': external_contours_img,
                'contornos_internos': internal_contours_img,
                'todos_contornos': all_contours_img,
                'contornos_rellenos': filled_contours_img
            },
            'metrics': metrics,
            'contours': contours,
            'hierarchy': hierarchy
        }
            
    except Exception as e:
        print(f"Error en dxf_to_morphology: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Crear una imagen de error para mostrar
        h, w = (500, 500)  # Tamaño predeterminado
        if dxf_image is not None and len(dxf_image.shape) >= 2:
            h, w = dxf_image.shape[:2]
        
        error_img = np.ones((h, w, 3), dtype=np.uint8) * 255  # Fondo blanco
        cv2.rectangle(error_img, (50, 50), (w-50, h-50), (0, 0, 255), 2)  # Borde rojo
        cv2.putText(error_img, "ERROR", (w//2-100, h//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.putText(error_img, str(e), (50, h//2+50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        
        # Devolver la imagen de error para todas las salidas
        blank_gray = np.ones((h, w), dtype=np.uint8) * 255
        return {
            'images': {
                'original': error_img,
                'binaria': blank_gray,
                'contornos_externos': error_img,
                'contornos_internos': error_img,
                'todos_contornos': error_img,
                'contornos_rellenos': blank_gray
            },
            'metrics': {
                'contornos_externos': {'cantidad': 0, 'perimetro_total': 0, 'area_total': 0, 'detalles': []},
                'contornos_internos': {'cantidad': 0, 'perimetro_total': 0, 'area_total': 0, 'detalles': []}
            },
            'contours': [],
            'hierarchy': None
        }