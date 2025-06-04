import sys
import os
import math
import numpy as np
import cv2
import ezdxf
import dxf_morphology  # Importar el módulo de análisis morfológico
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                              QWidget, QLabel, QFileDialog, QMessageBox, QTabWidget,
                              QToolBar, QStatusBar, QAction)
from PySide6.QtCore import Qt, QSize, Signal, QPointF, QTransform
from PySide6.QtGui import QPixmap, QImage, QColor, QPainter, QPen

class DXFRenderer(QWidget):
    """Widget personalizado para renderizar archivos DXF utilizando QPainter"""
    
    statusMessage = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Configuración inicial
        self.setMinimumSize(600, 400)
        
        # Variables para DXF
        self.dxf_doc = None
        self.dxf_entities = []
        self.bounds = (0, 0, 0, 0)
        
        # Variables para el visor
        self.zoom_factor = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        
        # Colores para entidades
        self.background_color = QColor(0, 0, 0)
        self.entity_colors = {
            'LINE': QColor(255, 255, 255),
            'CIRCLE': QColor(0, 255, 255),
            'ARC': QColor(255, 0, 255),
            'TEXT': QColor(255, 255, 0),
            'LWPOLYLINE': QColor(0, 255, 0),
            'POLYLINE': QColor(100, 255, 100),
            'default': QColor(255, 200, 0)
        }
        
        # Estadísticas
        self.entity_counts = {}
        
        # Inicializar transformación
        self.transform = QTransform()
    
    def load_dxf(self, filepath):
        """Cargar y procesar un archivo DXF"""
        try:
            self.statusMessage.emit(f"Cargando archivo: {os.path.basename(filepath)}...")
            
            # Cargar el documento DXF
            self.dxf_doc = ezdxf.readfile(filepath)
            self.dxf_entities = []
            self.entity_counts = {}
            
            # Obtener el modelspace
            msp = self.dxf_doc.modelspace()
            
            # Inicializar coordenadas para límites
            x_coords = []
            y_coords = []
            
            # Obtener todos los tipos de entidades compatibles
            for entity in msp:
                entity_type = entity.dxftype()
                
                # Contar entidades por tipo
                if entity_type in self.entity_counts:
                    self.entity_counts[entity_type] += 1
                else:
                    self.entity_counts[entity_type] = 1
                
                # Almacenar entidad para dibujar
                self.dxf_entities.append(entity)
                
                # Extraer coordenadas para calcular límites
                if entity_type == 'LINE':
                    x_coords.extend([entity.dxf.start[0], entity.dxf.end[0]])
                    y_coords.extend([entity.dxf.start[1], entity.dxf.end[1]])
                
                elif entity_type == 'CIRCLE':
                    center = entity.dxf.center
                    radius = entity.dxf.radius
                    x_coords.extend([center[0] - radius, center[0] + radius])
                    y_coords.extend([center[1] - radius, center[1] + radius])
                
                elif entity_type == 'ARC':
                    center = entity.dxf.center
                    radius = entity.dxf.radius
                    x_coords.extend([center[0] - radius, center[0] + radius])
                    y_coords.extend([center[1] - radius, center[1] + radius])
                
                elif entity_type in ['LWPOLYLINE', 'POLYLINE']:
                    # Manejar ambas polilíneas de manera unificada
                    if hasattr(entity, 'get_points'):
                        # Para LWPOLYLINE
                        for point in entity.get_points():
                            x_coords.append(point[0])
                            y_coords.append(point[1])
                    elif hasattr(entity, 'vertices'):
                        # Para POLYLINE
                        for vertex in entity.vertices():
                            try:
                                # Para polilíneas 2D
                                x_coords.append(vertex[0])
                                y_coords.append(vertex[1])
                            except:
                                # Para polilíneas 3D
                                if hasattr(vertex, 'dxf'):
                                    x_coords.append(vertex.dxf.location[0])
                                    y_coords.append(vertex.dxf.location[1])
                
                elif entity_type == 'SPLINE':
                    # Usar puntos de control para límites aproximados
                    for point in entity.control_points:
                        x_coords.append(point[0])
                        y_coords.append(point[1])
                
                elif entity_type == 'ELLIPSE':
                    center = entity.dxf.center
                    major_axis = entity.dxf.major_axis
                    ratio = entity.dxf.ratio
                    # Aproximar límites de la elipse
                    major_radius = math.sqrt(major_axis[0]**2 + major_axis[1]**2)
                    minor_radius = major_radius * ratio
                    x_coords.extend([center[0] - major_radius, center[0] + major_radius])
                    y_coords.extend([center[1] - minor_radius, center[1] + minor_radius])
                
                elif entity_type == 'POINT':
                    point = entity.dxf.location
                    x_coords.append(point[0])
                    y_coords.append(point[1])
                
                elif entity_type == 'TEXT' or entity_type == 'MTEXT':
                    # Solo posición de inserción
                    if hasattr(entity.dxf, 'insert'):
                        point = entity.dxf.insert
                        x_coords.append(point[0])
                        y_coords.append(point[1])
                
                elif entity_type == 'SOLID' or entity_type == '3DFACE':
                    # Usar los vértices
                    for i in range(1, 5):
                        if hasattr(entity.dxf, f'vtx{i}'):
                            point = getattr(entity.dxf, f'vtx{i}')
                            x_coords.append(point[0])
                            y_coords.append(point[1])
                
                elif entity_type == 'HATCH':
                    # Usar el contorno externo
                    for path in entity.paths:
                        for vertex in path.vertices:
                            x_coords.append(vertex[0])
                            y_coords.append(vertex[1])
            
            # Calcular límites
            if x_coords and y_coords:
                min_x = min(x_coords)
                min_y = min(y_coords)
                max_x = max(x_coords)
                max_y = max(y_coords)
                self.bounds = (min_x, min_y, max_x, max_y)
            else:
                self.bounds = (-100, -100, 100, 100)
            
            # Mostrar estadísticas
            stats = ", ".join([f"{k}: {v}" for k, v in self.entity_counts.items()])
            self.statusMessage.emit(f"Archivo cargado: {os.path.basename(filepath)} - Total: {len(self.dxf_entities)} entidades - {stats}")
            
            # Resetear vista
            self.reset_view()
            
            # Actualizar widget
            self.update()
            return True
            
        except Exception as e:
            self.statusMessage.emit(f"Error al cargar archivo DXF: {str(e)}")
            return False
    
    def reset_view(self):
        """Resetear la vista para mostrar todo el contenido centrado y maximizado"""
        if not self.dxf_entities:
            return
        
        # Obtener tamaño de widget y límites de DXF
        widget_width = self.width()
        widget_height = self.height()
        
        # Obtener los límites reales del dibujo
        min_x, min_y, max_x, max_y = self.bounds
        dxf_width = max_x - min_x
        dxf_height = max_y - min_y
        
        # Calcular factor de zoom para ajustar a pantalla con margen
        if dxf_width > 0 and dxf_height > 0:
            zoom_x = (widget_width - 80) / dxf_width  # Más margen horizontal
            zoom_y = (widget_height - 80) / dxf_height  # Más margen vertical
            self.zoom_factor = min(zoom_x, zoom_y)  # Usar el factor más restrictivo
        else:
            self.zoom_factor = 1.0
        
        # Calcular el centro real del objeto basado en sus límites
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        # Calcular offset para centrar el objeto en la ventana
        # Nota: Para Y, necesitamos considerar que el eje Y está invertido en QPainter
        self.pan_offset_x = widget_width / 2 - center_x * self.zoom_factor
        self.pan_offset_y = widget_height / 2 + center_y * self.zoom_factor  # Signo cambiado para corregir centrado en Y
        
        # Actualizar transformación
        self.update_transform()
        
        # Actualizar widget
        self.update()
    
    def update_transform(self):
        """Actualizar la matriz de transformación para el renderizado"""
        # Crear una nueva transformación
        transform = QTransform()
        
        # Aplicar traslación para centrar el dibujo
        transform.translate(self.pan_offset_x, self.pan_offset_y)
        
        # Aplicar escala (Y invertido porque en QPainter el eje Y va hacia abajo)
        # Nota: Mantenemos el signo negativo en Y para que el dibujo no aparezca invertido verticalmente
        transform.scale(self.zoom_factor, -self.zoom_factor)
        
        # Guardar la transformación
        self.transform = transform
    
    def paintEvent(self, event):
        """Renderizar el contenido DXF"""
        # Crear painter
        painter = QPainter(self)
        
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            
            if not self.dxf_entities:
                # Si no hay entidades, solo dibujar fondo
                painter.fillRect(self.rect(), self.background_color)
                painter.drawText(self.rect(), Qt.AlignCenter, "Sin archivo DXF cargado")
                return
            
            # Dibujar fondo
            painter.fillRect(self.rect(), self.background_color)
            
            # Aplicar transformación
            painter.setTransform(self.transform)
            
            # Dibujar todas las entidades
            for entity in self.dxf_entities:
                self.draw_entity(painter, entity)
        finally:
            # Asegurar que el painter siempre se cierre correctamente
            painter.end()
    
    def draw_entity(self, painter, entity):
        """Dibujar una entidad DXF específica"""
        entity_type = entity.dxftype()
        
        # Configurar pluma basada en tipo de entidad
        color = self.entity_colors.get(entity_type, self.entity_colors['default'])
        pen = QPen(color)
        pen.setWidth(0)  # Ancho cosmético, se escala con el zoom
        painter.setPen(pen)
        
        # Dibujar según tipo
        if entity_type == 'LINE':
            start = entity.dxf.start
            end = entity.dxf.end
            painter.drawLine(QPointF(start[0], start[1]), QPointF(end[0], end[1]))
        
        elif entity_type == 'CIRCLE':
            center = entity.dxf.center
            radius = entity.dxf.radius
            painter.drawEllipse(QPointF(center[0], center[1]), radius, radius)
        
        elif entity_type == 'ARC':
            center = entity.dxf.center
            radius = entity.dxf.radius
            start_angle = entity.dxf.start_angle
            end_angle = entity.dxf.end_angle
            
            # Convertir ángulos de grados a radianes
            start_rad = math.radians(start_angle)
            end_rad = math.radians(end_angle)
            
            # Asegurar que el ángulo final es mayor que el inicial
            if end_rad < start_rad:
                end_rad += 2 * math.pi
            
            # Calcular puntos para el arco (más preciso que usar drawArc)
            points = []
            steps = max(20, int(radius * abs(end_rad - start_rad) * 0.5))  # Más puntos para arcos grandes
            
            for i in range(steps + 1):
                # Interpolar entre ángulo inicial y final
                angle = start_rad + (end_rad - start_rad) * i / steps
                x = center[0] + radius * math.cos(angle)
                y = center[1] + radius * math.sin(angle)
                points.append(QPointF(x, y))
            
            # Dibujar arco como segmentos de línea
            for i in range(len(points) - 1):
                painter.drawLine(points[i], points[i+1])
            
            # No más modo debug
        
        elif entity_type == 'LWPOLYLINE':
            points = []
            
            # Obtener todos los vértices
            for point in entity.get_points():
                points.append(QPointF(point[0], point[1]))
            
            # Dibujar las líneas conectadas
            if points:
                # Dibujar líneas conectadas
                for i in range(len(points) - 1):
                    painter.drawLine(points[i], points[i + 1])
                
                # Si es un polígono cerrado, conectar el último punto con el primero
                if entity.closed and len(points) > 1:
                    painter.drawLine(points[-1], points[0])
        
        elif entity_type == 'POLYLINE':
            points = []
            
            # Obtener todos los vértices dependiendo del tipo de polilínea
            if hasattr(entity, 'vertices'):
                for vertex in entity.vertices():
                    try:
                        # Para polilíneas 2D
                        points.append(QPointF(vertex[0], vertex[1]))
                    except:
                        # Para polilíneas 3D
                        if hasattr(vertex, 'dxf'):
                            loc = vertex.dxf.location
                            points.append(QPointF(loc[0], loc[1]))
            
            # Dibujar las líneas conectadas
            if points:
                for i in range(len(points) - 1):
                    painter.drawLine(points[i], points[i + 1])
                
                # Si es un polígono cerrado, conectar el último punto con el primero
                if hasattr(entity, 'closed') and entity.closed and len(points) > 1:
                    painter.drawLine(points[-1], points[0])
        
        elif entity_type == 'SPLINE':
            # Convertir spline a segmentos aproximados
            points = []
            steps = 100  # Ajustar según la precisión deseada
            
            if hasattr(entity, 'approximate'):
                try:
                    # Usar método de aproximación si está disponible
                    for point in entity.approximate(steps):
                        points.append(QPointF(point[0], point[1]))
                except:
                    # Fallback a puntos de control
                    for point in entity.control_points:
                        points.append(QPointF(point[0], point[1]))
            else:
                # Fallback a puntos de control
                for point in entity.control_points:
                    points.append(QPointF(point[0], point[1]))
            
            # Dibujar segmentos de línea
            if points:
                for i in range(len(points) - 1):
                    painter.drawLine(points[i], points[i + 1])
        
        elif entity_type == 'ELLIPSE':
            center = entity.dxf.center
            major_axis = entity.dxf.major_axis
            ratio = entity.dxf.ratio
            
            # Calcular ángulos
            start_param = entity.dxf.start_param
            end_param = entity.dxf.end_param
            
            # Convertir a puntos para dibujar
            points = []
            steps = 72  # Ajustar según la precisión deseada
            
            for i in range(steps + 1):
                # Interpolar parámetro
                param = start_param + (end_param - start_param) * i / steps
                # Calcular punto en la elipse
                # Convertir el generador a lista para poder acceder a sus elementos
                vertices = list(entity.vertices([param]))
                if vertices:
                    point = vertices[0]
                    points.append(QPointF(point[0], point[1]))
            
            # Dibujar como segmentos de línea
            for i in range(len(points) - 1):
                painter.drawLine(points[i], points[i + 1])
        
        elif entity_type == 'POINT':
            point = entity.dxf.location
            # Dibujar un punto como una cruz pequeña
            size = 2
            painter.drawLine(QPointF(point[0]-size, point[1]), QPointF(point[0]+size, point[1]))
            painter.drawLine(QPointF(point[0], point[1]-size), QPointF(point[0], point[1]+size))
        
        elif entity_type == 'SOLID' or entity_type == '3DFACE':
            # Dibujar como un polígono
            points = []
            for i in range(1, 5):
                if hasattr(entity.dxf, f'vtx{i}'):
                    point = getattr(entity.dxf, f'vtx{i}')
                    points.append(QPointF(point[0], point[1]))
            
            # Dibujar el polígono
            if len(points) >= 3:
                # Crear un path para el polígono
                path = QPainterPath()
                path.moveTo(points[0])
                for i in range(1, len(points)):
                    path.lineTo(points[i])
                path.closeSubpath()
                
                # Dibujar contorno y relleno
                painter.setBrush(QBrush(color.lighter(150), Qt.SolidPattern))
                painter.drawPath(path)
        
        elif entity_type == 'HATCH':
            # Dibujar contornos de sombreado
            for path in entity.paths:
                if path.vertices:
                    points = [QPointF(v[0], v[1]) for v in path.vertices]
                    
                    # Dibujar las líneas del contorno
                    for i in range(len(points) - 1):
                        painter.drawLine(points[i], points[i + 1])
                    
                    # Cerrar el contorno si es necesario
                    if path.is_closed and len(points) > 1:
                        painter.drawLine(points[-1], points[0])
    
    def wheelEvent(self, event):
        """Manejar eventos de rueda del mouse para zoom manteniendo el punto bajo el cursor"""
        # Obtener posición del mouse
        mouse_pos = event.position()
        
        # Calcular factor de zoom (10% de cambio)
        zoom_delta = 1.1
        
        # Determinar dirección del zoom
        if event.angleDelta().y() > 0:
            # Zoom in
            new_zoom = self.zoom_factor * zoom_delta
        else:
            # Zoom out
            new_zoom = self.zoom_factor / zoom_delta
        
        # Limitar el zoom para evitar valores extremos
        new_zoom = max(0.01, min(100, new_zoom))
        
        # Obtener coordenadas DXF bajo el cursor antes del zoom
        old_pos = self.transform.inverted()[0].map(QPointF(mouse_pos.x(), mouse_pos.y()))
        
        # Guardar el zoom anterior y aplicar el nuevo
        old_zoom = self.zoom_factor
        self.zoom_factor = new_zoom
        
        # Ajustar offset para mantener el punto bajo el cursor en la misma posición
        # Esto es crucial para que el zoom se centre en el punto donde está el cursor
        self.pan_offset_x = mouse_pos.x() - (old_pos.x() * new_zoom)
        self.pan_offset_y = mouse_pos.y() + (old_pos.y() * new_zoom)  # Signo cambiado para corregir centrado en Y
        
        # Actualizar transformación
        self.update_transform()
        
        # Actualizar widget
        self.update()
    
        
    def convert_to_opencv_image(self):
        """Convierte la visualización actual a una imagen compatible con OpenCV"""
        # Crear un QImage con el tamaño actual del widget
        width = self.width()
        height = self.height()
        image = QImage(width, height, QImage.Format_RGB32)
        
        # Llenar con el color de fondo
        image.fill(self.background_color)
        
        # Crear un QPainter para dibujar en la imagen
        painter = QPainter(image)
        
        # Activar antialiasing para mejor calidad
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Aplicar la misma transformación que se usa en paintEvent
        # para asegurar que la imagen OpenCV coincida con lo que se ve en pantalla
        painter.setTransform(self.transform)
        
        # Dibujar todas las entidades
        for entity in self.dxf_entities:
            self.draw_entity(painter, entity)
        
        # Finalizar el pintor
        painter.end()
        
        # Convertir QImage a formato OpenCV (numpy array)
        # OpenCV usa formato BGR, mientras que Qt usa RGB
        
        # Método compatible con versiones recientes de PySide6
        # donde image.bits() devuelve un objeto memoryview
        bytes_per_line = image.bytesPerLine()
        buffer = image.constBits().tobytes()
        arr = np.frombuffer(buffer, dtype=np.uint8).reshape((height, bytes_per_line // 4, 4))
        arr = arr[:, :width, :3]  # Recortar al ancho correcto y eliminar canal alfa
        
        # Convertir de RGB a BGR (formato OpenCV)
        bgr_image = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        
        return bgr_image

class DXFViewerApp(QMainWindow):
    """Aplicación principal para visualización de archivos DXF"""
    
    def __init__(self):
        super().__init__()
        
        # Configuración de la ventana
        self.setWindowTitle("Visualizador DXF Profesional")
        self.setMinimumSize(800, 600)
        
        # Crear componentes (simplificado)
        self.create_widgets()
        self.create_toolbars()
        self.create_statusbar()
        
        # Variables
        self.current_file = None
        self.last_directory = os.path.join(os.getcwd(), "Samples")  # Inicializar con la carpeta Samples
    
    def create_widgets(self):
        """Crear widgets de la interfaz"""
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Visor DXF (ocupa todo el espacio)
        self.dxf_viewer = DXFRenderer()
        self.dxf_viewer.statusMessage.connect(self.show_status_message)
        self.dxf_viewer.setMinimumHeight(500)  # Altura mínima para el visor
        main_layout.addWidget(self.dxf_viewer, 1)  # Proporción 1
    
    
    def create_toolbars(self):
        """Crear barras de herramientas simplificadas"""
        # Barra principal
        main_toolbar = QToolBar("Herramientas principales")
        main_toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(main_toolbar)
        
        # Botón abrir
        open_action = QAction("Abrir DXF", self)
        open_action.triggered.connect(self.open_file)
        main_toolbar.addAction(open_action)
        
        main_toolbar.addSeparator()
        
        # Botón reset view
        reset_view_action = QAction("Centrar Vista", self)
        reset_view_action.triggered.connect(self.dxf_viewer.reset_view)
        main_toolbar.addAction(reset_view_action)
        
        main_toolbar.addSeparator()
        
        # Botón para análisis morfológico
        morphology_action = QAction("Análisis Morfológico", self)
        morphology_action.triggered.connect(self.perform_morphology_analysis)
        main_toolbar.addAction(morphology_action)
    
    def create_statusbar(self):
        """Crear barra de estado simplificada"""
        self.statusBar().showMessage("Listo para cargar archivo DXF")
    
    def show_status_message(self, message):
        """Mostrar mensaje en la barra de estado"""
        self.statusBar().showMessage(message)
    
    def open_file(self):
        """Abrir un archivo DXF"""
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Abrir archivo DXF", self.last_directory, "Archivos DXF (*.dxf);;Todos los archivos (*)",
            options=options
        )
        
        if filepath:
            # Actualizar el último directorio utilizado
            self.last_directory = os.path.dirname(filepath)
            
            if self.dxf_viewer.load_dxf(filepath):
                self.current_file = filepath
                self.setWindowTitle(f"Visualizador DXF Profesional - {os.path.basename(filepath)}")
    
    
    def perform_morphology_analysis(self):
        """Realizar análisis morfológico utilizando OpenCV y mostrar resultados en una ventana con pestañas"""
        try:
            # Verificar si hay un archivo DXF cargado
            if not self.dxf_viewer.dxf_entities:
                QMessageBox.warning(self, "Advertencia", "No hay archivo DXF cargado para analizar.")
                return
            
            # Mostrar mensaje de estado
            self.statusBar().showMessage("Realizando análisis morfológico...")
            
            # Obtener la imagen en formato OpenCV
            cv_image = self.dxf_viewer.convert_to_opencv_image()
            
            # Verificar que la imagen no esté vacía
            if cv_image is None or cv_image.size == 0:
                QMessageBox.warning(self, "Error", "No se pudo convertir la imagen DXF a formato OpenCV.")
                return
            
            # Mostrar información sobre la imagen
            print(f"Imagen convertida: forma={cv_image.shape}, tipo={cv_image.dtype}")
            print(f"Valores: min={np.min(cv_image)}, max={np.max(cv_image)}, media={np.mean(cv_image):.2f}")
            
            # Realizar análisis morfológico con umbral automático
            best_result = dxf_morphology.dxf_to_morphology(cv_image)
            
            # Obtener métricas para mostrar en la barra de estado
            metrics = best_result.get('metrics', {})
            externos = metrics.get('contornos_externos', {}).get('cantidad', 0)
            internos = metrics.get('contornos_internos', {}).get('cantidad', 0)
            best_contours_count = externos + internos
            
            # Verificar que se obtuvieron resultados
            if not best_result or 'images' not in best_result:
                QMessageBox.warning(self, "Error", "El análisis morfológico no produjo resultados.")
                return
            
            # Crear y mostrar la ventana de resultados
            results_window = MorphologyResultsWindow(best_result, self)
            results_window.setWindowTitle("Resultados del Análisis Morfológico")
            results_window.resize(800, 600)
            results_window.show()
            
            # Mostrar mensaje de éxito
            self.statusBar().showMessage(f"Análisis morfológico completado. Se encontraron {best_contours_count} contornos.")
            
        except Exception as e:
            # Mostrar mensaje de error
            import traceback
            traceback.print_exc()
            error_msg = f"Error al realizar el análisis morfológico: {str(e)}"
            self.statusBar().showMessage(error_msg)
            QMessageBox.critical(self, "Error", error_msg)

class MorphologyResultsWindow(QMainWindow):
    """Ventana para mostrar los resultados del análisis morfológico en pestañas"""
    
    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.results = results
        self.setup_ui()
    
    def setup_ui(self):
        """Configurar la interfaz de usuario con pestañas para cada imagen y métricas"""
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QVBoxLayout(central_widget)
        
        # Crear widget de pestañas
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Obtener imágenes y métricas
        images = self.results.get('images', {})
        metrics = self.results.get('metrics', {})
        contours = self.results.get('contours', [])
        hierarchy = self.results.get('hierarchy', None)
        
        # Crear pestañas para cada imagen
        self.create_image_tab("Original", images.get('original'), "Imagen DXF original")
        self.create_image_tab("Binaria", images.get('binaria'), "Imagen en escala de grises")
        
        # Pestaña para contorno exterior
        self.create_exterior_contour_tab(
            "Contorno exterior",
            images.get('binaria'),
            contours,
            hierarchy
        )
        
        # Pestaña para agujeros
        self.create_holes_tab(
            "Agujeros",
            images.get('binaria'),
            contours,
            hierarchy
        )
        
        # Pestaña para agujeros hijos
        self.create_hole_children_tab(
            "Agujeros Hijos",
            images.get('binaria'),
            contours,
            hierarchy
        )
    
    def create_image_tab(self, title, image, description):
        """Crear una pestaña con una imagen y descripción lado a lado"""
        if image is None:
            return
        
        # Crear widget para la pestaña
        tab = QWidget()
        layout = QHBoxLayout(tab)  # Layout horizontal
        
        # Panel izquierdo para la imagen
        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)
        
        # Convertir imagen OpenCV a QImage
        if len(image.shape) == 3:  # Imagen a color
            height, width, channels = image.shape
            bytes_per_line = channels * width
            # Convertir de BGR a RGB para Qt
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            q_img = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:  # Imagen en escala de grises
            height, width = image.shape
            q_img = QImage(image.data, width, height, width, QImage.Format_Grayscale8)
        
        # Crear QLabel para mostrar la imagen
        image_label = QLabel()
        image_label.setPixmap(QPixmap.fromImage(q_img))
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setScaledContents(False)
        
        # Añadir imagen al layout
        image_layout.addWidget(image_label, 1)
        layout.addWidget(image_panel, 7)  # 70% del ancho
        
        # Panel derecho para la descripción
        desc_panel = QWidget()
        desc_layout = QVBoxLayout(desc_panel)
        
        # Añadir descripción
        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        desc_layout.addWidget(desc_label)
        desc_layout.addStretch(1)  # Espacio flexible al final
        
        layout.addWidget(desc_panel, 3)  # 30% del ancho
        
        # Añadir pestaña
        self.tab_widget.addTab(tab, title)
    
    def create_exterior_contour_tab(self, title, gray_image, contours, hierarchy):
        """Crear una pestaña que muestra solo el contorno exterior con su longitud"""
        if gray_image is None or contours is None or len(contours) == 0 or hierarchy is None:
            return
        
        # Crear widget para la pestaña
        tab = QWidget()
        layout = QHBoxLayout(tab)  # Layout horizontal
        
        # Panel izquierdo para la imagen
        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)
        
        # Crear una imagen en blanco para dibujar el contorno exterior
        h, w = gray_image.shape[:2] if len(gray_image.shape) > 2 else gray_image.shape
        exterior_img = np.ones((h, w), dtype=np.uint8) * 255  # Fondo blanco
        
        # Variables para almacenar información del contorno exterior
        exterior_contour = None
        exterior_perimeter = 0
        exterior_index = -1
        
        # Encontrar el contorno exterior (el más grande con h[3] == -1)
        max_area = 0
        for i, contour in enumerate(contours):
            # Verificar si es un contorno externo
            if hierarchy[0][i][3] == -1:  # Es un contorno externo
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    exterior_contour = contour
                    exterior_index = i
        
        # Si encontramos un contorno exterior, dibujarlo y calcular su perímetro
        if exterior_contour is not None:
            # Dibujar solo el contorno (no rellenarlo)
            cv2.drawContours(exterior_img, [exterior_contour], 0, 0, 2)  # Línea negra de grosor 2
            
            # Calcular perímetro
            exterior_perimeter = cv2.arcLength(exterior_contour, True)
        
        # Convertir a QImage
        q_img = QImage(exterior_img.data, w, h, w, QImage.Format_Grayscale8)
        
        # Crear QLabel para mostrar la imagen
        image_label = QLabel()
        image_label.setPixmap(QPixmap.fromImage(q_img))
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setScaledContents(False)
        
        # Añadir imagen al layout
        image_layout.addWidget(image_label, 1)
        layout.addWidget(image_panel, 7)  # 70% del ancho
        
        # Panel derecho para la información
        info_panel = QWidget()
        info_layout = QVBoxLayout(info_panel)
        
        # Añadir información del contorno exterior
        if exterior_contour is not None:
            info_text = f"""
            <h3>Contorno Exterior</h3>
            <p>Longitud del perímetro: <b>{exterior_perimeter:.2f} píxeles</b></p>
            <p>Área: {max_area:.2f} píxeles cuadrados</p>
            <p>Este es el contorno exterior principal del objeto, representado con una línea negra.</p>
            """
        else:
            info_text = """
            <h3>Contorno Exterior</h3>
            <p>No se encontró ningún contorno exterior.</p>
            """
        
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        info_label.setTextFormat(Qt.RichText)
        info_layout.addWidget(info_label)
        info_layout.addStretch(1)
        
        layout.addWidget(info_panel, 3)  # 30% del ancho
        
        # Añadir pestaña
        self.tab_widget.addTab(tab, title)
    
    def create_holes_tab(self, title, gray_image, contours, hierarchy):
        """Crear una pestaña que muestra los agujeros (contornos hijos directos del contorno exterior)"""
        if gray_image is None or contours is None or len(contours) == 0 or hierarchy is None:
            return
        
        # Crear widget para la pestaña
        tab = QWidget()
        layout = QHBoxLayout(tab)  # Layout horizontal
        
        # Panel izquierdo para la imagen
        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)
        
        # Crear una imagen en blanco para dibujar los agujeros
        h, w = gray_image.shape[:2] if len(gray_image.shape) > 2 else gray_image.shape
        holes_img = np.ones((h, w), dtype=np.uint8) * 255  # Fondo blanco
        
        # Encontrar el contorno exterior principal (el más grande con h[3] == -1)
        exterior_indices = []
        max_area = 0
        main_exterior_index = -1
        
        for i, contour in enumerate(contours):
            # Verificar si es un contorno externo
            if hierarchy[0][i][3] == -1:  # Es un contorno externo
                exterior_indices.append(i)
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    main_exterior_index = i
        
        # Encontrar los agujeros (hijos directos del contorno exterior principal)
        holes = []
        for i, contour in enumerate(contours):
            # Verificar si es hijo directo del contorno exterior principal
            if hierarchy[0][i][3] == main_exterior_index:
                holes.append(contour)
                # Dibujar el contorno del agujero
                cv2.drawContours(holes_img, [contour], 0, 0, 2)  # Línea negra de grosor 2
        
        # Convertir a QImage
        q_img = QImage(holes_img.data, w, h, w, QImage.Format_Grayscale8)
        
        # Crear QLabel para mostrar la imagen
        image_label = QLabel()
        image_label.setPixmap(QPixmap.fromImage(q_img))
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setScaledContents(False)
        
        # Añadir imagen al layout
        image_layout.addWidget(image_label, 1)
        layout.addWidget(image_panel, 7)  # 70% del ancho
        
        # Panel derecho para la información
        info_panel = QWidget()
        info_layout = QVBoxLayout(info_panel)
        
        # Añadir información de los agujeros
        info_text = f"""
        <h3>Agujeros</h3>
        <p>Cantidad de agujeros: <b>{len(holes)}</b></p>
        <p>Estos son los contornos internos (huecos) que son hijos directos del contorno exterior principal.</p>
        <p>Cada agujero está representado con una línea negra.</p>
        """
        
        if len(holes) > 0:
            # Añadir información sobre cada agujero
            info_text += "<p>Detalles de los agujeros:</p><ul>"
            for i, hole in enumerate(holes):
                area = cv2.contourArea(hole)
                perimeter = cv2.arcLength(hole, True)
                info_text += f"<li>Agujero {i+1}: Área = {area:.2f}, Perímetro = {perimeter:.2f}</li>"
            info_text += "</ul>"
        else:
            info_text += "<p>No se encontraron agujeros en la imagen.</p>"
        
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        info_label.setTextFormat(Qt.RichText)
        info_layout.addWidget(info_label)
        info_layout.addStretch(1)
        
        layout.addWidget(info_panel, 3)  # 30% del ancho
        
        # Añadir pestaña
        self.tab_widget.addTab(tab, title)
    
    def create_hole_children_tab(self, title, gray_image, contours, hierarchy):
        """Crear una pestaña que muestra los hijos de los agujeros (nivel 2 en la jerarquía)"""
        if gray_image is None or contours is None or len(contours) == 0 or hierarchy is None:
            return
        
        # Crear widget para la pestaña
        tab = QWidget()
        layout = QHBoxLayout(tab)  # Layout horizontal
        
        # Panel izquierdo para la imagen
        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)
        
        # Crear una imagen en blanco para dibujar los hijos de los agujeros
        h, w = gray_image.shape[:2] if len(gray_image.shape) > 2 else gray_image.shape
        hole_children_img = np.ones((h, w), dtype=np.uint8) * 255  # Fondo blanco
        
        # Encontrar el contorno exterior principal (el más grande con h[3] == -1)
        exterior_indices = []
        max_area = 0
        main_exterior_index = -1
        
        for i, contour in enumerate(contours):
            # Verificar si es un contorno externo
            if hierarchy[0][i][3] == -1:  # Es un contorno externo
                exterior_indices.append(i)
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    main_exterior_index = i
        
        # Encontrar los agujeros (hijos directos del contorno exterior principal)
        hole_indices = []
        for i, contour in enumerate(contours):
            # Verificar si es hijo directo del contorno exterior principal
            if hierarchy[0][i][3] == main_exterior_index:
                hole_indices.append(i)
        
        # Encontrar los hijos de los agujeros (nivel 2 en la jerarquía)
        hole_children = []
        hole_children_info = []
        
        for i, contour in enumerate(contours):
            # Verificar si es hijo de un agujero
            if hierarchy[0][i][3] in hole_indices:
                hole_children.append(contour)
                parent_idx = hierarchy[0][i][3]
                
                # Dibujar el contorno del hijo del agujero
                cv2.drawContours(hole_children_img, [contour], 0, 0, 2)  # Línea negra de grosor 2
                
                # Calcular métricas
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                # Guardar información
                hole_children_info.append({
                    'index': i,
                    'parent_index': parent_idx,
                    'area': area,
                    'perimeter': perimeter
                })
        
        # Convertir a QImage
        q_img = QImage(hole_children_img.data, w, h, w, QImage.Format_Grayscale8)
        
        # Crear QLabel para mostrar la imagen
        image_label = QLabel()
        image_label.setPixmap(QPixmap.fromImage(q_img))
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setScaledContents(False)
        
        # Añadir imagen al layout
        image_layout.addWidget(image_label, 1)
        layout.addWidget(image_panel, 7)  # 70% del ancho
        
        # Panel derecho para la información
        info_panel = QWidget()
        info_layout = QVBoxLayout(info_panel)
        
        # Añadir información de los hijos de los agujeros
        info_text = f"""
        <h3>Agujeros Hijos</h3>
        <p>Cantidad de hijos de agujeros: <b>{len(hole_children)}</b></p>
        <p>Estos son los contornos que están dentro de los agujeros (nivel 2 en la jerarquía).</p>
        <p>Cada hijo de agujero está representado con una línea negra.</p>
        """
        
        if len(hole_children) > 0:
            # Añadir información sobre cada hijo de agujero
            info_text += "<p>Detalles de los hijos de agujeros:</p><ul>"
            for i, info in enumerate(hole_children_info):
                parent_idx = info['parent_index']
                area = info['area']
                perimeter = info['perimeter']
                info_text += f"<li>Hijo {i+1}: Área = {area:.2f}, Perímetro = {perimeter:.2f}, Padre = Agujero {hole_indices.index(parent_idx)+1}</li>"
            info_text += "</ul>"
        else:
            info_text += "<p>No se encontraron hijos de agujeros en la imagen.</p>"
        
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        info_label.setTextFormat(Qt.RichText)
        info_layout.addWidget(info_label)
        info_layout.addStretch(1)
        
        layout.addWidget(info_panel, 3)  # 30% del ancho
        
        # Añadir pestaña
        self.tab_widget.addTab(tab, title)

def main():
    app = QApplication(sys.argv)
    
    # Establecer estilo de la aplicación
    app.setStyle('Fusion')
    
    # Crear y mostrar la ventana principal
    window = DXFViewerApp()
    window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()