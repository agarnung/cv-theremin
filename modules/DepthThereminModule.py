# modules/DepthThereminModule.py

import numpy as np
import os

class DepthTheremin:
    def __init__(self, 
                 camera_matrix_path=None, 
                 dist_coeffs_path=None, 
                 hand_real_width=18.0,  # ancho real de la mano en cm (valor aproximado)
                 d_min=30, d_max=250,    # distancias mínimas y máximas esperadas en cm
                 min_frequency=200, max_frequency=600):
        if camera_matrix_path is None or dist_coeffs_path is None:
            # Obtener el directorio raíz del proyecto
            # __file__ es la ruta de este archivo (DepthThereminModule.py)
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # Construir la ruta hacia la carpeta cameraCalibration/results
            camera_matrix_path = os.path.join(base_dir, "cameraCalibration", "results", "camera_matrix.npy")
            dist_coeffs_path = os.path.join(base_dir, "cameraCalibration", "results", "dist_coeffs.npy")
        # Cargar parámetros de calibración
        self.camera_matrix = np.load(camera_matrix_path)
        self.dist_coeffs = np.load(dist_coeffs_path)
        # Se asume que la distancia focal en x es representativa (en píxeles)
        self.focal_length = self.camera_matrix[0, 0]
        self.hand_real_width = hand_real_width  # valor real en cm (puede ajustarse según la mano)
        self.d_min = d_min
        self.d_max = d_max
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency

    def compute_depth(self, hand_bbox):
        """
        Estima la profundidad (distancia D) usando el ancho de la caja delimitadora de la mano.
        
        Parámetro:
            hand_bbox: tupla (x, y, w, h) proveniente del detector de manos.
            
        Retorna:
            depth: distancia estimada en cm.
        """
        w_image = hand_bbox[2]  # ancho en píxeles
        if w_image == 0:
            return self.d_max  # valor por defecto en caso de error
        # Aplicar el modelo pinhole: D = (f * W_real) / W_imagen
        depth = (self.focal_length * self.hand_real_width) / w_image
        return depth

    def depth_to_frequency(self, depth):
        """
        Mapea la profundidad estimada a una frecuencia.
        Se asume que una mano más cercana (menor D) produce una frecuencia mayor.
        """
        # Asegurar que la profundidad esté dentro de los límites
        depth = np.clip(depth, self.d_min, self.d_max)
        # Relación lineal inversa:
        freq = self.min_frequency + (self.max_frequency - self.min_frequency) * ((self.d_max - depth) / (self.d_max - self.d_min))
        return freq

    def compute_tone_depth(self, hand_bbox):
        """
        Combina los métodos anteriores para obtener la frecuencia y la profundidad.
        
        Retorna:
            frequency: frecuencia calculada.
            depth: profundidad estimada en cm.
        """
        depth = self.compute_depth(hand_bbox)
        frequency = self.depth_to_frequency(depth)
        return frequency, depth
