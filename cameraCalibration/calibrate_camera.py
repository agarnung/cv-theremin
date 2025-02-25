# cameraCalibration/calibrate_camera.py

import cv2 as cv
import numpy as np
import sys
import os

# Agregar el directorio raíz al sys.path para poder importar módulos
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from modules.CameraModule import Camera

def main():
    # Configuración del patrón de calibración
    pattern_size = (9, 6)  # Número de esquinas internas (columnas, filas) del tablero de ajedrez
    square_size = 30       # Tamaño real de cada cuadrado (por ejemplo, 30 mm)
    
    # Criterios de terminación para refinar la posición de las esquinas
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Preparar los puntos 3D del objeto:
    # Por ejemplo, (0,0,0), (square_size, 0, 0), ..., ((pattern_size[0]-1)*square_size, (pattern_size[1]-1)*square_size, 0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size  # Escalar según el tamaño real del cuadrado
    
    # Listas para almacenar los puntos 3D y 2D de todas las imágenes válidas
    objpoints = []  # Puntos en el mundo real
    imgpoints = []  # Puntos en la imagen

    # Inicializar la cámara usando el módulo Camera
    camera = Camera(2)  # Cambiar el índice según tu dispositivo
    
    captured_count = 0
    minimum_required = 10  # Se requiere al menos 10 imágenes válidas para calibrar

    print("Iniciando calibración de la cámara.")
    print("Alinea el tablero de ajedrez en el campo de visión.")
    print("Presiona 'c' para capturar una imagen cuando el patrón sea detectado.")
    print("Presiona 'q' para finalizar la captura (mínimo requerido:", minimum_required, "imágenes).")
    
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error al leer la imagen de la cámara.")
            break
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret_corners, corners = cv.findChessboardCorners(gray, pattern_size, None)
        
        if ret_corners:
            # Refinar la detección de esquinas
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # Dibujar el patrón detectado en la imagen
            cv.drawChessboardCorners(frame, pattern_size, corners2, ret_corners)
            cv.putText(frame, "Patron detectado. Presione 'c' para capturar", (30, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv.putText(frame, "Patron no detectado", (30, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Mostrar la imagen con la información
        cv.putText(frame, f"Capturadas: {captured_count}", (30, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv.imshow("Calibracion", frame)
        
        # Leer la tecla presionada
        key = cv.waitKey(1) & 0xFF
        if key == ord('c') and ret_corners:
            # Si se presiona 'c' y se detectó el patrón, guardar la imagen
            objpoints.append(objp)
            imgpoints.append(corners2)
            captured_count += 1
            print(f"Capturada imagen {captured_count}.")
        elif key == ord('q'):
            if captured_count < minimum_required:
                print(f"Aún no se capturaron suficientes imágenes de calibración (se requieren al menos {minimum_required}).")
            else:
                # Finalizar el bucle si se presiona 'q'
                break

    # Liberar la cámara y cierra las ventanas de OpenCV
    camera.release()
    cv.destroyAllWindows()
    
    # Realizar la calibración utilizando los puntos obtenidos
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("\nCalibracion completada.")
    print("Matriz de la cámara (intrinsecos):\n", mtx)
    print("Coeficientes de distorsion:\n", dist)
    
    # Calcular el error de re-proyección para evaluar la precisión
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    mean_error /= len(objpoints)
    print("Error de re-proyeccion: ", mean_error)
    
    # Definir la ruta absoluta para guardar los resultados en 'cameraCalibration/results/'
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Guardar la matriz de la cámara y los coeficientes de distorsión en archivos .npy dentro de 'results'
    mtx_path = os.path.join(results_dir, "camera_matrix.npy")
    dist_path = os.path.join(results_dir, "dist_coeffs.npy")
    
    np.save(mtx_path, mtx)
    np.save(dist_path, dist)
    print(f"Parámetros guardados en '{mtx_path}' y '{dist_path}'.")
    print("Calibración completada.")

if __name__ == "__main__":
    main()
