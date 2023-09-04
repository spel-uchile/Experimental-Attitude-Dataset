
import cv2
import os
import errno


def save_frame(folder, video_file):
    # Crear un objeto de captura de video
    cap = cv2.VideoCapture(folder + video_file)

    try:
        os.mkdir(folder + video_file.split('.')[0])
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Inicializar un contador de frames
    frame_count = 0

    # Bucle para leer los frames del video
    while cap.isOpened():
        # Leer el siguiente frame
        ret, frame = cap.read()

        # Verificar si se ha llegado al final del video
        if not ret:
            break

        # Incrementar el contador de frames
        frame_count += 1

        # Guardar el frame como una imagen
        frame_file = f"frame{frame_count}.png"
        cv2.imwrite(folder + video_file.split('.')[0] + '/' + frame_file, frame)

    # Liberar los recursos
    cap.release()