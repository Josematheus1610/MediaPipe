#O MediaPipe Face Mesh fornece uma malha facial com 468 pontos de referência (landmarks).
#Cada ponto tem um identificador único e uma posição normalizada dentro do rosto detectado.
#Para visualizar e entender a posição é possivel consultar o diagrama das landmarks fornecido pela MediaPipe.
import time
import cv2
import pyautogui
import mediapipe as mp
import math

# Inicializando os módulos do MediaPipe para detecção de face
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Função para calcular a distância euclidiana entre dois pontos
def calcular_distancia(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

# Inicializando a webcam
cap = cv2.VideoCapture(0)

# Configurando o detector de pontos faciais do MediaPipe
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar a imagem da webcam.")
            break

        #invertendo a imagem na tela
        frame = cv2.flip(frame, 1)
        # Convertendo a imagem para RGB, pois o MediaPipe utiliza esse formato
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Processando a imagem e detectando os pontos faciais
        results = face_mesh.process(rgb_frame)

        # Verificando se há alguma face detectada
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Desenhando os pontos faciais na imagem
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

                # Obtendo as coordenadas dos olhos
                h, w, _ = frame.shape  # Altura e largura do frame

                # Ponto superior e inferior do olho esquerdo
                olho_esquerdo_superior = face_landmarks.landmark[160]  # Ponto superior
                olho_esquerdo_inferior = face_landmarks.landmark[24]  # Ponto inferior

                # Convertendo coordenadas normalizadas para pixels
                olho_esq_sup = (int(olho_esquerdo_superior.x * w), int(olho_esquerdo_superior.y * h))
                olho_esq_inf = (int(olho_esquerdo_inferior.x * w), int(olho_esquerdo_inferior.y * h))

                # Desenhando pontos no olho
                cv2.circle(frame, olho_esq_sup, 3, (255, 0, 0), -1)
                cv2.circle(frame, olho_esq_inf, 3, (0, 255, 0), -1)

                # Calculando a distância entre o ponto superior e inferior
                distancia_olho_esquerdo = calcular_distancia(olho_esq_sup, olho_esq_inf)

                # Exibindo a distância no frame
                cv2.putText(frame, f'Distancia: {distancia_olho_esquerdo:.2f}', (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Definindo um limiar para identificar se o olho está fechado
                if distancia_olho_esquerdo < 12:

                    # Você pode ajustar esse valor conforme necessário
                    cv2.putText(frame, 'Olho Esquerdo Fechado', (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


                    #chama 1 segundo de bloqueio após apertar a tecla windows


                else:
                    cv2.putText(frame, 'Olho Esquerdo Aberto', (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Exibindo o resultado com os pontos faciais
        cv2.imshow('Face Mesh', frame)

        # Encerrando a execução ao pressionar a tecla 'q'
        if cv2.waitKey(30) & 0xFF == 27:
            break

# Liberando a captura da webcam e fechando as janelas
cap.release()
cv2.destroyAllWindows()
