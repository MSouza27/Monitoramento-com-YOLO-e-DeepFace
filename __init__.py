import cv2
from ultralytics import YOLO
from deepface import DeepFace
import math
import time

# Carregar o modelo YOLOv8 pré-treinado
model = YOLO('yolov8n.pt')

# Inicializar a câmera ou vídeo
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Erro ao abrir a câmera ou o vídeo.")
    exit()

# Parâmetros
velocidade_limite = 80  # em km/h
linha_sinal_vermelho = 400  # Posição vertical no frame representando o sinal
rastreamento_veiculos = {}
proximo_id = 1  # ID único incremental para veículos
estatisticas_emocoes = {"feliz": 0, "triste": 0, "neutro": 0, "irritado": 0, "surpreso": 0, "medo": 0, "nojo": 0}
idades_detectadas = []  # Lista para armazenar idades detectadas
emocoes_pt = {
    "happy": "feliz",
    "sad": "triste",
    "neutral": "neutro",
    "angry": "irritado",
    "surprise": "surpreso",
    "fear": "medo",
    "disgust": "nojo"
}

# Constantes para cálculo de distância
FOCAL_LENGTH = 800
KNOWN_WIDTH = 2.0  # Largura média de um carro em metros

# Função para calcular a distância
def calcular_distancia(largura_pixel):
    if largura_pixel == 0:
        return None
    return (KNOWN_WIDTH * FOCAL_LENGTH) / largura_pixel

# Função para calcular a velocidade
def calcular_velocidade(distancia_anterior, distancia_atual, tempo_decorrido):
    if tempo_decorrido > 0:
        return abs(distancia_atual - distancia_anterior) / tempo_decorrido * 3.6  # Converter para km/h
    return 0

# Função para rastrear objetos entre frames
def rastrear_objetos(rastreamento_veiculos, detections, proximo_id, tempo_atual):
    novos_rastreamentos = {}
    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        centro_x = (x1 + x2) // 2
        centro_y = (y1 + y2) // 2
        largura = x2 - x1
        encontrado = False

        # Verificar se há um objeto próximo para associar
        for id, dados in rastreamento_veiculos.items():
            posicao_anterior = dados["posicao"]
            if math.hypot(centro_x - posicao_anterior[0], centro_y - posicao_anterior[1]) < 50:
                distancia = calcular_distancia(largura)
                tempo_decorrido = tempo_atual - dados["tempo"]
                velocidade = calcular_velocidade(dados["distancia"], distancia, tempo_decorrido)

                novos_rastreamentos[id] = {
                    "posicao": (centro_x, centro_y),
                    "box": (x1, y1, x2, y2),
                    "label": dados["label"],
                    "distancia": distancia,
                    "velocidade": velocidade,
                    "tempo": tempo_atual
                }
                encontrado = True
                break

        # Se não foi encontrado, atribuir um novo ID
        if not encontrado:
            novos_rastreamentos[proximo_id] = {
                "posicao": (centro_x, centro_y),
                "box": (x1, y1, x2, y2),
                "label": model.names[int(box.cls[0])],
                "distancia": calcular_distancia(largura),
                "velocidade": 0,
                "tempo": tempo_atual
            }
            proximo_id += 1

    return novos_rastreamentos, proximo_id

# Loop principal
frame_count = 0
processar_cada_n_frames = 5  # Processar apenas 1 a cada N frames

while True:
    ret, frame = camera.read()
    if not ret:
        print("Erro ao capturar o vídeo.")
        break

    frame_count += 1
    if frame_count % processar_cada_n_frames != 0:
        continue  # Ignorar frame para acelerar o processamento

    tempo_atual = time.time()

    # Reduzir a resolução do frame
    frame = cv2.resize(frame, (640, 360))

    # Realizar a detecção de objetos
    results = model.predict(source=frame, show=False, conf=0.5)

    # Atualizar rastreamento de veículos
    rastreamento_veiculos, proximo_id = rastrear_objetos(rastreamento_veiculos, results[0].boxes, proximo_id, tempo_atual)

    # Exibir informações de cada veículo rastreado
    for id, dados in rastreamento_veiculos.items(): 
        x1, y1, x2, y2 = dados["box"]
        distancia = dados["distancia"]
        velocidade = dados["velocidade"]

        # Desenhar caixa delimitadora e exibir ID, distância e velocidade
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {id}", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Distancia: {distancia:.2f}m", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Velocidade: {velocidade:.2f}km/h", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Detectar rostos e emoções
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector_faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    rostos = detector_faces.detectMultiScale(cinza, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in rostos:
        rosto = frame[y:y+h, x:x+w]

        try:
            analise = DeepFace.analyze(rosto, actions=["emotion", "age"], enforce_detection=False)
            emocao_en = analise[0]["dominant_emotion"]
            emocao_pt = emocoes_pt.get(emocao_en, "indefinido")
            idade = int(analise[0]["age"])

            estatisticas_emocoes[emocao_pt] += 1
            idades_detectadas.append(idade)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
            cv2.putText(frame, f'Emocao: {emocao_pt}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            cv2.putText(frame, f'Idade: {idade}', (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        except Exception as e:
            print(f"Erro ao analisar emoção/idade: {e}")

    # Mostrar o frame com as detecções
    cv2.imshow("Monitoramento", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()
