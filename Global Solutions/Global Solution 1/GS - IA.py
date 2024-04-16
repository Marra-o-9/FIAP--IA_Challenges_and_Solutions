#Henrique Marra Barbosa - 97672
#Arthur Hieda Cunha - 551882
#Lucas Bueno Taets Gustavo - 552162

# Bibliotecas utilizadas
import cv2
import cvlib as cv
import numpy as np
from urllib.request import urlopen
from cvlib.object_detection import draw_bbox

# URL da transmissão ao vivo
url = 'http://187.3.137.133/cam-hi.jpg'

# Classes dos objetos a serem detectados
classes = ['perecivel', 'nao_perecivel']

def process_frame(frame):
    # Detecta objetos comuns na imagem usando o modelo YOLOv3-tiny
    bbox, label, conf = cv.detect_common_objects(frame, confidence=0.6, model='yolo')
    
    # Desenha caixas delimitadoras e rótulos nos objetos detectados
    output = draw_bbox(frame, bbox, label, conf, classes=classes)
    
    return output

def main():
    # Nomeia a janela que será inserida a câmera
    cv2.namedWindow("Deteccao de Alimentos", cv2.WINDOW_AUTOSIZE)
    
    while True:
        # Obtém o quadro de imagem da transmissão ao vivo
        img_resp = urlopen(url)
        img_array = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_array, -1)
        
        # Processa o quadro de imagem para detecção de objetos
        output = process_frame(frame)
        
        # Exibe o quadro de imagem com as detecções
        cv2.imshow("Detecção de Alimentos", output)
        key = cv2.waitKey(1)
        
        # Encerra o programa ao pressionar a tecla 'q'
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
