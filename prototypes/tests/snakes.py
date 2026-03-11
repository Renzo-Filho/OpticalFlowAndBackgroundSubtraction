import cv2
import numpy as np
import time

try:
    from skimage.segmentation import active_contour
    from skimage.color import rgb2gray
    from skimage.filters import gaussian
except ImportError:
    print("ERRO: Por favor, instale o pacote via terminal: pip install scikit-image")
    exit()

def main():
    cap = cv2.VideoCapture(4)
    bg_base = None
    scale = 0.5 # Trabalhar em meia resolução é vital para as matemáticas da Snake em tempo real

    print("Pressione 'b' para capturar o fundo.")
    print("Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        
        # Reduzir a imagem para processamento rápido
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        h, w = small_frame.shape[:2]

        if bg_base is None:
            cv2.putText(frame, "Aguardando Captura (b)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Snakes - Contornos Ativos", frame)
        else:
            # =======================================================
            # 1. GERAÇÃO DA MÁSCARA BRUTA (O seu método atual)
            # =======================================================
            f_ycc = cv2.cvtColor(small_frame, cv2.COLOR_BGR2YCrCb)
            b_ycc = cv2.cvtColor(bg_base, cv2.COLOR_BGR2YCrCb)
            diff = cv2.absdiff(f_ycc, b_ycc)
            
            score = 0.3 * diff[..., 0] + 1.2 * diff[..., 1] + 1.2 * diff[..., 2]
            score_u8 = np.clip(score, 0, 255).astype(np.uint8)
            _, mask = cv2.threshold(score_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Limpeza básica morfológica
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            # =======================================================
            # 2. EXTRAÇÃO DO CONTORNO INICIAL
            # =======================================================
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            canvas = small_frame.copy()
            
            if contours:
                # Pegar apenas o maior contorno (a pessoa)
                c = max(contours, key=cv2.contourArea)
                
                # O scikit-image precisa de um array (N, 2) no formato (y, x) em vez de (x, y)
                # Vamos reamostrar o contorno para ter um número fixo de pontos espaçados
                # para que a "física" do elástico funcione bem.
                c = c.reshape(-1, 2)
                init_snake = np.zeros_like(c)
                init_snake[:, 0] = c[:, 1] # y
                init_snake[:, 1] = c[:, 0] # x
                
                # Desenhar o contorno bruto a VERMELHO
                cv2.polylines(canvas, [c], True, (0, 0, 255), 2)

                # =======================================================
                # 3. APLICAÇÃO DA SNAKE (Otimização de Energia)
                # =======================================================
                t0 = time.time()
                
                # A imagem base para a Snake precisa ser suavizada para que os gradientes atraiam o elástico
                img_gray = rgb2gray(small_frame)
                img_smooth = gaussian(img_gray, sigma=2.0)
                
                # Parâmetros de Szeliski:
                # alpha (Elasticidade): Tenta encolher a curva. Valores altos = elástico muito tenso.
                # beta (Rigidez): Tenta manter a curva sem quinas. Valores altos = curva como arame.
                # w_edge (Força Externa): Atração para as bordas da imagem.
                snake = active_contour(
                    img_smooth, 
                    init_snake, 
                    alpha=0.015,  
                    beta=10.0,    
                    w_line=0,     
                    w_edge=1.0,   
                    gamma=0.001,  
                    max_num_iter=10 # Mantemos iterações baixas para ser em tempo real
                )
                
                tempo_snake = (time.time() - t0) * 1000

                # =======================================================
                # 4. VISUALIZAÇÃO
                # =======================================================
                # Converter de volta para (x, y) do OpenCV
                snake_cv = np.zeros_like(snake)
                snake_cv[:, 0] = snake[:, 1] # x
                snake_cv[:, 1] = snake[:, 0] # y
                snake_cv = snake_cv.astype(np.int32)
                
                # Desenhar a Snake otimizada a VERDE
                cv2.polylines(canvas, [snake_cv], True, (0, 255, 0), 2)
                
                cv2.putText(canvas, f"Otsu (Vermelho) -> Snake (Verde): {tempo_snake:.1f}ms", 
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Aumentar para o tamanho original para exibir
            out_full = cv2.resize(canvas, (w * int(1/scale), h * int(1/scale)))
            cv2.imshow("Snakes - Contornos Ativos", out_full)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('b'):
            time.sleep(1)
            ret, bg_frame = cap.read()
            bg_frame = cv2.flip(bg_frame, 1)
            bg_base = cv2.resize(bg_frame, None, fx=scale, fy=scale)
            print("Background Capturado!")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()