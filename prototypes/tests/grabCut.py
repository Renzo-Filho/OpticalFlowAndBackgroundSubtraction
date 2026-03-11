import cv2
import numpy as np
import time

def main():
    cap = cv2.VideoCapture(4)
    bg_base = None
    
    # Modelos de memória exigidos pelo GrabCut (Matrizes 1x65 de float)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    print("Pressione 'b' para capturar o fundo.")
    print("Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Reduz a resolução para o GrabCut rodar mais rápido (Muito importante!)
        scale = 0.5
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale)

        if bg_base is None:
            cv2.putText(frame, "Aguardando Captura (b)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Comparacao de Segmentacao", frame)
        else:
            # 1. PREPARAÇÃO (Igual ao seu código atual)
            f_ycc = cv2.cvtColor(small_frame, cv2.COLOR_BGR2YCrCb)
            b_ycc = cv2.cvtColor(bg_base, cv2.COLOR_BGR2YCrCb)
            diff = cv2.absdiff(f_ycc, b_ycc)
            
            # Score de movimento focado nas cores
            score = 0.3 * diff[..., 0] + 1.2 * diff[..., 1] + 1.2 * diff[..., 2]
            score_u8 = np.clip(score, 0, 255).astype(np.uint8)

            # =======================================================
            # MÉTODO A: O SEU ATUAL (Otsu + Morfologia)
            # =======================================================
            t0_otsu = time.time()
            _, mask_otsu = cv2.threshold(score_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
            mask_otsu = cv2.morphologyEx(mask_otsu, cv2.MORPH_CLOSE, kernel, iterations=1)
            tempo_otsu = (time.time() - t0_otsu) * 1000

            # =======================================================
            # MÉTODO B: NOVO GRABCUT (Baseado em Graph Cuts)
            # =======================================================
            t0_grab = time.time()
            
            gc_mask = np.zeros(small_frame.shape[:2], np.uint8)
            
            # Mapeando o seu 'score' para a Energia do Grafo:
            gc_mask[:] = cv2.GC_PR_BGD             
            gc_mask[score_u8 > 20] = cv2.GC_PR_FGD 
            gc_mask[score_u8 > 80] = cv2.GC_FGD    
            gc_mask[score_u8 < 5] = cv2.GC_BGD     

            # Verificação de segurança: checar se há amostras de fundo E de frente
            has_bg = np.any((gc_mask == cv2.GC_BGD) | (gc_mask == cv2.GC_PR_BGD))
            has_fg = np.any((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD))

            if has_bg and has_fg:
                # Executa a otimização apenas se tivermos dados suficientes
                cv2.grabCut(small_frame, gc_mask, None, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)
                mask_grabcut = np.where((gc_mask == 1) | (gc_mask == 3), 255, 0).astype('uint8')
            else:
                # Se não há movimento/pessoa, a máscara é toda preta (fundo)
                mask_grabcut = np.zeros(small_frame.shape[:2], np.uint8)
                
            tempo_grab = (time.time() - t0_grab) * 1000

            # =======================================================
            # VISUALIZAÇÃO LADO A LADO
            # =======================================================
            # Retorna as máscaras para o tamanho original
            mask_otsu_full = cv2.resize(mask_otsu, (w, h))
            mask_grab_full = cv2.resize(mask_grabcut, (w, h))

            # Converte para BGR para colar textos coloridos
            vis_otsu = cv2.cvtColor(mask_otsu_full, cv2.COLOR_GRAY2BGR)
            vis_grab = cv2.cvtColor(mask_grab_full, cv2.COLOR_GRAY2BGR)

            cv2.putText(vis_otsu, f"Metodo Otsu: {tempo_otsu:.1f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(vis_grab, f"Metodo GrabCut: {tempo_grab:.1f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Cola as duas imagens lado a lado
            combined = np.hstack((vis_otsu, vis_grab))
            cv2.imshow("Comparacao de Segmentacao", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('b'):
            # Captura a sala vazia em resolução reduzida
            time.sleep(1)
            ret, bg_frame = cap.read()
            bg_frame = cv2.flip(bg_frame, 1)
            bg_base = cv2.resize(bg_frame, None, fx=scale, fy=scale)
            print("Background Capturado!")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()