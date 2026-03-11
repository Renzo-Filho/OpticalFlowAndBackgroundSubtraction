import cv2
import numpy as np
import time
import mediapipe as mp

def main():
    cap = cv2.VideoCapture(4)
    bg_base = None
    
    # =======================================================
    # INICIALIZAÇÃO DOS MODELOS
    # =======================================================
    # 1. GrabCut: Modelos de memória (Matrizes 1x65 de float)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # 2. OpenCV CNT (Requer opencv-contrib-python)
    bgs_algo = cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=15, useHistory=True)

    # 3. MediaPipe Selfie Segmentation
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    segmenter = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    print("Pressione 'b' para capturar o fundo (necessário para Otsu e GrabCut).")
    print("Pressione 'q' para sair.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Reduz a resolução para os algoritmos rodarem mais rápido
        scale = 0.5
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale)

        # O CNT aprende continuamente, então podemos alimentá-lo mesmo antes de capturar o fundo
        t0_cnt = time.time()
        mask_cnt = bgs_algo.apply(small_frame)
        mask_cnt = cv2.morphologyEx(mask_cnt, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        tempo_cnt = (time.time() - t0_cnt) * 1000

        # O MediaPipe não precisa de fundo, processa o frame atual diretamente
        t0_mp = time.time()
        frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        results = segmenter.process(frame_rgb)
        # Binariza a máscara do MediaPipe (0 a 255)
        mask_mp = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
        tempo_mp = (time.time() - t0_mp) * 1000

        if bg_base is None:
            cv2.putText(frame, "Aguardando Captura (b) para Otsu/GrabCut", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Comparacao de Segmentacao", frame)
        else:
            # =======================================================
            # PREPARAÇÃO OTSU / GRABCUT
            # =======================================================
            f_ycc = cv2.cvtColor(small_frame, cv2.COLOR_BGR2YCrCb)
            b_ycc = cv2.cvtColor(bg_base, cv2.COLOR_BGR2YCrCb)
            diff = cv2.absdiff(f_ycc, b_ycc)
            
            score = 0.3 * diff[..., 0] + 1.2 * diff[..., 1] + 1.2 * diff[..., 2]
            score_u8 = np.clip(score, 0, 255).astype(np.uint8)

            # =======================================================
            # MÉTODO A: OTSU + Morfologia
            # =======================================================
            t0_otsu = time.time()
            _, mask_otsu = cv2.threshold(score_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
            mask_otsu = cv2.morphologyEx(mask_otsu, cv2.MORPH_CLOSE, kernel, iterations=1)
            tempo_otsu = (time.time() - t0_otsu) * 1000

            # =======================================================
            # MÉTODO B: GRABCUT
            # =======================================================
            t0_grab = time.time()
            gc_mask = np.zeros(small_frame.shape[:2], np.uint8)
            
            gc_mask[:] = cv2.GC_PR_BGD             
            gc_mask[score_u8 > 20] = cv2.GC_PR_FGD 
            gc_mask[score_u8 > 80] = cv2.GC_FGD    
            gc_mask[score_u8 < 5] = cv2.GC_BGD     

            has_bg = np.any((gc_mask == cv2.GC_BGD) | (gc_mask == cv2.GC_PR_BGD))
            has_fg = np.any((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD))

            if has_bg and has_fg:
                cv2.grabCut(small_frame, gc_mask, None, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)
                mask_grabcut = np.where((gc_mask == 1) | (gc_mask == 3), 255, 0).astype('uint8')
            else:
                mask_grabcut = np.zeros(small_frame.shape[:2], np.uint8)
                
            tempo_grab = (time.time() - t0_grab) * 1000

            # =======================================================
            # VISUALIZAÇÃO EM GRADE 2x2
            # =======================================================
            # Retorna todas as máscaras para o tamanho original
            mask_otsu_full = cv2.resize(mask_otsu, (w, h))
            mask_grab_full = cv2.resize(mask_grabcut, (w, h))
            mask_cnt_full = cv2.resize(mask_cnt, (w, h))
            mask_mp_full = cv2.resize(mask_mp, (w, h))

            # Converte para BGR para colar textos coloridos
            vis_otsu = cv2.cvtColor(mask_otsu_full, cv2.COLOR_GRAY2BGR)
            vis_grab = cv2.cvtColor(mask_grab_full, cv2.COLOR_GRAY2BGR)
            vis_cnt = cv2.cvtColor(mask_cnt_full, cv2.COLOR_GRAY2BGR)
            vis_mp = cv2.cvtColor(mask_mp_full, cv2.COLOR_GRAY2BGR)

            # Adiciona os textos com os tempos de execução
            cv2.putText(vis_otsu, f"1. Otsu: {tempo_otsu:.1f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(vis_grab, f"2. GrabCut: {tempo_grab:.1f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(vis_cnt, f"3. OpenCV CNT: {tempo_cnt:.1f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
            cv2.putText(vis_mp, f"4. MediaPipe: {tempo_mp:.1f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

            # Monta a grade (Top: Otsu e GrabCut | Bottom: CNT e MediaPipe)
            top_row = np.hstack((vis_otsu, vis_grab))
            bottom_row = np.hstack((vis_cnt, vis_mp))
            combined = np.vstack((top_row, bottom_row))
            
            # Redimensiona a janela final para caber na tela (já que dobramos o tamanho)
            combined_display = cv2.resize(combined, (w, h))
            cv2.imshow("Comparacao de Segmentacao (2x2)", combined_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('b'):
            time.sleep(1)
            ret, bg_frame = cap.read()
            bg_frame = cv2.flip(bg_frame, 1)
            bg_base = cv2.resize(bg_frame, None, fx=scale, fy=scale)
            print("Background Capturado para Otsu e GrabCut!")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()