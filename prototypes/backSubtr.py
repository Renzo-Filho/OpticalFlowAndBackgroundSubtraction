import cv2
import numpy as np
import time
import os

# =========================
# Configurações Gerais
# =========================
BACKGROUND_IMAGE_PATH = "fundo-equacoes.png"  # (opcional, não usado diretamente aqui)

# Duração de cada efeito na rotação automática
EFFECT_SECONDS = 10

# Ajustes de Performance e Visual
FLOW_SCALE = 0.5           # 0.5 = calcula fluxo em meia resolução (mais rápido/suave)
FLOW_BLUR_K = (15, 15)     # Tamanho do blur aplicado nos vetores de movimento (suavidade)
NOISE_THRESHOLD = 2.0      # Ignora movimentos com magnitude menor que isso (limpa o fundo)

# Configurações do HUD
DEBUG_OVERLAY = True       # Começa com texto na tela?

# =========================
# Utilidades de UI
# =========================
def overlay_text(img, text, y=40, scale=0.9, color=(255, 255, 255)):
    cv2.putText(
        img, text, (20, y),
        cv2.FONT_HERSHEY_SIMPLEX, scale,
        color, 2, cv2.LINE_AA
    )

def overlay_hud(img, effect_name, remaining, extra="", debug=True):
    if not debug:
        return
    h, w = img.shape[:2]
    overlay_text(img, f"EFEITO: {effect_name}", y=35, scale=1.0, color=(0, 255, 255))
    overlay_text(img, f"Prox. em: {remaining:0.1f}s", y=70, scale=0.7)

    if extra:
        overlay_text(img, extra, y=105, scale=0.6, color=(200, 200, 200))

    overlay_text(
        img,
        "Teclas: n/p=Mudar | d=HUD | r=Reset | b=Captura Fundo | q/Esc=Sair",
        y=h-20, scale=0.6
    )

# =========================
# Engine de Optical Flow Otimizada
# =========================
def calculate_smooth_flow(prev_gray, curr_gray):
    """
    Calcula Optical Flow Farneback com downscaling e blur.
    Retorna o fluxo redimensionado para o tamanho original.
    """
    h, w = curr_gray.shape

    # 1. Downscale
    prev_small = cv2.resize(prev_gray, None, fx=FLOW_SCALE, fy=FLOW_SCALE)
    curr_small = cv2.resize(curr_gray, None, fx=FLOW_SCALE, fy=FLOW_SCALE)

    # 2. Farneback
    flow_small = cv2.calcOpticalFlowFarneback(
        prev_small, curr_small,
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    # 3. Blur nos vetores (coerência espacial)
    flow_small = cv2.GaussianBlur(flow_small, FLOW_BLUR_K, 5.0)

    # 4. Upscale
    flow = cv2.resize(flow_small, (w, h))

    # Ajusta magnitude por causa do redimensionamento
    flow *= (1.0 / FLOW_SCALE)

    return flow

# =========================
# Background Subtraction (modelo fixo)
# =========================

def capture_background_average(cap, num_frames=150, settle_ms=300):
    """
    Captura um modelo de background como média de 'num_frames' frames.
    - settle_ms: espera curta para autoexposição estabilizar após você apertar 'b'
    Retorna bg_base (uint8 BGR) ou None se falhar.
    """
    # Dá um tempinho para estabilizar exposição/white balance
    t0 = time.time()
    while (time.time() - t0) * 1000.0 < settle_ms:
        cap.read()

    acc = None
    got = 0

    for _ in range(num_frames):
        ret, fr = cap.read()
        if not ret:
            continue
        fr = cv2.flip(fr, 1)  # manter consistente com seu pipeline
        fr_f = fr.astype(np.float32)
        if acc is None:
            acc = fr_f
        else:
            acc += fr_f
        got += 1

    if acc is None or got == 0:
        return None

    bg = (acc / got)
    return np.clip(bg, 0, 255).astype(np.uint8)

def make_foreground_mask(frame_bgr, bg_base_bgr):
    """
    Gera máscara de foreground (pessoa) a partir de um modelo fixo de background.
    Retorna máscara uint8 0..255 (255 = foreground).

    Pós-processamento:
      - opening leve (remove ruído)
      - closing holes (fecha buracos internos)
      - dilatação final (engorda o foreground)
    """
    BIG_STRUCT_ELEMENT_SIZE = 31  # <<< PARÂMETRO ÚNICO
    SMALL_STRUCT_ELEMENT_SIZE = 11  # <<< PARÂMETRO ÚNICO

    # Garante mesmo tamanho
    if bg_base_bgr.shape[:2] != frame_bgr.shape[:2]:
        bg_base_bgr = cv2.resize(
            bg_base_bgr,
            (frame_bgr.shape[1], frame_bgr.shape[0])
        )

    # Diferença em YCrCb (mais estável que BGR)
    f = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    b = cv2.cvtColor(bg_base_bgr, cv2.COLOR_BGR2YCrCb)

    diff = cv2.absdiff(f, b)

    dy  = diff[..., 0].astype(np.float32)
    dcr = diff[..., 1].astype(np.float32)
    dcb = diff[..., 2].astype(np.float32)

    # Score de diferença
    score = 0.5 * dy + 1.0 * dcr + 1.0 * dcb
    score_u8 = np.clip(score, 0, 255).astype(np.uint8)

    # Threshold automático (Otsu)
    _, mask = cv2.threshold(
        score_u8, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # =========================
    # Pós-processamento morfológico
    # =========================
    big_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (BIG_STRUCT_ELEMENT_SIZE, BIG_STRUCT_ELEMENT_SIZE)
    )

    small_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (BIG_STRUCT_ELEMENT_SIZE, BIG_STRUCT_ELEMENT_SIZE)
    )

    # 1) Opening leve – remove ruído isolado
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_k, iterations=1)

    # 2) Closing holes – fecha buracos internos (rosto, tronco)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, big_k, iterations=2)

    # 3) Dilatação final – engorda o foreground
    mask = cv2.dilate(mask, big_k, iterations=2)

    # Feather final para bordas suaves
    mask = cv2.GaussianBlur(mask, (11, 11), 0)

    return mask




# =========================
# Efeitos Visuais
# =========================
def effect_flow_color_polished(frame_bgr, prev_gray, gray, state):
    """
    Visualização limpa do fluxo. Fundo preto, movimento colorido.
    """
    flow = calculate_smooth_flow(prev_gray, gray)
    h, w = gray.shape
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    if "hsv" not in state or state["hsv"].shape[:2] != (h, w):
        state["hsv"] = np.zeros((h, w, 3), dtype=np.uint8)
        state["hsv"][..., 1] = 255

    hsv = state["hsv"]

    hsv[..., 0] = ang * 180 / np.pi / 2
    val = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    mask_static = mag < NOISE_THRESHOLD
    val[mask_static] = 0

    hsv[..., 2] = val.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def effect_fluid_paint(frame_bgr, prev_gray, gray, state):
    """
    Simulação visual de fluido via Advecção.
    (versão original: pinta em tela inteira)
    """
    flow = calculate_smooth_flow(prev_gray, gray)
    h, w = gray.shape

    if "fluid_canvas" not in state or state["fluid_canvas"].shape[:2] != (h, w):
        state["fluid_canvas"] = np.zeros((h, w, 3), dtype=np.float32)

    # Advecção
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = grid_x - flow[..., 0]
    map_y = grid_y - flow[..., 1]
    fluid_next = cv2.remap(state["fluid_canvas"], map_x, map_y, interpolation=cv2.INTER_LINEAR)

    # Injeção
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv_inject = np.zeros((h, w, 3), dtype=np.uint8)
    hsv_inject[..., 0] = ang * 180 / np.pi / 2
    hsv_inject[..., 1] = 255
    hsv_inject[..., 2] = 255
    bgr_inject = cv2.cvtColor(hsv_inject, cv2.COLOR_HSV2BGR).astype(np.float32)

    inject_mask = (mag > NOISE_THRESHOLD).astype(np.float32)
    inject_mask = cv2.GaussianBlur(inject_mask, (9, 9), 0)
    if inject_mask.ndim == 2:
        inject_mask = inject_mask[..., None]

    DECAY = 0.96 
    fluid_next = (fluid_next * DECAY) + (bgr_inject * inject_mask * 0.3)

    state["fluid_canvas"] = fluid_next
    return np.clip(fluid_next, 0, 255).astype(np.uint8)

def effect_fluid_paint_bg_only(frame_bgr, prev_gray, gray, state):
    """
    FLUID_PAINT_BG (MODIFICADO):
    - Usa modelo fixo de background (state["bg_base"]) para gerar máscara de foreground.
    - Saída:
        * Foreground = pessoa recortada (frame filtrado pela máscara)
        * Background = efeito de fluido (advecção) aplicado só no fundo
    """
    h, w = gray.shape

    # If we calculated a mask in main (Flow or Static), use it!
    if "active_mask" in state and state["active_mask"] is not None:
        # The mask in state is typically 0 or 255
        fg_mask_0_255 = state["active_mask"]
    
    # Fallback to old behavior (Static BG) if no active mask found
    elif state.get("bg_base") is not None:
        fg_mask_0_255 = make_foreground_mask(frame_bgr, state["bg_base"])
    else:
        # If we have neither, we can't do the effect
        return frame_bgr

    # 2) Converte para [0,1] e 3 canais
    fg = (fg_mask_0_255.astype(np.float32) / 255.0)   # (H,W)
    fg3 = fg[..., None]                               # (H,W,1)
    bg3 = 1.0 - fg3

    # 3) Optical flow
    flow = calculate_smooth_flow(prev_gray, gray)

    # 4) Canvas fluido
    if "fluid_canvas" not in state or state["fluid_canvas"].shape[:2] != (h, w):
        state["fluid_canvas"] = np.zeros((h, w, 3), dtype=np.float32)

    # 5) Advecção (mover tinta)
    ADVECT_GAIN = 2.0
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = grid_x - ADVECT_GAIN * flow[..., 0]
    map_y = grid_y - ADVECT_GAIN * flow[..., 1]
    fluid_next = cv2.remap(
        state["fluid_canvas"], map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    # 6) Injeção baseada no movimento (somente no BACKGROUND)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv_inject = np.zeros((h, w, 3), dtype=np.uint8)
    hsv_inject[..., 0] = ang * 180 / np.pi / 2
    hsv_inject[..., 1] = 255
    hsv_inject[..., 2] = 255
    bgr_inject = cv2.cvtColor(hsv_inject, cv2.COLOR_HSV2BGR).astype(np.float32)

    inject_mask = (mag > NOISE_THRESHOLD).astype(np.float32)   # (H,W)
    inject_mask = cv2.GaussianBlur(inject_mask, (9, 9), 0)     # (H,W)

    # Restringe ao background: (1 - fg)
    inject_mask *= (1.0 - fg)                                  # (H,W)
    if inject_mask.ndim == 2:
        inject_mask = inject_mask[..., None]                   # (H,W,1)

    DECAY = 0.985  # 0.96
    fluid_next = (fluid_next * DECAY) + (bgr_inject * inject_mask * 0.30)

    # Zera fluido no foreground pra não vazar atrás do corpo
    fluid_next *= bg3

    state["fluid_canvas"] = fluid_next

    # 7) Composição final:
    # Foreground = pessoa recortada
    # Background = fluido
    frame_f = frame_bgr.astype(np.float32)
    foreground_filtered = frame_f * fg3
    background_effect = np.clip(fluid_next, 0, 255).astype(np.float32) * bg3

    out = foreground_filtered + background_effect
    return np.clip(out, 0, 255).astype(np.uint8)

def effect_grid_warp(frame_bgr, prev_gray, gray, state):
    """
    Deforma uma grade virtual baseada no movimento (Wireframe).
    """
    flow = calculate_smooth_flow(prev_gray, gray)
    h, w = gray.shape
    out = np.zeros_like(frame_bgr)

    STEP = 40
    AMP = 3.0
    COLOR = (0, 255, 255)

    for x in range(0, w, STEP):
        pts = []
        for y in range(0, h, 10):
            dx, dy = flow[y, min(x, w-1)]
            pts.append([x + dx * AMP, y + dy * AMP])
        pts_arr = np.array(pts, np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts_arr], False, COLOR, 1, cv2.LINE_AA)

    for y in range(0, h, STEP):
        pts = []
        for x in range(0, w, 10):
            dx, dy = flow[y, min(x, w-1)]
            pts.append([x + dx * AMP, y + dy * AMP])
        pts_arr = np.array(pts, np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts_arr], False, COLOR, 1, cv2.LINE_AA)

    return out

def effect_simple_arrows(frame_bgr, prev_gray, gray, state):
    """
    Versão clássica de setas do optical flow,
    com fator de amplificação do vetor.
    """
    OF_VECTOR_FACTOR = 4.0  # <<< NOVO PARÂMETRO

    flow = calculate_smooth_flow(prev_gray, gray)
    out = frame_bgr.copy()
    h, w = gray.shape

    step = 30

    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            fx, fy = flow[y, x]

            # Ignora vetores pequenos (ruído)
            if (fx * fx + fy * fy) < (NOISE_THRESHOLD ** 2):
                continue

            # Amplificação do vetor
            fx_plot = fx * OF_VECTOR_FACTOR
            fy_plot = fy * OF_VECTOR_FACTOR

            cv2.arrowedLine(
                out,
                (x, y),
                (int(x + fx_plot), int(y + fy_plot)),
                (0, 255, 0),
                1,
                tipLength=0.3
            )

    return out


def effect_motion_trail(frame_bgr, prev_gray, gray, state):
    """
    Rastro (ghosting) apenas onde há movimento detectado pelo Optical Flow.
    """
    flow = calculate_smooth_flow(prev_gray, gray)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    h, w = frame_bgr.shape[:2]
    if ("trail_acc" not in state) or (state["trail_acc"] is None) or (state["trail_acc"].shape[:2] != (h, w)):
        state["trail_acc"] = frame_bgr.astype(np.float32)

    acc = state["trail_acc"]
    frame_f = frame_bgr.astype(np.float32)

    motion_mask = (mag > NOISE_THRESHOLD).astype(np.float32)
    motion_mask = cv2.GaussianBlur(motion_mask, (9, 9), 0)
    motion_mask = motion_mask[..., None]

    TRAIL_LENGTH = 0.90
    trail_blend = cv2.addWeighted(acc, TRAIL_LENGTH, frame_f, (1.0 - TRAIL_LENGTH), 0)
    output = (trail_blend * motion_mask) + (frame_f * (1.0 - motion_mask))

    state["trail_acc"] = output
    return np.clip(output, 0, 255).astype(np.uint8)

def effect_long_trail(frame_bgr, prev_gray, gray, state):
    """
    Rastro sólido e grosso com dilatação.
    """
    flow = calculate_smooth_flow(prev_gray, gray)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    h, w = frame_bgr.shape[:2]
    if ("long_trail_acc" not in state) or (state["long_trail_acc"] is None) or (state["long_trail_acc"].shape[:2] != (h, w)):
        state["long_trail_acc"] = frame_bgr.astype(np.float32)

    acc = state["long_trail_acc"]
    frame_f = frame_bgr.astype(np.float32)

    THRESHOLD_LOCAL = 1.0
    motion_mask = (mag > THRESHOLD_LOCAL).astype(np.float32)

    kernel_dilate = np.ones((15, 15), np.uint8)
    motion_mask = cv2.dilate(motion_mask, kernel_dilate, iterations=1)
    motion_mask = cv2.GaussianBlur(motion_mask, (5, 5), 0)
    motion_mask = motion_mask[..., None]

    DECAY = 0.995
    acc_next = (frame_f * motion_mask) + (acc * DECAY * (1.0 - motion_mask))
    state["long_trail_acc"] = acc_next

    return np.clip(acc_next, 0, 255).astype(np.uint8)

def effect_show_mask(frame_bgr, prev_gray, gray, state):
    """
    Wrapper para visualizar a máscara de foreground como um efeito.
    """
    return show_mask_effect(frame_bgr, state)

def show_mask_effect(frame_bgr, state):
    """
    Mostra (visualiza) a máscara binária do background subtraction.
    - Se não houver bg_base, retorna uma imagem preta com aviso.
    - Caso exista bg_base, exibe a máscara (0/255) em BGR para o imshow.
    """
    h, w = frame_bgr.shape[:2]

    if state.get("bg_base") is None:
        out = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(out, "Sem bg_base: pressione 'b' p/ capturar o fundo",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)
        return out

    mask_0_255 = make_foreground_mask(frame_bgr, state["bg_base"])

    # Se você quiser ver a máscara binária (sem blur), pode binarizar aqui:
    # _, mask_0_255 = cv2.threshold(mask_0_255, 127, 255, cv2.THRESH_BINARY)

    # Converte para 3 canais para o cv2.imshow
    return cv2.cvtColor(mask_0_255, cv2.COLOR_GRAY2BGR)


def overlay_mask_debug(frame, mask_u8):
    """
    Visualizes the mask in the corner of the screen for comparison.
    """
    h, w = frame.shape[:2]
    small_h, small_w = h // 4, w // 4
    
    # Create red overlay for the mask
    mask_small = cv2.resize(mask_u8, (small_w, small_h))
    mask_color = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
    mask_color[..., 0] = 0 # Remove Blue
    mask_color[..., 1] = 0 # Remove Green
    # Result is Red channel only

    # Overlay on bottom-right
    frame[h-small_h:h, w-small_w:w] = mask_color
    cv2.rectangle(frame, (w-small_w, h-small_h), (w, h), (255, 255, 255), 1)

# =========================
# Background Subtraction via Optical Flow
# =========================

def make_mask_from_flow_simple(flow, threshold=2.0):
    """
    PHASE 1: Generates a foreground mask based purely on instantaneous motion.
    Pros: Doesn't need a static background model.
    Cons: The person disappears if they stop moving (The 'Statue' problem).
    """
    # 1. Calculate Magnitude of the flow vectors
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # 2. Thresholding (Motion > Threshold = Foreground)
    _, mask = cv2.threshold(mag, threshold, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)

    # 3. Morphology (Clean up noise and fill holes)
    # Using the same kernel size as your static method for fair comparison
    kernel_size = 21 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Remove small noise dots
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # Fill small holes inside the moving object
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Dilate slightly to ensure coverage
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask

def make_mask_from_flow_robust(flow, state, threshold=2.0, decay=0.90):
    """
    Method B: Generates mask based on MOTION (Optical Flow).
    Uses 'decay' to keep the mask active for a short time after motion stops.
    """
    h, w = flow.shape[:2]
    
    # === FIX START ===
    # We must check if it is None BEFORE we check for .shape
    if state.get("flow_acc") is None or state["flow_acc"].shape != (h, w):
        state["flow_acc"] = np.zeros((h, w), dtype=np.float32)
    # === FIX END ===

    # 1. Calculate Magnitude
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # 2. Accumulate (Memory)
    # Take the MAX of current motion vs decayed history
    state["flow_acc"] = np.maximum(mag, state["flow_acc"] * decay)
    
    # 3. Threshold
    _, mask = cv2.threshold(state["flow_acc"], threshold, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)

    # 4. Clean up noise (Morphology)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask




def main():
    window_name = "Optical Flow Study"
    debug_overlay = DEBUG_OVERLAY

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Câmera não encontrada.")
        return

    ret, frame0 = cap.read()
    if not ret:
        print("Erro: não consegui ler frame inicial.")
        return

    h0, w0 = frame0.shape[:2]

    # Estado compartilhado
    state = {
        "hsv": np.zeros((h0, w0, 3), dtype=np.uint8),
        "fluid_canvas": np.zeros((h0, w0, 3), dtype=np.float32),
        "trail_acc": None,
        "long_trail_acc": None,
        "bg_base": None,        # Static Model
        "flow_acc": None,       # Motion Model
        "use_flow_mask": False  # Toggle: False=Static, True=OpticalFlow
    }
    state["hsv"][..., 1] = 255

    # Lista de Efeitos
    effects = [
        ("SHOW_MASK", effect_show_mask, EFFECT_SECONDS),
        ("FLUID_PAINT_BG", effect_fluid_paint_bg_only, 30), # Increased duration for testing
        ("ARROWS", effect_simple_arrows, 15),
    ]

    idx = 0
    effect_start = time.time()
    gray_prev = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

    print("=== CONTROLS ===")
    print(" 'm' : Toggle Mask Method (Static vs. Optical Flow)")
    print(" 'b' : Capture Static Background (step out first!)")
    print(" 'd' : Toggle HUD")
    print(" 'q' : Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Pre-process
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Calculate Flow (Essential for everything)
        flow = calculate_smooth_flow(gray_prev, gray)

        now = time.time()
        elapsed = now - effect_start
        name, func, duration = effects[idx]
        remaining = max(0, duration - elapsed)

        if elapsed >= duration:
            idx = (idx + 1) % len(effects)
            effect_start = now
            # Reset visual accumulators (not models)
            state["fluid_canvas"][:] = 0
            state["trail_acc"] = None

        # =========================================
        # STUDIED LOGIC: Generate the Mask
        # =========================================
        active_mask = None
        method_name = ""

        if state["use_flow_mask"]:
            # METHOD B: Optical Flow
            active_mask = make_mask_from_flow_robust(flow, state, threshold=NOISE_THRESHOLD)
            method_name = "METHOD: Optical Flow (Motion)"
            
            # Update state with this mask so effects can use it
            # We "hack" the bg_base logic by passing the mask directly if needed, 
            # but for FLUID_PAINT_BG we need to ensure it uses this mask.
            # *Note: standard FLUID_PAINT_BG in your code calculates its own mask from bg_base.
            # For this study, we are just visualizing the difference mostly.*
        else:
            # METHOD A: Static Background
            if state["bg_base"] is not None:
                active_mask = make_foreground_mask(frame, state["bg_base"])
                method_name = "METHOD: Static BG Subtraction"
            else:
                active_mask = np.zeros_like(gray)
                method_name = "METHOD: Static (No BG captured)"
                

        state["active_mask"] = active_mask

        # =========================================
        # Render Effect
        # =========================================
        try:
            # If we are in SHOW_MASK mode, we force the display of our ACTIVE mask
            if name == "SHOW_MASK":
                out = cv2.cvtColor(active_mask, cv2.COLOR_GRAY2BGR)
            else:
                out = func(frame, gray_prev, gray, state)
        except Exception as e:
            print(f"Erro no efeito {name}: {e}")
            out = frame

        # Overlay the active mask in the corner (Red) for comparison
        if debug_overlay:
            overlay_mask_debug(out, active_mask)

        # HUD
        overlay_hud(out, name, remaining, method_name, debug=debug_overlay)
        
        cv2.imshow(window_name, out)
        gray_prev = gray.copy()

        # =========================================
        # Inputs
        # =========================================
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('d'):
            debug_overlay = not debug_overlay
        elif key == ord('n'):
            idx = (idx + 1) % len(effects)
            effect_start = time.time()
        elif key == ord('m'):
            # THE SWITCH
            state["use_flow_mask"] = not state["use_flow_mask"]
            print(f"Switched Method. Flow Mode: {state['use_flow_mask']}")
        elif key == ord('b'):
            print("Capturing background in 3 seconds...")
            bg = capture_background_average(cap, num_frames=100, settle_ms=1000)
            if bg is not None:
                state["bg_base"] = bg
                print("Background Captured!")
            else:
                print("Capture failed.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()