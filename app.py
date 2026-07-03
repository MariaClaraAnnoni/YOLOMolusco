import streamlit as st
import cv2
import numpy as np
import base64
import time
from typing import List
from PIL import Image
from ultralytics import YOLO, SAM

# ------------------------------------------------------------------
# 1. Configuração da Página e Estilo (CSS + Imagem de Fundo)
# ------------------------------------------------------------------
st.set_page_config(page_title="YOLOMolusco", page_icon="🍍", layout="wide")

@st.cache_data
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return "" # Retorna vazio se a imagem não for encontrada

# Nome da sua imagem de fundo (coloque na mesma pasta do GitHub)
img_base64 = get_base64_of_bin_file("fundo.png") 

st.markdown(f"""
<style>
    .stApp {{
        background-image: url("data:image/png;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-color: #032b43; /* Cor de fallback */
    }}
    .block-container {{
        background-color: rgba(3, 43, 67, 0.85);
        padding: 2rem;
        border-radius: 15px;
        margin-top: 2rem;
    }}
    h1, h2, h3, p, label {{ color: white !important; }}
    [data-testid="stMetric"] {{
        background-color: rgba(255, 255, 255, 0.5) !important;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.5);
        text-align: center;
    }}
    [data-testid="stMetricLabel"] * {{ color: #032b43 !important; font-weight: 800; font-size: 24px !important; justify-content: center; }}
    [data-testid="stMetricValue"] > div {{ color: #26708a !important; font-weight: 900; justify-content: center; }}
    [data-testid="stFileUploadDropzone"] {{
        background-color: rgba(255, 255, 255, 0.1);
        border: 2px dashed #f4fbd1;
        border-radius: 20px;
    }}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# 2. Carregamento dos Modelos (Cache)
# ------------------------------------------------------------------
@st.cache_resource
def load_models():
    # Carrega o YOLO treinado (deve estar no GitHub)
    yolo = YOLO("best1.pt")
    yolo.model.names = {0: "Patrick", 1: "SpongeBob", 2: "Squidward"}
    
    # Carrega o MobileSAM (baixará automaticamente na nuvem, ignorando o limite de 100MB do GitHub)
    sam = SAM("mobile_sam.pt")
    
    return yolo, sam

yolo_model, sam_model = load_models()

# ------------------------------------------------------------------
# 3. Funções de Segmentação
# ------------------------------------------------------------------
def seg_otsu(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened

def seg_grabcut(roi: np.ndarray) -> np.ndarray:
    mask = np.zeros(roi.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (2, 2, roi.shape[1] - 4, roi.shape[0] - 4)
    cv2.grabCut(roi, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    return mask2 * 255

def seg_watershed(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(roi, markers.copy())
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    mask[markers == 1] = 255
    mask = cv2.bitwise_not(mask)
    return mask

def seg_kmeans(roi: np.ndarray, k: int = 3) -> np.ndarray:
    z = roi.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, _ = cv2.kmeans(z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    h, w = roi.shape[:2]
    center_label = labels.reshape(h, w)[h // 2, w // 2]
    mask = (labels.reshape(h, w) == center_label).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def seg_canny_fill(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)
    return mask

def seg_sam(img_bgr: np.ndarray, box: List[int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    # Usa o sam_model carregado no cache
    results = sam_model(img_bgr, bboxes=[box], verbose=False)
    if results[0].masks is None:
        return seg_otsu(img_bgr[y1:y2, x1:x2])
    mask_full = results[0].masks.data[0].cpu().numpy()
    mask_full = (mask_full * 255).astype(np.uint8)
    return mask_full[y1:y2, x1:x2]

SEGMENTATION_METHODS = {
    "Otsu": lambda roi, img, box: seg_otsu(roi),
    "Watershed": lambda roi, img, box: seg_watershed(roi),
    "GrabCut": lambda roi, img, box: seg_grabcut(roi),
    "K-Means": lambda roi, img, box: seg_kmeans(roi, 3),
    "Canny Fill": lambda roi, img, box: seg_canny_fill(roi),
    "SAM": lambda roi, img, box: seg_sam(img, box),
}

# ------------------------------------------------------------------
# 4. Interface do Usuário (Frontend)
# ------------------------------------------------------------------
st.title("YOLOMolusco - Laboratório de Identificação 🌊")
st.markdown("Carregue imagens para identificar personagens em tempo real.")

col_metodo, _ = st.columns([1, 3])
with col_metodo:
    metodo_escolhido = st.selectbox("Método de Segmentação", list(SEGMENTATION_METHODS.keys()))

uploaded_file = st.file_uploader("Solte sua imagem aqui (PNG, JPG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner("Analisando dados no laboratório..."):
        t0 = time.time()
        
        # Leitura da imagem
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Inferência YOLO
        results = yolo_model(img_bgr, conf=0.5)
        boxes = results[0].boxes
        
        # Prepara as imagens de saída
        det_img = img_rgb.copy()      # Imagem 2: Apenas as caixas do YOLO
        overlay_img = img_rgb.copy()  # Imagem 3: Apenas o preenchimento verde (sem caixas)
        confidences = []
        
        if boxes is not None:
            for b in boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                cls_id = int(b.cls.item() if hasattr(b.cls, "item") else b.cls)
                conf = float(b.conf.item() if hasattr(b.conf, "item") else b.conf)
                name = yolo_model.names[cls_id]
                confidences.append(conf)
                
                # 1. Imagem de Detecção: Desenha a Bounding Box e o Texto
                cv2.rectangle(det_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(det_img, f"{name} {conf:.0%}", (x1, max(y1 - 12, 0)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Segmentação na ROI
                roi = img_bgr[y1:y2, x1:x2]
                if roi.size > 0:
                    try: 
                        # Chama a função correspondente no dicionário
                        mask = SEGMENTATION_METHODS[metodo_escolhido](roi, img_bgr, [x1, y1, x2, y2])
                    except Exception as e: 
                        mask = seg_otsu(roi) # Fallback
                    
                    if mask.shape[:2] != roi.shape[:2]:
                        mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]))
                    
                    # 2. Imagem de Overlay: Aplica apenas o preenchimento da máscara
                    overlay_roi = overlay_img[y1:y2, x1:x2].copy()
                    overlay_roi[mask == 255] = (0, 255, 0)
                    overlay_img[y1:y2, x1:x2] = cv2.addWeighted(img_rgb[y1:y2, x1:x2], 0.5, overlay_roi, 0.5, 0)

        # Cálculo de Métricas
        tempo_ms = round((time.time() - t0) * 1000, 1)
        media_confianca = round((sum(confidences) / len(confidences) * 100), 1) if confidences else 0
        
        # Exibição dos Cartões
        st.write("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Confiança", f"{media_confianca}%")
        m2.metric("Tempo de processamento", f"{tempo_ms} ms")
        m3.metric("Personagens detectados", f"{len(confidences)}")
        st.write("---")
        
        # Exibição das 3 Imagens
        st.subheader("Resultados da Análise")
        img1, img2, img3 = st.columns(3)
        with img1: 
            st.image(img_rgb, caption="Imagem Original", use_container_width=True)
        with img2: 
            st.image(det_img, caption="Detecção (YOLO)", use_container_width=True)
        with img3: 
            st.image(overlay_img, caption=f"Overlay + Segmentação ({metodo_escolhido.upper()})", use_container_width=True)