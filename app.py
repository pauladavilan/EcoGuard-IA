import os
import requests
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import streamlit as st

# Configuración de la página
st.set_page_config(page_title="EcoGuard IA", page_icon="🛡️", layout="wide")

st.title("🛡️ EcoGuard IA - Triaje Multimodal de Vigilancia Epidemiológica")
st.write("Detección automatizada de especies exóticas y análisis de riesgo sanitario en redes sociales.")

# Descarga automática del modelo
@st.cache_resource
def descargar_modelo_ia():
    proto_url = "https://raw.githubusercontent.com/pauladavilan/EcoGuard-IA/main/deploy.prototxt"
    model_url = "https://raw.githubusercontent.com/pauladavilan/EcoGuard-IA/main/mobilenet.caffemodel"
    
    if not os.path.exists("deploy.prototxt") or os.path.getsize("deploy.prototxt") < 1000:
        r = requests.get(proto_url, allow_redirects=True)
        with open("deploy.prototxt", "wb") as f:
            f.write(r.content)
            
    if not os.path.exists("mobilenet.caffemodel") or os.path.getsize("mobilenet.caffemodel") < 1000000:
        r = requests.get(model_url, allow_redirects=True)
        with open("mobilenet.caffemodel", "wb") as f:
            f.write(r.content)
    
    # Carga del modelo usando cv2.dnn.readNet (compatible con todas las versiones)
    return cv2.dnn.readNet("mobilenet.caffemodel", "deploy.prototxt")

# Cargar modelo
try:
    net = descargar_modelo_ia()
except Exception as e:
    st.error(f"Error al cargar el modelo de visión artificial: {e}")

# Clases de MobileNet SSD
CLASSES = ["fondo", "avión", "bicicleta", "ave", "barco",
           "botella", "autobús", "automóvil", "gato", "silla",
           "vaca", "mesa", "perro", "caballo", "motocicleta",
           "persona", "planta en maceta", "oveja", "sofá", "tren", "monitor"]

col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Módulo de Visión Artificial")
    uploaded_file = st.file_uploader("Cargar imagen del espécimen / publicación...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_column_width=True)
        
        # Procesamiento con OpenCV
        image_np = np.array(image.convert('RGB'))
        (h, w) = image_np.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image_np, (300, 300)), 0.007843, (300, 300), 127.5)
        
        net.setInput(blob)
        detections = net.forward()
        
        detecciones_morfologicas = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                idx = int(detections[0, 0, i, 1])
                etiqueta = CLASSES[idx] if idx < len(CLASSES) else "Desconocido"
                detecciones_morfologicas.append((etiqueta, float(confidence)))
        
        st.write("**Detecciones morfológicas preliminares:**")
        if detecciones_morfologicas:
            for et, conf in detecciones_morfologicas:
                st.info(f"Clase detectada: **{et}** (Confianza: {conf*100:.1f}%)")
        else:
            st.warning("No se detectaron siluetas morfológicas claras (Posible oclusión por rejas/cautiverio).")

with col2:
    st.subheader("📝 Módulo de Texto y Análisis de Riesgo")
    texto_pub = st.text_area("Texto de la publicación o anuncio:", value="Se vende cachorro de jaguar ocelote en excelente estado, entrega inmediata por DM.")
    
    palabras_clave = ["vendo", "vende", "precio", "dm", "ejemplar", "entrega", "envíos", "exótico", "barato", "jaula"]
    coincidencias = [palabra for palabra in palabras_clave if palabra in texto_pub.lower()]
    
    st.write("**Análisis del Procesamiento de Lenguaje Natural (PLN):**")
    if coincidencias:
        st.error(f"🚨 **ALERTA EPIDEMIOLÓGICA:** Se identificaron patrones de comercio ilícito ({', '.join(coincidencias)}).")
        st.markdown("**Triaje prioritario:** Nivel 1 - Inspección Sanitaria Requerida.")
    else:
        st.success("✅ No se detectan indicadores explícitos de venta en el texto.")
        

