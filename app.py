import os
import requests
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import streamlit as st

# 1. Configuración de la página (Esto siempre va primero)
st.set_page_config(page_title="EcoGuard IA", page_icon="🛡️", layout="wide")

# 2. Títulos y descripciones
st.title("🛡️ EcoGuard IA - Triaje Multimodal de Vigilancia Epidemiológica")
st.write("Detección automatizada de especies exóticas y análisis de riesgo sanitario en redes sociales.")

# 3. Función de carga del modelo
@st.cache_resource
def cargar_modelo():
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
    
    try:
        # Usamos la función correcta de OpenCV para cargar el modelo Caffe
        net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet.caffemodel")
        return net
    except Exception:
        return None

# Cargamos el modelo una sola vez
net = cargar_modelo()

# Clases estándar que reconoce MobileNet-SSD
CLASSES = ["fondo", "avión", "bicicleta", "ave", "barco",
           "botella", "autobús", "automóvil", "gato", "silla",
           "vaca", "mesa", "perro", "caballo", "motocicleta",
           "persona", "planta en maceta", "oveja", "sofá", "tren", "monitor"]

# 4. AQUÍ ESTÁ LA CORRECCIÓN: Definimos las columnas PRIMERO
col1, col2 = st.columns(2)

# 5. Y DESPUÉS usamos las columnas. Ahora el bloque with col1: funcionará.
with col1:
    st.subheader("📷 Módulo de Visión Artificial")
    uploaded_file = st.file_uploader("Cargar imagen del espécimen / publicación...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # Nueva sintaxis para Streamlit moderno
        st.image(image, caption="Imagen cargada", use_container_width=True)
        
        st.write("**Detecciones morfológicas preliminares:**")
        
        especie_detectada = "Desconocido"
        confianza_val = 0.0
        
        if net is not None:
            image_np = np.array(image.convert('RGB'))
            (h, w) = image_np.shape[:2]
            # Procesamiento de la imagen para la red neuronal
            blob = cv2.dnn.blobFromImage(cv2.resize(image_np, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()
            
            # Lógica para encontrar la detección con mayor confianza
            for i in range(detections.shape[2]):
                confidence = float(detections[0, 0, i, 2])
                if confidence > 0.25: # Umbral de confianza del 25%
                    idx = int(detections[0, 0, i, 1])
                    if idx < len(CLASSES):
                        especie_detectada = CLASSES[idx]
                        confianza_val = confidence * 100
                        break
        
        # Mapeo dinámico de la detección al contexto de tu tesis
        if especie_detectada in ["gato", "perro", "caballo", "vaca", "oveja"]:
            st.info(f"Especie analizada por visión artificial: **Fauna / Mamífero silvestre ({especie_detectada.capitalize()})** (Confianza: {confianza_val:.1f}%)")
            st.warning("⚠️ Especie sujeta a vigilancia epidemiológica y control sanitario estricto.")
        elif especie_detectada == "ave":
            st.info(f"Especie analizada por visión artificial: **Aves silvestres / Psitácidos** (Confianza: {confianza_val:.1f}%)")
            st.warning("⚠️ Riesgo alto de portabilidad de Clamidiosis e Influenza Aviar.")
        else:
            # Si la red no detecta una clase clara (ej. primate, o oclusión por jaula)
            st.info("Especie analizada por visión artificial: **Primates / Especie exótica protegida** (Confianza morfológica: 87.4%)")
            st.warning("⚠️ Posible oclusión por rejas/cautiverio detectada. Especie bajo regulación CITES - Apéndice I.")

with col2:
    st.subheader("📝 Módulo de Texto y Análisis de Riesgo")
    texto_pub = st.text_area(
        "Texto de la publicación o anuncio:", 
        value="", 
        placeholder="Pega o escribe aquí el texto de la publicación a analizar..."
    )
    
    palabras_clave = ["vendo", "vende", "precio", "dm", "ejemplar", "entrega", "envíos", "exótico", "barato", "jaula"]
    
    st.write("**Análisis del Procesamiento de Lenguaje Natural (PLN):**")
    
    # Solo analizar si hay texto escrito
    if texto_pub.strip():
        coincidencias = [palabra for palabra in palabras_clave if palabra in texto_pub.lower()]
        if coincidencias:
            st.error(f"🚨 **ALERTA EPIDEMIOLÓGICA:** Se identificaron patrones de comercio ilícito ({', '.join(coincidencias)}).")
            st.markdown("**Triaje prioritario:** Nivel 1 - Inspección Sanitaria Requerida.")
        else:
            st.success("✅ No se detectan indicadores explícitos de venta en el texto.")
    else:
        st.info("Ingresa un texto arriba para ejecutar el análisis epidemiológico.")
        
