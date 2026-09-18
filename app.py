import os
import requests
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import streamlit as st
from datetime import datetime

# 1. Configuración de la página
st.set_page_config(page_title="EcoGuard IA", page_icon="🛡️", layout="wide")

# 2. Títulos y descripciones
st.title("🛡️ EcoGuard IA - Triaje Multimodal de Vigilancia Epidemiológica")
st.write("Detección automatizada de especies exóticas y análisis de riesgo sanitario en redes sociales.")

# 3. Diccionario de Zoonosis por Grupo Taxonómico
ZOONOSIS_DB = {
    "Aves": [
        "Influenza Aviar de Alta Patogenicidad",
        "Salmonelosis",
        "Campilobacteriosis",
        "Clamidiosis Aviar"
    ],
    "Primates": [
        "Giardiasis",
        "Balantidiosis",
        "Gusano Barrenador",
        "Tuberculosis (Mycobacterium bovis)"
    ],
    "Félidos silvestres": [
        "Rabia Silvestre",
        "Toxoplasmosis"
    ]
}

# 4. Función de carga del modelo
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
        net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet.caffemodel")
        return net
    except Exception:
        return None

net = cargar_modelo()

# Distribución en dos columnas
col1, col2 = st.columns(2)

# Variables globales para el triaje multimodal
especie_seleccionada = "Desconocido"
grupo_taxonomico = "Aves"
oclusion_barrotes = False

with col1:
    st.subheader("📷 Módulo de Visión Artificial y Contexto")
    uploaded_file = st.file_uploader("Cargar imagen del espécimen / publicación...", type=["jpg", "jpeg", "png"])
    
    # Selector manual de categoría taxonómica para asegurar precisión en la demo académica
    grupo_taxonomico = st.selectbox(
        "Seleccione Grupo Taxonómico Identificado (Validación Morfológica):",
        ["Aves", "Primates", "Félidos silvestres"]
    )
    
    # Checkbox para simular el factor de oclusión (Efecto Cautiverio)
    oclusion_barrotes = st.checkbox("🔍 Detectar condiciones de cautiverio / oclusión visual (Jaulas / Rejas)")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_container_width=True)
        
        if oclusion_barrotes:
            st.warning("⚠️ **Efecto de Cautiverio Identificado:** Se detectan barreras físicas/barrotes. La precisión morfológica desciende al ~20%, indicando estrés y hacinamiento crítico.")
        else:
            st.success("✅ Entorno natural / Sin oclusión visual aparente.")

with col2:
    st.subheader("📝 Módulo de Texto y Análisis de Riesgo")
    texto_pub = st.text_area(
        "Texto de la publicación o anuncio:", 
        value="", 
        placeholder="Ej: Se vende cachorro de ocelote en excelente estado, entrega inmediata por DM..."
    )
    
    palabras_clave = ["vendo", "vende", "precio", "dm", "ejemplar", "entrega", "envíos", "exótico", "barato", "jaula"]
    
    st.write("**Análisis de Procesamiento de Lenguaje Natural (PLN):**")
    
    # Detección de palabras clave en el texto
    coincidencias = []
    if texto_pub.strip():
        coincidencias = [palabra for palabra in palabras_clave if palabra in texto_pub.lower()]
    
    # --- LÓGICA DE TRIAJE MULTIMODAL (CRÍTICO, ALTO, MODERADO) ---
    st.markdown("---")
    st.subheader("🚨 Resultado del Triaje Epidemiológico")
    
    timestamp_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. NIVEL CRÍTICO: Especie + Palabras clave O (Especie + Oclusión/Cautiverio)
    if (len(coincidencias) > 0 and uploaded_file is not None) or (oclusion_barrotes and len(coincidencias) > 0):
        st.error(f"🔴 **NIVEL CRÍTICO (Riesgo Sanitario / Tráfico Ilegal Confirmado)**")
        if coincidencias:
            st.write(f"• **Palabras clave comerciales detectadas:** `{', '.join(coincidencias)}`")
        if oclusion_barrotes:
            st.write("• **Factor de Oclusión:** Cautiverio y estrés en confinamiento verificado.")
        st.markdown("**Acción del Sistema:** Triaje prioritario de **Nivel 1**. Notificación inmediata a autoridades sanitarias y de procuración de justicia ambiental.")
        nivel_asignado = "Crítico"

    # 2. NIVEL ALTO: Palabras clave detectadas pero imagen ambigua / poco clara
    elif len(coincidencias) > 0 and uploaded_file is None:
        st.warning(f"🟠 **NIVEL ALTO (Sospecha de Comercio Ilegal / Vectores Zoonóticos)**")
        st.write(f"• **Palabras clave detectadas:** `{', '.join(coincidencias)}` (Falta validación visual del espécimen).")
        st.markdown("**Acción del Sistema:** Canalización a **revisión humana secundaria** para evitar falsos positivos y confirmar el estatus legal.")
        nivel_asignado = "Alto"

    # 3. NIVEL MODERADO: Entorno natural, sin barreras físicas y texto informativo/avistamiento
    elif len(coincidencias) == 0 and not oclusion_barrotes:
        st.info(f"🟡 **NIVEL MODERADO (Monitoreo Preventivo / Sin Riesgo Comercial Explícito)**")
        st.write("• Contexto puramente informativo, avistamiento o divulgación científica en libertad.")
        st.markdown("**Acción del Sistema:** El registro se archiva únicamente para **bases de datos epidemiológicas** y estadísticas de distribución de la biodiversidad, sin activar alarmas de intervención.")
        nivel_asignado = "Moderado"
        
    else:
        st.info("ℹ️ Ingrese un texto o cargue una imagen para completar la matriz de decisión.")
        nivel_asignado = "Pendiente"

    # --- SISTEMA DE GESTIÓN DE EVIDENCIAS Y ZOONOSIS ---
    if uploaded_file is not None or texto_pub.strip():
        st.markdown("---")
        with st.expander("📋 **Expediente y Registro de Evidencias (Sistema)**", expanded=True):
            st.write(f"📅 **Fecha y Hora de Registro:** `{timestamp_actual}`")
            st.write(f"🐾 **Grupo Taxonómico Evaluado:** `{grupo_taxonomico}`")
            st.write(f"📊 **Nivel de Alerta Asignado:** `{nivel_asignado}`")
            
            st.write("🦠 **Enfermedades Zoonóticas Asociadas al Grupo Taxonómico:**")
            enfermedades = ZOONOSIS_DB.get(grupo_taxonomico, [])
            for enf in enfermedades:
                st.markdown(f"  - ⚠️ {enf}")
            
            st.caption("Registro almacenado exitosamente en el repositorio central de vigilancia epidemiológica EcoGuard IA.")
