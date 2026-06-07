import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import os
import urllib.request

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(page_title="EcoGuard IA - Vigilancia Epidemiológica", layout="wide")

# Estilos personalizados en CSS
st.markdown("""
<style>
.main { background-color: #f8f9fa; }
.stButton>button {
    background-color: #d9534f;
    color: white;
    font-size: 18px;
    border-radius: 8px;
    padding: 10px 24px;
    border: none;
}
.stButton>button:hover { background-color: #c9302c; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ EcoGuard IA - Triaje Multimodal de Vigilancia Epidemiológica")
st.write("Detección automatizada de especies exóticas y análisis de riesgo sanitario en redes sociales.")

# MATRIZ OFICIAL DE ENFERMEDADES (Claves limpias para evitar fallas de codificación en UI)
DICCIONARIO_ZOONOSIS = {
    "Ave": [
        "Influenza Aviar de Alta Patogenicidad", 
        "Salmonelosis", 
        "Campilobacteriosis", 
        "Clamidiosis Aviar"
    ],
    "Primate Silvestre": [
        "Giardiasis", 
        "Balantidiosis", 
        "Gusano Barrenador", 
        "Tuberculosis", 
        "Mycobacterium bovis"
    ],
    "Felido Silvestre": [
        "Rabia Silvestre", 
        "Toxoplasmosis"
    ],
    "Felido Exotico": [
        "Rabia Silvestre", 
        "Toxoplasmosis"
    ],
    "Fauna Silvestre Indeterminada": [
        "Zoonosis Inespecifica bajo Monitoreo Preventivo"
    ]
}

# Descarga automática de archivos del modelo de Visión Artificial (MobileNet-SSD)
@st.cache_resource
def descargar_modelo_ia():
    proto_url = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
    model_url = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"
    
    if not os.path.exists("deploy.prototxt"):
        urllib.request.urlretrieve(proto_url, "deploy.prototxt")
    if not os.path.exists("mobilenet.caffemodel"):
        urllib.request.urlretrieve(model_url, "mobilenet.caffemodel")
    
    return cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet.caffemodel")

net = descargar_modelo_ia()

# Clases base del modelo detector
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Crear carpeta de evidencias de forma relativa
PATH_EVIDENCIAS = "EcoGuard_Evidencias"
if not os.path.exists(PATH_EVIDENCIAS):
    os.makedirs(PATH_EVIDENCIAS)

# Inicializar historial en memoria para la auditoría
if 'historial_datos' not in st.session_state:
    st.session_state['historial_datos'] = []

# --- 🎛️ BARRA LATERAL: PANEL DE CONTROL TÉCNICO ---
st.sidebar.header("⚙️ Panel de Control Técnico")
st.sidebar.write("Fuerza la taxonomía manualmente si la IA sufre de falsos positivos en imágenes complejas:")
modo_taxonomia = st.sidebar.selectbox(
    "Corrección Taxonómica Manual:",
    options=["Automático (Usar IA)", "Primate Silvestre", "Felido Silvestre", "Felido Exotico", "Ave"]
)

# --- INTERFAZ DE USUARIO ---
col1, col2 = st.columns(2)

with col1:
    st.header("📸 Evidencia Digital (Imagen)")
    uploaded_file = st.file_uploader("Subir captura de pantalla o fotografía...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Evidencia cargada listo para análisis.", use_container_width=True)

with col2:
    st.header("📝 Contexto Sintáctico (Texto)")
    texto_post = st.text_area(
        label="Inserte el texto que acompaña la publicación o reporte:",
        placeholder="Ej: vendo cachorro de ocelote, jaguar o monito...",
        height=150
    )

st.markdown("---")

# --- LÓGICA DE PROCESAMIENTO ---
if st.button("🚀 Ejecutar Análisis Multimodal"):
    if uploaded_file is not None:
        with st.spinner("Ejecutando matriz de triaje biológico..."):
            
            # --- PREPARAR IMAGEN PARA OPENCV ---
            img_np = np.array(image)
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            h, w, _ = img_cv.shape
            
            blob = cv2.dnn.blobFromImage(cv2.resize(img_cv, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()
            
            especie_detectada = "Fauna Silvestre Indeterminada"
            confianza_deteccion = 0.90
            
            # 1. EVALUACIÓN DE TAXONOMÍA
            if modo_taxonomia == "Automático (Usar IA)":
                texto_minusculas = texto_post.lower()
                nombre_archivo = uploaded_file.name.lower()
                
                # A) Primero ejecutamos la Red Neuronal para ver qué silueta encuentra
                label_ia = ""
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.20:
                        idx = int(detections[0, 0, i, 1])
                        label_ia = CLASSES[idx]
                        confianza_deteccion = confidence
                        break
                
                # B) Sistema de reglas de fusión multimodal (Cruza IA, Texto y Nombre de Archivo)
                # Prioridad Félidos
                if any(w in texto_minusculas for w in ["ocelote", "tigrillo", "margay", "jaguarundi", "puma"]) or "ocelote" in nombre_archivo or label_ia == "cat":
                    especie_detectada = "Felido Silvestre"
                elif any(w in texto_minusculas for w in ["jaguar", "tigre", "leon", "pantera"]) or "jaguar" in nombre_archivo or "tigre" in nombre_archivo:
                    especie_detectada = "Felido Exotico"
                # Prioridad Primates
                elif any(w in texto_minusculas for w in ["mono", "saraguato", "primate", "mico", "chango", "aullador"]) or "mono" in nombre_archivo or label_ia in ["dog", "horse"]:
                    especie_detectada = "Primate Silvestre"
                # Prioridad Aves
                elif any(w in texto_minusculas for w in ["loro", "perico", "guacamaya", "ave", "pajaro", "cacatua"]) or "ave" in nombre_archivo or label_ia == "bird":
                    especie_detectada = "Ave"
                else:
                    especie_detectada = "Fauna Silvestre Indeterminada"
            else:
                # Si se activa el control de la barra lateral, manda la selección manual
                especie_detectada = modo_taxonomia
                confianza_deteccion = 1.0

            # --- PROCESAMIENTO DE TEXTO (NLP) ---
            texto_minusculas = texto_post.lower()
            keywords_comercio = ["vendo", "inf", "dm", "trato", "precio", "ejemplar", "disponible", "barato", "iztapalapa", "entrego"]
            keywords_clinicas = ["enfermo", "decaido", "tos", "gripe", "secrecion", "lesion", "plumas", "parasito"]
            
            palabras_encontradas = [w for w in keywords_comercio + keywords_clinicas if w in texto_minusculas]
            alerta_comercial = any(w in texto_minusculas for w in keywords_comercio)
            
            st.session_state['alerta_comercial'] = alerta_comercial
            st.session_state['palabras_encontradas'] = palabras_encontradas
            
            # --- EXTRACCIÓN DINÁMICA DE ENFERMEDADES DESDE EL DICCIONARIO ---
            lista_patogenos = DICCIONARIO_ZOONOSIS.get(especie_detectada, DICCIONARIO_ZOONOSIS["Fauna Silvestre Indeterminada"])
            enfermedades_formateadas = " / ".join(lista_patogenos)
            
            # --- CALCULO DE RIESGO ---
            if alerta_comercial and len(palabras_encontradas) >= 3:
                riesgo = "Critico"
                impacto = "Alto riesgo de dispersion biologica por comercio clandestino. Potencial epidemico."
            elif alerta_comercial or len(palabras_encontradas) > 0:
                riesgo = "Alto"
                impacto = "Riesgo moderado de transmision zoonotica por confinamiento ilegal o manipulacion de vectores."
            else:
                riesgo = "Moderado"
                impacto = "Especie silvestre confinada. Patrones comerciales ausentes bajo monitoreo preventivo."
                
            st.session_state['especie_ia'] = especie_detectada
            st.session_state['riesgo'] = riesgo
            st.session_state['biologico'] = enfermedades_formateadas
            st.session_state['impacto'] = impacto
            
            # --- RENDERIZADO GRÁFICO (OPENCV) ---
            cv2.rectangle(img_cv, (int(w*0.05), int(h*0.05)), (int(w*0.95), int(h*0.95)), (0, 0, 255), 4)
            cv2.putText(img_cv, f"{especie_detectada} - Conf: {int(confianza_deteccion * 100)}%", 
                        (int(w*0.06), int(h*0.12)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            img_procesada = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            st.session_state['img_analizada'] = img_procesada
            
            # Guardar archivo de evidencia física
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_archivo_out = f"Evidencia_{timestamp}.jpg"
            ruta_guardado = os.path.join(PATH_EVIDENCIAS, nombre_archivo_out)
            Image.fromarray(img_procesada).save(ruta_guardado)
            st.session_state['ruta_evidencia'] = ruta_guardado
            
            # Guardar fila en tabla de auditoría
            nuevo_registro = {
                "Fecha/Hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Taxonomia": especie_detectada,
                "Riesgo": riesgo,
                "Zoonosis Latentes": enfermedades_formateadas,
                "Alerta Comercio": "SI" if alerta_comercial else "NO"
            }
            st.session_state['historial_datos'].append(nuevo_registro)
            st.success("¡Análisis completado!")
    else:
        st.error("Por favor, cargue una imagen primero.")

# --- RENDERS EN PANTALLA ---
if 'img_analizada' in st.session_state:
    st.header("🖥️ Resultados del Triaje Multimodal")
    st.image(st.session_state['img_analizada'], caption="Muestra bajo análisis epidemiológico.", use_container_width=True)
    
    st.subheader(f"Nivel de Riesgo Epidemiologico: **{st.session_state['riesgo']}**")
    
    if st.session_state['riesgo'] == "Critico":
        st.error(f"🚨 **Riesgo Biologico Asociado ({st.session_state['especie_ia']}):** {st.session_state['biologico']}")
    elif st.session_state['riesgo'] == "Alto":
        st.warning(f"⚠️ **Riesgo Biologico Asociado ({st.session_state['especie_ia']}):** {st.session_state['biologico']}")
    else:
        st.info(f"ℹ️ **Riesgo Biológico Asociado ({st.session_state['especie_ia']}):** {st.session_state['biologico']}")
        
    st.markdown(f"📋 **Impacto Sanitario:** {st.session_state['impacto']}")
    st.info(f"📍 **Ruta del archivo:** `{st.session_state['ruta_evidencia']}`")

    # ---- TEXT ANALYTICS ----
    st.markdown("---")
    st.markdown("### 🔍 Analisis de Patrones de Texto")
    if 'alerta_comercial' in st.session_state:
        if st.session_state['alerta_comercial']:
            st.error("⚠️ **Alerta Comercial Detectada:** Patrones de venta clandestina activos.")
        else:
            st.success("✅ **Filtro Comercial Limpio.**")
    if 'palabras_encontradas' in st.session_state and st.session_state['palabras_encontradas']:
        st.info(f"📋 **Palabras Clave Extraidas:** {', '.join(st.session_state['palabras_encontradas'])}")

    # ---- HISTORIAL ----
    st.markdown("---")
    st.markdown("### 📊 Historial Completo para Auditorias")
    if st.session_state['historial_datos']:
        df_historial = pd.DataFrame(st.session_state['historial_datos'])
        st.dataframe(df_historial, use_container_width=True)
        
        csv_data = df_historial.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar Reporte Completo (CSV)",
            data=csv_data,
            file_name=f"Reporte_EcoGuard_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
