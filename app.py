# -----------------------------------------------------------------------------
# 3. CARGA INTELIGENTE Y SEGURA DEL MODELO DE INTELIGENCIA ARTIFICIAL
# -----------------------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    prototxt_path = "deploy.prototxt"
    caffemodel_path = "mobilenet.caffemodel"
    
    # URLs directas de respaldo estables para Caffe MobileNet-SSD
    url_proto = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
    url_model = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/mobilenet.caffemodel"
    
    try:
        # Descargar prototxt si no existe o pesa muy poco
        if not os.path.exists(prototxt_path) or os.path.getsize(prototxt_path) < 1000:
            r = requests.get(url_proto, timeout=10)
            if r.status_code == 200:
                with open(prototxt_path, "wb") as f:
                    f.write(r.content)
                    
        # Descargar caffemodel si no existe o pesa muy poco (el archivo pesa ~23MB)
        if not os.path.exists(caffemodel_path) or os.path.getsize(caffemodel_path) < 1000000:
            r = requests.get(url_model, timeout=30)
            if r.status_code == 200:
                with open(caffemodel_path, "wb") as f:
                    f.write(r.content)

        # Intentar leer la red si ambos archivos son válidos
        if os.path.exists(prototxt_path) and os.path.exists(caffemodel_path):
            net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
            return net
    except Exception as e:
        st.error(f"Error al inicializar el modelo de red neuronal: {e}")
        
    return None

net = cargar_modelo()
