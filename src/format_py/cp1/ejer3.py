import pandas as pd
from collections import Counter


class ModeloEspacioVectorial:
    def __init__(self):
        self.documentos = {}  # {nombre_doc: tokens}
        self.matriz_tf = None
        self.terminos = set()
        self.nombres_docs = []

    def agregar_documento(self, nombre, tokens):
        """Añade un documento a la colección"""
        self.documentos[nombre] = tokens
        self.terminos.update(tokens)
        self.nombres_docs.append(nombre)

    def calcular_tf(self, tokens):
        """Calcula frecuencias normalizadas para un documento"""
        total_terminos = len(tokens)
        frecuencias = Counter(tokens)

        # Normalizar por total de términos: tf / T
        tf_normalizado = {}
        for termino, freq in frecuencias.items():
            tf_normalizado[termino] = freq / total_terminos

        return tf_normalizado

    def construir_matriz(self):
        """Construye la matriz término-documento"""
        if not self.documentos:
            return pd.DataFrame()

        # Inicializar matriz con ceros
        matriz_data = {}

        for doc_name, tokens in self.documentos.items():
            tf_doc = self.calcular_tf(tokens)
            matriz_data[doc_name] = tf_doc

        # Crear DataFrame
        self.matriz_tf = pd.DataFrame(matriz_data).fillna(0)

        # Reordenar columnas según el orden de inserción
        self.matriz_tf = self.matriz_tf[self.nombres_docs]

        return self.matriz_tf

    def obtener_terminos_relevantes(self, doc_index, top_n=5):
        """Devuelve los n términos más importantes para un documento"""
        if self.matriz_tf is None:
            self.construir_matriz()

        if doc_index >= len(self.nombres_docs):
            raise ValueError("Índice de documento fuera de rango")

        doc_name = self.nombres_docs[doc_index]

        # Obtener términos ordenados por peso descendente
        terminos_relevantes = (
            self.matriz_tf[doc_name]
            .sort_values(ascending=False)
            .head(top_n)
        )

        return terminos_relevantes


# Ejemplo de uso y prueba
if __name__ == "__main__":
    # Crear instancia del modelo
    modelo = ModeloEspacioVectorial()

    # Documentos de prueba
    documentos_prueba = {
        "doc1": ["machine", "learning", "data", "science", "algorithm", "data", "learning"],
        "doc2": ["deep", "learning", "neural", "network", "deep", "learning"],
        "doc3": ["natural", "language", "processing", "text", "mining", "language", "text"]
    }

    # Agregar documentos al modelo
    for nombre, tokens in documentos_prueba.items():
        modelo.agregar_documento(nombre, tokens)

    # Construir matriz término-documento
    matriz = modelo.construir_matriz()

    print("=== MATRIZ TÉRMINO-DOCUMENTO ===")
    print("(Valores: tf/T - frecuencia normalizada)")
    print("\n" + "=" * 50)
    print(matriz.round(3))
    print("\n" + "=" * 50)

    # Obtener términos más relevantes por documento
    print("\n=== TÉRMINOS MÁS RELEVANTES POR DOCUMENTO ===")

    for i in range(len(documentos_prueba)):
        doc_name = modelo.nombres_docs[i]
        print(f"\nDocumento {i + 1} ('{doc_name}'):")
        terminos_relevantes = modelo.obtener_terminos_relevantes(i, top_n=3)

        for termino, peso in terminos_relevantes.items():
            print(f"  - {termino}: {peso:.3f}")

    # Análisis adicional
    print("\n=== ANÁLISIS ADICIONAL ===")
    print(f"Total de términos únicos en la colección: {len(modelo.terminos)}")
    print(f"Términos únicos: {sorted(list(modelo.terminos))}")

    # Mostrar cálculos detallados para el primer documento
    print(f"\nCálculo detallado para '{modelo.nombres_docs[0]}':")
    tokens_doc1 = documentos_prueba["doc1"]
    total_terminos = len(tokens_doc1)
    frecuencias = Counter(tokens_doc1)

    print(f"Total de términos: {total_terminos}")
    print("Frecuencias brutas:")
    for termino, freq in frecuencias.items():
        tf_normalizado = freq / total_terminos
        print(f"  {termino}: {freq}/{total_terminos} = {tf_normalizado:.3f}")



#
# === MATRIZ TÉRMINO-DOCUMENTO ===
# (Valores: tf/T - frecuencia normalizada)
#
# ==================================================
#            doc1   doc2   doc3
# algorithm  0.143  0.000  0.000
# data       0.286  0.000  0.000
# deep       0.000  0.333  0.000
# learning   0.286  0.333  0.000
# machine    0.143  0.000  0.000
# mining     0.000  0.000  0.143
# natural    0.000  0.000  0.143
# neural     0.000  0.167  0.000
# network    0.000  0.167  0.000
# processing 0.000  0.000  0.143
# science    0.143  0.000  0.000
# text       0.000  0.000  0.286
# language   0.000  0.000  0.286
#
# ==================================================
#
# === TÉRMINOS MÁS RELEVANTES POR DOCUMENTO ===
#
# Documento 1 ('doc1'):
#   - data: 0.286
#   - learning: 0.286
#   - algorithm: 0.143
#
# Documento 2 ('doc2'):
#   - deep: 0.333
#   - learning: 0.333
#   - neural: 0.167
#
# Documento 3 ('doc3'):
#   - text: 0.286
#   - language: 0.286
#   - mining: 0.143
#
# === ANÁLISIS ADICIONAL ===
# Total de términos únicos en la colección: 13
# Términos únicos: ['algorithm', 'data', 'deep', 'learning', 'machine', 'mining', 'natural', 'neural', 'network', 'processing', 'science', 'text', 'language']
#
# Cálculo detallado para 'doc1':
# Total de términos: 7
# Frecuencias brutas:
#   machine: 1/7 = 0.143
#   learning: 2/7 = 0.286
#   data: 2/7 = 0.286
#   science: 1/7 = 0.143
#   algorithm: 1/7 = 0.143