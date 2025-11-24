import pandas as pd
from collections import Counter
import numpy as np
import math


class ModeloEspacioVectorialTFIDF:
    def __init__(self):
        self.documentos = {}
        self.matriz_tf = None
        self.matriz_tfidf = None
        self.terminos = set()
        self.nombres_docs = []
        self.idf = {}

    def agregar_documento(self, nombre, tokens):
        """Añade un documento a la colección"""
        self.documentos[nombre] = tokens
        self.terminos.update(tokens)
        self.nombres_docs.append(nombre)

    def calcular_tf(self, tokens):
        """Calcula frecuencias normalizadas para un documento"""
        total_terminos = len(tokens)
        frecuencias = Counter(tokens)

        tf_normalizado = {}
        for termino, freq in frecuencias.items():
            tf_normalizado[termino] = freq / total_terminos

        return tf_normalizado

    def calcular_idf(self):
        """Calcula IDF para todos los términos usando la fórmula suavizada"""
        N = len(self.documentos)  # Número total de documentos

        for termino in self.terminos:
            # df(t): número de documentos que contienen el término t
            df_t = sum(1 for doc_tokens in self.documentos.values() if termino in doc_tokens)

            # idf(t) = log(N / (df(t) + 1)) + 1 (suavizado)
            self.idf[termino] = math.log(N / (df_t + 1)) + 1

    def construir_matriz_tf(self):
        """Construye la matriz término-documento con TF normalizado"""
        if not self.documentos:
            return pd.DataFrame()

        matriz_data = {}

        for doc_name, tokens in self.documentos.items():
            tf_doc = self.calcular_tf(tokens)
            matriz_data[doc_name] = tf_doc

        self.matriz_tf = pd.DataFrame(matriz_data).fillna(0)
        self.matriz_tf = self.matriz_tf[self.nombres_docs]

        return self.matriz_tf

    def construir_matriz_tfidf(self):
        """Construye la matriz TF-IDF"""
        if self.matriz_tf is None:
            self.construir_matriz_tf()

        self.calcular_idf()

        # Crear matriz TF-IDF: tf-idf(d,t) = tf(d,t) × idf(t)
        self.matriz_tfidf = self.matriz_tf.copy()

        for termino in self.matriz_tfidf.index:
            if termino in self.idf:
                self.matriz_tfidf.loc[termino] = self.matriz_tfidf.loc[termino] * self.idf[termino]

        return self.matriz_tfidf

    def obtener_terminos_relevantes(self, doc_index, top_n=5, use_tfidf=True):
        """Devuelve los n términos más importantes para un documento"""
        if use_tfidf:
            if self.matriz_tfidf is None:
                self.construir_matriz_tfidf()
            matriz = self.matriz_tfidf
        else:
            if self.matriz_tf is None:
                self.construir_matriz_tf()
            matriz = self.matriz_tf

        if doc_index >= len(self.nombres_docs):
            raise ValueError("Índice de documento fuera de rango")

        doc_name = self.nombres_docs[doc_index]

        terminos_relevantes = (
            matriz[doc_name]
            .sort_values(ascending=False)
            .head(top_n)
        )

        return terminos_relevantes

    def obtener_estadisticas_idf(self):
        """Devuelve estadísticas del cálculo IDF"""
        if not self.idf:
            self.calcular_idf()

        return pd.Series(self.idf).sort_values()


# Sistema extendido con TF-IDF
class SistemaProcesamientoTextoAvanzado:
    def __init__(self):
        self.procesador = ProcesadorAvanzado()
        self.modelo = ModeloEspacioVectorialTFIDF()
        self.documentos_originales = {}
        self.emails_por_documento = {}

    def agregar_documento(self, nombre, texto):
        """Agrega un documento al sistema"""
        self.documentos_originales[nombre] = texto

        # Procesar el texto
        tokens_lematizados, emails = self.procesador.limpiar_y_lematizar(texto)

        # Generar bigramas
        bigramas = self.procesador.generar_bigramas(tokens_lematizados)

        # Combinar tokens simples y bigramas
        tokens_completos = tokens_lematizados + bigramas

        # Agregar emails como términos especiales
        tokens_completos.extend(emails)

        # Guardar emails para reporte
        self.emails_por_documento[nombre] = emails

        # Agregar al modelo vectorial
        self.modelo.agregar_documento(nombre, tokens_completos)

    def generar_reporte_completo(self):
        """Genera el reporte completo con comparación TF vs TF-IDF"""
        print("=== SISTEMA AVANZADO DE PROCESAMIENTO DE TEXTO (TF-IDF) ===")
        print("=" * 70)

        # 1. Mostrar emails detectados
        print("\n1. EMAILS DETECTADOS POR DOCUMENTO:")
        print("-" * 40)
        for doc_name, emails in self.emails_por_documento.items():
            print(f"📧 {doc_name}: {emails if emails else 'No se detectaron emails'}")

        # 2. Construir matrices
        matriz_tf = self.modelo.construir_matriz_tf()
        matriz_tfidf = self.modelo.construir_matriz_tfidf()

        # 3. Mostrar comparación TF vs TF-IDF
        print(f"\n2. COMPARACIÓN: TÉRMINOS MÁS RELEVANTES (TF vs TF-IDF):")
        print("-" * 65)

        for i, doc_name in enumerate(self.modelo.nombres_docs):
            print(f"\n📄 DOCUMENTO: {doc_name}")

            print("   TF (Frecuencia Normalizada):")
            terminos_tf = self.modelo.obtener_terminos_relevantes(i, 5, use_tfidf=False)
            for j, (termino, peso) in enumerate(terminos_tf.items(), 1):
                print(f"      {j}. {termino}: {peso:.4f}")

            print("   TF-IDF (Ponderación Global):")
            terminos_tfidf = self.modelo.obtener_terminos_relevantes(i, 5, use_tfidf=True)
            for j, (termino, peso) in enumerate(terminos_tfidf.items(), 1):
                print(f"      {j}. {termino}: {peso:.4f}")

        # 4. Mostrar matrices
        print(f"\n3. MATRIZ TF (Frecuencias Normalizadas):")
        print("-" * 45)
        print(f"Dimensiones: {matriz_tf.shape[0]} términos × {matriz_tf.shape[1]} documentos")
        print("\nMatriz TF (primeras 12 filas):")
        print(matriz_tf.head(12).round(4))

        print(f"\n4. MATRIZ TF-IDF (Ponderación Global):")
        print("-" * 45)
        print(f"Dimensiones: {matriz_tfidf.shape[0]} términos × {matriz_tfidf.shape[1]} documentos")
        print("\nMatriz TF-IDF (primeras 12 filas):")
        print(matriz_tfidf.head(12).round(4))

        # 5. Mostrar estadísticas IDF
        print(f"\n5. ESTADÍSTICAS IDF (Inverse Document Frequency):")
        print("-" * 50)
        estadisticas_idf = self.modelo.obtener_estadisticas_idf()
        print("Valores IDF para algunos términos:")
        print(estadisticas_idf.head(10))
        print("...")
        print(estadisticas_idf.tail(10))

        # 6. Análisis de términos únicos vs comunes
        print(f"\n6. ANÁLISIS DE TÉRMINOS:")
        print("-" * 30)

        # Términos con IDF alto (términos únicos)
        terminos_unicos = estadisticas_idf[estadisticas_idf > 1.2].index.tolist()
        print(f"Términos más únicos (IDF > 1.2): {terminos_unicos[:10]}")

        # Términos con IDF bajo (términos comunes)
        terminos_comunes = estadisticas_idf[estadisticas_idf < 0.8].index.tolist()
        print(f"Términos más comunes (IDF < 0.8): {terminos_comunes[:10]}")

        return {
            'matriz_tf': matriz_tf,
            'matriz_tfidf': matriz_tfidf,
            'estadisticas_idf': estadisticas_idf,
            'terminos_relevantes_tf': {
                doc: self.modelo.obtener_terminos_relevantes(i, 5, False)
                for i, doc in enumerate(self.modelo.nombres_docs)
            },
            'terminos_relevantes_tfidf': {
                doc: self.modelo.obtener_terminos_relevantes(i, 5, True)
                for i, doc in enumerate(self.modelo.nombres_docs)
            }
        }


# CASO DE PRUEBA CON TF-IDF
if __name__ == "__main__":
    # Crear sistema avanzado
    sistema_avanzado = SistemaProcesamientoTextoAvanzado()

    # Documentos de prueba (mismos del ejercicio anterior para comparación)
    documentos = {
        "tecnologia": """
        El machine learning y la inteligencia artificial están transformando la industria. 
        Los algoritmos de deep learning permiten reconocimiento de imágenes avanzado. 
        Para consultas técnicas contactar a: soporte@techcompany.com 
        La computación en la nube y big data son esenciales para el análisis de datos.
        Python se ha convertido en el lenguaje preferido para data science.
        El machine learning requiere grandes cantidades de datos de calidad.
        """,

        "salud": """
        La medicina preventiva y los avances en telemedicina mejoran la calidad de vida. 
        Investigadores en genómica estudian terapias personalizadas contra el cáncer.
        Contacto para estudios clínicos: estudios@hospitalmoderno.org
        La nutrición balanceada y ejercicio regular previenen enfermedades cardiovasculares.
        La salud mental es igual de importante que la física.
        La medicina moderna utiliza machine learning para diagnóstico temprano.
        """,

        "educacion": """
        La educación online ha revolucionado el acceso al conocimiento global. 
        Las plataformas de e-learning permiten aprendizaje personalizado adaptado a cada estudiante.
        Información sobre cursos: info@academiadigital.edu
        La gamificación y realidad aumentada crean experiencias educativas inmersivas.
        El desarrollo de habilidades digitales es crucial para el futuro laboral.
        El machine learning se está incorporando en herramientas educativas.
        """
    }

    # Procesar documentos
    for nombre, texto in documentos.items():
        sistema_avanzado.agregar_documento(nombre, texto)

    # Generar reporte completo con TF-IDF
    reporte = sistema_avanzado.generar_reporte_completo()

    # Ejemplo de cálculo manual para demostración
    print("\n" + "=" * 70)
    print("DEMOSTRACIÓN DEL CÁLCULO TF-IDF:")
    print("=" * 70)

    # Mostrar cálculo paso a paso para un término
    termino_ejemplo = "aprendizaje_automatizado"  # machine learning en español lematizado
    if termino_ejemplo in sistema_avanzado.modelo.idf:
        N = len(sistema_avanzado.modelo.documentos)
        df_t = sum(1 for doc_tokens in sistema_avanzado.modelo.documentos.values()
                   if termino_ejemplo in doc_tokens)

        print(f"\nCálculo para el término: '{termino_ejemplo}'")
        print(f"N (total documentos) = {N}")
        print(f"df(t) (documentos con el término) = {df_t}")
        print(
            f"IDF(t) = log({N} / ({df_t} + 1)) + 1 = log({N}/{df_t + 1}) + 1 = {sistema_avanzado.modelo.idf[termino_ejemplo]:.4f}")

        # Mostrar TF en cada documento
        print("\nTF en cada documento:")
        for doc_name in sistema_avanzado.modelo.nombres_docs:
            tf_val = sistema_avanzado.modelo.matriz_tf.loc[
                termino_ejemplo, doc_name] if termino_ejemplo in sistema_avanzado.modelo.matriz_tf.index else 0
            tfidf_val = sistema_avanzado.modelo.matriz_tfidf.loc[
                termino_ejemplo, doc_name] if termino_ejemplo in sistema_avanzado.modelo.matriz_tfidf.index else 0
            print(f"  {doc_name}: TF = {tf_val:.4f}, TF-IDF = {tfidf_val:.4f}")