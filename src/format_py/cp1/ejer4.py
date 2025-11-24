import re
import pandas as pd
from collections import Counter
import spacy
from nltk.util import ngrams
from nltk.corpus import stopwords as nltk_stopwords
import nltk

# Descargar stopwords si no están disponibles
try:
    nltk_stopwords.words('spanish')
    nltk_stopwords.words('english')
except:
    nltk.download('stopwords')


class ProcesadorAvanzado:
    def __init__(self):
        self.nlp = spacy.load("es_core_news_sm")
        self.stopwords_es = set(nltk_stopwords.words('spanish'))
        self.stopwords_en = set(nltk_stopwords.words('english'))
        self.stopwords = self.stopwords_es.union(self.stopwords_en)

    def extraer_emails(self, texto):
        patron = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(patron, texto)

    def limpiar_y_lematizar(self, texto):
        # Extraer emails primero y preservarlos
        emails = self.extraer_emails(texto)

        # Remover emails del texto para procesamiento normal
        texto_sin_emails = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', texto)

        # Procesar con spaCy
        doc = self.nlp(texto_sin_emails.lower())

        # Lematizar y filtrar
        tokens_lematizados = []
        for token in doc:
            if (token.is_alpha and
                    not token.is_stop and
                    token.text not in self.stopwords and
                    len(token.text) > 2):
                tokens_lematizados.append(token.lemma_)

        return tokens_lematizados, emails

    def generar_bigramas(self, tokens):
        return ['_'.join(bg) for bg in ngrams(tokens, 2)]

    def generar_trigramas(self, tokens):
        return ['_'.join(tg) for tg in ngrams(tokens, 3)]


class ModeloEspacioVectorial:
    def __init__(self):
        self.documentos = {}
        self.matriz_tf = None
        self.terminos = set()
        self.nombres_docs = []

    def agregar_documento(self, nombre, tokens):
        self.documentos[nombre] = tokens
        self.terminos.update(tokens)
        self.nombres_docs.append(nombre)

    def calcular_tf(self, tokens):
        total_terminos = len(tokens)
        frecuencias = Counter(tokens)

        tf_normalizado = {}
        for termino, freq in frecuencias.items():
            tf_normalizado[termino] = freq / total_terminos

        return tf_normalizado

    def construir_matriz(self):
        if not self.documentos:
            return pd.DataFrame()

        matriz_data = {}

        for doc_name, tokens in self.documentos.items():
            tf_doc = self.calcular_tf(tokens)
            matriz_data[doc_name] = tf_doc

        self.matriz_tf = pd.DataFrame(matriz_data).fillna(0)
        self.matriz_tf = self.matriz_tf[self.nombres_docs]

        return self.matriz_tf

    def obtener_terminos_relevantes(self, doc_index, top_n=5):
        if self.matriz_tf is None:
            self.construir_matriz()

        if doc_index >= len(self.nombres_docs):
            raise ValueError("Índice de documento fuera de rango")

        doc_name = self.nombres_docs[doc_index]

        terminos_relevantes = (
            self.matriz_tf[doc_name]
            .sort_values(ascending=False)
            .head(top_n)
        )

        return terminos_relevantes


class SistemaProcesamientoTexto:
    def __init__(self):
        self.procesador = ProcesadorAvanzado()
        self.modelo = ModeloEspacioVectorial()
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

    def generar_reporte(self):
        """Genera el reporte completo del sistema"""
        print("=== SISTEMA DE PROCESAMIENTO DE TEXTO ===")
        print("=" * 60)

        # 1. Mostrar emails detectados
        print("\n1. EMAILS DETECTADOS POR DOCUMENTO:")
        print("-" * 40)
        for doc_name, emails in self.emails_por_documento.items():
            print(f"📧 {doc_name}: {emails if emails else 'No se detectaron emails'}")

        # 2. Construir matriz y mostrar términos relevantes
        matriz = self.modelo.construir_matriz()

        print(f"\n2. TÉRMINOS MÁS RELEVANTES POR DOCUMENTO (Top 5):")
        print("-" * 55)

        for i, doc_name in enumerate(self.modelo.nombres_docs):
            print(f"\n📄 DOCUMENTO: {doc_name}")
            print("Términos más relevantes:")
            terminos_relevantes = self.modelo.obtener_terminos_relevantes(i, 5)

            for j, (termino, peso) in enumerate(terminos_relevantes.items(), 1):
                print(f"   {j}. {termino}: {peso:.4f}")

        # 3. Mostrar matriz completa
        print(f"\n3. MATRIZ TÉRMINO-DOCUMENTO COMPLETA:")
        print("-" * 40)
        print(f"Dimensiones: {matriz.shape[0]} términos × {matriz.shape[1]} documentos")
        print("\nMatriz (primeras 15 filas):")
        print(matriz.head(15).round(4))

        # 4. Estadísticas generales
        print(f"\n4. ESTADÍSTICAS DEL SISTEMA:")
        print("-" * 30)
        print(f"Total de documentos procesados: {len(self.documentos_originales)}")
        print(f"Total de términos únicos: {len(self.modelo.terminos)}")
        print(f"Total de emails detectados: {sum(len(emails) for emails in self.emails_por_documento.values())}")

        return {
            'matriz': matriz,
            'terminos_relevantes': {
                doc: self.modelo.obtener_terminos_relevantes(i, 5)
                for i, doc in enumerate(self.modelo.nombres_docs)
            },
            'emails': self.emails_por_documento
        }


# CASO DE PRUEBA
if __name__ == "__main__":
    # Crear sistema
    sistema = SistemaProcesamientoTexto()

    # Documentos de prueba sobre diferentes temas
    documentos = {
        "tecnologia": """
        El machine learning y la inteligencia artificial están transformando la industria. 
        Los algoritmos de deep learning permiten reconocimiento de imágenes avanzado. 
        Para consultas técnicas contactar a: soporte@techcompany.com 
        La computación en la nube y big data son esenciales para el análisis de datos.
        Python se ha convertido en el lenguaje preferido para data science.
        """,

        "salud": """
        La medicina preventiva y los avances en telemedicina mejoran la calidad de vida. 
        Investigadores en genómica estudian terapias personalizadas contra el cáncer.
        Contacto para estudios clínicos: estudios@hospitalmoderno.org
        La nutrición balanceada y ejercicio regular previenen enfermedades cardiovasculares.
        La salud mental es igual de importante que la física.
        """,

        "educacion": """
        La educación online ha revolucionado el acceso al conocimiento global. 
        Las plataformas de e-learning permiten aprendizaje personalizado adaptado a cada estudiante.
        Información sobre cursos: info@academiadigital.edu
        La gamificación y realidad aumentada crean experiencias educativas inmersivas.
        El desarrollo de habilidades digitales es crucial para el futuro laboral.
        """
    }

    # Procesar documentos
    for nombre, texto in documentos.items():
        sistema.agregar_documento(nombre, texto)

    # Generar reporte completo
    reporte = sistema.generar_reporte()

    # Información adicional para análisis
    print("\n" + "=" * 60)
    print("INFORMACIÓN ADICIONAL PARA ANÁLISIS:")
    print("=" * 60)

    # Mostrar algunos bigramas detectados
    print("\nEjemplos de bigramas generados:")
    for doc_name in documentos.keys():
        texto = documentos[doc_name]
        tokens, _ = sistema.procesador.limpiar_y_lematizar(texto)
        bigramas = sistema.procesador.generar_bigramas(tokens)
        print(f"\n{doc_name}: {bigramas[:5]}...")









