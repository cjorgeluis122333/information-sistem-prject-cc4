import re
from nltk.util import ngrams
import spacy
import subprocess
import sys


def verificar_instalar_spacy_model():
    """Verifica e instala el modelo de spaCy si es necesario"""
    try:
        nlp = spacy.load("es_core_news_sm")
        return nlp
    except OSError:
        print("Modelo es_core_news_sm no encontrado. Instalando...")
        try:
            # Intentar instalar el modelo
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "es_core_news_sm"])
            nlp = spacy.load("es_core_news_sm")
            print("Modelo instalado correctamente.")
            return nlp
        except Exception as e:
            print(f"Error instalando el modelo: {e}")
            print("\nSOLUCIÓN ALTERNATIVA:")
            print("1. Ejecuta en la terminal: python -m spacy download es_core_news_sm")
            print(
                "2. O instala con: pip install https://github.com/explosion/spacy-models/releases/download/es_core_news_sm-3.7.0/es_core_news_sm-3.7.0-py3-none-any.whl")
            return None


class ProcesadorAvanzado:
    def __init__(self):
        self.nlp = verificar_instalar_spacy_model()
        if self.nlp is None:
            raise Exception("No se pudo cargar el modelo de spaCy. Sigue las instrucciones de instalación.")

    def extraer_emails(self, texto):
        patron = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(patron, texto)

    def generar_ngramas(self, tokens, n):
        return list(ngrams(tokens, n))

    def lematizar(self, texto):
        if self.nlp is None:
            return texto.lower().split()  # Fallback básico
        doc = self.nlp(texto)
        return [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]

    def reconocer_entidades(self, texto):
        if self.nlp is None:
            return []  # Fallback si no hay modelo
        doc = self.nlp(texto)
        return [(ent.text, ent.label_) for ent in doc.ents]


# Ejemplo de uso
if __name__ == "__main__":
    try:
        procesador = ProcesadorAvanzado()

        texto_ejemplo = "Contacte a María Pérez en maria.perez@empresa.com. Trabaja en Google España."

        # Extraer emails
        emails = procesador.extraer_emails(texto_ejemplo)
        print("Emails detectados:", emails)

        # Lematización
        tokens_lematizados = procesador.lematizar(texto_ejemplo)
        print("Tokens lematizados:", tokens_lematizados)

        # Generar n-gramas
        bigramas = procesador.generar_ngramas(tokens_lematizados, 2)
        print("Bigramas:", bigramas)

        # Reconocer entidades
        entidades = procesador.reconocer_entidades(texto_ejemplo)
        print("Entidades detectadas:", entidades)

    except Exception as e:
        print(f"Error: {e}")
        print("\n--- INSTRUCCIONES DE INSTALACIÓN MANUAL ---")
        print("1. Abre tu terminal/command prompt")
        print("2. Activa tu entorno virtual si tienes uno")
        print("3. Ejecuta: python -m spacy download es_core_news_sm")
        print("4. Si falla, prueba con:")
        print("   pip install -U spacy")
        print("   python -m spacy download es_core_news_sm")