import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


class ProcesadorTextoBasico:
    def __init__(self):
        # Descargar recursos necesarios de NLTK
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

        # Inicializar componentes para español
        self.stemmer = SnowballStemmer('spanish')
        self.stop_words = set(stopwords.words('spanish'))
        self.puntuacion = set(string.punctuation)

    def procesar_texto(self, texto):
        # Tokenización y conversión a minúsculas
        tokens = word_tokenize(texto, language='spanish')
        tokens_minusculas = [token.lower() for token in tokens]

        # Filtrar puntuación y stopwords
        tokens_limpios = [
            token for token in tokens_minusculas
            if token not in self.puntuacion and token not in self.stop_words
        ]

        # Aplicar stemming
        stems = [self.stemmer.stem(token) for token in tokens_limpios]

        return {
            'tokens_originales': tokens_minusculas,
            'tokens_limpios': tokens_limpios,
            'stems': stems
        }


# Ejemplo de uso
if __name__ == "__main__":
    procesador = ProcesadorTextoBasico()
    texto_ejemplo = "¡Hola mundo! Este es un ejemplo de análisis de texto con NLTK."
    resultado = procesador.procesar_texto(texto_ejemplo)

    print("Tokens originales:", resultado['tokens_originales'])
    print("Tokens limpios:", resultado['tokens_limpios'])
    print("Stems:", resultado['stems'])
