import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# 1. Descarga segura de recursos lingüísticos
nltk.download('stopwords', quiet=True)

# Cargar modelo de spaCy optimizado (deshabilitando componentes innecesarios para mayor velocidad)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def text_preprocessing(text):
    """
    Función de limpieza y lematización B2B.
    Procesa textos operativos eliminando ruido y reduciendo palabras a su raíz léxica.
    """
    doc = nlp(text)
    # Lematización excluyendo stopwords nativas de spaCy
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return ' '.join(tokens)

def prep_data(filepath='data/imdb_reviews.tsv'):
    """
    Pipeline de preparación de datos: Carga, Limpieza y Vectorización.
    """
    print("Iniciando carga de base de datos de textos operativos...")
    df = pd.read_csv(filepath, sep='\t', dtype={'votes': 'Int64'})

    # Separación estricta de lotes para evitar Fuga de Datos (Data Leakage)
    df_train = df[df['ds_part'] == 'train'].copy()
    df_test = df[df['ds_part'] == 'test'].copy()

    print("Lematizando corpus de entrenamiento (IQC - Control de Calidad de Entrada)...")
    # Nota: Si tu equipo es lento, esta parte tomará unos minutos.
    df_train['clean_text'] = df_train['review'].apply(text_preprocessing)
    
    print("Lematizando corpus de validación (Test)...")
    df_test['clean_text'] = df_test['review'].apply(text_preprocessing)

    print("Vectorizando textos operativos (TF-IDF)...")
    stop_words = set(stopwords.words('english'))
    tfidf_vect = TfidfVectorizer(stop_words=list(stop_words))

    # PREVENCIÓN DE FUGAS: fit_transform SOLO en entrenamiento
    X_train = tfidf_vect.fit_transform(df_train['clean_text'])
    # Transform en prueba usando las reglas aprendidas
    X_test = tfidf_vect.transform(df_test['clean_text'])

    y_train = df_train['pos']
    y_test = df_test['pos']

    # Persistencia del vectorizador para despliegue en producción (API)
    joblib.dump(tfidf_vect, 'src/tfidf_vectorizer.joblib')
    print("Activos de vectorización guardados exitosamente (.joblib).")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Prueba local rápida para verificar que el pipeline compila
    prep_data()