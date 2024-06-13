import fitz
import docx2txt
import nltk
import re
from nltk.corpus import stopwords
import numpy as np

nltk.download("stopwords")


# Extraer el texto de un pdf
def extractTextPDF(filename):
    text = ""
    with fitz.open(f"uploads/{filename}") as pdf:
        for page in pdf:
            text += page.get_text()
    return text


# Extraer el texto de un docx
def extractTextWord(filename):
    text = docx2txt.process(f"uploads/{filename}")
    return text


# Limpiar Curriculum
def cleanResume(resumeText):
    resumeText = str(resumeText)
    resumeText = re.sub(r"http\S+\s*|https\S+\s", "", resumeText)  # Elimina Urls
    resumeText = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "", resumeText)  # Elimina Emails
    resumeText = re.sub(
        r"[^\w\s-]", "", resumeText
    )  # Elimina todos los signos de puntuacion menos el guion
    resumeText = re.sub(r"\n", " ", resumeText)  # Elimina los saltos de linea
    resumeText = re.sub(r"V-\d{8}", "", resumeText)  # Eliminar cedula
    resumeText = re.sub(
        r"(0412|0414|0424|0416|0212)\d{7}", "", resumeText
    )  # Eliminar numeros de telefono

    # Convertir todo a minusculas
    resumeText = resumeText.lower()

    # Eliminar palabras vacias (stopwords)
    stopWords = set(stopwords.words("spanish"))
    words = resumeText.split()
    resumeText = " ".join([word for word in words if word not in stopWords])

    return resumeText


def eval_curriculum(cleanText, subjectSelected, clf, tfidf):
    # Vectorización del currículum
    input_features = tfidf.transform(cleanText)

    # Vectorización de la materia
    subject_tfidf = tfidf.transform([subjectSelected])

    # Combinar las características de los currículos y las materias
    input_features = np.hstack((input_features.toarray(), subject_tfidf.toarray()))

    # Predicción de la calidad
    try:
        calidad_predicha = clf.predict(input_features)[0]
    except Exception as e:
        return "Error", f"Error en la predicción: {e}", 0

    # Mapeo de etiquetas numéricas a etiquetas de calidad
    etiquetas_calidad = {0: "Bajo", 1: "Regular", 2: "Alto"}
    calidad_texto = etiquetas_calidad[calidad_predicha]

    # Explicación de la predicción (puedes personalizarla)
    explicacion = f"El currículum para la materia '{subjectSelected}' se clasifica como '{calidad_texto}'"

    return calidad_texto, explicacion
