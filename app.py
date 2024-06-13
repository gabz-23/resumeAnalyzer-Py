from flask import Flask, render_template, request, redirect, url_for, flash
from utils.utils import cleanResume, extractTextPDF, extractTextWord, eval_curriculum
from werkzeug.utils import secure_filename
import pickle
import os


app = Flask(__name__)

# Cargar el modelo y el vectorizador TF-IDF
with open("clf.pkl", "rb") as f:
    clf = pickle.load(f)
with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "docx"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SECRET_KEY"] = "xxxxxx"

dicionario = {
    25: "programacion",
    23: "procesamiento-datos",
    35: "teoria-grafos",
    1: "analisis-sistemas",
    36: "teoria-sistemas",
    4: "base-datos",
    2: "arquitectura-computador",
    8: "diseno-sistemas",
    31: "sistemas-operativos",
    15: "implantacion-sistemas",
    29: "simulacion-modelo",
    27: "redes",
    17: "inteligencia-artificial",
    3: "auditoria-sistemas",
    33: "teleprocesos",
    13: "gerencia-proyectos",
    9: "educacion-ambiental",
    14: "hombre-sociedad-ciencia-tecnologia",
    7: "dibujo-tecnico",
    11: "geometria-analitica",
    16: "ingles",
    19: "matematica",
    28: "seminario",
    0: "algebra-lineal",
    6: "defensa-integral-nacion",
    10: "fisica",
    26: "quimica",
    22: "probabilidad-estadistica",
    34: "teoria-desiciones",
    30: "sistemas-administrativos",
    5: "calculo-numerico",
    18: "logica-matematica",
    32: "sistemas-produccion",
    21: "optimizacion-no-lineal",
    24: "procesos-estocasticos",
    20: "metodologia-investigacion",
    12: "gerencia-informatica",
}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/", methods=["POST"])
def curriculum():

    # Obtener el archivo subido
    curriculum_file = request.files.getlist("file")

    # Validar que se haya subido un archivo
    for file in curriculum_file:
        if not file:
            flash("No se ha subido ningun archivo", "error")
            return redirect(url_for("index"))
        else:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            else:
                flash(
                    f"La extensión del archivo {file.filename} no esta permitida",
                    "error",
                )
                return redirect(url_for("index"))

    # Curriclums procesados
    curriculums = []

    # Procesar cada currículum
    for file in curriculum_file:
        fileExtension = filename.rsplit(".", 1)[1].lower()

        # Extraer y limpiar el texto del currículum
        if fileExtension == "pdf":
            resumeText = extractTextPDF(file.filename)
            cleanText = cleanResume(resumeText)
        elif fileExtension == "docx":
            resumeText = extractTextWord(file.filename)
            cleanText = cleanResume(resumeText)

        # Vectorizar el currículum limpio (utilizando el vectorizador existente)
        resumeVectorized = tfidf.transform([cleanText])

        # Realizar la predicción (utilizando el modelo existente)
        materia = clf.predict(resumeVectorized)[0]

        # Obtener las probabilidades posteriores de cada clase (materia)
        probabilidades = clf.predict_proba(resumeVectorized)[0]

        # Encontrar la materia con mayor probabilidad y su valor
        probabilidad_mas_probable = round(probabilidades.max() * 100, 1)

        curriculums.append(
            {
                "materia": dicionario[materia],
                "probabilidad": probabilidad_mas_probable,
                "archivo": file.filename,
            }
        )

    return render_template(
        "index.html",
        curriculums=curriculums,
    )
