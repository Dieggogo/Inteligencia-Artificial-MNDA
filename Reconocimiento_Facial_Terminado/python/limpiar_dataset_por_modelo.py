import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SIZE = (160, 160)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def importar_tensorflow():
    try:
        import tensorflow as tf
    except ImportError:
        print("Falta la dependencia 'tensorflow'.")
        print("Instalala con: py -3.13 -m pip install -r requirements.txt")
        sys.exit(1)
    return tf


def resolver_ruta(valor):
    ruta = Path(valor)
    if ruta.is_absolute():
        return ruta
    return PROJECT_ROOT / ruta


def cargar_labels(labels_path):
    with labels_path.open("r", encoding="utf-8") as archivo:
        data = json.load(archivo)
    class_names = data.get("class_names", [])
    if not class_names:
        print("labels.json no contiene class_names.")
        sys.exit(1)
    return class_names


def cargar_modelo(tf, modelo_path):
    try:
        return tf.keras.models.load_model(modelo_path, compile=False)
    except TypeError as error:
        mensaje = str(error)
        if "preprocess_input" not in mensaje:
            raise
        return tf.keras.models.load_model(
            modelo_path,
            compile=False,
            safe_mode=False,
            custom_objects={
                "preprocess_input": tf.keras.applications.resnet50.preprocess_input,
            },
        )


def crear_parser():
    parser = argparse.ArgumentParser(
        description="Detecta imagenes sospechosas por etiqueta incorrecta usando un modelo entrenado."
    )
    parser.add_argument("--dataset", default="Dataset_equipo_v6")
    parser.add_argument("--modelo", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--salida", default="Dataset_sospechoso")
    parser.add_argument(
        "--umbral-mismatch",
        type=float,
        default=0.70,
        help="Si predice otra clase con confianza >= este valor, se marca sospechosa.",
    )
    parser.add_argument(
        "--umbral-baja-confianza",
        type=float,
        default=0.45,
        help="Si la confianza top1 es menor, se marca sospechosa por baja calidad/ambiguedad.",
    )
    parser.add_argument(
        "--mover",
        action="store_true",
        help="Mueve archivos sospechosos a la carpeta salida. Por defecto solo reporte.",
    )
    return parser


def listar_imagenes_por_clase(dataset_dir):
    for clase_dir in sorted(dataset_dir.iterdir()):
        if not clase_dir.is_dir():
            continue
        clase = clase_dir.name
        for archivo in sorted(clase_dir.iterdir()):
            if archivo.is_file() and archivo.suffix.lower() in IMAGE_EXTENSIONS:
                yield clase, archivo


def cargar_rostro(path_imagen):
    imagen_bgr = cv2.imread(str(path_imagen))
    if imagen_bgr is None:
        return None
    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(imagen_rgb, IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)


def predecir(model, rostro_rgb):
    entrada = np.expand_dims(rostro_rgb.astype(np.float32), axis=0)
    pred = model.predict(entrada, verbose=0)[0]
    idx = int(np.argmax(pred))
    conf = float(pred[idx])
    return idx, conf, pred


def mover_archivo(origen, dataset_dir, salida_dir, motivo):
    relativo = origen.relative_to(dataset_dir)
    destino = salida_dir / motivo / relativo
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(origen), str(destino))


def main():
    args = crear_parser().parse_args()
    dataset_dir = resolver_ruta(args.dataset)
    modelo_path = resolver_ruta(args.modelo)
    labels_path = resolver_ruta(args.labels)
    salida_dir = resolver_ruta(args.salida)

    if not dataset_dir.is_dir():
        print(f"No existe dataset: {dataset_dir}")
        sys.exit(1)
    if not modelo_path.is_file():
        print(f"No existe modelo: {modelo_path}")
        sys.exit(1)
    if not labels_path.is_file():
        print(f"No existe labels: {labels_path}")
        sys.exit(1)

    tf = importar_tensorflow()
    class_names = cargar_labels(labels_path)
    class_to_idx = {nombre: i for i, nombre in enumerate(class_names)}
    model = cargar_modelo(tf, modelo_path)

    total = 0
    sospechosas_mismatch = []
    sospechosas_baja_conf = []
    errores_lectura = 0

    for clase_real, path_imagen in listar_imagenes_por_clase(dataset_dir):
        if clase_real not in class_to_idx:
            continue
        rostro = cargar_rostro(path_imagen)
        if rostro is None:
            errores_lectura += 1
            continue
        total += 1
        idx_pred, conf, pred = predecir(model, rostro)
        clase_pred = class_names[idx_pred]
        idx_real = class_to_idx[clase_real]
        conf_real = float(pred[idx_real])

        if clase_pred != clase_real and conf >= args.umbral_mismatch:
            sospechosas_mismatch.append((path_imagen, clase_real, clase_pred, conf, conf_real))
        elif conf < args.umbral_baja_confianza:
            sospechosas_baja_conf.append((path_imagen, clase_real, clase_pred, conf, conf_real))

    print(f"Total imagenes evaluadas: {total}")
    print(f"Errores de lectura: {errores_lectura}")
    print(f"Sospechosas por mismatch fuerte: {len(sospechosas_mismatch)}")
    print(f"Sospechosas por baja confianza: {len(sospechosas_baja_conf)}")
    print()

    if sospechosas_mismatch:
        print("Top 20 mismatch fuerte:")
        for item in sospechosas_mismatch[:20]:
            path_imagen, real, pred, conf, conf_real = item
            print(
                f"- {path_imagen} | real={real} pred={pred} "
                f"conf_pred={conf:.2%} conf_real={conf_real:.2%}"
            )
        print()

    if sospechosas_baja_conf:
        print("Top 20 baja confianza:")
        for item in sospechosas_baja_conf[:20]:
            path_imagen, real, pred, conf, conf_real = item
            print(
                f"- {path_imagen} | real={real} pred={pred} "
                f"conf_pred={conf:.2%} conf_real={conf_real:.2%}"
            )
        print()

    if args.mover:
        for path_imagen, *_ in sospechosas_mismatch:
            if path_imagen.exists():
                mover_archivo(path_imagen, dataset_dir, salida_dir, "mismatch_fuerte")
        for path_imagen, *_ in sospechosas_baja_conf:
            if path_imagen.exists():
                mover_archivo(path_imagen, dataset_dir, salida_dir, "baja_confianza")
        print(f"Imagenes movidas a: {salida_dir}")
    else:
        print("Modo reporte: no se movio ningun archivo.")
        print("Si quieres moverlas: agrega --mover")


if __name__ == "__main__":
    main()
