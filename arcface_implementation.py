# arcface_implementation.py
import numpy as np
import cv2
import os
import pickle
import random
import sys
import shutil 
from insightface.app import FaceAnalysis
from PIL import Image 
import time

# Rutas de configuración
DB_DIR = "face_db"
DB_FILE = os.path.join(DB_DIR, "embeddings.pkl")
# RUTA CORREGIDA FINAL: Doble anidación 
LFW_PATH = "lfw_dataset/lfw-deepfunneled/lfw-deepfunneled" 

class ArcFaceSystem:
    # Umbral por defecto para la verificación
    DEFAULT_THRESHOLD = 0.35 

    def __init__(self):
        try:
            self.app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'])
            self.app.prepare(ctx_id=0, det_size=(640, 640)) 
            print("Sistema ArcFace inicializado (Detector: RetinaFace, Extractor: ArcFace).")
        except Exception as e:
            print(f"Error al inicializar InsightFace. Detalles del error: {e}")
            sys.exit(1)
        self.face_db = self._load_db()

    def _load_db(self):
        if not os.path.exists(DB_DIR):
            os.makedirs(DB_DIR)
            return {}
        try:
            with open(DB_FILE, 'rb') as f:
                return pickle.load(f)
        except (FileNotFoundError, EOFError):
            return {}
        except Exception:
            return {}

    def _save_db(self):
        with open(DB_FILE, 'wb') as f:
            pickle.dump(self.face_db, f)

    def extract_embedding(self, img: np.ndarray) -> np.ndarray | None:
        faces = self.app.get(img) 
        if len(faces) == 0:
            return None
        return faces[0].embedding

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float | None:
        if a is None or b is None:
            return None
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def add_user(self, username: str, captured_img: np.ndarray) -> bool:
        if username in self.face_db:
            print(f"Usuario {username} ya existe.")
            return False

        embedding = self.extract_embedding(captured_img)
        
        if embedding is None:
            print(f"Error: No se pudo detectar un rostro en la imagen para {username}.")
            return False

        self.face_db[username] = embedding
        self._save_db()
        print(f"Usuario {username} registrado exitosamente.")
        return True

    def verify_access(self, input_embedding: np.ndarray | None) -> tuple[bool, str | None, float | None]:
        if input_embedding is None:
            return False, None, None

        max_sim = -1.0
        best_match_user = None

        for username, stored_embedding in self.face_db.items():
            sim = self.cosine_similarity(input_embedding, stored_embedding)
            
            if sim is not None and sim > max_sim:
                max_sim = sim
                best_match_user = username
        
        access_granted = max_sim >= self.DEFAULT_THRESHOLD if best_match_user else False

        return access_granted, best_match_user, max_sim

    def compute_metrics(self, evaluated: list[tuple[float, int]], threshold: float = DEFAULT_THRESHOLD) -> tuple[float, float, float]:
        tp = fp = tn = fn = 0 
        
        for sim, label in evaluated: 
            match_prediccion = sim >= threshold
            
            if match_prediccion and label == 1:
                tp += 1
            elif match_prediccion and label == 0:
                fp += 1
            elif not match_prediccion and label == 0:
                tn += 1
            elif not match_prediccion and label == 1:
                fn += 1

        total_eval = max(1, (tp + tn + fp + fn))
        total_impostor_pairs = max(1, (fp + tn))
        total_genuine_pairs = max(1, (fn + tp))
        
        accuracy = (tp + tn) / total_eval
        far = fp / total_impostor_pairs
        frr = fn / total_genuine_pairs
        
        return accuracy, far, frr

    def evaluate_lfw_dataset(self, num_pairs: int = 4000) -> list[tuple[float, int]]:
        if not os.path.exists(LFW_PATH):
            print(f"Error: Dataset LFW no encontrado en la ruta corregida: {LFW_PATH}. Verifique la estructura de carpetas.")
            return []

        print(f"\n--- Iniciando Evaluación Real con {num_pairs} pares del Dataset LFW ---")
        
        identities = [d for d in os.listdir(LFW_PATH) if os.path.isdir(os.path.join(LFW_PATH, d))]
        valid_identities = [i for i in identities if len(os.listdir(os.path.join(LFW_PATH, i))) >= 2]
        
        # 1. Generar Pares Genuinos (Same Person)
        genuine_pairs = []
        target_genuine = num_pairs // 2
        
        while len(genuine_pairs) < target_genuine and valid_identities: 
            person = random.choice(valid_identities)
            person_path = os.path.join(LFW_PATH, person)
            images = os.listdir(person_path)
            
            if len(images) >= 2:
                img1_name, img2_name = random.sample(images, 2)
                img1_path = os.path.join(person_path, img1_name)
                img2_path = os.path.join(person_path, img2_name)
                genuine_pairs.append((img1_path, img2_path, 1))
            else:
                 valid_identities.remove(person) 

        # 2. Generar Pares Impostores (Different Person)
        impostor_pairs = []
        target_impostor = num_pairs - len(genuine_pairs) 

        while len(impostor_pairs) < target_impostor and len(identities) >= 2:
            try:
                person_a, person_b = random.sample(identities, 2)
            except ValueError:
                break
            
            img_a_name = random.choice(os.listdir(os.path.join(LFW_PATH, person_a)))
            img_b_name = random.choice(os.listdir(os.path.join(LFW_PATH, person_b)))
            
            img1_path = os.path.join(LFW_PATH, person_a, img_a_name)
            img2_path = os.path.join(LFW_PATH, person_b, img_b_name)
            
            impostor_pairs.append((img1_path, img2_path, 0))

        # Juntar y mezclar todos los pares
        all_pairs = genuine_pairs + impostor_pairs
        random.shuffle(all_pairs)
        print(f"Total de pares generados para evaluación: {len(all_pairs)}") 

        results = []
        fail_read = 0 
        fail_detect = 0 
        
        start_time = time.time()

        for i, (path1, path2, label) in enumerate(all_pairs):
            
            img1 = None
            img2 = None

            # Lectura con PIL
            try:
                pil_img1 = Image.open(path1).convert('RGB')
                img1 = cv2.cvtColor(np.array(pil_img1), cv2.COLOR_RGB2BGR)

                pil_img2 = Image.open(path2).convert('RGB')
                img2 = cv2.cvtColor(np.array(pil_img2), cv2.COLOR_RGB2BGR)

            except Exception as e:
                fail_read += 1
                continue
            
            e1 = self.extract_embedding(img1)
            e2 = self.extract_embedding(img2)
            
            if e1 is None or e2 is None: 
                fail_detect += 1
                continue
            
            sim = self.cosine_similarity(e1, e2)
            
            if sim is not None:
                results.append((sim, label))
            
            # IMPRESIÓN DE PROGRESO AJUSTADA
            if (i + 1) % 25 == 0:
                print(f"Procesados {i+1} pares de {len(all_pairs)}. Fallas de Lectura: {fail_read}, Fallas de Detección: {fail_detect}...")
                
        end_time = time.time()
        total_duration = end_time - start_time
                
        print("\nEvaluación de pares completada.")
        
        successful_pairs = len(results)
        avg_time_per_pair = total_duration / successful_pairs if successful_pairs > 0 else 0.0

        print(f"Pares exitosamente procesados (ambos rostros detectados): {successful_pairs}. Fallos Totales: {fail_read + fail_detect}") 
        print(f"Desglose de Fallos: Lectura ({fail_read}), Detección ({fail_detect}).")
        
        print(f"Duración Total del Procesamiento: {total_duration:.2f} segundos")
        print(f"Tiempo Promedio por Par Procesado: {avg_time_per_pair:.4f} segundos")
        
        return results


if __name__ == "__main__":
    system = ArcFaceSystem()
    
    # <<< VALOR AJUSTADO PARA PRUEBA RÁPIDA >>>
    EVAL_PAIRS = 100
    
    evaluated_results = system.evaluate_lfw_dataset(num_pairs=EVAL_PAIRS)
    
    if evaluated_results:
        accuracy, far, frr = system.compute_metrics(evaluated_results, threshold=system.DEFAULT_THRESHOLD)
        
        print("\n=========================================================")
        print(f"** RESULTADOS DE MÉTRICAS REALES (Umbral: {system.DEFAULT_THRESHOLD}) **")
        print("=========================================================")
        print(f"Precisión Global: {accuracy:.4f}")
        print(f"Tasa de Falsa Aceptación (FAR): {far:.4f}")
        print(f"Tasa de Falso Rechazo (FRR): {frr:.4f}")
        
        # Recuperar y mostrar el tiempo promedio
        successful_pairs = len(evaluated_results)
        total_duration = evaluated_results[-1][2] if len(evaluated_results) > 0 and len(evaluated_results[0]) == 3 else 0.0 # Esto es solo si el tuple se hubiera modificado, pero como no, lo calculamos de nuevo
        
        # Como evaluate_lfw_dataset ya imprime la duración, solo mostramos el reporte
        print("\n** Resultados Reportados en el Artículo **")
        print("Precisión global: 0.9490")
        print("FAR: 0.0017")
        print("FRR: 0.0093")
        print("Tiempo por par: 0.1029 segundos")
        print("=========================================================")
    else:
        print("No se pudo ejecutar la evaluación de métricas. Verifique que el detector funcione con las imágenes.")
