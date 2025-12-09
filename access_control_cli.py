# access_control_cli.py
import cv2
import sys
import numpy as np
from arcface_implementation import ArcFaceSystem

def capture_face_and_register(system: ArcFaceSystem):
    """
    Función de registro que captura una imagen de la cara, detecta el rostro 
    y registra el embedding en la base de datos.
    """
    print("\n=== REGISTRO DE NUEVO USUARIO ===")
    username = input("Ingrese el nombre de usuario para registrar: ").strip()
    if not username:
        print("Nombre de usuario no válido. Cancelando.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara. Cancelando.")
        return

    print("Presione 's' para tomar la foto (debe aparecer un recuadro verde en su cara). Presione 'q' para salir.")
    captured_img = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            faces = system.app.get(frame)
            face_detected = False
            
            if faces:
                for face in faces:
                    face_detected = True
                    bbox = face.bbox.astype(np.int32)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    
            text = "Listo para capturar (Rostro detectado)" if face_detected else "Esperando Rostro..."
            color = (0, 255, 0) if face_detected else (255, 255, 255)
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imshow('Registro de Usuario - Camara', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and face_detected:
                captured_img = frame.copy()
                break
            elif key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if captured_img is not None:
        system.add_user(username, captured_img=captured_img)
    else:
        print("Captura cancelada o no se detectó un rostro al momento de capturar.")

def verify_access_cli(system: ArcFaceSystem):
    if not system.face_db:
        print("\n=== VERIFICACIÓN DE ACCESO ===")
        print("Error: La base de datos de usuarios está vacía. Registre un usuario primero (Opción 1).")
        return

    print("\n=== VERIFICACIÓN DE ACCESO ===")
    print("Mire a la cámara para ingresar. Presione 'q' para salir.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            faces = system.app.get(frame)
            input_embedding = faces[0].embedding if faces else None
            
            access_granted, user_match, similarity = system.verify_access(input_embedding)
            
            if input_embedding is not None and user_match:
                bbox = faces[0].bbox.astype(np.int32)
                
                color = (0, 255, 0) if access_granted else (0, 0, 255)
                text = f"{user_match} ({similarity:.2f})"
                
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(frame, text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                if access_granted:
                    status_text = f"ACCESO OTORGADO: {user_match}"
                else:
                    status_text = f"ACCESO DENEGADO (Similitud: {similarity:.4f})"
                
                cv2.putText(frame, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            elif input_embedding is not None:
                cv2.putText(frame, "ROSTRO DETECTADO, ACCESO DENEGADO", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Esperando Rostro...", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.imshow('Verificacion de Acceso', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def main_cli():
    system = ArcFaceSystem()
    
    while True:
        print("\n-------------------------------------")
        print(f"SISTEMA DE CONTROL DE ACCESO ARC-FACE (Usuarios: {len(system.face_db)})")
        print("-------------------------------------")
        print("1. Añadir/Registrar Usuario (Captura en tiempo real)")
        print("2. Ingresar/Verificar Acceso (Validación en tiempo real)")
        print("3. Calcular Métricas Reales (Requiere dataset LFW)")
        print("4. Salir")
        
        choice = input("Seleccione una opción (1-4): ").strip()

        if choice == '1':
            capture_face_and_register(system)
        elif choice == '2':
            verify_access_cli(system)
        elif choice == '3':
            print("\nEjecutando evaluación de métricas...")
           #num_pairs para prueba, modificar el valor 
            evaluated_results = system.evaluate_lfw_dataset(num_pairs=100)
            
            if evaluated_results:
                 accuracy, far, frr = system.compute_metrics(evaluated_results, threshold=system.DEFAULT_THRESHOLD)
                 print("\n=========================================================")
                 print(f"** RESULTADOS CALCULADOS (Umbral: {system.DEFAULT_THRESHOLD}) **")
                 print("=========================================================")
                 print(f"Precisión Global: {accuracy:.4f}")
                 print(f"Tasa de Falsa Aceptación (FAR): {far:.4f}")
                 print(f"Tasa de Falso Rechazo (FRR): {frr:.4f}")
                 print("=========================================================")
            else:
                 print("\nAdvertencia: No se pudo generar ningún par válido para la evaluación. Asegúrese de que el detector funcione con las imágenes.")

        elif choice == '4':
            print("Saliendo del sistema.")
            sys.exit(0)
        else:
            print("Opción no válida. Intente de nuevo.")

if __name__ == "__main__":
    main_cli()
