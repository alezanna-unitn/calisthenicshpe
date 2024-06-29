import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import csv
import json
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import math

model_path = r'C:\Users\aless\TestMediapipe\mediapipe\examples\pose_landmarker\python\pose_landmarker_lite.task'
video_path = r'C:\Users\aless\TestMediapipe\mediapipe\examples\pose_landmarker\python\setconpersona.mp4'

# Define the tolerance for horizontal and vertical alignment
horizontal_tolerance = 80
vertical_tolerance = 80

# Define the tolerance for squad position
squad_x_tolerance = 80
squad_y_tolerance = 80

def determine_pose(right_hip, right_ankle, right_shoulder, horizontal_tolerance, vertical_tolerance, squad_x_tolerance, squad_y_tolerance): #DA RIVEDERE (LA SQUADRA MI DA SEMPRE VERTICALE)
    
        # Check for horizontal alignment
    if abs(right_hip[1] - right_shoulder[1]) <= horizontal_tolerance and \
        abs(right_shoulder[1] - right_ankle[1]) <= horizontal_tolerance:
            return "ORIZZONTALE"

        # Check for vertical alignment
    if abs(right_hip[0] - right_shoulder[0]) <= vertical_tolerance and \
        abs(right_shoulder[0] - right_ankle[0]) <= vertical_tolerance:
            return "VERTICALE"

        # Check for squad position
    if abs(right_hip[0] - right_shoulder[0]) <= squad_x_tolerance and \
        abs(right_hip[1] - right_ankle[1]) <= squad_y_tolerance:
            return "SQUADRA"

    return "-Non sono riuscito ad elaborare la posizione-"

def normalizzate_a_assolute(normalized_x, normalized_y, width, height):
    absolute_x = int(normalized_x * width)
    absolute_y = int(normalized_y * height)
    return absolute_x, absolute_y

def extract_coordinates_from_mp(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    coordinates = []

    for frame_data in data:
        frame_number = frame_data.get("frame_number", 0)
        for node_name in ['bacino', 'spalla', 'caviglia']:
            node = frame_data.get(node_name, [])
            if node:
                coordinates.append((node_name,int(frame_number), node['x'], node['y']))

    return coordinates

def extract_coordinates_from_annotated(file_path): 
    with open(file_path, 'r') as file:
        data = json.load(file)

    coordinates = []

    for annotation in data.get('annotations', []):
        frames = annotation.get('frames', {})

        for frame_number, frame_data in frames.items():
            for node in frame_data.get('skeleton', {}).get('nodes', []):
                node_name = node.get('name', '')
                if node_name in ['bacino', 'spalla', 'caviglia']:
                    coordinates.append((node_name,int(frame_number), node['x'], node['y']))
                    

    return coordinates

def calculate_differences(mp_coordinates, annotated_coordinates, joint_rows):

    mp_coordinates = np.array(mp_coordinates)
    
    annotated_coordinates = np.array(annotated_coordinates)
    print(f"MediaPipe coordinates: {mp_coordinates}")
    print(f"Annotated coordinates: {annotated_coordinates}\n")
   # Trova i frame comuni tra i due video
    common_frames = set(mp_coordinates[:, 1]).intersection(set(annotated_coordinates[:,1]))
    #ordina i frame in ordine crescente
    common_frames = sorted(common_frames)
    
    print(f"Common frames: {common_frames}")
    # Inizializza array per le differenze
    frame_counter = 0
    differences = []
    #differences2 = []

    # Itera solo sui frame comuni
    for frame_number in common_frames:
        mp_frame = mp_coordinates[mp_coordinates[:,1] == frame_number]
        annotated_frame = annotated_coordinates[annotated_coordinates[:,1] == frame_number]  
        #print (f"mp_frame: {mp_frame}")
        #print (f"annotated_frame: {annotated_frame}")

        if annotated_frame.shape[1] == 0:
            print(f"Warning: No annotated coordinates found for frame {frame_number}")
            continue  # Skip frames with missing annotated data

        for mp_joint in mp_frame:
            
            mp_joint_name = mp_joint[0]  # Access joint name from MediaPipe data
            #print(f"MediaPipe joint: {mp_joint}")
            #print(f"Joint name: {mp_joint_name}")

            # Find the corresponding joint name in the annotated data
            #annotated_joint_name = joint_rows.get(mp_joint_name)
            if not mp_joint_name in joint_rows:
                print(f"Warning: Joint '{mp_joint_name}' not found in database")
                continue  # Skip joints not found in the database

            # Filter the annotated frame for the matching joint
            annotated_joint = annotated_frame[annotated_frame[:, 0] == mp_joint_name]
            #print(f"Annotated joint: {annotated_joint}")

            if annotated_joint.shape[0] == 0:
                print(f"Warning: Joint '{mp_joint_name}' not found in annotated frame {frame_number}")
                continue  # Skip joints missing from the current annotated frame
            
            annotated_x = float(annotated_joint[0][2])  # Assuming x is at index 2
            annotated_y = float(annotated_joint[0][3])  # Assuming y is at index 3


            # Extract and combine joint data SONO STRINGHE! MI SERVONO NUMERI PER SOTTRARRE!!!
            processed_joint = {
                'frame_number': float(frame_number),
                'joint_name': mp_joint_name,  # Maintain MediaPipe joint name
                'x': float(mp_joint[2]),
                'y': float(mp_joint[3]),
                #'occluded': mp_joint[4],  # Assuming occluded flag exists in MediaPipe data
                # Add any relevant information from annotated data if available
                # (e.g., 'annotated_joint_name' if desired)
            }
            #print(f"Processed joint: {processed_joint}")
            #converti frame, x e y di mp_joint e annotated_joint in float

            #processed_joint['frame_number'] = float(processed_joint['frame_number'])
            #calcola la differenza tra i processed joint e gli annotated joint
            diff_frame = np.array([processed_joint['x'], processed_joint['y']]) - np.array([annotated_x, annotated_y])
            if np.all(np.abs(diff_frame) <= 100):
                differences.append(diff_frame)
            
            #print(f"Processed joint: {processed_joint}")
           
        # Calculate the difference between the two joint positions
        #diff_frame = mp_frame[:,2:] - annotated_frame[:,2:]
         

    differences = np.array(differences)
    return differences 
        
def calcola_distanza_media(mp_coordinates,annotated_coordinates):
    # Filtra i frame considerando solo quelli nel range specificato
    #frames_selezionati = [frame for frame in frames if frame[0] in frame_range]
    # Estrai i frame comuni tra i dati ZED e quelli annotati
    mp_coordinates = np.array(mp_coordinates)
    annotated_coordinates = np.array(annotated_coordinates)
    #print(f"MediaPipe coordinates: {mp_coordinates}")
    #print(f"Annotated coordinates: {annotated_coordinates}")
    common_frames = set(mp_coordinates[:, 1]).intersection(set(annotated_coordinates[:,1]))

    # Verifica se ci sono frame nel range specificato
    if not common_frames:
        return None

    distanze_per_frame_bacino_spalla = []
    distanze_per_frame_bacino_caviglia = []
    distanze_per_frame_spalla_caviglia = []
    for frame in common_frames:
        # Seleziona le coordinate ZED per il frame corrente
        mp_frame = mp_coordinates[mp_coordinates[:,0] == frame]
        
        if mp_frame.shape[0] > 0:  # Assicurati che ci siano coordinate ZED per questo frame
            bacino_x, bacino_y = mp_frame[0, 1:]
            spalla_x, spalla_y = mp_frame[1, 1:]
            caviglia_x, caviglia_y = mp_frame[2, 1:]
        #bacino = np.array([zed_frame["bacino"][0], zed_frame["bacino"][1]])
        #spalla = joint_rows["spalla"]
        #caviglia = joint_rows["caviglia"]
        
        bacino=np.array([bacino_x,bacino_y])
        spalla=np.array([spalla_x,spalla_y])
        caviglia=np.array([caviglia_x,caviglia_y])

        print(f"\nBacino: {bacino}")
        print(f"\nSpalla: {spalla}")
        print(f"\nCaviglia: {caviglia}\n")
        
        
        print("#############################################")

        distanza_bacino_spalla = np.linalg.norm(bacino - spalla)
        distanza_bacino_caviglia = np.linalg.norm(bacino - caviglia)
        distanza_spalla_caviglia = np.linalg.norm(spalla - caviglia)

        distanze_per_frame_bacino_spalla.append(distanza_bacino_spalla)
        distanze_per_frame_bacino_caviglia.append(distanza_bacino_caviglia)
        distanze_per_frame_spalla_caviglia.append(distanza_spalla_caviglia)

    # Calcola la media totale delle distanze per tutti i frame selezionati
        
    #print(f"\nDistanze per frame bacino-spalla: {distanze_per_frame_bacino_spalla}")
    #print(f"\nDistanze per frame bacino-caviglia: {distanze_per_frame_bacino_caviglia}")
    media_totale_distanze_bacino_spalla = np.mean(distanze_per_frame_bacino_spalla)
    media_totale_distanze_bacino_caviglia = np.mean(distanze_per_frame_bacino_caviglia)
    media_totale_distanze_spalla_caviglia = np.mean(distanze_per_frame_spalla_caviglia)

    return media_totale_distanze_bacino_spalla, media_totale_distanze_bacino_caviglia, media_totale_distanze_spalla_caviglia

def calculate_mean_errors_per_joint(errors_per_joint):
    
    # Calculate squared errors
    squared_errors = np.square(errors_per_joint)

    # Initialize empty matrix for MSE values
    mpjpe_matrix = np.zeros((3, 2))

    for row_index in range(3):
        # Extract the corresponding indices for the current row
        indices = np.arange(row_index, len(errors_per_joint), step=3)
        mean_row = np.mean(squared_errors[indices], axis=0)
        mpjpe_matrix[row_index] = mean_row

    mpjpe=[]
    for riga in mpjpe_matrix:
        x=riga[0]
        y=riga[1]
        distanza=np.sqrt(x+y)
        mpjpe.append(distanza)

    return mpjpe

def mse(errors_per_joint):
    # Calculate squared errors
    squared_errors = np.square(errors_per_joint)

    # Initialize empty matrix for MSE values
    mse_matrix = np.zeros((3, 2))

    # Calculate MSE for each row (first element, fourth element, ...)
    for row_index in range(3):
        # Extract the corresponding indices for the current row
        indices = np.arange(row_index, len(errors_per_joint), step=3)

        # Calculate MSE for the current row
        mse_row = np.mean(squared_errors[indices], axis=0)

        # Set the MSE values in the matrix
        mse_matrix[row_index] = mse_row

    return mse_matrix

def dist_euclidea(zed_coordinates,annotated_coordinates):
    #print(f"\nZED Coordinates: {zed_coordinates}")
    dist=[]
    zed_coordinates = np.array(zed_coordinates)
    annotated_coordinates = np.array(annotated_coordinates)
    common_frames = set(zed_coordinates[:, 1]) & set(annotated_coordinates[:,1])
    common_frames = sorted(common_frames)
    for frame_number in common_frames:
        zed_frame = zed_coordinates[zed_coordinates[:,1] == frame_number]
        #annotated_frame = annotated_coordinates[annotated_coordinates[:,1] == frame_number]
        #print(f"\nZED Frame: {zed_frame}")
        bacino_x, bacino_y = zed_frame[0, 2:]
        spalla_x, spalla_y = zed_frame[1, 2:]
        caviglia_x, caviglia_y = zed_frame[2, 2:]
        #print(f"\nBacino: {float(bacino_x), bacino_y}")
        bacino=np.array([float(bacino_x),float(bacino_y)])
        spalla=np.array([float(spalla_x),float(spalla_y)])
        caviglia=np.array([float(caviglia_x),float(caviglia_y)])
    
        bsx = float(bacino_x) - float(spalla_x)
        bsy = float(bacino_y) - float(spalla_y)
        bcx = float(bacino_x) - float(caviglia_x)
        bcy = float(bacino_y) - float(caviglia_y)
        scx = float(spalla_x) - float(caviglia_x)
        scy = float(spalla_y) - float(caviglia_y)

        dist_bacino_spalla = math.sqrt(bsx**2 + bsy**2)
        dist_bacino_caviglia = math.sqrt(bcx**2 + bcy**2)
        dist_spalla_caviglia = math.sqrt(scx**2 + scy**2)
        #print(f"\nDistanza bacino-spalla: {dist_bacino_spalla}")
        #print(f"\nDistanza bacino-caviglia: {dist_bacino_caviglia}")
        #print(f"\nDistanza spalla-caviglia: {dist_spalla_caviglia}")
        if dist_bacino_spalla < 250 and dist_bacino_caviglia < 250 and dist_spalla_caviglia < 500:
            dist.append([dist_bacino_spalla, dist_bacino_caviglia, dist_spalla_caviglia])
    newdist = np.array(dist)
    #print(f"\nDistanze euclidee: {newdist}")
    distbs=np.mean(newdist[:,0])
    distbc=np.mean(newdist[:,1])
    distsc=np.mean(newdist[:,2])
    newdist = np.array([distbs, distbc, distsc])
    #return newdist
    return distbs, distbc, distsc, newdist, bacino, spalla, caviglia

def calcola_angolo(vettore1, vettore2, vettore3):
    #prodotto_scalare = np.dot(vettore1, vettore2)
    #norma_vettore1 = np.linalg.norm(vettore1)
    #norma_vettore2 = np.linalg.norm(vettore2)
    coseno_angolo = (vettore1**2 + vettore2**2 - vettore3**2) / (2 * vettore1 * vettore2)
    #coseno_angolo = prodotto_scalare / (norma_vettore1 * norma_vettore2)
    angolo_radianti = np.arccos(np.clip(coseno_angolo, -1.0, 1.0))
    angolo_gradi = np.degrees(angolo_radianti)

    return angolo_gradi
# Create PoseLandmarkerOptions
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)

json_data_list = []
frame_iniziale = 0
frame_finale = 400

# Initialize MediaPipe Pose and Drawing utilities
#mp_pose = mp.solutions.pose
#mp_drawing = mp.solutions.drawing_utils
#pose = mp_pose.Pose()

with PoseLandmarker.create_from_options(options) as landmarker:
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    frame_number = 0
    csv_data = []
    key_wait = 10

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        result = mp.solutions.pose.Pose().process(frame_rgb)

        #json_data = {'frame_number': frame_number, 'landmarks': []}
        
        # Verifica se il numero del frame è nell'intervallo desiderato
        if frame_iniziale <= frame_number <= frame_finale:

        # Draw the pose landmarks on the frame
            if result.pose_landmarks:
                """
                for idx, landmark in enumerate(result.pose_landmarks.landmark):
                    landmark_data = {'id': idx, 'name': mp.solutions.pose.PoseLandmark(idx).name, 'x': landmark.x, 'y': landmark.y}
                    json_data['landmarks'].append(landmark_data)
                """
                # Converti il dizionario in una stringa JSON
                #json_string = json.dumps(json_data, indent=2)
                #print(json_string)
                # mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                left_hip = result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
                left_ankle = result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ANKLE]
                left_shoulder = result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
                # Converti le coordinate normalizzate in coordinate assolute
                width, height = frame.shape[1], frame.shape[0]
                left_hip.x, left_hip.y = normalizzate_a_assolute(left_hip.x, left_hip.y, width, height)
                left_ankle.x, left_ankle.y = normalizzate_a_assolute(left_ankle.x, left_ankle.y, width, height)
                left_shoulder.x, left_shoulder.y = normalizzate_a_assolute(left_shoulder.x, left_shoulder.y, width, height)
                # Aggiungi le coordinate al dizionario
                frame_data = {
                    'frame_number': frame_number,
                    'bacino': {'x': left_hip.x, 'y': left_hip.y},
                    'spalla': {'x': left_shoulder.x, 'y': left_shoulder.y},
                    'caviglia': {'x': left_ankle.x, 'y': left_ankle.y}

                }
                json_data_list.append(frame_data)

                # Add the landmark coordinates to the list and print them
                #write_landmarks_to_csv([left_hip, left_ankle, left_shoulder], frame_number, csv_data)
                

                annotated_frame = frame.copy()
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated_frame, result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
            
                # Display the frame with pose landmarks
                cv2.imshow('Pose Estimation', annotated_frame)
            # Esci dal loop se hai raggiunto il frame finale
        frame_number += 1
        if frame_number > frame_finale:
            break

        # Exit if 'q' keypyt
        #if cv2.waitKey(1) & 0xFF == ord('q'):
           # break

        key = cv2.waitKey(key_wait)
        if key == 113: # for 'q' key
            print("Exiting...") 
            break
        if key == 109: # for 'm' key
            if (key_wait>0):
                print("Pause")
                key_wait = 0 
            else : 
                print("Restart")
                key_wait = 10
    # Salva i dati in un file JSON 
    #COMMENTA QUANDO HO GIà IL FILE JSON
    with open('mp.json', 'w') as json_file:
        json.dump(json_data_list, json_file, indent=2)


    mp_coordinates = extract_coordinates_from_mp('mp.json')
    annotated_coordinates = extract_coordinates_from_annotated('verticalecortochiuso.json')

    #alignment_time = calculate_alignment_time(mp_coordinates)
    #print(f"\nAlignment time: {alignment_time}")

    joint_rows = {
        'bacino',
        'spalla',
        'caviglia'
    }

    result = calculate_differences(mp_coordinates, annotated_coordinates, joint_rows)
    #print(result)
    
    #result = calculate_differences(zed_coordinates, annotated_coordinates, joint_rows) #non stampa perchè non ci sono frame comuni (non ho gli annotati)
    #print("\nShape of differences:", result.shape)

    mean_errors = calculate_mean_errors_per_joint(result)
    print("\nMean Per Joint Position Error for MediaPipe:")
    print(mean_errors)
 
    #msquaree = mse(result2,n_com)
    msquaree = mse(result)
    print("\nMean squared errors for MediaPipe:")
    print(msquaree)
   
    media1,media2,media3,dist,bacino,spalla,caviglia = dist_euclidea(mp_coordinates,annotated_coordinates)
    print(f"\nBacino: {bacino}")
    print(f"\nSpalla: {spalla}")
    print(f"\nCaviglia: {caviglia}")
    print(f"\nDistanze euclidee: Spalla-Bacino: '{round(dist[0],2)}', Bacino-Caviglia: '{round(dist[1],2)}', Spalla-Caviglia: '{round(dist[2],2)}'")
    angolo_bacino = calcola_angolo(media1, media2, media3)
    angolo = round(angolo_bacino, 0)
    print(f"\nAngolo bacino: {angolo}") #AGGIUNGERE CONDIZIONE: SE POSA = SQUADRA, L'ANGOLO GIUSTO DIVENTA 90
    posa = determine_pose(bacino, caviglia, spalla, horizontal_tolerance, vertical_tolerance, squad_x_tolerance, squad_y_tolerance)
    if posa == "SQUADRA":
        malus = 90-15-angolo
    else:
        malus=180-15-angolo 
    if malus<0:
        malus=0
    if malus>100:
        malus=100
    print(f"\nIl MALUS che riceverà l'atleta per questa Skill {posa} sarà del {malus}% sul valore della Skill!\n")
   
    # Rilascia le risorse
    cap.release()
    cv2.destroyAllWindows()