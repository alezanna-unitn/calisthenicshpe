########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
   This sample shows how to detect a human bodies and draw their 
   modelised skeleton in an OpenGL window
"""
import cv2
import sys
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
import numpy as np
import argparse
import json
import time
import matplotlib.pyplot as plt
import math

# Define the tolerance for horizontal and vertical alignment
horizontal_tolerance = 80
vertical_tolerance = 80

# Define the tolerance for squad position
squad_x_tolerance = 80
squad_y_tolerance = 80

def determine_pose(right_hip, right_ankle, right_shoulder, horizontal_tolerance, vertical_tolerance, squad_x_tolerance, squad_y_tolerance):
    
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

def parse_args(init):
    if len(opt.input_svo_file)>0 and opt.input_svo_file.endswith(".svo"):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))
    elif len(opt.ip_address)>0 :
        ip_str = opt.ip_address
        if ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4 and len(ip_str.split(':'))==2:
            init.set_from_stream(ip_str.split(':')[0],int(ip_str.split(':')[1]))
            print("[Sample] Using Stream input, IP : ",ip_str)
        elif ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4:
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP : ",ip_str)
        else :
            print("Unvalid IP format. Using live stream")
    if ("HD2K" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD2K
        print("[Sample] Using Camera in resolution HD2K")
    elif ("HD1200" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1200
        print("[Sample] Using Camera in resolution HD1200")
    elif ("HD1080" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1080
        print("[Sample] Using Camera in resolution HD1080")
    elif ("HD720" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using Camera in resolution HD720")
    elif ("SVGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.SVGA
        print("[Sample] Using Camera in resolution SVGA")
    elif ("VGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.VGA
        print("[Sample] Using Camera in resolution VGA")
    elif len(opt.resolution)>0: 
        print("[Sample] No valid resolution entered. Using default")
    else : 
        print("[Sample] Using default resolution")

def extract_keypoints_json(bodies, fc, kp_data, json_path, image_scale):

    if bodies.is_tracked and bodies.body_list:      

        right_hip = bodies.body_list[0].keypoint_2d[11]
        right_shoulder = bodies.body_list[0].keypoint_2d[5]
        right_ankle = bodies.body_list[0].keypoint_2d[13]

        # Apply scaling factor to each keypoint coordinate
        right_hip[0] *= image_scale[0]  # Scale x-coordinate of right hip
        right_hip[1] *= image_scale[1]  # Scale y-coordinate of right hip
        right_shoulder[0] *= image_scale[0]  # Scale x-coordinate of right shoulder
        right_shoulder[1] *= image_scale[1]  # Scale y-coordinate of right shoulder
        right_ankle[0] *= image_scale[0]  # Scale x-coordinate of right ankle
        right_ankle[1] *= image_scale[1]  # Scale y-coordinate of right ankle

        # Converte le coordinate in liste
        right_hip_list = [right_hip[0], right_hip[1]]
        right_shoulder_list = [right_shoulder[0], right_shoulder[1]]
        right_ankle_list = [right_ankle[0], right_ankle[1]]

        # Salva le coordinate in un file JSON
        kp_data.append ({
            "frame": fc,
            #"caviglia": right_ankle_list,
            "bacino": right_hip_list,
            "spalla": right_shoulder_list,
            "caviglia": right_ankle_list
            
        })
        
        #PRIMA SCRIVE IL FILE JSON (TOGLIERE IL COMMENTO DI QUESTO CODICE) DOPO DI CHE LEGGE IL FILE E CALCOLA LE DIFFERENZE (COMMENTA IL CODICE)
                    
        #with open(json_path, "w") as json_file:
         #   json.dump(kp_data, json_file, indent=2)    

def load_json(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data

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

def extract_coordinates_from_zed(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    coordinates = []

    for frame_data in data:
        frame_number = frame_data.get("frame", 0)
        for node_name in ['bacino', 'spalla', 'caviglia']:
            node = frame_data.get(node_name, [])
            if node:
                coordinates.append((node_name,int(frame_number), node[0], node[1]))
    #print(f"\nZED Coordinates: {coordinates}")
    return coordinates

def calculate_differences(zed_coordinates, annotated_coordinates, joint_rows):#, groups):

    zed_coordinates = np.array(zed_coordinates)
    annotated_coordinates = np.array(annotated_coordinates)
    print(f"\nZED Coordinates: {zed_coordinates}")
    print (f"\nAnnotated Coordinates: {annotated_coordinates}")
   # Trova i frame comuni tra i due video
    #common_frames = set(zed_coordinates[:, 0]).intersection(set(annotated_coordinates[:,0]))
    common_frames = set(zed_coordinates[:, 1]) & set(annotated_coordinates[:,1])
    common_frames = sorted(common_frames)
    print(f"\nCommon Frames: {common_frames}")

    # Inizializza array per le differenze
    differences = []

    for frame_number in common_frames:
        zed_frame = zed_coordinates[zed_coordinates[:,1] == frame_number]
        annotated_frame = annotated_coordinates[annotated_coordinates[:,1] == frame_number]  
        #print (f"zed_frame: {zed_frame}")
        #print (f"annotated_frame: {annotated_frame}")

        if annotated_frame.shape[1] == 0:
            print(f"Warning: No annotated coordinates found for frame {frame_number}")
            continue  # Skip frames with missing annotated data

        for zed_joint in zed_frame:
            
            zed_joint_name = zed_joint[0]  # Access joint name from MediaPipe data
            #print(f"ZED joint: {zed_joint}")
            #print(f"Joint name: {mp_joint_name}")

            # Find the corresponding joint name in the annotated data
            #annotated_joint_name = joint_rows.get(mp_joint_name)
            if not zed_joint_name in joint_rows:
                print(f"Warning: Joint '{zed_joint_name}' not found in database")
                continue  # Skip joints not found in the database

            # Filter the annotated frame for the matching joint
            annotated_joint = annotated_frame[annotated_frame[:, 0] == zed_joint_name]
            #print(f"Annotated joint: {annotated_joint}")

            if annotated_joint.shape[0] == 0:
                print(f"Warning: Joint '{zed_joint_name}' not found in annotated frame {frame_number}")
                continue  # Skip joints missing from the current annotated frame
            
            annotated_x = float(annotated_joint[0][2])  # Assuming x is at index 2
            annotated_y = float(annotated_joint[0][3])  # Assuming y is at index 3


            # Extract and combine joint data SONO STRINGHE! MI SERVONO NUMERI PER SOTTRARRE!!!
            processed_joint = {
                'frame_number': float(frame_number),
                'joint_name': zed_joint_name,  # Maintain MediaPipe joint name
                'x': float(zed_joint[2]),
                'y': float(zed_joint[3]),
                #'occluded': mp_joint[4],  # Assuming occluded flag exists in MediaPipe data
                # Add any relevant information from annotated data if available
                # (e.g., 'annotated_joint_name' if desired)
            }
            #print(f"Processed joint: {processed_joint}")
            #converti frame, x e y di mp_joint e annotated_joint in float
            #set a threshold for the outliars THESHOLLLDDDDD!!!
            
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

def calculate_mean_errors_per_joint(errors_per_joint):
    # Calculate the mean error along axis 0 (frames)

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

def mse(errors_per_joint): ##PERCHè COSI ALTO L'MSE DI SPALLA???
   
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
        if dist_bacino_spalla < 250 and dist_bacino_caviglia < 250 and dist_spalla_caviglia < 500 and dist_bacino_spalla > 0 and dist_bacino_caviglia > 0 and dist_spalla_caviglia > 0:
            dist.append([dist_bacino_spalla, dist_bacino_caviglia, dist_spalla_caviglia])
    newdist = np.array(dist)
    print(f"\nDistanze euclidee: {newdist}")
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

def calcolo_malus(angolo,posa,bacino,spalla,caviglia):
    if posa == "ORIZZONTALE":
        malus = 90-15-angolo
    else:
        malus=180-15-angolo 
    if malus<0:
        malus=0
    if malus>100:
        malus=100
    threshold=15
    #print malus
    print(f"\nIl MALUS {malus}% sul valore della Skill!")
    if malus == 0 and posa == "VERTICALE" and (spalla[0] > caviglia[0]-threshold or spalla[0] < caviglia[0]+threshold):
        euclidea = math.sqrt((caviglia[0]-spalla[0])**2 + (caviglia[1]-spalla[1])**2)
        #stampa spalla
        print(f"\nSpalla X : {spalla[0]}")
        #stampa caviglia
        print(f"\nCaviglia X: {caviglia[0]}")
        print(f"\nEuclidea: {euclidea}")
        distx = abs(caviglia[0]-spalla[0])
        print(f"\nDistanza x: {distx}")
        #teorema di pitagora conoscedo l'ipotenusa e un cateto
        disty = math.sqrt(euclidea**2 - distx**2)
        print(f"\nDistanza y: {disty}")
        #angolo sotteso tra cateto maggiore e ipotenusa
        angolo = 90-math.degrees(math.atan(disty/distx))
        malus=angolo
        print(f"\nAngolo bacino: {angolo}")
    if malus == 0 and posa == "ORIZZONTALE" and (spalla[1] > caviglia[1]-threshold or spalla[1] < caviglia[1]+threshold):
        euclidea = math.sqrt((caviglia[0]-spalla[0])**2 + (caviglia[1]-spalla[1])**2)
        disty = abs(caviglia[1]-spalla[1])
        #teorema di pitagora conoscedo l'ipotenusa e un cateto
        distx = math.sqrt(euclidea**2 - disty**2)
        #angolo sotteso tra cateto maggiore e ipotenusa
        angolo = 90-math.degrees(math.atan(distx/disty))
        malus=angolo
    print(f"\nMalus: {malus}")
    if malus <10:
        malus=0
    if malus >10 and malus <20:
        malus=15
    if malus >20 and malus <35:
        malus=25
    if malus >35 and malus <65:
        malus=50
    if malus >65:
        malus=100

    return malus

def main():
    print("Running Body Tracking sample ... Press 'q' to quit, or 'm' to pause or restart")

    # Create a Camera object
    zed = sl.Camera()
    svo_output_path = "output_video.svo"  # Imposta il percorso desiderato per il file SVO

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    
    parse_args(init_params)

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    svo_params = sl.RecordingParameters(svo_output_path)

    # Inizia la registrazione SVO
    err2 = zed.enable_recording(svo_params)
    if err2 != sl.ERROR_CODE.SUCCESS:
       sys.stdout.write(repr(err2))
       zed.close()
       exit()

    # Enable Positional tracking (mandatory for object detection)
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)
    
    body_param = sl.BodyTrackingParameters()
    body_param.enable_tracking = True                # Track people across images flow
    body_param.enable_body_fitting = False            # Smooth skeleton move
    body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST 
    body_param.body_format = sl.BODY_FORMAT.BODY_18  # Choose the BODY_FORMAT you wish to use

    # Enable Object Detection module
    zed.enable_body_tracking(body_param)

    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    body_runtime_param.detection_confidence_threshold = 40

    fps = zed.get_camera_information().camera_configuration.fps #time stamp temporale di acquisizione del frame
    #con il time stamp non mi interessa il frame rate del video, ma il time stamp temporale di acquisizione del frame
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (1280, 720))  # Adjust the resolution as needed
    
    # Get ZED camera information
    camera_info = zed.get_camera_information()
    # 2D viewer utilities
    display_resolution = sl.Resolution(min(camera_info.camera_configuration.resolution.width, 1280), min(camera_info.camera_configuration.resolution.height, 720))
    image_scale = [display_resolution.width / camera_info.camera_configuration.resolution.width
                 , display_resolution.height / camera_info.camera_configuration.resolution.height]
    # Create a scaling factor based on the display resolution
    #scale_factor_x = display_resolution.width / camera_info.camera_configuration.resolution.width
    #scale_factor_y = display_resolution.height / camera_info.camera_configuration.resolution.height
    bodies = sl.Bodies()
    image = sl.Mat()
    key_wait = 10
    frame_count = 0
    keypoints_data = []
    # Create OpenGL viewer
    viewer = gl.GLViewer()
    viewer.init(camera_info.camera_configuration.calibration_parameters.left_cam, body_param.enable_tracking,body_param.body_format)
    # Create ZED objects filled in the main loop
    
    
    while viewer.is_available():
        # Grab an image
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            #svo_position = zed.get_svo_position()
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            # Retrieve bodies
            zed.retrieve_bodies(bodies, body_runtime_param)
            # Update GL view
            viewer.update_view(image, bodies) 
            # Update OCV view
            image_left_ocv = image.get_data()
            #image_left_ocv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            #Estrai le coordinate delle giunture e salvale in un file JSON
            
            extract_keypoints_json(bodies, frame_count, keypoints_data, "zed", image_scale)
            #extract_keypoints_json_with_pose_recognition(bodies, frame_count, keypoints_data, "output_keypoints_zed.json", image_scale, zed_coordinates, annotated_coordinates)
            #print("Keypoints extracted for the current frame.")
            frame_count += 1

            cv_viewer.render_2D(image_left_ocv,image_scale, bodies.body_list, body_param.enable_tracking, body_param.body_format)
            cv2.imshow("ZED | 2D View", image_left_ocv)
            out.write(image_left_ocv)
            #svo_params.position = zed.get_svo_position()
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
    viewer.exit()
    # Scrivi i dati nel file JSON
    with open("zed.json", "w") as json_file_all:
       json.dump(keypoints_data, json_file_all, indent=2)

    annotated_coordinates = extract_coordinates_from_annotated('verticalecorto.json')
    zed_coordinates = extract_coordinates_from_zed('zed.json')
    
    #print("Annotated Coordinates:", annotated_coordinates)
    #print("ZED Coordinates:", zed_coordinates)
   
    joint_rows = {
                "bacino": 0,
                "spalla": 1,
                "caviglia": 2
            }

    result = calculate_differences(zed_coordinates, annotated_coordinates, joint_rows)
    #print(result)
    
    #result = calculate_differences(zed_coordinates, annotated_coordinates, joint_rows) #non stampa perchè non ci sono frame comuni (non ho gli annotati)
    #print("\nShape of differences:", result.shape)

    mean_errors = calculate_mean_errors_per_joint(result)
    print("\nMean Per Joint Position Error for ZED:")
    print(mean_errors)
 
    #msquaree = mse(result2,n_com)
    msquaree = mse(result)
    print("\nMean squared errors for ZED:")
    print(msquaree)
   
    media1,media2,media3,dist,bacino,spalla,caviglia = dist_euclidea(zed_coordinates,annotated_coordinates)
    print(f"\nBacino: {bacino}, Spalla: {spalla}, Caviglia: {caviglia}")
    print(f"\nDistanze euclidee: Spalla-Bacino: '{round(dist[0],2)}', Bacino-Caviglia: '{round(dist[1],2)}', Spalla-Caviglia: '{round(dist[2],2)}'")
    angolo_bacino = calcola_angolo(media1, media2, media3)
    angolo = round(angolo_bacino, 0)
    print(f"\nAngolo bacino: {angolo}")

    posa = determine_pose(bacino, caviglia, spalla, horizontal_tolerance, vertical_tolerance, squad_x_tolerance, squad_y_tolerance)
    malus=calcolo_malus(angolo,posa,bacino,spalla,caviglia)
    print(f"\nIl MALUS che riceverà l'atleta per questa Skill {posa} sarà del {malus}% sul valore della Skill!\n")
    
    
    image.free(sl.MEM.CPU)
    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.disable_recording()
    zed.close()
    out.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file, if you want to replay it',default = '')
    parser.add_argument('--output_svo_file', type=str)
    parser.add_argument('--ip_address', type=str, help='IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup', default = '')
    parser.add_argument('--resolution', type=str, help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA', default = '')
    opt = parser.parse_args()
    if len(opt.input_svo_file)>0 and len(opt.ip_address)>0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit()
    main() 