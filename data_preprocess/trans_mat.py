data_path = "/home/longbin/workspace/animate/dataset/MPII/1/011775108"
data_path = "/home/longbin/workspace/animate/dataset/MPII/1/067189274"
import mediapipe as mp
import os
import h5py
import cdflib
import cv2
import os
import numpy as np
from scipy.interpolate import splprep, splev
from ultralytics import YOLO
import scipy

model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model

def read_mat(data_path):
    mat_data = scipy.io.loadmat(data_path)
    print(mat_data['RELEASE']['act'])
    for i, (anno, act_flag) in enumerate(
        zip(mat_data['RELEASE']['annolist'][0, 0][0],
            mat_data['RELEASE']['act'][0, 0][0])):

        img_fn = anno['image']['name'][0, 0][0]
        train_flag = act_flag['act_name']
        print(img_fn, train_flag)



    # for key, value in mat_data.items():
    #     if key.startswith('__'):
    #         continue  # 忽略内部变量
    #     print(f"Variable: {key}")
    #     print(f"Type: {type(value)}")
    #     print(f"Data: {value}")
    # # print(mat_data.keys())
    # print(mat_data['data'])



def human36_visual(base_path, actor_id, video_name, thre=100, window_size=20, bbox_expand=0):
    
    image_path = os.path.join(base_path, actor_id, "Images", video_name)
    anno_path = os.path.join(base_path, actor_id, "MyPoseFeatures", "D2_Positions", video_name[:-4]+".cdf")
    filename = anno_path
    cdf_file = cdflib.CDF(filename)
    img_paths = sorted(os.listdir(image_path))
    output_path = os.path.join(base_path, actor_id, "Annotations", video_name)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    variables = cdf_file.cdf_info()
    pose = cdf_file["pose"]
    length = pose.shape[1]

    print("Variables: ", variables)

    keypoints = pose[0].reshape(length,-1,2)
    hips = keypoints[:,0,:]

    for i in range(0,length,window_size):
        if i+window_size<length:
            print(f"start frame: {i}")
            distance = (hips[i+window_size][0]-hips[i][0])**2+(hips[i+window_size][1]-hips[i][1])**2
            if distance>thre:
                inital_img_path = os.path.join(image_path,img_paths[i])
                inital_img = cv2.imread(inital_img_path)

                final_img_path = os.path.join(image_path,img_paths[i+window_size])
                final_img = cv2.imread(final_img_path)
                mask_img = np.zeros(inital_img.shape, dtype=np.uint8)

                bbox_start, bbox_end = yolo_box_detect(model, final_img_path)
                init_bbox_start, init_bbox_end = yolo_box_detect(model, inital_img_path)

                if bbox_expand>0:
                    center = [int(bbox_start[0]+bbox_end[0])/2, int(bbox_start[1]+bbox_end[1])/2]
                    width = center[0]-bbox_start[0]
                    height = center[1]-bbox_start[1]
                    expand_width = width*(1+bbox_expand)
                    expand_height = height*(1+bbox_expand)
                    bbox_start = (center[0]-expand_width,center[1]-expand_height)
                    bbox_end = (center[0]+expand_width,center[1]+expand_height)

                print(f"read image for {img_paths[i+window_size]}")
                trajectory_img = draw_trajectory(hips[i:i+window_size,:], final_img)
                trajectory_black = draw_trajectory(hips[i:i+window_size,:], mask_img)
                inital_trajectory_img = draw_trajectory(hips[i:i+window_size,:], inital_img)

                print("draw box !!!!!!")
                cv2.rectangle(trajectory_img, bbox_start, bbox_end, color=(0,0,255), thickness=3)
                cv2.rectangle(trajectory_black, bbox_start, bbox_end, color=(0,0,255), thickness=3)
                cv2.rectangle(inital_trajectory_img, init_bbox_start, init_bbox_end, color=(0,0,255), thickness=3)
                

                save_img_path =  os.path.join(output_path, f"anno_{i:04d}_{i+window_size:04d}.jpg")
                cv2.imwrite(save_img_path, trajectory_img)
                cv2.imwrite(save_img_path[:-4]+"_prev.jpg", inital_trajectory_img)  
                cv2.imwrite(save_img_path[:-4]+"_black.jpg", trajectory_black)                
            print(distance)


    return hips

def draw_trajectory(keypoints, image):

    tck, u = splprep(keypoints.T, s=0)
    u_new = np.linspace(u.min(), u.max(), 50)
    x_new, y_new = splev(u_new, tck, der=0)

    smooth_points = np.vstack((x_new, y_new)).T.astype(np.int32)
    print(smooth_points.shape)

    for i in range(len(smooth_points) - 1):
        if i == len(smooth_points)-2:
            print("end i")
            cv2.arrowedLine(image, tuple(smooth_points[i]), tuple(smooth_points[i + 1]), (0, 0, 255), 2, tipLength=10)
        else:
            cv2.line(image, tuple(smooth_points[i]), tuple(smooth_points[i + 1]), (0, 0, 255), 2)
    
    return image

def split_video(video_path, output_path):

    output_dir = os.path.join(output_path,os.path.basename(video_path))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video FPS: {fps}")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()

    print(f"Total frames extracted: {frame_count}")
    return fps, width, height

def yolo_box_detect(model, image_path):
    bboxs_starts = []
    bboxs_ends = []

    # Run batched inference on a list of images
    # results = model([image_path])  # return a list of Results objects
    results = model.predict(image_path, conf=0.5)
    
    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
    boxes = boxes.xyxy.tolist()[0]
    x1, y1, x2, y2 = boxes
    bboxs_start = (int(x1), int(y1))
    bboxs_end = (int(x2), int(y2))

    # print(bboxs_start)
    # bboxs_starts.append(bboxs_start)
    # bboxs_ends.append(bboxs_end)


    return bboxs_start, bboxs_end

def box_detect(image_path):
    BaseOptions = mp.tasks.BaseOptions
    ObjectDetector = mp.tasks.vision.ObjectDetector
    ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path='/home/longbin/workspace/mediapipe/efficientdet_lite0.tflite'),
        max_results=5,
        running_mode=VisionRunningMode.IMAGE)
    
    bboxs_start = []
    bboxs_end = []

    with ObjectDetector.create_from_options(options) as detector:
        # Load the input image from an image file.
        mp_image = mp.Image.create_from_file(image_path)

        detection_result = detector.detect(mp_image)
        for detection in detection_result.detections:
            category_name = detection.categories[0].category_name
            if category_name == "person":
                bbox = detection.bounding_box
                print(bbox)

                start_point = bbox.origin_x, bbox.origin_y
                end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
                bboxs_start.append(start_point)
                bboxs_end.append(end_point)

    return bboxs_start, bboxs_end

read_mat("/home/longbin/workspace/animate/dataset/MPII/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat")
# human36_visual("/home/longbin/workspace/animate/dataset/human36_video","S1","Walking 1.54138969.mp4")
# split_video("human36_video/S1/Videos/Walking 1.54138969.mp4","human36_video/S1/Images")

# yolo_box_detect("/home/longbin/workspace/animate/dataset/human36_video/S1/Images/Walking 1.54138969.mp4/frame_0120.jpg")

