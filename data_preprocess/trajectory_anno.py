import os
import h5py
import cdflib
import cv2
import os
import numpy as np
from scipy.interpolate import splprep, splev
from ultralytics import YOLO
import shutil

model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model

def human36_visual(base_path, actor_id, video_name, output_path, thre=300, window_size=20, step_size=10, bbox_expand=0):
    
    image_path = os.path.join(base_path, actor_id, "Images", video_name)
    anno_path = os.path.join(base_path, actor_id, "MyPoseFeatures", "D2_Positions", video_name[:-4]+".cdf")
    filename = anno_path
    cdf_file = cdflib.CDF(filename)
    img_paths = sorted(os.listdir(image_path))
    visual_count = 0

    # output_path = os.path.join(base_path, actor_id, "Annotations", video_name)

    target_path = os.path.join(output_path, "target", actor_id+"_"+video_name)
    traj_path = os.path.join(output_path, "trajectory", actor_id+"_"+video_name)
    initial_path = os.path.join(output_path, "initial", actor_id+"_"+video_name)
    anno_path = os.path.join(output_path, "anno", actor_id+"_"+video_name)

    if not os.path.exists(target_path):
        os.makedirs(target_path)
    if not os.path.exists(traj_path):
        os.makedirs(traj_path)
    if not os.path.exists(initial_path):
        os.makedirs(initial_path)
    if not os.path.exists(anno_path):
        os.makedirs(anno_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    variables = cdf_file.cdf_info()
    pose = cdf_file["pose"]
    length = pose.shape[1]

    print("Variables: ", variables)

    keypoints = pose[0].reshape(length,-1,2)
    hips = keypoints[:,0,:]

    for i in range(0,length,step_size):
        if i+window_size<length:
            print(f"start frame: {i}")
            distance = (hips[i+window_size][0]-hips[i][0])**2+(hips[i+window_size][1]-hips[i][1])**2
            if distance>thre:
                visual_count += 1
                inital_img_path = os.path.join(image_path,img_paths[i])
                inital_img = cv2.imread(inital_img_path)

                final_img_path = os.path.join(image_path,img_paths[i+window_size])
                final_img = cv2.imread(final_img_path)
                mask_img = np.zeros(inital_img.shape, dtype=np.uint8)

                bbox_start, bbox_end = yolo_box_detect(model, final_img_path)
                if bbox_start==None:
                    continue
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
                # trajectory_img = draw_trajectory(hips[i:i+window_size,:], final_img)
                # trajectory_black = draw_trajectory(hips[i:i+window_size,:], mask_img)
                trajectory_black = draw_trajectory_line(hips[i:i+window_size,:], mask_img)
                # inital_trajectory_img = draw_trajectory(hips[i:i+window_size,:], inital_img)

                print("draw box !!!!!!")
                # cv2.rectangle(trajectory_img, bbox_start, bbox_end, color=(0,0,255), thickness=3)
                cv2.rectangle(trajectory_black, init_bbox_start, init_bbox_end, color=(0,0,255), thickness=1)
                cv2.rectangle(inital_img, init_bbox_start, init_bbox_end, color=(0,0,255), thickness=1)

                save_final_path =  os.path.join(target_path, f"{i:04d}_{i+window_size:04d}.jpg")
                save_trajectory_path =  os.path.join(traj_path, f"{i:04d}_{i+window_size:04d}.jpg")
                save_start_path =  os.path.join(initial_path, f"{i:04d}_{i+window_size:04d}.jpg")
                save_bbox_path = os.path.join(anno_path, f"{i:04d}_{i+window_size:04d}.json")

                cv2.imwrite(save_final_path, final_img)
                cv2.imwrite(save_start_path, inital_img)  
                cv2.imwrite(save_trajectory_path, trajectory_black) 
                           
            print(distance)
    return hips, visual_count

def sample_sub_box(image_path, start_id, window):
    images = sorted(os.listdir(image_path))
    bboxs = []

    for i in range(start_id, start_id+24*window, window):
        results = model.predict(os.path.join(image_path, images[i]), conf=0.5)
        for result in results:
            box_norm = result.boxes.xyxyn.cpu().numpy()  # Boxes object for bounding box outputs
            bboxs.append(box_norm)

    bboxs = np.stack(bboxs).squeeze(1)
    print(bboxs.shape)
    np.save("human_round.npy", bboxs)
    # print(bboxs)

def downsample_video(images_path, output_path, start_idx=0, chunk_size=100):
    
    images = sorted(os.listdir(images_path))
    # output_path = 
    count = 0
    for image_idx in range(start_idx, len(images)):
        if image_idx%5 == 0:
            chunk_idx = count//chunk_size
            chunk_output_path = output_path+f"_{chunk_idx}"
            
            if not os.path.exists(chunk_output_path):
                os.makedirs(chunk_output_path)
                
            print(os.path.join(images_path, images[image_idx]))
            shutil.copy(os.path.join(images_path, images[image_idx]), chunk_output_path)
            count+=1

def draw_trajectory(keypoints, image):

    tck, u = splprep(keypoints.T, s=0)
    u_new = np.linspace(u.min(), u.max(), 50)
    x_new, y_new = splev(u_new, tck, der=0)

    smooth_points = np.vstack((x_new, y_new)).T.astype(np.int32)
    print(smooth_points.shape)

    for i in range(len(smooth_points) - 1):
        if i == len(smooth_points)-2:
            print("end i")
            # cv2.arrowedLine(image, tuple(smooth_points[i-3]), tuple(smooth_points[i + 1]), (0, 0, 255), 2, tipLength=10)
            draw_arrow(image, smooth_points[i-10], smooth_points[i + 1], 12, 40, (0, 0, 255), 2)
        cv2.line(image, tuple(smooth_points[i]), tuple(smooth_points[i + 1]), (0, 0, 255), 2)
    
    return image

def draw_trajectory_line(keypoints, image):

    smooth_points = keypoints.astype(np.int32)
    
    cv2.arrowedLine(image, tuple(smooth_points[0]), tuple(smooth_points[-1]), (0, 0, 255), 2, tipLength=0.3)
    
    return image

def draw_arrow(image, start_point, end_point, arrow_length, arrow_angle, color, thickness):

    # cv2.line(image, start_point, end_point, color, thickness)
    
    direction = np.array([end_point[0] - start_point[0], end_point[1] - start_point[1]])
    length = np.linalg.norm(direction)
    direction = direction / length
    
    left_wing = np.array([np.cos(np.radians(arrow_angle)), np.sin(np.radians(arrow_angle))])
    right_wing = np.array([np.cos(np.radians(-arrow_angle)), np.sin(np.radians(-arrow_angle))])
    
    left_wing_point = end_point - arrow_length * (direction * np.cos(np.radians(arrow_angle)) + left_wing * np.sin(np.radians(arrow_angle)))
    right_wing_point = end_point - arrow_length * (direction * np.cos(np.radians(arrow_angle)) + right_wing * np.sin(np.radians(arrow_angle)))
    
    cv2.line(image, end_point, tuple(left_wing_point.astype(int)), color, thickness)
    cv2.line(image, end_point, tuple(right_wing_point.astype(int)), color, thickness)



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
    if len(boxes.xyxy.tolist())<=0:
        return None, None
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

def visual_all(base_path):
    all_count = 0
    folders = sorted(os.listdir(base_path))
    for spk in folders:
        print(spk)
        spk_path = os.path.join(base_path, spk, "Images")
        videos = os.listdir(spk_path)
        for vid in videos:
            hips, count = human36_visual(base_path, spk, vid, "36m_Anno_line")
            all_count += count
    print(f"all count is {all_count}")
    
# videos_path = "/data/longbinji/animate-images/dataset/human36_video/S5/Videos"

# for vid in sorted(os.listdir(videos_path)):
#     split_video(os.path.join(videos_path, vid), "human36_video/S5/Images")

human36_path = "/data/longbinji/animate-images/dataset/human36_video"
down_path = "/data/longbinji/animate-images/dataset/human36_video_down_chunks"

for person in os.listdir(human36_path):
    human36_person = os.path.join(human36_path, person)
    # down_person_path = os.path.join(down_path, person+"_"+vid)
    
    for vid in os.listdir(human36_person+"/Images"):   
        down_person_path = os.path.join(down_path, person+"_"+vid)       
        downsample_video(os.path.join(human36_person+"/Images", vid), down_person_path, 80)
            
        
# sample_sub_box("/data/longbinji/animate-images/dataset/human36_video/S7/Images/WalkDog 1.54138969.mp4", 140, 5)

# videos_path = "/data/longbinji/animate-images/dataset/human36_video/S8/Videos"

# for vid in sorted(os.listdir(videos_path)):
#     split_video(os.path.join(videos_path, vid), "human36_video/S8/Images")

# visual_all("/data/longbinji/animate-images/dataset/human36_video")

# human36_visual("/home/longbin/workspace/animate/dataset/human36_video","S1","Walking 1.54138969.mp4", "36m_Anno")
# split_video("human36_video/S1/Videos/Walking 1.54138969.mp4","human36_video/S1/Images")

# yolo_box_detect("/home/longbin/workspace/animate/dataset/human36_video/S1/Images/Walking 1.54138969.mp4/frame_0120.jpg")

