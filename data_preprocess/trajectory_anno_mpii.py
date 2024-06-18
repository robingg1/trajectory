data_path = "/home/longbin/workspace/animate/dataset/MPII/1/011775108"
data_path = "/home/longbin/workspace/animate/dataset/MPII/1/067189274"
# import mediapipe as mp
import os
import h5py
import cdflib
import cv2
import os
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.interpolate import interp1d
from ultralytics import YOLO
from tqdm import tqdm
import json
import shutil
import torch

# model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model
model = YOLO("yolov8l-pose.pt")

def mpii_visual(base_path, video_name, anno_path, output_path, thre=500, window_size=4, step_size=1, bbox_expand=0):
    
    image_path = os.path.join(base_path, video_name)
    
    visual_count = 0
    cur_anno = os.path.join(anno_path, video_name+"_filtered_data.json")
    with open(cur_anno, "r") as f:
        data_file = json.load(f)

    print(data_file.keys())
        
    pose = data_file[image_path][1]
    bbox = data_file[image_path][0]

    img_paths = sorted(os.listdir(image_path))
    # output_path = os.path.join(output_path, "Annotations", video_name)

    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)

    target_path = os.path.join(output_path, "target", video_name)
    traj_path = os.path.join(output_path, "trajectory", video_name)
    initial_path = os.path.join(output_path, "initial", video_name)
    anno_path = os.path.join(output_path, "anno", video_name)

    if not os.path.exists(target_path):
        os.makedirs(target_path)
    if not os.path.exists(traj_path):
        os.makedirs(traj_path)
    if not os.path.exists(initial_path):
        os.makedirs(initial_path)
    if not os.path.exists(anno_path):
        os.makedirs(anno_path)




    length = len(pose)

    hips = np.array(pose)
    bbox = np.array(bbox)

    for i in range(0,length,step_size):
        if i+window_size<length:
            print(f"start frame: {i}")
            try:
                print(hips[i])
                distance = (hips[i+window_size][0][0]-hips[i][0][0])**2+(hips[i+window_size][0][1]-hips[i][0][1])**2
                # tck, u = splprep(hips[i:i+window_size,0,:].T, s=0)
            except Exception as e:
                print(e)
                print("error with distance computation, possibly missing hip keypoint in window.")
                continue
            if distance>thre and distance<15000:
                visual_count += 1
                inital_img_path = os.path.join(image_path,img_paths[i])
                inital_img = cv2.imread(inital_img_path)

                final_img_path = os.path.join(image_path,img_paths[i+window_size])
                final_img = cv2.imread(final_img_path)
                mask_img = np.zeros(inital_img.shape, dtype=np.uint8)

                print(f"read image for {img_paths[i+window_size]}")
                hips_window = hips[i:i+window_size]
                persons = len(hips_window[0])

                # trajectory_img = final_img
                trajectory_black = mask_img
                inital_trajectory_img = inital_img

                for idx in range(persons):
                    mid_window = int(window_size/2)
                    sequence_hip = hips[i:i+window_size,idx,:]
                    # np.stack((hips[i,idx,:], hips[i+mid_window,idx,:], hips[i+window_size,idx,:])) 
                    # print(sequence_hip.shape)
                    # trajectory_img = draw_trajectory(sequence_hip, trajectory_img)
                    trajectory_black = draw_trajectory(sequence_hip, trajectory_black)
                    # inital_trajectory_img = draw_trajectory(sequence_hip, inital_trajectory_img)

                    bbox_start, bbox_end = bbox[i+window_size,idx][:2], bbox[i+window_size,idx][2:]
                    init_bbox_start, init_bbox_end = bbox[i,idx][:2], bbox[i,idx][2:]

                    bbox_start = (int(bbox_start[0]), int(bbox_start[1]))
                    bbox_end = (int(bbox_end[0]), int(bbox_end[1]))
                    init_bbox_start = (int(init_bbox_start[0]), int(init_bbox_start[1]))
                    init_bbox_end = (int(init_bbox_end[0]), int(init_bbox_end[1]))
                    print(bbox_start, bbox_end)

                    if bbox_expand>0:
                        center = [int(bbox_start[0]+bbox_end[0])/2, int(bbox_start[1]+bbox_end[1])/2]
                        width = center[0]-bbox_start[0]
                        height = center[1]-bbox_start[1]
                        expand_width = width*(1+bbox_expand)
                        expand_height = height*(1+bbox_expand)
                        bbox_start = (center[0]-expand_width,center[1]-expand_height)
                        bbox_end = (center[0]+expand_width,center[1]+expand_height)

                    # cv2.rectangle(trajectory_img, bbox_start, bbox_end, color=(0,0,255), thickness=1)
                    cv2.rectangle(trajectory_black, init_bbox_start, init_bbox_end, color=(0,0,255), thickness=1)
                    cv2.rectangle(inital_trajectory_img, init_bbox_start, init_bbox_end, color=(0,0,255), thickness=1)

                save_final_path =  os.path.join(target_path, f"{i:04d}_{i+window_size:04d}.jpg")
                save_trajectory_path =  os.path.join(traj_path, f"{i:04d}_{i+window_size:04d}.jpg")
                save_start_path =  os.path.join(initial_path, f"{i:04d}_{i+window_size:04d}.jpg")
                save_bbox_path = os.path.join(anno_path, f"{i:04d}_{i+window_size:04d}.json")

                cv2.imwrite(save_final_path, final_img)
                cv2.imwrite(save_start_path, inital_trajectory_img)  
                cv2.imwrite(save_trajectory_path, trajectory_black) 

                # with open(save_bbox_path, "w") as f:


            print(distance)


    return hips, visual_count

def draw_trajectory_interpolate(keypoints, image):
    print(keypoints)

    x = keypoints[:, 0]
    y = keypoints[:, 1]

    f = interp1d(x, y, kind='cubic')

    x_new = np.linspace(x.min(), x.max(), 100)
    y_new = f(x_new)
    # tck, u = splprep(keypoints.T, s=0)
    # u_new = np.linspace(u.min(), u.max(), 50)
    # x_new, y_new = splev(u_new, tck, der=0)

    smooth_points = np.vstack((x_new, y_new)).T.astype(np.int32)
    print(smooth_points.shape)

    for i in range(len(smooth_points) - 1):
        if i == len(smooth_points)-2:
            print("end i")
            cv2.arrowedLine(image, tuple(smooth_points[i-10]), tuple(smooth_points[i + 1]), (0, 0, 255), 2, tipLength=1)
            # draw_arrow(image, smooth_points[0], smooth_points[i + 1], 10, 30, (0, 0, 255), 2)
        else:
            cv2.line(image, tuple(smooth_points[i]), tuple(smooth_points[i + 1]), (0, 0, 255), 2)
    
    return image

def draw_trajectory(keypoints, image):
    smooth_points = keypoints.astype(np.int32)
    
    cv2.arrowedLine(image, tuple(smooth_points[0]), tuple(smooth_points[-1]), (0, 0, 255), 2, tipLength=0.3)
    
    return image

def draw_arrow(image, start_point, end_point, arrow_length, arrow_angle, color, thickness):

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
    boxes = boxes.xyxy.tolist()[0]
    x1, y1, x2, y2 = boxes
    bboxs_start = (int(x1), int(y1))
    bboxs_end = (int(x2), int(y2))

    # print(bboxs_start)
    # bboxs_starts.append(bboxs_start)
    # bboxs_ends.append(bboxs_end)


    return bboxs_start, bboxs_end

def object_tracking(video_path):
    first_img = True
    res_list = []
    imgs = sorted(os.listdir(video_path))
    for img_path in imgs:
        if img_path.endswith(".jpg"):
            img = cv2.imread(os.path.join(video_path, img_path))
            res_boxes = []
            if first_img:
                results = model.track(img, persist=False, conf=0.6)
                first_img = False
            else:
                results = model.track(img, persist=True, conf=0.6)
            boxes = results[0].boxes.xyxy
            for result in results[0].boxes:
                if result.cls == 0:
                    res_boxes.append(result.xywh)
            
            # print(res_boxes)
            res_list.append(boxes.tolist())
    check_moving(res_list, 10)
    return res_list

def pose_tracking(video_path):
    first_img = True
    res_list = []
    res_points = []
    imgs = sorted(os.listdir(video_path))
    for img_path in imgs:
        if img_path.endswith(".jpg"):
            img = cv2.imread(os.path.join(video_path, img_path))
            height, width = img.shape[0], img.shape[1]
            res_boxes = []
            res_keypoint = []
            if first_img:
                results = model.track(img, persist=False, conf=0.5)
                first_img = False
            else:
                results = model.track(img, persist=True, conf=0.5)
            # print(results)
            keypoints = results[0].keypoints.xy

            if keypoints.shape[1]==0 or torch.sum(keypoints[:,11:13]==0)>0:
                print("no detections or break detections !!!!")
                # hip = (keypoints[:,12]+keypoints[:,13])/2
                res_list.append([])
                res_points.append([])
            else:
                hip = (keypoints[:,11]+keypoints[:,12])/2

                for i in range(len(results[0].boxes)):
                    result = results[0].boxes[i]
                    if result.cls == 0:
                        res_boxes.append(result.xyxy.tolist()[0])
                        res_keypoint.append(hip[i].tolist())

                res_list.append(sorted(res_boxes))
                res_points.append(sorted(res_keypoint))
    print(res_list, res_points)
    check_moving(res_points, height, width, 10)

    return res_list, res_points, height, width

def tracking_all(videos, save_path=None, save_json_path=None):
    if not os.path.exists(save_json_path):
        os.makedirs(save_json_path)

    single_videos = os.listdir(videos)
    print(len(single_videos))
    # filtered_res = {}
    for single_video in tqdm(single_videos):
        filtered_res = {}
        tracked_people, tracked_points, img_h, img_w = pose_tracking(os.path.join(videos, single_video))
        inital_count = len(tracked_people[0])
        can_use = True
        count = 0
        for frame_res in tracked_people:
            if inital_count>3:
                can_use = False
                break
            if len(frame_res) == inital_count:
                count += 1
                continue
            elif len(frame_res)>inital_count or len(frame_res)<inital_count:
                print(f"multiple people: {len(frame_res)}")
                if count>10:
                    if check_moving(tracked_points, img_h, img_w, 10):
                        print(f"partial stable video: {os.path.join(videos, single_video)} with {count} framess")
                        filtered_res[os.path.join(videos, single_video)] = [tracked_people[:count], tracked_points[:count]]
                        if save_path!=None:
                            copy_partial(os.path.join(videos, single_video), os.path.join(save_path, single_video), count)
                        with open(os.path.join(save_json_path, single_video+'_filtered_data.json'), 'w') as json_file:
                            json.dump(filtered_res, json_file, indent=4)
                can_use = False
                break
        if can_use:
            if check_moving(tracked_points, img_h, img_w, 10):
                print(f"stable person video: {os.path.join(videos, single_video)}")
                # filtered_res.append(os.path.join(videos, single_video))
                filtered_res[os.path.join(videos, single_video)] = [tracked_people, tracked_points]
                if save_path!=None:
                    shutil.copytree(os.path.join(videos, single_video), os.path.join(save_path, single_video))
                print(filtered_res.keys())
                with open(os.path.join(save_json_path, single_video+'_filtered_data.json'), 'w') as json_file:
                    json.dump(filtered_res, json_file, indent=4)
                # return filtered_res

    # with open(os.path.join(save_path, 'filtered_data.json'), 'w') as json_file:
    #     json.dump(filtered_res, json_file, indent=4)
    # return filtered_res

def copy_partial(src, det, count):
    if not os.path.exists(det):
        os.makedirs(det)
    files = sorted(os.listdir(src))
    for i in range(count):
        filename = files[i]
        shutil.copy2(os.path.join(src,filename), os.path.join(det,filename))

def check_moving(bbox_res, height, width, count=None):
    print(f"height: {height}, width: {width}")
    if len(bbox_res[0])==0 or len(bbox_res[count])==0:
        return False

    inital = bbox_res[0][0]
    if count != None:
        final = bbox_res[count][0]
    else:
        final = bbox_res[-1][0]

    # print(inital)
    # print(final)

    if len(inital) == 4: # boxes input
        center_inital = [(inital[0]+inital[2])/2, (inital[1]+inital[3])/2]
        center_final = [(final[0]+final[2])/2, (final[1]+final[3])/2]
        distance = np.sqrt((center_final[0]-center_inital[0])**2 + (center_final[1]-center_inital[1])**2) 
    elif len(inital) == 2: # hip keypoint input
        distance = np.sqrt((final[0]-inital[0])**2 + (final[1]-inital[1])**2) 
        norm_distance = np.sqrt(((final[0]-inital[0])/width)**2 + ((final[1]-inital[1])/height)**2) 
    print(distance, norm_distance)

    return norm_distance>0.06 and norm_distance<0.4
        

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

def process_draw(data_path):
    subset_paths = os.listdir(data_path)
    for subset_path in subset_paths:
        sub_path = os.path.join(data_path, subset_path)
        folders = os.listdir(data_path)
        all_count = 0
        for folder in folders:
            if os.path.isdir(os.path.join("/home/longbin/workspace/animate/dataset/MPII/1", folder)):
                hips, count = mpii_visual("/home/longbin/workspace/animate/dataset/MPII/1", folder, "MPII/processed_1_json", "MPII/Anno_inter")
                print(f"visual count {count}")
                all_count += count
        print(all_count)

# data_path = "/data/longbinji/animate-images/dataset/MPII/processed_5"
# folders = os.listdir(data_path)
# all_count = 0
# for folder in folders:
#     if os.path.isdir(os.path.join("/data/longbinji/animate-images/dataset/MPII/5", folder)):
#         hips, count = mpii_visual("/data/longbinji/animate-images/dataset/MPII/5", folder, "MPII/processed_5_json", "MPII/Anno_new")
#         print(f"visual count {count}")
#         all_count += count
# print(all_count)

# mpii_visual("/home/longbin/workspace/animate/dataset/MPII/1", "000919705", "MPII/processed_1/filtered_data.json", "MPII/anno_1")
# mpii_visual("/home/longbin/workspace/animate/dataset/MPII/1", "053211349", "MPII/processed_1/filtered_data.json", "MPII/anno_1")
# /home/longbin/workspace/animate/dataset/MPII/processed_1/002246931

# split_video("human36_video/S1/Videos/Walking 1.54138969.mp4","human36_video/S1/Images")
# pose_tracking("/home/longbin/workspace/animate/dataset/MPII/1/022864281")
# pose_tracking("/home/longbin/workspace/animate/dataset/MPII/1/096244729")
# /home/longbin/workspace/animate/dataset/MPII/1/096361998

tracking_all("/data/longbinji/animate-images/dataset/MPII/1", "/data/longbinji/animate-images/dataset/MPII/processed_1_norm", "/data/longbinji/animate-images/dataset/MPII/processed_1_norm_json")
tracking_all("/data/longbinji/animate-images/dataset/MPII/2", "/data/longbinji/animate-images/dataset/MPII/processed_2_norm", "/data/longbinji/animate-images/dataset/MPII/processed_2_norm_json")
tracking_all("/data/longbinji/animate-images/dataset/MPII/3", "/data/longbinji/animate-images/dataset/MPII/processed_3_norm", "/data/longbinji/animate-images/dataset/MPII/processed_3_norm_json")
tracking_all("/data/longbinji/animate-images/dataset/MPII/4", "/data/longbinji/animate-images/dataset/MPII/processed_4_norm", "/data/longbinji/animate-images/dataset/MPII/processed_4_norm_json")
tracking_all("/data/longbinji/animate-images/dataset/MPII/5", "/data/longbinji/animate-images/dataset/MPII/processed_5_norm", "/data/longbinji/animate-images/dataset/MPII/processed_5_norm_json")
# tracking_all("/home/longbin/workspace/animate/dataset/MPII/2", "/home/longbin/workspace/animate/dataset/MPII/processed_2", "/home/longbin/workspace/animate/dataset/MPII/processed_2_json")
# tracking_all("/home/longbin/workspace/animate/dataset/MPII/3", "/home/longbin/workspace/animate/dataset/MPII/processed_3", "/home/longbin/workspace/animate/dataset/MPII/processed_3_json")
# tracking_all("/home/longbin/workspace/animate/dataset/MPII/2", "/home/longbin/workspace/animate/dataset/MPII/processed_2")
# yolo_box_detect("/home/longbin/whome/longbin/workspace/animate/dataset/MPII/2/096910263orkspace/animate/dataset/human36_video/S1/Images/Walking 1.54138969.mp4/frame_0120.jpg")

