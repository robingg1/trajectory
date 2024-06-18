data_path = "/home/longbin/workspace/animate/dataset/MPII/1/011775108"
data_path = "/home/longbin/workspace/animate/dataset/MPII/1/067189274"
# import mediapipe as mp
import os
import h5py
import cv2
import os
import numpy as np
# from scipy.interpolate import splprep, splev
# from scipy.interpolate import interp1d
from utils import frame_utils
import imageio
from ultralytics import YOLO
from tqdm import tqdm
import json
import shutil
import torch
from unimatch.unimatch import UniMatch
from utils.file_io import extract_video
from glob import glob
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose


# model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model
model = YOLO("yolov8l-pose.pt")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

unimodel = UniMatch(feature_channels=128,
                     num_scales=2,
                     upsample_factor=4,
                     num_head=1,
                     ffn_dim_expansion=4,
                     num_transformer_layers=6,
                     reg_refine=True,
                     task='flow').to(device)

checkpoint = torch.load('/data/longbinji/animate-images/unimatch/pretrained/gmflow-scale2-mixdata-train320x576-9ff1c094.pth', map_location='cpu')

unimodel.load_state_dict(checkpoint['model'], strict=False)

cotracker_model = CoTrackerPredictor(checkpoint="cotracker2.pth")

model_configs = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
}

encoder = 'vitb' # or 'vitb', 'vits'
depth_anything = DepthAnything(model_configs[encoder])
depth_anything.load_state_dict(torch.load(f'depth_anything_{encoder}14.pth'))
depth_anything = depth_anything.to(device).eval()
# total_params = sum(param.numel() for param in depth_anything.parameters())
# print('Total parameters: {:.2f}M'.format(total_params / 1e6))

# SAM
sam_checkpoint = "./sam/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

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

# def down_sample(image_path, window=5):


def mask_from_segmentation(seg_path, output_path):
    all_masks = []
    all_bboxs = []
    # basename = os.path.split(seg_path)[1]
    # output_path = os.path.join(output_dir, basename)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    seg_images = sorted(os.listdir(seg_path))
    for seg_image in seg_images:
        bboxs = []
        masks = []
        seg_mask = cv2.imread(os.path.join(seg_path, seg_image))
        gray_mask = cv2.cvtColor(seg_mask, cv2.COLOR_BGR2GRAY)
    
        masks_and_bboxes = []
        unique_colors = np.unique(seg_mask.reshape(-1, seg_mask.shape[2]), axis=0)
        # print(unique_colors)
        mask_num = 0
        
        for color in unique_colors:
            if sum(color)!=0:
                fake_mask = np.zeros(seg_mask.shape)
                color_mask = cv2.inRange(seg_mask, color, color)

                non_zero_coords = np.argwhere(color_mask)

                top_left = non_zero_coords.min(axis=0)
                bottom_right = non_zero_coords.max(axis=0)
                
                x, y = top_left[1], top_left[0]
                w, h = bottom_right[1], bottom_right[0]
                
                bboxs.append(np.array([x, y, w, h]))
                print(color_mask.shape)
                save_color_mask = np.expand_dims(color_mask, axis=0)
                masks.append(save_color_mask)
                
        bboxs = np.vstack(bboxs)
        masks = np.vstack(masks)
        print(bboxs.shape)
        print(masks.shape)
        
        if len(all_bboxs) == 0 or len(bboxs) == len(all_bboxs[-1]):
            all_bboxs.append(bboxs)
            all_masks.append(masks)
        else:
            print("object number mismatch, truncate saving")
            break

    all_bboxs = np.array(all_bboxs)
    all_masks = np.array(all_masks)
    print(all_bboxs.shape)
    np.save(os.path.join(output_path, "masks.npy"), all_masks)
    np.save(os.path.join(output_path, "bboxs.npy"), all_bboxs)

    return all_bboxs, all_masks

def read_video_from_path(folder_path):
    image_files = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    
    frames = []
    for image_file in image_files:
        # 构建每个图像文件的完整路径
        image_path = os.path.join(folder_path, image_file)
        try:
            # 读取图像文件
            image = imageio.imread(image_path)
            frames.append(np.array(image))
        except Exception as e:
            print(f"Error reading image file {image_file}: {e}")
            continue
        
    # 将所有帧堆叠成一个三维 NumPy 数组
    if frames:
        return np.stack(frames)
    else:
        print("No valid image files found in the folder.")
        return None

# mask_from_segmentation("/data/longbinji/animate-images/dataset/DAVIS/Annotations/480p/bike-packing", "/data/longbinji/animate-images/dataset/DAVIS/bbox_mask/480p/bike-packing")

# DAVIS_path = "/data/longbinji/animate-images/dataset/DAVIS"

# for file_path in tqdm(os.listdir(os.path.join(DAVIS_path, "Annotations", "480p"))):
#     print(file_path)
#     mask_from_segmentation(os.path.join(DAVIS_path, "Annotations", "480p", file_path), os.path.join(DAVIS_path, "bbox_mask", "480p", file_path))
def depth_estimation(img_path):
    
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    filenames = os.listdir(img_path)
    filenames = [os.path.join(img_path, filename) for filename in filenames if not filename.startswith('.')]
    filenames.sort()
    depths = []
    
    for filename in tqdm(filenames):
        raw_image = cv2.imread(filename)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        
        h, w = image.shape[:2]
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            depth = depth_anything(image)
        
        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        
        depth = depth.cpu().numpy()
        depths.append(depth)
        
    depths = np.array(depths)
        
    return depths

def mpii_visual_flow(base_path, video_name, anno_path, output_path, thre=500, window_size=4, window_step=2, step_size=1, traj_num=10, point_step=10, src_bboxs=None, src_masks=None, anno_depth=True):
    if src_bboxs!=None:
        print("loading extracted bboxs")
        bboxs = np.load(src_bboxs)
        print(bboxs.shape)
        image_path = os.path.join(base_path, video_name)
    elif anno_path == None:
        image_path = os.path.join(base_path, video_name)
        results = model.predict(str(image_path), conf=0.5)
        res_bboxs = []
    
        # Process results list
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            res_bboxs.append(boxes.xyxy.cpu().numpy())
        res_bboxs = np.array(res_bboxs)
        bboxs = res_bboxs
        
        image_path = os.path.join(base_path, video_name)
    else:
        process_base_path = "processed_"+os.path.split(base_path)[1]+"_norm"
        new_base_path = os.path.join(os.path.split(base_path)[0], process_base_path)
        
        ori_image_path = os.path.join(base_path, video_name)
        image_path = os.path.join(new_base_path, video_name)
        
        visual_count = 0
        cur_anno = os.path.join(anno_path, video_name+"_filtered_data.json")
        with open(cur_anno, "r") as f:
            data_file = json.load(f)

        print(data_file.keys())
        
        bboxs = data_file[ori_image_path][0]

    output_path = os.path.join(output_path, video_name)

    traj_path = os.path.join(output_path, "traj")
    images_path = os.path.join(output_path, "images")

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(traj_path):
        os.makedirs(traj_path)
    # if not os.path.exists(images_path):
    #     os.makedirs(initial_path)

    flows = inference_flow(unimodel,
                       inference_dir=image_path,
                       inference_video=None,
                       output_path=output_path,
                       padding_factor=32,
                       bboxs=bboxs,
                       inference_size=None,
                       save_flo_flow=False,
                       attn_type='swin',
                       attn_splits_list=[2, 8],
                       corr_radius_list=[-1, 4],
                       prop_radius_list=[-1, 1],
                       pred_bidir_flow=False,
                       pred_bwd_flow=False,
                       num_reg_refine=6,
                       fwd_bwd_consistency_check=False,
                       save_video=False,
                       concat_flow_img=False,
                       )
    
    if anno_depth:
        depths = depth_estimation(image_path)

    flows = np.stack(flows, axis=0) # t*H*W*2
    # print(flows.shape)
    # u_flow = flows[:, :, :, 0]
    # v_flow = flows[:, :, :, 1]
    # sigma = (2, 2, 2)
    # u_flow_smoothed = gaussian_filter(u_flow, sigma=sigma)
    # v_flow_smoothed = gaussian_filter(v_flow, sigma=sigma)

    # flows = np.stack((u_flow_smoothed, v_flow_smoothed), axis=-1)
    
    filenames = sorted(glob(image_path + '/*.png') + glob(image_path + '/*.jpg'))
    
    if src_masks!=None:
        loaded_masks = np.load(src_masks)
        length_mask = len(loaded_masks)
        print(f"length: {loaded_masks.shape} !!!!!!!!!!!!!!")
        filenames = filenames[:length_mask]

    for window_id in range(0, len(filenames)-window_step, window_step):
        if not window_id>=len(filenames)-window_size:
            flow_res = []
            for test_id in range(window_id, window_size+window_id, step_size):

                if len(flow_res) == 0:
                    ori_image1 = Image.open(filenames[test_id])
                    ori_image2 = Image.open(filenames[test_id+step_size])

                    ori_image1 = np.array(ori_image1).astype(np.uint8)
                    ori_image1 = cv2.cvtColor(ori_image1, cv2.COLOR_RGB2BGR)

                    ori_image2 = np.array(ori_image2).astype(np.uint8)
                    ori_image2 = cv2.cvtColor(ori_image2, cv2.COLOR_RGB2BGR)
                    
                    mask_image = np.zeros(ori_image1.shape)

                    sample_x = []
                    sample_y = []

                    if src_masks!=None:
                        print(loaded_masks.shape, "~!!!!!!!!!!!!!!")
                        masks = loaded_masks[test_id]
                        masks = torch.tensor(masks)
                        print(masks.shape, "~!!!!!!!!!!!!!!")
                        masks = masks.unsqueeze(1)
                        print(masks.shape)
                    else:
                        print("No extracted masks, use SAM for segmentation")

                        predictor.set_image(ori_image1)

                        input_boxes = torch.tensor([bboxs[test_id]], device=predictor.device)
                        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, ori_image1.shape[:2])
                        masks, _, _ = predictor.predict_torch(
                            point_coords=None,
                            point_labels=None,
                            boxes=transformed_boxes,
                            multimask_output=False,
                        )
                        masks = masks.cpu()
                        print(masks.shape)

                    all_points = []

                    # for bbox in bboxs[test_id]:
                    for i in range(len(bboxs[test_id])):
                        bbox = bboxs[test_id][i]
                        int_bbox = [int(x) for x in bbox]
                        # h_bot, w_left, h_top, w_right = int_bbox

                        w_left, h_bot, w_right, h_top = int_bbox
                        print(int_bbox)
                        # print(image1.size())
                        cur_sample_x, cur_sample_y = np.mgrid[point_step:w_right-point_step:point_step, point_step:h_top-point_step:point_step]

                        sample_x.append(cur_sample_x.ravel())
                        sample_y.append(cur_sample_y.ravel())

                        # sample_x = np.concatenate(sample_x, axis=0)
                        # sample_y = np.concatenate(sample_y, axis=0)

                        # sampling_points = np.vstack((sample_x.ravel(), sample_y.ravel())).T # n*2

                        mask = masks[i][0]
                        print("mask shape ", mask.shape)
                        valid_indices = mask.nonzero()
                        print(valid_indices.shape)
                        # valid_indices_set = set(map(tuple, valid_indices.tolist()))
                        # # print(valid_indices_set)
                        # print(sampling_points[0].tolist())
                        # in_mask_samples = [point for point in sampling_points if tuple(point.tolist()) in valid_indices_set]
                        # in_mask_samples = np.array(in_mask_samples)
                        # print("inside points ", in_mask_samples.shape)
                    
                        sample_num = np.random.randint(2,traj_num) 
                        indices = np.random.choice(valid_indices.shape[0], sample_num, replace=False)
                        partial_sampling_points = valid_indices[indices]
                        print(partial_sampling_points.shape)
                        all_points.append(partial_sampling_points)
                        print("draw box !!!!!!!!!")

                        # cv2.rectangle(ori_image1, (w_left, h_bot), (w_right, h_top), color=(0,0,255), thickness=2)
                        # cv2.rectangle(mask_image, (w_left, h_bot), (w_right, h_top), color=(0,0,255), thickness=2)

                    # sample_x = np.concatenate(sample_x, axis=0)
                    # sample_y = np.concatenate(sample_y, axis=0)

                    # sampling_points = np.vstack((sample_x.ravel(), sample_y.ravel())).T
                    
                    # sample_num = np.random.randint(2,traj_num) 
                    # indices = np.random.choice(sampling_points.shape[0], sample_num, replace=False)
                    # sampling_points = sampling_points[indices]
                    all_points = np.vstack(all_points)
                    print(all_points.shape)
                    print(all_points[0])
                    all_points = all_points[:, [1, 0]]
                    print(all_points[0])
                    flow_res.append(all_points)

                new_points = np.round(flow_res[-1] + flows[test_id][flow_res[-1][:, 1], flow_res[-1][:, 0]]).astype(int)

                new_xs = new_points[:, 1]
                new_ys = new_points[:, 0]
                new_xs[new_xs>=flows[0].shape[0]] = flows[0].shape[0]-1
                new_ys[new_ys>=flows[0].shape[1]] = flows[0].shape[1]-1
                new_xs[new_xs<=0] = 0
                new_ys[new_ys<=0] = 0
                new_points_fixed = np.stack((new_ys, new_xs), axis=1)
                # print(new_points_fixed)
                flow_res.append(new_points_fixed)
                
            line_color = (0, 0, 255)

            for i in range(len(flow_res)-1):
                for (x1, y1), (x2, y2) in zip(flow_res[i], flow_res[i+1]):
                    if anno_depth:
                        cur_depth = depths[i+1][y2, x2]
                        print(f"depth: {cur_depth}, !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
                        line_color = (int(cur_depth * 255), int(cur_depth * 255), 255)
                    # if i == len(flow_res)-2:
                    #     # cv2.arrowedLine(mask_image, (x1, y1), (int(x2), int(y2)), (0, 0, 255), 2)
                    #     cv2.circle(mask_image, (int(x2), int(y2)), 2, (0, 255, 0), -1)
                    cv2.line(mask_image, (x1, y1), (int(x2), int(y2)), line_color, 2)
                    
            sigma = 3
            kernel_size = (99, 99)
            fs = np.transpose(mask_image, (2, 0, 1))
            fg = np.empty_like(fs)
            for c in range(fs.shape[0]):
                fg[c] = cv2.GaussianBlur(fs[c], kernel_size, sigma)
            fg = np.transpose(fg, (1, 2, 0))
            fg = cv2.normalize(fg, None, 0, 255, cv2.NORM_MINMAX)
            
            for (x2, y2) in flow_res[-1]:
                cv2.circle(fg, (int(x2), int(y2)), 3, (0, 255, 0), -1)

            output_traj_file = os.path.join(output_path, 'traj', '%04d_%04d.png' % (test_id, test_id+window_size))
            # output_initial_file = os.path.join(output_path, 'initial', '%04d_%04d.png' % test_id, test_id+window_size)
            # output_traj_file = os.path.join(output_path, 'target', '%04d_%04d.png' % test_id, test_id+window_size)
            cv2.imwrite(output_traj_file, fg)

    shutil.copytree(image_path, os.path.join(output_path, 'images'))
    np.save(os.path.join(output_path, "bbox.npy"), bboxs)
    

def mpii_visual_cotracker(base_path, video_name, anno_path, output_path, thre=500, window_size=4, window_step=2, step_size=1, traj_num=10, point_step=10, src_bboxs=None, src_masks=None):
    if src_bboxs!=None:
        print("loading extracted bboxs")
        bboxs = np.load(src_bboxs)
        print(bboxs.shape)
        image_path = os.path.join(base_path, video_name)
    elif anno_path == None:
        image_path = os.path.join(base_path, video_name)
        results = model.predict(str(image_path), conf=0.5)
        res_bboxs = []
    
        # Process results list
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            res_bboxs.append(boxes.xyxy.cpu().numpy())
        res_bboxs = np.array(res_bboxs)
        bboxs = res_bboxs
        
        image_path = os.path.join(base_path, video_name)
    else:
        process_base_path = "processed_"+os.path.split(base_path)[1]+"_norm"
        new_base_path = os.path.join(os.path.split(base_path)[0], process_base_path)
        
        ori_image_path = os.path.join(base_path, video_name)
        image_path = os.path.join(new_base_path, video_name)
        
        visual_count = 0
        cur_anno = os.path.join(anno_path, video_name+"_filtered_data.json")
        with open(cur_anno, "r") as f:
            data_file = json.load(f)

        print(data_file.keys())
        
        bboxs = data_file[ori_image_path][0]
        
    pre_box = bboxs[0]

    output_path = os.path.join(output_path, video_name)

    traj_path = os.path.join(output_path, "traj")
    images_path = os.path.join(output_path, "images")
    
    video = read_video_from_path(image_path)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    
    tracker_model = cotracker_model.to(device)
    video = video.to(device)
    
    filenames = sorted(glob(image_path + '/*.png') + glob(image_path + '/*.jpg'))
    
    ori_image1 = Image.open(filenames[0])
    ori_image1 = np.array(ori_image1).astype(np.uint8)
    ori_image1 = cv2.cvtColor(ori_image1, cv2.COLOR_RGB2BGR)
    
    if src_masks!=None:
        loaded_masks = np.load(src_masks)
        length_mask = len(loaded_masks)
        print(f"length: {loaded_masks.shape} !!!!!!!!!!!!!!")
        filenames = filenames[:length_mask]
        masks = torch.tensor(loaded_masks)
    else:
        predictor.set_image(ori_image1)
        input_boxes = torch.tensor(pre_box, device=predictor.device)
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, ori_image1.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        masks = masks.cpu().float()
        
        print(masks.shape)
        
    pred_tracks = []
        
    for i in range(len(pre_box)):
        mask = masks[i][0]
        segm_mask = mask[None, None]
        print(mask.shape, segm_mask.shape)
        
        cur_pred_tracks, pred_visibility = tracker_model(
            video,
            grid_size=100,
            grid_query_frame=0,
            backward_tracking="store_true",
            segm_mask=segm_mask
        )
        print(cur_pred_tracks.shape)
        
        pred_tracks.append(cur_pred_tracks.cpu().numpy())
        
    pred_tracks = np.concatenate(pred_tracks, axis=2)
    print(pred_tracks.shape, "@@@@@@@@@@@@@@")
    
    if pred_tracks.shape[2]==0:
        return None

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(traj_path):
        os.makedirs(traj_path)

    for window_id in range(0, len(filenames)-window_step, window_step):
        if not window_id>=len(filenames)-window_size:
            
            mask_image = np.zeros(ori_image1.shape)
            
            for i in range(len(bboxs[window_id])):
                bbox = bboxs[window_id][i]
                int_bbox = [int(x) for x in bbox]

                w_left, h_bot, w_right, h_top = int_bbox
                
                cv2.rectangle(mask_image, (w_left, h_bot), (w_right, h_top), color=(0,0,255), thickness=2)

            sample_num = np.random.randint(2, traj_num) 
            indices = np.random.choice(pred_tracks.shape[2], sample_num*len(bboxs[window_id]), replace=False)
            sampling_trackers = pred_tracks[:, :, indices, :]
            print(sample_num, sampling_trackers.shape, "111111111")
            sampling_trackers = sampling_trackers[0, window_id:window_id+window_size]
            print(sample_num, sampling_trackers.shape, "111111111")
            
            for i in range(len(sampling_trackers)-1):
                for (x1, y1), (x2, y2) in zip(sampling_trackers[i], sampling_trackers[i+1]):
                    # print(x1, y1, x2, y2)
                    if i == len(sampling_trackers)-2:
                        # cv2.arrowedLine(mask_image, (x1, y1), (int(x2), int(y2)), (0, 0, 255), 2)
                        cv2.circle(mask_image, (int(x2), int(y2)), 2, (0, 255, 0), -1)
                    cv2.line(mask_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            output_traj_file = os.path.join(output_path, 'traj', '%04d_%04d.png' % (window_id, window_id+window_size))
            # output_initial_file = os.path.join(output_path, 'initial', '%04d_%04d.png' % test_id, test_id+window_size)
            # output_traj_file = os.path.join(output_path, 'target', '%04d_%04d.png' % test_id, test_id+window_size)
            cv2.imwrite(output_traj_file, mask_image)

    shutil.copytree(image_path, os.path.join(output_path, 'images'))
    np.save(os.path.join(output_path, "bbox.npy"), bboxs)

            

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

    return norm_distance>0.08
        

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

@torch.no_grad()
def inference_flow(model,
                   inference_dir,
                   inference_video=None,
                   output_path='output',
                   padding_factor=8,
                   bboxs=None,
                   inference_size=None,
                   save_flo_flow=False,  # save raw flow prediction as .flo
                   attn_type='swin',
                   attn_splits_list=None,
                   corr_radius_list=None,
                   prop_radius_list=None,
                   num_reg_refine=1,
                   pred_bidir_flow=False,
                   pred_bwd_flow=False,
                   fwd_bwd_consistency_check=False,
                   save_video=False,
                   concat_flow_img=False,
                   ):
    """ Inference on a directory or a video """
    model.eval()

    if fwd_bwd_consistency_check:
        assert pred_bidir_flow

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if save_video:
        assert inference_video is not None

    fixed_inference_size = inference_size
    transpose_img = False

    if inference_video is not None:
        filenames, fps = extract_video(inference_video)  # list of [H, W, 3]
    else:
        print(inference_dir)
        filenames = sorted(glob(inference_dir + '/*.png') + glob(inference_dir + '/*.jpg'))
    print('%d images found' % len(filenames))

    vis_flow_preds = []
    ori_imgs = []

    window_size = 7
    step_size = 1
    traj_num = 8

    flows = []

    for test_id in range(0, len(filenames)-1):
        print(test_id, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        if (test_id + 1) % 50 == 0:
            print('predicting %d/%d' % (test_id + 1, len(filenames)))

        if inference_video is not None:
            image1 = filenames[test_id]
            image2 = filenames[test_id + 1]
        else:
            image1 = frame_utils.read_gen(filenames[test_id])
            image2 = frame_utils.read_gen(filenames[test_id + step_size])

        # if len(flow_res) == 0:
        #     ori_image1 = image1
        #     ori_image1 = np.array(ori_image1).astype(np.uint8)
        #     ori_image1 = cv2.cvtColor(ori_image1, cv2.COLOR_RGB2BGR)
        
        image1 = np.array(image1).astype(np.uint8)
        image2 = np.array(image2).astype(np.uint8)

        if len(image1.shape) == 2:  # gray image
            image1 = np.tile(image1[..., None], (1, 1, 3))
            image2 = np.tile(image2[..., None], (1, 1, 3))
        else:
            image1 = image1[..., :3]
            image2 = image2[..., :3]

        if concat_flow_img:
            ori_imgs.append(image1)

        image1 = torch.from_numpy(image1).permute(2, 0, 1).float().unsqueeze(0).to(device)
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float().unsqueeze(0).to(device)

        # the model is trained with size: width > height
        if image1.size(-2) > image1.size(-1):
            image1 = torch.transpose(image1, -2, -1)
            image2 = torch.transpose(image2, -2, -1)
            transpose_img = True

        nearest_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
                        int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]
        step = 10

        # if len(flow_res) == 0:
        #     sample_x = []
        #     sample_y = []
        #     for bbox in bboxs[test_id]:
        #         int_bbox = [int(x) for x in bbox]
        #         # h_bot, w_left, h_top, w_right = int_bbox

        #         w_left, h_bot, w_right, h_top = int_bbox
        #         print(int_bbox)
        #         print(image1.size())
        #         cur_sample_x, cur_sample_y = np.mgrid[w_left+step//2:w_right-step//2:step, h_bot+step//2:h_top-step//2:step]
        #         sample_x.append(cur_sample_x.ravel())
        #         sample_y.append(cur_sample_y.ravel())
        #         print("draw box !!!!!!!!!")

        #         cv2.rectangle(ori_image1, (w_left, h_bot), (w_right, h_top), color=(0,0,255), thickness=2)

        #     sample_x = np.concatenate(sample_x, axis=0)
        #     sample_y = np.concatenate(sample_y, axis=0)
            
        #     sample_num = np.random.randint(2,traj_num) 
        #     indices = np.random.choice(sample_x.shape[0], sample_num, replace=False)
        #     sample_x = sample_x[indices]
        #     sample_y = sample_y[indices]

        #     sampling_points = np.vstack((sample_x.ravel(), sample_y.ravel())).T
            
        #     flow_res.append(sampling_points)

        # resize to nearest size or specified size
        inference_size = nearest_size if fixed_inference_size is None else fixed_inference_size

        assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
        ori_size = image1.shape[-2:]

        # resize before inference
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                                align_corners=True)
            image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                                align_corners=True)

        if pred_bwd_flow:
            image1, image2 = image2, image1

        results_dict = model(image1, image2,
                            attn_type=attn_type,
                            attn_splits_list=attn_splits_list,
                            corr_radius_list=corr_radius_list,
                            prop_radius_list=prop_radius_list,
                            num_reg_refine=num_reg_refine,
                            task='flow',
                            pred_bidir_flow=pred_bidir_flow,
                            )

        flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]

        # resize back
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                    align_corners=True)
            flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
            flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

        if transpose_img:
            flow_pr = torch.transpose(flow_pr, -2, -1)

        flow = flow_pr[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]
        flows.append(flow)

    return flows



# DAVIS_path = "/data/longbinji/animate-images/dataset/DAVIS"

# for file_path in tqdm(os.listdir(os.path.join(DAVIS_path, "Annotations", "480p"))):
#     # file_path = "car-turn"
#     bbox_path = os.path.join(DAVIS_path, "bbox_mask", "480p", file_path, "bboxs.npy")
#     mask_path = os.path.join(DAVIS_path, "bbox_mask", "480p", file_path, "masks.npy")
#     # mask_from_segmentation(os.path.join(DAVIS_path, "Annotations", "480p", file_path), os.path.join(DAVIS_path, "bbox_mask", "480p", file_path))
#     hip = mpii_visual_flow(os.path.join(DAVIS_path, "JPEGImages", "480p"), file_path, None, "MPII/Anno_flow_DAVIS_gaussian", src_bboxs=bbox_path, src_masks=mask_path)
    
# human36m_path = "/data/longbinji/animate-images/dataset/human36_video_down_chunks"

# for file_path in tqdm(os.listdir(human36m_path)):
#     hip = mpii_visual_cotracker(os.path.join(human36m_path), file_path, None, "MPII/Anno_tracker_human36m")

# for file_path in tqdm(os.listdir(human36m_path)):
#     hip = mpii_visual_flow(os.path.join(human36m_path), file_path, None, "MPII/Anno_flow_human36m")
    

    

# data_path = "/data/longbinji/animate-images/dataset/MPII/processed_1_norm"
# folders = os.listdir(data_path)
# for folder in folders:
#     if os.path.isdir(os.path.join("/data/longbinji/animate-images/dataset/MPII/1", folder)):
#         hip = mpii_visual_flow("/data/longbinji/animate-images/dataset/MPII/1", folder, "MPII/processed_1_norm_json", "MPII/Anno_flow_mpii_gau")

data_path = "/data/longbinji/animate-images/dataset/MPII/processed_2_norm"
folders = os.listdir(data_path)
for folder in folders:
    if os.path.isdir(os.path.join("/data/longbinji/animate-images/dataset/MPII/2", folder)):
        hip = mpii_visual_flow("/data/longbinji/animate-images/dataset/MPII/2", folder, "MPII/processed_2_norm_json", "MPII/Anno_flow_mpii_gau")

data_path = "/data/longbinji/animate-images/dataset/MPII/processed_3_norm"
folders = os.listdir(data_path)
for folder in folders:
    if os.path.isdir(os.path.join("/data/longbinji/animate-images/dataset/MPII/3", folder)):
        hip = mpii_visual_flow("/data/longbinji/animate-images/dataset/MPII/3", folder, "MPII/processed_3_norm_json", "MPII/Anno_flow_mpii_gau")


# mpii_visual("/home/longbin/workspace/animate/dataset/MPII/process_1_norm", "000919705", "MPII/processed_1_norm", "MPII/anno_flow")
# mpii_visual("/home/longbin/workspace/animate/dataset/MPII/1", "053211349", "MPII/processed_1/filtered_data.json", "MPII/anno_1")
# /home/longbin/workspace/animate/dataset/MPII/processed_1/002246931

# split_video("human36_video/S1/Videos/Walking 1.54138969.mp4","human36_video/S1/Images")
# pose_tracking("/home/longbin/workspace/animate/dataset/MPII/1/022864281")
# pose_tracking("/home/longbin/workspace/animate/dataset/MPII/1/096244729")
# /home/longbin/workspace/animate/dataset/MPII/1/096361998


# tracking_all("/data/longbinji/animate-images/dataset/MPII/3", "/data/longbinji/animate-images/dataset/MPII/processed_3_norm", "/data/longbinji/animate-images/dataset/MPII/processed_3_norm_json")
# tracking_all("/home/longbin/workspace/animate/dataset/MPII/2", "/home/longbin/workspace/animate/dataset/MPII/processed_2", "/home/longbin/workspace/animate/dataset/MPII/processed_2_json")
# tracking_all("/home/longbin/workspace/animate/dataset/MPII/3", "/home/longbin/workspace/animate/dataset/MPII/processed_3", "/home/longbin/workspace/animate/dataset/MPII/processed_3_json")
# tracking_all("/home/longbin/workspace/animate/dataset/MPII/2", "/home/longbin/workspace/animate/dataset/MPII/processed_2")
# yolo_box_detect("/home/longbin/whome/longbin/workspace/animate/dataset/MPII/2/096910263orkspace/animate/dataset/human36_video/S1/Images/Walking 1.54138969.mp4/frame_0120.jpg")

