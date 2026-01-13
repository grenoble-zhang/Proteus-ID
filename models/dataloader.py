import gc
import json
import math
import os
import random
import threading
from contextlib import contextmanager
from typing import Iterator, Optional, Sized

import zlib
import blosc
import cv2
import decord
import numpy as np
import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from decord import VideoReader
from func_timeout import FunctionTimedOut
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import ToPILImage
from tqdm import tqdm


logger = get_logger(__name__)

log_lock = threading.Lock()

def log_error_to_file(error_message, video_path):
    with log_lock:
        with open("error_log.txt", "a") as f:
            f.write(f"Error: {error_message}\n")
            f.write(f"Video Path: {video_path}\n")
            f.write("-" * 50 + "\n")

def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil

@contextmanager
def VideoReader_contextmanager(*args, **kwargs):
    vr = VideoReader(*args, **kwargs)
    try:
        yield vr
    finally:
        del vr
        gc.collect()

def get_valid_segments(valid_frame, tolerance=5):
    valid_positions = sorted(set(valid_frame['face']).union(set(valid_frame['head'])))

    valid_segments = []
    current_segment = [valid_positions[0]]

    for i in range(1, len(valid_positions)):
        if valid_positions[i] - valid_positions[i - 1] <= tolerance:
            current_segment.append(valid_positions[i])
        else:
            valid_segments.append(current_segment)
            current_segment = [valid_positions[i]]

    if current_segment:
        valid_segments.append(current_segment)

    return valid_segments


def get_frame_indices_adjusted_for_face(valid_frames, n_frames):
    valid_length = len(valid_frames)
    if valid_length >= n_frames:
        return valid_frames[:n_frames]

    additional_frames_needed = n_frames - valid_length
    repeat_indices = []

    for i in range(additional_frames_needed):
        index_to_repeat = i % valid_length
        repeat_indices.append(valid_frames[index_to_repeat])

    all_indices = valid_frames + repeat_indices
    all_indices.sort()

    return all_indices


def generate_frame_indices_for_face(n_frames, sample_stride, valid_frame, tolerance=7, skip_frames_start_percent=0.0, skip_frames_end_percent=1.0, skip_frames_start=0, skip_frames_end=0):
    valid_segments = get_valid_segments(valid_frame, tolerance)
    selected_segment = max(valid_segments, key=len)

    valid_length = len(selected_segment)
    if skip_frames_start_percent != 0.0 or skip_frames_end_percent != 1.0:
        # print("use skip frame percent")
        valid_start = int(valid_length * skip_frames_start_percent)
        valid_end = int(valid_length * skip_frames_end_percent)
    elif skip_frames_start != 0 or skip_frames_end != 0:
        # print("use skip frame")
        valid_start = skip_frames_start
        valid_end = valid_length - skip_frames_end
    else:
        # print("no use skip frame")
        valid_start = 0
        valid_end = valid_length

    if valid_length <= n_frames:
        return get_frame_indices_adjusted_for_face(selected_segment, n_frames), valid_length
    else:
        adjusted_length = valid_end - valid_start
        if adjusted_length <= 0:
            print(f"video_length: {valid_length}, adjusted_length: {adjusted_length}, valid_start:{valid_start}, skip_frames_end: {valid_end}")
            raise ValueError("Skipping too many frames results in no frames left to sample.")

        clip_length = min(adjusted_length, (n_frames - 1) * sample_stride + 1)
        start_idx_position = random.randint(valid_start, valid_end - clip_length)
        start_frame = selected_segment[start_idx_position]

        selected_frames = []
        for i in range(n_frames):
            next_frame = start_frame + i * sample_stride
            if next_frame in selected_segment:
                selected_frames.append(next_frame)
            else:
                break

        if len(selected_frames) < n_frames:
            return get_frame_indices_adjusted_for_face(selected_frames, n_frames), len(selected_frames)

        return selected_frames, len(selected_frames)

def frame_has_required_confidence(bbox_data, frame, ID, conf_threshold=0.88):
    frame_str = str(frame)
    if frame_str not in bbox_data:
        return False

    frame_data = bbox_data[frame_str]

    face_conf = any(
        item['confidence'] > conf_threshold and item['new_track_id'] == ID
        for item in frame_data.get('face', [])
    )

    head_conf = any(
        item['confidence'] > conf_threshold and item['new_track_id'] == ID
        for item in frame_data.get('head', [])
    )

    return face_conf and head_conf

def load_binary_masks(bin_file_path):
    """
    从二进制文件加载掩码
    
    参数:
    bin_file_path: 二进制文件路径
    
    返回:
    掩码数组
    """
    with open(bin_file_path, 'rb') as f:
        # 读取形状信息（假设是3维：帧数，高度，宽度）
        shape = np.fromfile(f, dtype=np.int32, count=3)
        # 读取掩码数据，使用uint16类型
        binary_masks = np.fromfile(f, dtype=np.uint8)
        binary_masks = binary_masks.reshape(shape)
    return binary_masks


def decompress_3d_direct(input_file='3d_direct.bin'):
    """解压缩由compress_3d_direct压缩的3D数组"""
    with open(input_file, 'rb') as f:
        # 读取形状信息（3个int32值）
        shape_bytes = f.read(3 * 4)
        shape_info = np.frombuffer(shape_bytes, dtype=np.int32)
        
        # 读取数据类型信息
        dtype_len_bytes = f.read(4)  # 读取dtype字符串长度(1个int32)
        dtype_len = np.frombuffer(dtype_len_bytes, dtype=np.int32)[0]
        dtype_str = f.read(dtype_len).decode('utf-8')
        data_type = np.dtype(dtype_str)
        
        # 读取压缩数据
        compressed_data = f.read()
        
        # 解压缩blosc数据
        decompressed_bytes = blosc.decompress(compressed_data)
        
        # 将字节转换回numpy数组
        result = np.frombuffer(decompressed_bytes, dtype=data_type)
        
        # 重塑为原始形状
        result = result.reshape(tuple(shape_info))
        
        return result.astype(np.uint8)

def load_compressed_masks(file_path):
    with open(file_path, 'rb') as f:
        # 读取形状信息
        shape_info = np.fromfile(f, dtype=np.int32, count=3)
        num_frames, height, width = shape_info
        
        # 读取压缩数据长度和压缩数据
        compressed_size = np.fromfile(f, dtype=np.int32, count=1)[0]
        compressed_data = f.read(compressed_size)
        
        # 解压缩
        decompressed_data = zlib.decompress(compressed_data)
        masks_array = np.frombuffer(decompressed_data, dtype=np.uint8).reshape(num_frames, height, width)
        
        return masks_array

def check_dictionaries(dict1, dict2, dict1_name="dict1", dict2_name="dict2"):
    """
    检查两个字典的键和值是否完全一致
    
    参数:
    dict1, dict2: 要比较的两个字典
    dict1_name, dict2_name: 字典的名称，用于输出
    
    返回:
    bool: 两个字典是否完全一致
    """
    # 检查键是否一致
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    
    if keys1 != keys2:
        print(f"键不一致:")
        print(f"仅在 {dict1_name} 中的键: {keys1 - keys2}")
        print(f"仅在 {dict2_name} 中的键: {keys2 - keys1}")
        return False
    
    # 检查每个键对应的值是否一致
    all_equal = True
    for key in keys1:
        values1 = dict1[key]
        values2 = dict2[key]
        
        # 检查值列表的长度是否一致
        if len(values1) != len(values2):
            print(f"键 '{key}' 的值列表长度不一致: {dict1_name}={len(values1)}, {dict2_name}={len(values2)}")
            all_equal = False
            continue
        
        # 逐个检查值是否一致
        for i, (val1, val2) in enumerate(zip(values1, values2)):
            # 使用numpy的array_equal来比较数组
            if not np.array_equal(val1, val2):
                print(f"键 '{key}' 的第 {i} 个值不一致")
                # 输出不一致的地方的详细信息
                if val1.shape != val2.shape:
                    print(f"  形状不一致: {dict1_name}={val1.shape}, {dict2_name}={val2.shape}")
                else:
                    # 计算不一致的元素数量
                    num_diff = np.sum(val1 != val2)
                    percent_diff = 100.0 * num_diff / val1.size
                    print(f"  有 {num_diff} 个元素不一致 ({percent_diff:.2f}%)")
                    
                    # 如果数组较小，可以展示具体的差异
                    if val1.size < 100:
                        print(f"  {dict1_name} 值: {val1}")
                        print(f"  {dict2_name} 值: {val2}")
                        print(f"  差异位置: {np.where(val1 != val2)}")
                
                all_equal = False
    
    if all_equal:
        print(f"{dict1_name} 和 {dict2_name} 完全一致")
    else:
        print(f"{dict1_name} 和 {dict2_name} 存在差异")
    
    return all_equal

def check_dictionaries_equality(dict1, dict2, dict1_name="dict1", dict2_name="dict2"):
    """
    检查两个字典的键和值是否完全一致，并返回不一致值的索引位置
    
    参数:
    dict1, dict2: 要比较的两个字典
    dict1_name, dict2_name: 字典的名称，用于输出
    
    返回:
    bool: 两个字典是否完全一致
    dict: 包含不一致值的索引位置信息
    """
    # 初始化用于存储差异位置的字典
    differences = {}
    
    # 检查键是否一致
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    
    if keys1 != keys2:
        print(f"键不一致:")
        print(f"仅在 {dict1_name} 中的键: {keys1 - keys2}")
        print(f"仅在 {dict2_name} 中的键: {keys2 - keys1}")
        differences['different_keys'] = {
            f'only_in_{dict1_name}': list(keys1 - keys2),
            f'only_in_{dict2_name}': list(keys2 - keys1)
        }
        return False, differences
    
    # 检查每个键对应的值是否一致
    all_equal = True
    for key in keys1:
        values1 = dict1[key]
        values2 = dict2[key]
        
        # 检查值列表的长度是否一致
        if len(values1) != len(values2):
            print(f"键 '{key}' 的值列表长度不一致: {dict1_name}={len(values1)}, {dict2_name}={len(values2)}")
            if key not in differences:
                differences[key] = {}
            differences[key]['length_mismatch'] = {
                f'{dict1_name}_length': len(values1),
                f'{dict2_name}_length': len(values2)
            }
            all_equal = False
            continue
        
        # 逐个检查值是否一致
        for i, (val1, val2) in enumerate(zip(values1, values2)):
            # 使用numpy的array_equal来比较数组
            if not np.array_equal(val1, val2):
                print(f"键 '{key}' 的第 {i} 个值不一致")
                # 输出不一致的地方的详细信息
                if key not in differences:
                    differences[key] = {}
                
                if i not in differences[key]:
                    differences[key][i] = {}
                
                if val1.shape != val2.shape:
                    print(f"  形状不一致: {dict1_name}={val1.shape}, {dict2_name}={val2.shape}")
                    differences[key][i]['shape_mismatch'] = {
                        f'{dict1_name}_shape': val1.shape,
                        f'{dict2_name}_shape': val2.shape
                    }
                else:
                    # 找出不一致的元素索引位置
                    diff_indices = np.where(val1 != val2)
                    
                    # 计算不一致的元素数量
                    num_diff = np.sum(val1 != val2)
                    percent_diff = 100.0 * num_diff / val1.size
                    print(f"  有 {num_diff} 个元素不一致 ({percent_diff:.2f}%)")
                    
                    differences[key][i]['diff_indices'] = [tuple(idx) for idx in np.array(diff_indices).T]
                    differences[key][i]['diff_count'] = int(num_diff)
                    differences[key][i]['diff_percent'] = float(percent_diff)
                    
                    # 如果数组较小，可以展示具体的差异
                    if val1.size < 100:
                        print(f"  {dict1_name} 值: {val1}")
                        print(f"  {dict2_name} 值: {val2}")
                        print(f"  差异位置: {diff_indices}")
                        
                        # 记录具体的差异值
                        diff_values = {}
                        for idx in zip(*diff_indices):
                            idx_str = str(idx)
                            diff_values[idx_str] = {
                                f'{dict1_name}_value': float(val1[idx]) if np.isscalar(val1[idx]) else val1[idx].tolist(),
                                f'{dict2_name}_value': float(val2[idx]) if np.isscalar(val2[idx]) else val2[idx].tolist()
                            }
                        differences[key][i]['diff_values'] = diff_values
                
                all_equal = False
    
    if all_equal:
        print(f"{dict1_name} 和 {dict2_name} 完全一致")
    else:
        print(f"{dict1_name} 和 {dict2_name} 存在差异")
    
    return all_equal, differences

def select_mask_frames_from_index(batch_frame, original_batch_frame, valid_id, corresponding_data, control_sam2_frame,
                                  valid_frame, bbox_data, base_dir, min_distance=3, min_frames=1, max_frames=5,
                                  mask_type='face', control_mask_type='head', dense_masks=False,
                                  ensure_control_frame=True, human_valid_id=0):
    """
    Selects frames with corresponding mask images while ensuring a minimum distance constraint between frames,
    and that the frames exist in both batch_frame and valid_frame.

    Parameters:
        base_path (str): Base directory where the JSON files and mask results are located.
        min_distance (int): Minimum distance between selected frames.
        min_frames (int): Minimum number of frames to select.
        max_frames (int): Maximum number of frames to select.
        mask_type (str): Type of mask to select frames for ('face' or 'head').
        control_mask_type (str): Type of mask used for control frame selection ('face' or 'head').

    Returns:
        dict: A dictionary where keys are IDs and values are lists of selected mask PNG paths.
    """
    # Helper function to randomly select frames with at least X frames apart
    def select_frames_with_distance_constraint(frames, num_frames, min_distance, control_frame, bbox_data, ID,
                                               ensure_control_frame=True, fallback=True):
        """
        Selects frames with a minimum distance constraint. If not enough frames can be selected, a fallback plan is applied.

        Parameters:
            frames (list): List of frame indices to select from.
            num_frames (int): Number of frames to select.
            min_distance (int): Minimum distance between selected frames.
            control_frame (int): The control frame that must always be included.
            fallback (bool): Whether to apply a fallback strategy if not enough frames meet the distance constraint.

        Returns:
            list: List of selected frames.
        """
        conf_thresholds = [0.95, 0.94, 0.93, 0.92, 0.91, 0.90]
        if ensure_control_frame:
            selected_frames = [control_frame]  # Ensure control frame is always included
        else:
            valid_initial_frames = []
            for conf_threshold in conf_thresholds:
                valid_initial_frames = [
                    f for f in frames
                    if frame_has_required_confidence(bbox_data, f, ID, conf_threshold=conf_threshold)
                ]
                if valid_initial_frames:
                    break
            if valid_initial_frames:
                selected_frames = [random.choice(valid_initial_frames)]
            else:
                # If no frame meets the initial confidence, fall back to a random frame (or handle as per your preference)
                selected_frames = [random.choice(frames)]

        available_frames = [f for f in frames if f != selected_frames[0]]  # Exclude control frame for random selection

        random.shuffle(available_frames)  # Shuffle to introduce randomness

        while available_frames and len(selected_frames) < num_frames:
            last_selected_frame = selected_frames[-1]

            valid_choices = []
            for conf_threshold in conf_thresholds:
                valid_choices = [
                    f for f in available_frames
                    if abs(f - last_selected_frame) >= min_distance and
                       frame_has_required_confidence(bbox_data, f, ID, conf_threshold=conf_threshold)
                ]
                if valid_choices:
                    break

            if valid_choices:
                frame = random.choice(valid_choices)
                available_frames.remove(frame)
                selected_frames.append(frame)
            else:
                if fallback:
                    # Fallback strategy: uniformly distribute remaining frames if distance constraint cannot be met
                    remaining_needed = num_frames - len(selected_frames)
                    remaining_frames = available_frames[:remaining_needed]

                    # Distribute the remaining frames evenly if possible
                    if remaining_frames:
                        step = max(1, len(remaining_frames) // remaining_needed)
                        evenly_selected = remaining_frames[::step][:remaining_needed]
                        selected_frames.extend(evenly_selected)
                    break
                else:
                    break  # No valid choices remain and no fallback strategy is allowed

        if len(selected_frames) < num_frames:
            return None

        return selected_frames

    # Convert batch_frame list to a set to remove duplicates
    batch_frame_set = set(batch_frame)

    # Dictionary to store selected mask PNGs
    selected_masks_dict = {}
    selected_bin_dict = {}
    selected_bboxs_dict = {}
    dense_masks_dict = {}
    human_masks_dict = {}
    dense_bin_dict = {}
    selected_frames_dict = {}

    # ID
    try:
        mask_valid_frames = valid_frame[mask_type]  # Select frames based on the specified mask type --> face
        control_valid_frames = valid_frame[control_mask_type]  # Control frames for control_mask_type  --> head
    except KeyError:
        if mask_type not in valid_frame.keys():
            print(f"no valid {mask_type}")
        if control_mask_type not in valid_frame.keys():
            print(f"no valid {control_mask_type}")

    # Get the control frame for the control mask type 控制帧 ex. 131 head
    control_frame = control_sam2_frame[valid_id][control_mask_type]

    # Filter frames to only those which are in both valid_frame and batch_frame_set
    valid_frames = []
    # valid_frames = [frame for frame in mask_valid_frames if frame in control_valid_frames and frame in batch_frame_set]
    # if frame in valid_frames is in control_valid_frames and batch_frame_set, has head and face, append
    for frame in mask_valid_frames:
        if frame in control_valid_frames and frame in batch_frame_set:
            # Check if bbox_data has 'head' or 'face' for the frame
            if str(frame) in bbox_data:
                frame_data = bbox_data[str(frame)]
                if 'head' in frame_data or 'face' in frame_data:
                    valid_frames.append(frame)

    # Ensure the control frame is included in the valid frames 提高泛化性，是否将控制帧放入valid_frames中
    if ensure_control_frame and (control_frame not in valid_frames):
        valid_frames.append(control_frame)

    # Select a random number of frames between min_frames and max_frames
    num_frames_to_select = random.randint(min_frames, max_frames)
    selected_frames = select_frames_with_distance_constraint(valid_frames, num_frames_to_select, min_distance,
                                                             control_frame, bbox_data, valid_id, ensure_control_frame)

    # Store the selected frames as mask PNGs and bbox
    selected_masks_dict[valid_id] = []
    selected_bin_dict[valid_id] = []
    selected_bboxs_dict[valid_id] = []

    # Initialize the dense_masks_dict entry for the current ID
    dense_masks_dict[valid_id] = []
    human_masks_dict[valid_id] = []
    dense_bin_dict[valid_id] = []
    # Store the selected frames in the dictionary
    selected_frames_dict[valid_id] = selected_frames

    # 1:face, 2:head, 3:person

        
    bin_file_path = f"{base_dir}/{valid_id}/masks.bin"
    body_file_path = f"{base_dir}/{human_valid_id}/masks.bin"
    if dense_masks:
        if os.path.exists(bin_file_path):
            bin_all_masks = decompress_3d_direct(bin_file_path)
            for frame in original_batch_frame:
                bin_frame_idx = int(frame)
                dense_masks_dict[valid_id].append(np.where(bin_all_masks[bin_frame_idx] > 0, 1, 0).astype(np.uint8))
                
        if os.path.exists(body_file_path):
            human_bin_all_masks = decompress_3d_direct(body_file_path)
            for frame in original_batch_frame:
                bin_frame_idx = int(frame)
                human_masks_dict[valid_id].append(np.where(human_bin_all_masks[bin_frame_idx] > 0, 1, 0).astype(np.uint8))
                
        else:
            for frame in original_batch_frame:
                mask_data_path = f"{base_dir}/{valid_id}/annotated_frame_{int(frame):05d}.png"
                mask_array = np.array(Image.open(mask_data_path))
                binary_mask = np.where(mask_array > 0, 1, 0).astype(np.uint8)
                dense_masks_dict[valid_id].append(binary_mask)
                
            for frame in original_batch_frame:
                mask_data_path = f"{base_dir}/{human_valid_id}/annotated_frame_{int(frame):05d}.png"
                mask_array = np.array(Image.open(mask_data_path))
                binary_mask = np.where(mask_array > 0, 1, 0).astype(np.uint8)
                human_masks_dict[valid_id].append(binary_mask)
                
    if os.path.exists(bin_file_path):
        bin_all_masks = decompress_3d_direct(bin_file_path)
        for frame in selected_frames:
            bin_frame_idx = int(frame)
            selected_masks_dict[valid_id].append(np.where(bin_all_masks[bin_frame_idx] > 0, 1, 0).astype(np.uint8))
            try:
                for item in bbox_data[f"{frame}"]["head"]:
                    if item['new_track_id'] == int(valid_id):
                        temp_bbox = item['box']
                        break
            except (KeyError, StopIteration):
                try:
                    for item in bbox_data[f"{frame}"]["face"]:
                        if item['new_track_id'] == int(valid_id):
                            temp_bbox = item['box']
                            break
                except (KeyError, StopIteration):
                    temp_bbox = None

            selected_bboxs_dict[valid_id].append(temp_bbox)
    else:
        for frame in selected_frames:
            mask_data_path = f"{base_dir}/{valid_id}/annotated_frame_{frame:05d}.png"
            mask_array = np.array(Image.open(mask_data_path))
            binary_mask = np.where(mask_array > 0, 1, 0).astype(np.uint8)
            selected_masks_dict[valid_id].append(binary_mask)
        
            try:
                for item in bbox_data[f"{frame}"]["head"]:
                    if item['new_track_id'] == int(valid_id):
                        temp_bbox = item['box']
                        break
            except (KeyError, StopIteration):
                try:
                    for item in bbox_data[f"{frame}"]["face"]:
                        if item['new_track_id'] == int(valid_id):
                            temp_bbox = item['box']
                            break
                except (KeyError, StopIteration):
                    temp_bbox = None

            selected_bboxs_dict[valid_id].append(temp_bbox)

    return selected_frames_dict, selected_masks_dict, selected_bboxs_dict, dense_masks_dict, human_masks_dict

def pad_tensor(tensor, target_size, dim=0):
    padding_size = target_size - tensor.size(dim)
    if padding_size > 0:
        pad_shape = list(tensor.shape)
        pad_shape[dim] = padding_size
        padding_tensor = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, padding_tensor], dim=dim)
    else:
        return tensor[:target_size]

def crop_images(selected_frame_index, selected_bboxs_dict, video_reader, return_ori=False):
    """
    Crop images based on given bounding boxes and frame indices from a video.

    Args:
        selected_frame_index (list): List of frame indices to be cropped.
        selected_bboxs_dict (list of dict): List of dictionaries, each containing 'x1', 'y1', 'x2', 'y2' bounding box coordinates.
        video_reader (VideoReader or list of numpy arrays): Video frames accessible by index, where each frame is a numpy array (H, W, C).

    Returns:
        list: A list of cropped images in PIL Image format.
    """
    expanded_cropped_images = []
    original_cropped_images = []
    for frame_idx, bbox in zip(selected_frame_index, selected_bboxs_dict):
        # Get the specific frame from the video reader using the frame index
        frame = video_reader[frame_idx]  # torch.tensor # (H, W, C)

        # Extract bounding box coordinates and convert them to integers
        x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
        # Crop to minimize the bounding box to a square
        width = x2 - x1  # Calculate the width of the bounding box
        height = y2 - y1  # Calculate the height of the bounding box
        side_length = max(width, height)  # Determine the side length of the square (max of width or height)

        # Calculate the center of the bounding box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Calculate new coordinates for the square region centered around the original bounding box
        new_x1 = max(0, center_x - side_length // 2)  # Ensure x1 is within image bounds
        new_y1 = max(0, center_y - side_length // 2)  # Ensure y1 is within image bounds
        new_x2 = min(frame.shape[1], new_x1 + side_length)  # Ensure x2 does not exceed image width
        new_y2 = min(frame.shape[0], new_y1 + side_length)  # Ensure y2 does not exceed image height

        # Adjust coordinates if the cropped area is smaller than the desired side length
        # Ensure final width and height are equal, keeping it a square
        actual_width = new_x2 - new_x1
        actual_height = new_y2 - new_y1

        if actual_width < side_length:
            # Adjust x1 or x2 to ensure the correct side length, while staying in bounds
            if new_x1 == 0:
                new_x2 = min(frame.shape[1], new_x1 + side_length)
            else:
                new_x1 = max(0, new_x2 - side_length)

        if actual_height < side_length:
            # Adjust y1 or y2 to ensure the correct side length, while staying in bounds
            if new_y1 == 0:
                new_y2 = min(frame.shape[0], new_y1 + side_length)
            else:
                new_y1 = max(0, new_y2 - side_length)

        # Expand the square by 20%
        expansion_ratio = 0.2  # Define the expansion ratio
        expansion_amount = int(side_length * expansion_ratio)  # Calculate the number of pixels to expand by

        # Calculate expanded coordinates, ensuring they stay within image bounds
        expanded_x1 = max(0, new_x1 - expansion_amount)  # Expand left, ensuring x1 is within bounds
        expanded_y1 = max(0, new_y1 - expansion_amount)  # Expand up, ensuring y1 is within bounds
        expanded_x2 = min(frame.shape[1], new_x2 + expansion_amount)  # Expand right, ensuring x2 does not exceed bounds
        expanded_y2 = min(frame.shape[0], new_y2 + expansion_amount)  # Expand down, ensuring y2 does not exceed bounds

        # Ensure the expanded area is still a square
        expanded_width = expanded_x2 - expanded_x1
        expanded_height = expanded_y2 - expanded_y1
        final_side_length = min(expanded_width, expanded_height)

        # Adjust to ensure square shape if necessary
        if expanded_width != expanded_height:
            if expanded_width > expanded_height:
                expanded_x2 = expanded_x1 + final_side_length
            else:
                expanded_y2 = expanded_y1 + final_side_length

        expanded_cropped_rgb_tensor = frame[expanded_y1:expanded_y2, expanded_x1:expanded_x2, :]
        expanded_cropped_rgb = Image.fromarray(np.array(expanded_cropped_rgb_tensor)).convert('RGB')
        expanded_cropped_images.append(expanded_cropped_rgb)

        if return_ori:
            original_cropped_rgb_tensor = frame[new_y1:new_y2, new_x1:new_x2, :]
            original_cropped_rgb = Image.fromarray(np.array(original_cropped_rgb_tensor)).convert('RGB')
            original_cropped_images.append(original_cropped_rgb)
            return expanded_cropped_images, original_cropped_images

    return expanded_cropped_images, None

def process_cropped_images(expand_images_pil, original_images_pil, target_size=(480, 480)):
    """
    Process a list of cropped images in PIL format.

    Parameters:
    expand_images_pil (list of PIL.Image): List of cropped images in PIL format.
    target_size (tuple of int): The target size for resizing images, default is (480, 480).

    Returns:
    torch.Tensor: A tensor containing the processed images.
    """
    expand_face_imgs = []
    original_face_imgs = []
    if len(original_images_pil) != 0:
        for expand_img, original_img in zip(expand_images_pil, original_images_pil):
            expand_resized_img = expand_img.resize(target_size, Image.LANCZOS)
            expand_src_img = np.array(expand_resized_img)
            expand_src_img = np.transpose(expand_src_img, (2, 0, 1))
            expand_src_img = torch.from_numpy(expand_src_img).unsqueeze(0).float()
            expand_face_imgs.append(expand_src_img)

            original_resized_img = original_img.resize(target_size, Image.LANCZOS)
            original_src_img = np.array(original_resized_img)
            original_src_img = np.transpose(original_src_img, (2, 0, 1))
            original_src_img = torch.from_numpy(original_src_img).unsqueeze(0).float()
            original_face_imgs.append(original_src_img)

        expand_face_imgs = torch.cat(expand_face_imgs, dim=0)
        original_face_imgs = torch.cat(original_face_imgs, dim=0)
    else:
        for expand_img in expand_images_pil:
            expand_resized_img = expand_img.resize(target_size, Image.LANCZOS)
            expand_src_img = np.array(expand_resized_img)
            expand_src_img = np.transpose(expand_src_img, (2, 0, 1))
            expand_src_img = torch.from_numpy(expand_src_img).unsqueeze(0).float()
            expand_face_imgs.append(expand_src_img)
        expand_face_imgs = torch.cat(expand_face_imgs, dim=0)
        original_face_imgs = None

    return expand_face_imgs, original_face_imgs

class RandomSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.

    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """

    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator
        self._pos_start = 0

        if not isinstance(self.replacement, bool):
            raise TypeError(f"replacement should be a boolean value, but got replacement={self.replacement}")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(f"num_samples should be a positive integer value, but got num_samples={self.num_samples}")

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        else:
            for _ in range(self.num_samples // n):
                xx = torch.randperm(n, generator=generator).tolist()
                if self._pos_start >= n:
                    self._pos_start = 0
                # print("xx top 10", xx[:10], self._pos_start)
                for idx in range(self._pos_start, n):
                    yield xx[idx]
                    self._pos_start = (self._pos_start + 1) % n
                self._pos_start = 0
            yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]

    def __len__(self) -> int:
        return self.num_samples

class SequentialSampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """

    data_source: Sized

    def __init__(self, data_source: Sized) -> None:
        self.data_source = data_source
        self._pos_start = 0

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        for idx in range(self._pos_start, n):
            yield idx
            self._pos_start = (self._pos_start + 1) % n
        self._pos_start = 0

    def __len__(self) -> int:
        return len(self.data_source)

class ProteusID_Dataset(Dataset):
    def __init__(
            self,
            instance_data_root: Optional[str] = None,
            id_token: Optional[str] = None,
            height=480,
            width=720,
            max_num_frames=49,
            sample_stride=3,
            skip_frames_start_percent=0.0,
            skip_frames_end_percent=1.0,
            skip_frames_start=7,
            skip_frames_end=7,
            text_drop_ratio=-1,
            is_train_face=False,
            is_single_face=False,
            miss_tolerance=6,
            min_distance=3,
            min_frames=1,
            max_frames=5,
            is_cross_face=False,
            is_reserve_face=False,
    ):
        self.id_token = id_token or ""
        self.skip_frames_start_percent = skip_frames_start_percent
        self.skip_frames_end_percent   = skip_frames_end_percent
        self.skip_frames_start         = skip_frames_start
        self.skip_frames_end           = skip_frames_end
        self.is_train_face             = is_train_face
        self.is_single_face            = is_single_face

        if is_train_face:
            self.miss_tolerance     = miss_tolerance
            self.min_distance       = min_distance
            self.min_frames         = min_frames
            self.max_frames         = max_frames
            self.is_cross_face      = is_cross_face
            self.is_reserve_face    = is_reserve_face

        # Loading annotations from files
        print(f"loading annotations from {instance_data_root} ...")
        with open(instance_data_root, 'r') as f:
            folder_anno = [i.strip() for i in f.readlines() if len(i.strip()) > 0]

        self.instance_prompts = []
        self.identity_prompts = []
        self.instance_video_paths = []
        self.instance_annotation_base_paths = []
        for anno in tqdm(folder_anno):
            print(anno)
            with open(anno, 'r') as f:
                sub_list = json.load(f)
            for i in tqdm(sub_list):
                sub_root = i.get('video_root', None)
                path = os.path.join(sub_root, os.path.basename(i['path']))
                cap = i.get('cap', None)
                identity = i.get('identity', None)
                fps = i.get('fps', 0)
                duration = i.get('duration', 0)
                anno_base = i.get('anno_base', None)
                if fps * duration < 49.0:
                    continue

                self.instance_prompts.append(cap)
                self.identity_prompts.append(identity)
                self.instance_video_paths.append(path)
                self.instance_annotation_base_paths.append(anno_base)

        self.num_instance_videos = len(self.instance_video_paths)

        self.text_drop_ratio = text_drop_ratio

        # Video params
        self.sample_stride = sample_stride
        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width

    def _get_frame_indices_adjusted(self, video_length, n_frames):
        indices = list(range(video_length))
        additional_frames_needed = n_frames - video_length

        repeat_indices = []
        for i in range(additional_frames_needed):
            index_to_repeat = i % video_length
            repeat_indices.append(indices[index_to_repeat])

        all_indices = indices + repeat_indices
        all_indices.sort()

        return all_indices


    def _generate_frame_indices(self, video_length, n_frames, sample_stride, skip_frames_start_percent=0.0, skip_frames_end_percent=1.0, skip_frames_start=0, skip_frames_end=0):
        if skip_frames_start_percent != 0.0 or  skip_frames_end_percent != 1.0:
            print("use skip frame percent")
            valid_start = int(video_length * skip_frames_start_percent)
            valid_end = int(video_length * skip_frames_end_percent)
        elif skip_frames_start != 0 or skip_frames_end != 0:
            print("use skip frame")
            valid_start = skip_frames_start
            valid_end = video_length - skip_frames_end
        else:
            print("no use skip frame")
            valid_start = 0
            valid_end = video_length

        adjusted_length = valid_end - valid_start

        if adjusted_length <= 0:
            print(f"video_length: {video_length}, adjusted_length: {adjusted_length}, valid_start:{valid_start}, skip_frames_end: {valid_end}")
            raise ValueError("Skipping too many frames results in no frames left to sample.")

        if video_length <= n_frames:
            return self._get_frame_indices_adjusted(video_length, n_frames)
        else:
            # clip_length = min(video_length, (n_frames - 1) * sample_stride + 1)
            # start_idx = random.randint(0, video_length - clip_length)
            # frame_indices = np.linspace(start_idx, start_idx + clip_length - 1, n_frames, dtype=int).tolist()

            clip_length = min(adjusted_length, (n_frames - 1) * sample_stride + 1)
            start_idx = random.randint(valid_start, valid_end - clip_length)
            frame_indices = np.linspace(start_idx, start_idx + clip_length - 1, n_frames, dtype=int).tolist()
            return frame_indices

    def _short_resize_and_crop(self, frames, target_width, target_height):
        """
        Resize frames and crop to the specified size.

        Args:
            frames (torch.Tensor): Input frames of shape [T, H, W, C].
            target_width (int): Desired width.
            target_height (int): Desired height.

        Returns:
            torch.Tensor: Cropped frames of shape [T, target_height, target_width, C].
        """
        T, H, W, C = frames.shape
        aspect_ratio = W / H

        # Determine new dimensions ensuring they are at least target size
        if aspect_ratio > target_width / target_height:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
            if new_height < target_height:
                new_height = target_height
                new_width = int(target_height * aspect_ratio)
        else:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
            if new_width < target_width:
                new_width = target_width
                new_height = int(target_width / aspect_ratio)

        resize_transform = transforms.Resize((new_height, new_width))
        crop_transform = transforms.CenterCrop((target_height, target_width))

        frames_tensor = frames.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
        resized_frames = resize_transform(frames_tensor)
        cropped_frames = crop_transform(resized_frames)
        sample = cropped_frames.permute(0, 2, 3, 1)

        return sample

    def _resize_with_aspect_ratio(self, frames, target_width, target_height):
        """
            Resize frames while maintaining the aspect ratio by padding or cropping.

            Args:
                frames (torch.Tensor): Input frames of shape [T, H, W, C].
                target_width (int): Desired width.
                target_height (int): Desired height.

            Returns:
                torch.Tensor: Resized and padded frames of shape [T, target_height, target_width, C].
        """
        T, frame_height, frame_width, C = frames.shape
        aspect_ratio = frame_width / frame_height  # 1.77, 1280 720 -> 720 406
        target_aspect_ratio = target_width / target_height  # 1.50, 720 480 ->

        # If the frame is wider than the target, resize based on width
        if aspect_ratio > target_aspect_ratio:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)

        # Resize using batch processing
        frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]
        frames = F.interpolate(frames, size=(new_height, new_width), mode='bilinear', align_corners=False)

        # Calculate padding
        pad_top = (target_height - new_height) // 2
        pad_bottom = target_height - new_height - pad_top
        pad_left = (target_width - new_width) // 2
        pad_right = target_width - new_width - pad_left

        # Apply padding
        frames = F.pad(frames, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

        frames = frames.permute(0, 2, 3, 1)  # [T, H, W, C]

        return frames


    def _save_frame(self, frame, name="1.png"):
        # [H, W, C] -> [C, H, W]
        img = frame
        img = img.permute(2, 0, 1)
        to_pil = ToPILImage()
        img = to_pil(img)
        img.save(name)


    def _save_video(self, torch_frames, name="output.mp4"):
        from moviepy import ImageSequenceClip
        frames_np = torch_frames.cpu().numpy()
        if frames_np.dtype != 'uint8':
            frames_np = frames_np.astype('uint8')
        frames_list = list(frames_np)
        desired_fps = 24
        clip = ImageSequenceClip(frames_list, fps=desired_fps)
        clip.write_videofile(name, codec="libx264")


    def get_batch(self, idx):
        decord.bridge.set_bridge("torch")

        video_dir = self.instance_video_paths[idx]
        text = self.instance_prompts[idx]
        text_identity = self.identity_prompts[idx]
        
        train_transforms = transforms.Compose(
            [
                transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0),
            ]
        )

        with VideoReader_contextmanager(video_dir, num_threads=2) as video_reader:
            video_num_frames = len(video_reader)

            if self.is_train_face:
                reserve_face_imgs = None
                file_base_name = os.path.basename(video_dir.replace(".mp4", ""))

                # TODO: add optical flow mask
                anno_base_path = self.instance_annotation_base_paths[idx]
                valid_frame_path = os.path.join(anno_base_path, "track_masks_data", file_base_name, "valid_frame.json")
                control_sam2_frame_path = os.path.join(anno_base_path, "track_masks_data", file_base_name, "control_sam2_frame.json")
                corresponding_data_path = os.path.join(anno_base_path, "track_masks_data", file_base_name, "corresponding_data.json") # person number
                masks_data_path = os.path.join(anno_base_path, "track_masks_data", file_base_name, "tracking_mask_results")
                bboxs_data_path = os.path.join(anno_base_path, "refine_bbox_jsons", f"{file_base_name}.json")

                with open(corresponding_data_path, 'r') as f:
                    corresponding_data = json.load(f)

                with open(control_sam2_frame_path, 'r') as f:
                    control_sam2_frame = json.load(f)

                with open(valid_frame_path, 'r') as f:
                    valid_frame = json.load(f)

                with open(bboxs_data_path, 'r') as f:
                    bbox_data = json.load(f)

                if self.is_single_face:
                    if len(corresponding_data) != 1:
                        raise ValueError(f"Using single face, but {idx} is multi person.")

                # get random valid id
                valid_ids = [] # person number
                backup_ids = []
                for id_key, data in corresponding_data.items():
                    if 'face' in data and 'head' in data:
                        valid_ids.append(id_key)
                # There may be more than one person in the video, we randomly select an ID
                valid_id = random.choice(valid_ids) if valid_ids else (random.choice(backup_ids) if backup_ids else None)
                if valid_id is None:
                    raise ValueError("No valid ID found: both valid_ids and backup_ids are empty.")
                
                human_valid_id = corresponding_data[valid_id]["person"] if corresponding_data[valid_id]["person"] is not None else valid_id
                # get video
                # frame number of video
                total_index = list(range(video_num_frames))
                # select frames (49) from video  49, 3, 
                batch_index, _ = generate_frame_indices_for_face(self.max_num_frames, self.sample_stride, valid_frame[valid_id],
                                                                          self.miss_tolerance, self.skip_frames_start_percent, self.skip_frames_end_percent,
                                                                          self.skip_frames_start, self.skip_frames_end)
                # TODO: Used in a certain proportion
                if self.is_cross_face:
                    remaining_batch_index_index = [i for i in total_index if i not in batch_index]
                    try:
                        selected_frame_index, selected_masks_dict, selected_bboxs_dict, dense_masks_dict, human_masks_dict = select_mask_frames_from_index(
                                                                                                                            remaining_batch_index_index,
                                                                                                                            batch_index, valid_id,
                                                                                                                            corresponding_data, control_sam2_frame,
                                                                                                                            valid_frame[valid_id], bbox_data, masks_data_path,
                                                                                                                            min_distance=self.min_distance, min_frames=self.min_frames,
                                                                                                                            max_frames=self.max_frames, dense_masks=True,
                                                                                                                            ensure_control_frame=False,
                                                                                                                            human_valid_id = human_valid_id
                                                                                                                        )
                    except (ValueError, TypeError):
                        selected_frame_index, selected_masks_dict, selected_bboxs_dict, dense_masks_dict, human_masks_dict = select_mask_frames_from_index(
                                                                                                                            batch_index,
                                                                                                                            batch_index, valid_id,
                                                                                                                            corresponding_data, control_sam2_frame,
                                                                                                                            valid_frame[valid_id], bbox_data, masks_data_path,
                                                                                                                            min_distance=self.min_distance, min_frames=self.min_frames,
                                                                                                                            max_frames=self.max_frames, dense_masks=True,
                                                                                                                            ensure_control_frame=False,
                                                                                                                            human_valid_id = human_valid_id
                                                                                                                        )
                else:
                    selected_frame_index, selected_masks_dict, selected_bboxs_dict, dense_masks_dict, human_masks_dict = select_mask_frames_from_index(
                                                                                                                        batch_index,
                                                                                                                        batch_index, valid_id,
                                                                                                                        corresponding_data, control_sam2_frame,
                                                                                                                        valid_frame[valid_id], bbox_data, masks_data_path,
                                                                                                                        min_distance=self.min_distance, min_frames=self.min_frames,
                                                                                                                        max_frames=self.max_frames, dense_masks=True,
                                                                                                                        ensure_control_frame=True,
                                                                                                                        human_valid_id = human_valid_id
                                                                                                                    )
                    if self.is_reserve_face:
                        reserve_frame_index, _, reserve_bboxs_dict, _, human_masks_dict = select_mask_frames_from_index(
                                                                        batch_index,
                                                                        batch_index, valid_id,
                                                                        corresponding_data, control_sam2_frame,
                                                                        valid_frame[valid_id], bbox_data, masks_data_path,
                                                                        min_distance=3, min_frames=4,
                                                                        max_frames=4, dense_masks=False,
                                                                        ensure_control_frame=False,
                                                                        human_valid_id = human_valid_id
                                                                    )

                # get mask and aligned_face_img
                selected_frame_index = selected_frame_index[valid_id] #控制帧, ex. 131
                valid_frame = valid_frame[valid_id]
                selected_masks_dict = selected_masks_dict[valid_id]
                selected_bboxs_dict = selected_bboxs_dict[valid_id]
                dense_masks_dict = dense_masks_dict[valid_id]
                human_masks_dict = human_masks_dict[valid_id]
                
                if self.is_reserve_face:
                    reserve_frame_index = reserve_frame_index[valid_id]
                    reserve_bboxs_dict = reserve_bboxs_dict[valid_id]

                temp_dense_masks_tensor = torch.stack([torch.tensor(mask) for mask in dense_masks_dict])
                dense_masks_tensor = self._short_resize_and_crop(temp_dense_masks_tensor.unsqueeze(-1), self.width, self.height).squeeze(-1)  # [T, H, W] -> [T, H, W, 1] -> [T, H, W]
                
                
                human_temp_dense_masks_tensor = torch.stack([torch.tensor(mask) for mask in human_masks_dict])
                human_masks_tensor = self._short_resize_and_crop(human_temp_dense_masks_tensor.unsqueeze(-1), self.width, self.height).squeeze(-1)  # [T, H, W] -> [T, H, W, 1] -> [T, H, W]

                expand_images_pil, original_images_pil = crop_images(selected_frame_index, selected_bboxs_dict, video_reader, return_ori=True)
                expand_face_imgs, original_face_imgs = process_cropped_images(expand_images_pil, original_images_pil, target_size=(480, 480))
                if self.is_reserve_face:
                    reserve_images_pil, _ = crop_images(reserve_frame_index, reserve_bboxs_dict, video_reader, return_ori=False)
                    reserve_face_imgs, _ = process_cropped_images(reserve_images_pil, [], target_size=(480, 480))

                if len(expand_face_imgs) == 0 or len(original_face_imgs) == 0:
                    raise ValueError("No face detected in input image pool")

                # post process id related data
                expand_face_imgs = pad_tensor(expand_face_imgs, self.max_frames, dim=0)
                original_face_imgs = pad_tensor(original_face_imgs, self.max_frames, dim=0)
                selected_frame_index = torch.tensor(selected_frame_index)                         # torch.Size(([15, 13])          [N1]
                selected_frame_index = pad_tensor(selected_frame_index, self.max_frames, dim=0)
            else:
                batch_index = self._generate_frame_indices(video_num_frames, self.max_num_frames, self.sample_stride,
                                                            self.skip_frames_start_percent, self.skip_frames_end_percent,
                                                            self.skip_frames_start, self.skip_frames_end)

            try:
                frames = video_reader.get_batch(batch_index) # torch [T, H, W, C]
                frames = self._short_resize_and_crop(frames, self.width, self.height)  # [T, H, W, C]
            except FunctionTimedOut:
                raise ValueError(f"Read {idx} timeout.")
            except Exception as e:
                raise ValueError(f"Failed to extract frames from video. Error is {e}.")

            # Apply training transforms in batch
            frames = frames.float()
            frames = train_transforms(frames)
            pixel_values = frames.permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]
            raft_pixel_values = frames.permute(0, 3, 1, 2).contiguous() # [T, C, H, W]
            del video_reader

            # Random use no text generation （0,1）  text_drop_ratio=-1
            # if random.random() < self.text_drop_ratio:
            #     text = ''

        # pixel_values (49,3,480,720)
        # text
        # 'video'
        # video_dir 'asserts/demo_train_data/dataname/videos/81FyMPk-WIS_segment_0_243_noFacePose_0_243.mp4'
        # expand_face_imgs (1,3,480,480)
        # dense_masks_tensor (49,480,720)
        # selected_frame_index 131
        # reserve_face_imgs
        # original_face_imgs (1,3,480,480)
        
        
        if self.is_train_face:
            return  pixel_values, text, text_identity, 'video', video_dir, expand_face_imgs, dense_masks_tensor, selected_frame_index, reserve_face_imgs, original_face_imgs, raft_pixel_values, human_masks_tensor
        else:
            return pixel_values, text, 'video', video_dir

    def __len__(self):
        return self.num_instance_videos

    def __getitem__(self, idx):
        while True:
            sample = {}
            try:
                if self.is_train_face:
                    # pixel_values (49,3,480,720)
                    # text
                    # 'video'
                    # video_dir 'asserts/demo_train_data/dataname/videos/81FyMPk-WIS_segment_0_243_noFacePose_0_243.mp4'
                    # expand_face_imgs (1,3,480,480)
                    # dense_masks_tensor (49,480,720)
                    # selected_frame_index 131
                    # reserve_face_imgs
                    # original_face_imgs (1,3,480,480)
                    pixel_values, cap, text_identity, data_type, video_dir, expand_face_imgs, dense_masks_tensor, selected_frame_index, reserve_face_imgs, original_face_imgs, raft_pixel_values, human_masks_tensor = self.get_batch(idx)
                    sample["instance_prompt"] = self.id_token + cap
                    sample["identity_prompt"] = self.id_token + text_identity
                    sample["instance_video"] = pixel_values
                    sample["video_path"] = video_dir
                    if self.is_train_face:
                        sample["expand_face_imgs"] = expand_face_imgs
                        sample["dense_masks_tensor"] = dense_masks_tensor
                        sample["human_masks_tensor"] = human_masks_tensor
                        sample["selected_frame_index"] = selected_frame_index
                        if reserve_face_imgs is not None:
                            sample["reserve_face_imgs"] = reserve_face_imgs
                        if original_face_imgs is not None:
                            sample["original_face_imgs"] = original_face_imgs
                    sample["raft_video"] = raft_pixel_values
                else:
                    pixel_values, cap, data_type, video_dir, = self.get_batch(idx)
                    sample["instance_prompt"] = self.id_token + cap
                    sample["instance_video"] = pixel_values
                    sample["video_path"] = video_dir
                break
            except Exception as e:
                error_message = str(e)
                video_path = self.instance_video_paths[idx % len(self.instance_video_paths)]
                print(error_message, video_path)
                log_error_to_file(error_message, video_path)
                idx = random.randint(0, self.num_instance_videos - 1)
        return sample
