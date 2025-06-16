import os
import cv2
import argparse
import glob
import torch
import pdb
import numpy as np
from tqdm import tqdm
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import gpu_is_available, get_device
from scipy.ndimage import gaussian_filter1d
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray
from basicsr.utils.video_util import VideoReader, VideoWriter
from einops import rearrange

from utils import TDCF_OPT, TCFDD_OPT
from basicsr.utils.registry import ARCH_REGISTRY


def interpolate_sequence(sequence):
    interpolated_sequence = np.copy(sequence)
    missing_indices = np.isnan(sequence)

    if np.any(missing_indices):
        valid_indices = ~missing_indices
        x = np.arange(len(sequence))

        # Interpolate missing values using valid data points
        interpolated_sequence[missing_indices] = np.interp(
            x[missing_indices], x[valid_indices], sequence[valid_indices])

    return interpolated_sequence


def process_single(args, face_helper, input_path, ldmk_folder_path):
    input_img_list = []

    if input_path.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')): # input video path
        vidreader = VideoReader(input_path)
        image = vidreader.get_frame()
        while image is not None:
            input_img_list.append(image)
            image = vidreader.get_frame()
        fps = vidreader.get_fps() if args.save_video_fps is None else args.save_video_fps   
        vidreader.close()

        clip_name = os.path.basename(input_path)[:-4]
        result_root = os.path.join(args.output_path, clip_name)
    elif os.path.isdir(args.input_path): # input img folder
        # scan all the jpg and png images
        for img_path in sorted(glob.glob(os.path.join(input_path, '*.[jpJP][pnPN]*[gG]'))):
            input_img_list.append(cv2.imread(img_path))
        clip_name = os.path.basename(input_path)
        result_root = os.path.join(args.output_path, clip_name)
    else:
        raise TypeError(f'Unrecognized type of input video {input_path}.')

    if len(input_img_list) == 0:
        raise FileNotFoundError('No input image/video is found...\n'
                                '\tNote that --input_path for video should end with .mp4|.mov|.avi')

    # Smoothing aligned landmarks
    print('Detecting keypoints and smooth alignment ...')
    avg_landmarks = []
    with open(f"{ldmk_folder_path}/{clip_name}.txt", "r") as f:
        for line in f.readlines():
            line = line.strip().split()
            landmark = np.array([float(_) for _ in line]).reshape(5, 2)
            avg_landmarks.append(landmark)

    # Save cropped faces.
    output_path = os.path.join(args.output_path, f'{clip_name}')
    os.makedirs(output_path, mode=0o777, exist_ok=True)
    if args.save_video:
        writer = cv2.VideoWriter(os.path.join(output_path, f'{clip_name}.mp4'),
                                 fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps=args.save_video_fps,
                                 frameSize=(512, 512))

    for idx, img in enumerate(input_img_list):
        face_helper.clean_all()
        face_helper.read_image(img)
        face_helper.all_landmarks_5 = [avg_landmarks[idx]]
        face_helper.align_warp_face()

        img_abs_path = os.path.join(output_path, str(idx).zfill(8)+'.png')
        cv2.imwrite(img_abs_path, face_helper.cropped_faces[0], [cv2.IMWRITE_PNG_COMPRESSION, 0])

        if args.save_video:
            writer.write(face_helper.cropped_faces[0])

    if args.save_video:
        writer.release()

    print(f'All results are saved in {result_root}')

"""
CUDA_VISIBLE_DEVICES=6 python warp_images.py \
    -i vfhq_test_inpaint_input_512 \
    -o vfhq_test_inpaint_input_512_warped

CUDA_VISIBLE_DEVICES=6 python warp_images.py \
    -i /cpfs01/projects-HDD/cfff-721febfbdfb0_HDD/public/anna/workspaces/HanlinShang/55test/Interval1_512x512_LANCZOS4 \
    -o vfhq_test_inpaint_gt
"""
if __name__ == '__main__':
    device = get_device()
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', type=str, default='../dataset/real_LQ/',
                        help='no warped images')
    parser.add_argument('-o', '--output_path', type=str, default='results/',
                        help='Output folder. Default: results/')
    parser.add_argument('-l', '--ldmk_folder_path', type=str, required=True,
                        help='landmarks info folder.')
    parser.add_argument('--save_video', action='store_true',
                        help='Save output as video. Default: False')
    parser.add_argument('-s', '--upscale', type=int, default=1,
                        help='The final upsampling scale of the image. Default: 1')
    # large det_model: 'YOLOv5l', 'retinaface_resnet50'
    # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'
    parser.add_argument('--detection_model', type=str, default='retinaface_resnet50',
                        help='Face detector. Optional: retinaface_resnet50, retinaface_mobile0.25, YOLOv5l, YOLOv5n, dlib. \
                Default: retinaface_resnet50')
    parser.add_argument('--bg_tile', type=int, default=0,
                        help='Tile size for background sampler. Default: 400')
    parser.add_argument('--save_video_fps', type=float, default=24,
                        help='Frame rate for saving video. Default: 20')

    args = parser.parse_args()

    # ------------------ set up FaceRestoreHelper -------------------
    face_helper = FaceRestoreHelper(
        args.upscale,
        face_size=512,
        crop_ratio=(1, 1),
        det_model=args.detection_model,
        save_ext='png',
        use_parse=False,
        device=device)

    for _, clip_name in enumerate(tqdm(os.listdir(args.input_path))):
        process_single(args,
                       face_helper,
                       os.path.join(args.input_path, clip_name), args.ldmk_folder_path)