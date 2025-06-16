import os
import cv2
import argparse
import glob
import torch
import numpy as np
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.misc import gpu_is_available, get_device
from scipy.ndimage import gaussian_filter1d
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray
from basicsr.utils.video_util import VideoReader, VideoWriter
from einops import rearrange

from basicsr.utils.registry import ARCH_REGISTRY


def interpolate_sequence(sequence):
    interpolated_sequence = np.copy(sequence)
    missing_indices = np.isnan(sequence)

    if np.any(missing_indices):
        valid_indices = ~missing_indices
        x = np.arange(len(sequence))

        # Interpolate missing values using valid data points
        interpolated_sequence[missing_indices] = np.interp(
            x[missing_indices], x[valid_indices], sequence[valid_indices]
        )

    return interpolated_sequence


def set_realesrgan():
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.realesrgan_utils import RealESRGANer

    use_half = False
    if torch.cuda.is_available():  # set False in CPU/MPS mode
        # set False for GPUs that don't support f16
        no_half_gpu_list = ["1650", "1660"]
        if not True in [
            gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list
        ]:
            use_half = True

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path="./ckpts/realesrgan/RealESRGAN_x2plus.pth",
        model=model,
        tile=args.bg_tile,
        tile_pad=40,
        pre_pad=0,
        half=use_half,
    )

    if not gpu_is_available():  # CPU
        import warnings

        warnings.warn(
            "Running on CPU now! Make sure your PyTorch version matches your CUDA."
            "The unoptimized RealESRGAN is slow on CPU. "
            "If you want to disable it, please remove `--bg_upsampler` and `--face_upsample` in command.",
            category=RuntimeWarning,
        )
    return upsampler


if __name__ == "__main__":
    device = get_device()
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        default="None",
        help="Input image, video or folder. Default: inputs/whole_imgs",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="results/",
        help="Output folder. Default: results/",
    )
    parser.add_argument(
        "--save_video", action="store_true", help="Save output as video. Default: False"
    )
    parser.add_argument(
        "-s",
        "--upscale",
        type=int,
        default=2,
        help="The final upsampling scale of the image. Default: 1",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=20,
        help="Max length of per sub-clip depending of GPU memory. Default: 20",
    )
    parser.add_argument(
        "--has_aligned",
        action="store_true",
        help="Input are cropped and aligned faces. Default: False",
    )
    parser.add_argument(
        "--only_center_face",
        type=bool,
        default=True,
        help="Only restore the center face. Default: True",
    )
    parser.add_argument(
        "--draw_box",
        action="store_true",
        help="Draw the bounding box for the detected faces. Default: False",
    )
    parser.add_argument(
        "--detection_model",
        type=str,
        default="retinaface_resnet50",
        help="Face detector. Optional: retinaface_resnet50, retinaface_mobile0.25, YOLOv5l, YOLOv5n, dlib. \
                        Default: retinaface_resnet50",
    )
    parser.add_argument(
        "--bg_upsampler",
        type=str,
        default="None",
        help="Background upsampler. Optional: realesrgan",
    )
    parser.add_argument(
        "--face_upsample",
        action="store_true",
        help="Face upsampler after enhancement. Default: False",
    )
    parser.add_argument(
        "--bg_tile",
        type=int,
        default=400,
        help="Tile size for background sampler. Default: 400",
    )
    parser.add_argument(
        "--save_video_fps",
        type=float,
        default=20,
        help="Frame rate for saving video. Default: 20",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default="None", help="the loaded ckpt file path"
    )

    args = parser.parse_args()
    input_video = False

    ckpt_path = args.ckpt_path
    weight_parameter = 1.0

    # ------------------ set up background upsampler ------------------
    print("------------------ set up background upsampler ------------------")
    if args.bg_upsampler == "realesrgan":
        bg_upsampler = set_realesrgan()
    else:
        bg_upsampler = None

    # ------------------ set up face upsampler ------------------
    if args.face_upsample:
        if bg_upsampler is not None:
            face_upsampler = bg_upsampler
        else:
            face_upsampler = set_realesrgan()
    else:
        face_upsampler = None

    # ------------------ set up restorer -------------------
    net = ARCH_REGISTRY.get("TemporalCodeFormerDirDistMultiScale")(
        dim_embed=512,
        n_head=8,
        n_layers=9,
        codebook_size=1024,
        connect_list=["32", "64", "128", "256"],
        frame_length=5,
    ).to(device)

    checkpoint = torch.load(ckpt_path)["params_ema"]
    net.load_state_dict(checkpoint)
    net.eval()

    # ------------------ set up FaceRestoreHelper -------------------
    # large det_model: 'YOLOv5l', 'retinaface_resnet50'
    # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'
    if not args.has_aligned:
        print(f"Face detection model: {args.detection_model}")
    if bg_upsampler is not None:
        print(f"Background upsampling: True. Face upsampling: {args.face_upsample}")
    else:
        print(f"Background upsampling: False. Face upsampling: {args.face_upsample}")

    face_helper = FaceRestoreHelper(
        args.upscale,
        face_size=512,
        crop_ratio=(1, 1),
        det_model=args.detection_model,
        save_ext="png",
        use_parse=True,
        device=device,
    )

    # -------------------- start processing ---------------------
    input_img_list = []
    restored_img_list = []

    if args.input_path.endswith(
        ("mp4", "mov", "avi", "MP4", "MOV", "AVI")
    ):  # input video path
        vidreader = VideoReader(args.input_path)
        image = vidreader.get_frame()
        while image is not None:
            input_img_list.append(image)
            image = vidreader.get_frame()
        fps = (
            vidreader.get_fps() if args.save_video_fps is None else args.save_video_fps
        )
        vidreader.close()

        clip_name = os.path.basename(args.input_path)[:-4]
        result_root = os.path.join(args.output_path, clip_name)
        os.makedirs(result_root, exist_ok=True)

    elif os.path.isdir(args.input_path):  # input img folder
        # scan all the jpg and png images
        for img_path in sorted(
            glob.glob(os.path.join(args.input_path, "*.[jpJP][pnPN]*[gG]"))
        ):
            input_img_list.append(cv2.imread(img_path))
        clip_name = os.path.basename(args.input_path)
        result_root = os.path.join(args.output_path, clip_name)
        os.makedirs(result_root, exist_ok=True)

    else:
        raise TypeError(f"Unrecognized type of input video {args.input_path}.")

    if len(input_img_list) == 0:
        raise FileNotFoundError(
            "No input image/video is found...\n"
            "\tNote that --input_path for video should end with .mp4|.mov|.avi"
        )

    if not args.has_aligned:
        # Smoothing aligned landmarks
        print("Detecting keypoints and smooth alignment ...")
        raw_landmarks = []
        for i, img in enumerate(input_img_list):
            # clean all the intermediate results to process the next image
            face_helper.clean_all()
            face_helper.read_image(img)

            # get face landmarks for each face
            num_det_faces = face_helper.get_face_landmarks_5(
                only_center_face=args.only_center_face,
                resize=640,
                eye_dist_threshold=5,
                only_keep_largest=True,
            )

            if num_det_faces == 1:
                raw_landmarks.append(face_helper.all_landmarks_5[0].reshape((10,)))
            elif num_det_faces == 0:
                raw_landmarks.append(np.array([np.nan] * 10))

        raw_landmarks = np.array(raw_landmarks)
        for i in range(10):
            raw_landmarks[:, i] = interpolate_sequence(raw_landmarks[:, i])
        video_length = len(input_img_list)
        avg_landmarks = gaussian_filter1d(raw_landmarks, 5, axis=0).reshape(
            video_length, 5, 2
        )

    # Pack cropped faces.
    cropped_faces = []
    for i, img in enumerate(input_img_list):
        if not args.has_aligned:
            face_helper.clean_all()
            face_helper.read_image(img)
            face_helper.all_landmarks_5 = [avg_landmarks[i]]
            face_helper.align_warp_face()
        else:
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            face_helper.is_gray = is_gray(img, threshold=10)
            # if face_helper.is_gray:
                # print("Grayscale input: True")
            face_helper.cropped_faces = [img]

        cropped_face_t = img2tensor(
            face_helper.cropped_faces[0] / 255.0, bgr2rgb=True, float32=True
        )
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_faces.append(cropped_face_t)
    cropped_faces = torch.stack(cropped_faces, dim=0).unsqueeze(0).to(device)

    print("Restoring faces ...")
    with torch.no_grad():
        video_length = cropped_faces.shape[1]
        output = []
        for start_idx in range(0, video_length):
            pre_length = args.max_length // 2
            post_length = args.max_length - pre_length - 1

            padding_begin_idx = start_idx - pre_length
            padding_end_idx = start_idx + post_length

            if padding_begin_idx < 0:
                pre_padding = torch.zeros(
                    (
                        cropped_faces.shape[0],
                        -padding_begin_idx,
                        *cropped_faces.shape[2:],
                    ),
                    dtype=cropped_faces.dtype,
                    device=cropped_faces.device,
                )
                pre_padding = pre_padding + cropped_faces[:, 0:1]
                small_clip = torch.cat(
                    [pre_padding, cropped_faces[:, : padding_end_idx + 1, ...]], dim=1
                )
            elif padding_end_idx >= video_length:
                post_padding = torch.zeros(
                    (
                        cropped_faces.shape[0],
                        padding_end_idx - video_length + 1,
                        *cropped_faces.shape[2:],
                    ),
                    dtype=cropped_faces.dtype,
                    device=cropped_faces.device,
                )
                post_padding = post_padding + cropped_faces[:, -1:]
                small_clip = torch.cat(
                    [cropped_faces[:, padding_begin_idx:, ...], post_padding], dim=1
                )
            else:
                small_clip = cropped_faces[
                    :, padding_begin_idx : padding_end_idx + 1, ...
                ]

            small_clip = rearrange(
                small_clip, "b t c h w -> (b t) c h w", t=args.max_length
            )
            bt = small_clip.shape[0]
            res, _, _ = net(small_clip, w=weight_parameter)

            res = rearrange(res, "(b t) c h w -> b t c h w", t=args.max_length)

            res = res[:, pre_length : pre_length + 1, ...]
            output.append(res)

        output = torch.cat(output, dim=1).squeeze(0)
        assert output.shape[0] == video_length, "Differer number of frames"

        restored_faces = [tensor2img(x, rgb2bgr=True, min_max=(-1, 1)) for x in output]
        del output
        torch.cuda.empty_cache()

    print("Saving result ...")
    
    output_path = result_root
    os.makedirs(output_path, mode=0o777, exist_ok=True)
    if args.save_video:
        writer = cv2.VideoWriter(
            f"{output_path}.mp4",
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=args.save_video_fps,
            frameSize=(512, 512),
        )

    for idx, restored_img in enumerate(restored_faces):
        img_abs_path = os.path.join(output_path, str(idx).zfill(8) + ".png")
        cv2.imwrite(img_abs_path, restored_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        if args.save_video:
            writer.write(restored_img)

    if args.save_video:
        writer.release()
