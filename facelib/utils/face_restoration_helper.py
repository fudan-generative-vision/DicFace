import cv2
import numpy as np
import os
import torch
import pdb
import dlib
from torchvision.transforms.functional import normalize

from facelib.detection import init_detection_model
from facelib.parsing import init_parsing_model
from facelib.utils.misc import img2tensor, imwrite, is_gray, bgr2gray, adain_npy
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import get_device

dlib_model_url = {
    'face_detector': 'https://github.com/jnjaby/KEEP/releases/download/v0.1.0/mmod_human_face_detector-4cb19393.dat',
    'shape_predictor_5': 'https://github.com/jnjaby/KEEP/releases/download/v0.1.0/shape_predictor_5_face_landmarks-c4b1e980.dat'
}
# is the test part
dlib_model_path = {
    'face_detector': "./ckpts/dlib/mmod_human_face_detector.dat",
    'shape_predictor_5' : "./ckpts/dlib/shape_predictor_5_face_landmarks.dat"
}

def get_largest_face(det_faces, h, w):

    def get_location(val, length):
        if val < 0:
            return 0
        elif val > length:
            return length
        else:
            return val

    face_areas = []
    for det_face in det_faces:
        left = get_location(det_face[0], w)
        right = get_location(det_face[2], w)
        top = get_location(det_face[1], h)
        bottom = get_location(det_face[3], h)
        face_area = (right - left) * (bottom - top)
        face_areas.append(face_area)
    largest_idx = face_areas.index(max(face_areas))
    return det_faces[largest_idx], largest_idx


def get_center_face(det_faces, h=0, w=0, center=None):
    if center is not None:
        center = np.array(center)
    else:
        center = np.array([w / 2, h / 2])
    center_dist = []
    for det_face in det_faces:
        face_center = np.array(
            [(det_face[0] + det_face[2]) / 2, (det_face[1] + det_face[3]) / 2]
        )
        dist = np.linalg.norm(face_center - center)
        center_dist.append(dist)
    center_idx = center_dist.index(min(center_dist))
    return det_faces[center_idx], center_idx


class FaceRestoreHelper(object):
    """Helper for the face restoration pipeline (base class)."""

    def __init__(
        self,
        upscale_factor,
        face_size=512,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='png',
        template_3points=False,
        pad_blur=False,
        use_parse=False,
        device=None,
    ):
        self.template_3points = template_3points  # improve robustness
        self.upscale_factor = int(upscale_factor)
        # the cropped face ratio based on the square face
        self.crop_ratio = crop_ratio  # (h, w)
        assert (self.crop_ratio[0] >= 1 and self.crop_ratio[1] >= 1), 'crop ration only supports >=1'
        self.face_size = (
            int(face_size * self.crop_ratio[1]),
            int(face_size * self.crop_ratio[0]),
        )
        self.det_model = det_model

        if self.det_model == 'dlib':
            # standard 5 landmarks for FFHQ faces with 1024 x 1024
            self.face_template = np.array(
                [
                    [686.77227723, 488.62376238],
                    [586.77227723, 493.59405941],
                    [337.91089109, 488.38613861],
                    [437.95049505, 493.51485149],
                    [513.58415842, 678.5049505],
                ]
            )
            self.face_template = self.face_template / (1024 // face_size)
        elif self.template_3points:
            self.face_template = np.array([[192, 240], [319, 240], [257, 371]])
        else:
            # standard 5 landmarks for FFHQ faces with 512 x 512
            # facexlib
            self.face_template = np.array(
                [
                    [192.98138, 239.94708],
                    [318.90277, 240.1936],
                    [256.63416, 314.01935],
                    [201.26117, 371.41043],
                    [313.08905, 371.15118],
                ]
            )

            # dlib: left_eye: 36:41  right_eye: 42:47  nose: 30,32,33,34  left mouth corner: 48  right mouth corner: 54
            # self.face_template = np.array([[193.65928, 242.98541], [318.32558, 243.06108], [255.67984, 328.82894],
            #                                 [198.22603, 372.82502], [313.91018, 372.75659]])

        self.face_template = self.face_template * (face_size / 512.0)
        if self.crop_ratio[0] > 1:
            self.face_template[:, 1] += face_size * \
                (self.crop_ratio[0] - 1) / 2
        if self.crop_ratio[1] > 1:
            self.face_template[:, 0] += face_size * \
                (self.crop_ratio[1] - 1) / 2
        self.save_ext = save_ext
        self.pad_blur = pad_blur
        if self.pad_blur is True:
            self.template_3points = False

        self.all_landmarks_5 = []
        self.det_faces = []
        self.affine_matrices = []
        self.inverse_affine_matrices = []
        self.cropped_faces = []
        self.restored_faces = []
        self.pad_input_imgs = []

        if device is None:
            # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.device = get_device()
        else:
            self.device = device

        # init face detection model
        if self.det_model == 'dlib':
            self.face_detector, self.shape_predictor_5 = self.init_dlib(
                dlib_model_path['face_detector'], dlib_model_path['shape_predictor_5'])
        else:
            self.face_detector = init_detection_model(
                det_model, half=False, device=self.device)

        # init face parsing model
        self.use_parse = use_parse
        self.face_parse = init_parsing_model(
            model_name='parsenet', device=self.device)

    def set_upscale_factor(self, upscale_factor):
        self.upscale_factor = upscale_factor

    def read_image(self, img):
        """img can be image path or cv2 loaded image."""
        # self.input_img is Numpy array, (h, w, c), BGR, uint8, [0, 255]
        if isinstance(img, str):
            img = cv2.imread(img)

        if np.max(img) > 256:  # 16-bit image
            img = img / 65535 * 255
        if len(img.shape) == 2:  # gray image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # BGRA image with alpha channel
            img = img[:, :, 0:3]

        self.input_img = img
        self.is_gray = is_gray(img, threshold=10)
        if self.is_gray:
            print('Grayscale input: True')

        if min(self.input_img.shape[:2]) < 512:
            f = 512.0/min(self.input_img.shape[:2])
            self.input_img = cv2.resize(
                self.input_img, (0, 0), fx=f, fy=f, interpolation=cv2.INTER_LINEAR)

    def init_dlib(self, detection_path, landmark5_path):
        """Initialize the dlib detectors and predictors."""
        try:
            import dlib
        except ImportError:
            print('Please install dlib by running:' 'conda install -c conda-forge dlib')
        # detection_path = load_file_from_url(
        #     url=detection_path, model_dir='weights/dlib', progress=True, file_name=None)
        # landmark5_path = load_file_from_url(
        #     url=landmark5_path, model_dir='weights/dlib', progress=True, file_name=None)
        face_detector = dlib.cnn_face_detection_model_v1(detection_path)
        shape_predictor_5 = dlib.shape_predictor(landmark5_path)
        return face_detector, shape_predictor_5

    def get_face_landmarks_5_dlib(self,
                                  only_keep_largest=False,
                                  scale=1):
        det_faces = self.face_detector(self.input_img, scale)

        if len(det_faces) == 0:
            # print('No face detected. Try to increase upsample_num_times.')
            return 0
        else:
            if only_keep_largest:
                # print('Detect several faces and only keep the largest.')
                face_areas = []
                for i in range(len(det_faces)):
                    face_area = (det_faces[i].rect.right() - det_faces[i].rect.left()) * (
                        det_faces[i].rect.bottom() - det_faces[i].rect.top())
                    face_areas.append(face_area)
                largest_idx = face_areas.index(max(face_areas))
                self.det_faces = [det_faces[largest_idx]]
            else:
                self.det_faces = det_faces

        if len(self.det_faces) == 0:
            return 0

        for face in self.det_faces:
            shape = self.shape_predictor_5(self.input_img, face.rect)
            landmark = np.array([[part.x, part.y] for part in shape.parts()])
            self.all_landmarks_5.append(landmark)

        return len(self.all_landmarks_5)
    

    def get_face_landmarks_5(self,
                             only_keep_largest=False,
                             only_center_face=False,
                             resize=None,
                             blur_ratio=0.01,
                             eye_dist_threshold=None):
        if self.det_model == 'dlib':
            return self.get_face_landmarks_5_dlib(only_keep_largest)

        if resize is None:
            scale = 1
            input_img = self.input_img
        else:
            h, w = self.input_img.shape[0:2]
            scale = resize / min(h, w)
            scale = max(1, scale)  # always scale up
            h, w = int(h * scale), int(w * scale)
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            input_img = cv2.resize(
                self.input_img, (w, h), interpolation=interp)

        with torch.no_grad():
            bboxes = self.face_detector.detect_faces(input_img)

        if bboxes is None or bboxes.shape[0] == 0:
            return 0
        else:
            bboxes = bboxes / scale

        for bbox in bboxes:
            # remove faces with too small eye distance: side faces or too small faces
            eye_dist = np.linalg.norm([bbox[6] - bbox[8], bbox[7] - bbox[9]])
            if eye_dist_threshold is not None and (eye_dist < eye_dist_threshold):
                continue

            if self.template_3points:
                landmark = np.array([[bbox[i], bbox[i + 1]]
                                    for i in range(5, 11, 2)])
            else:
                landmark = np.array([[bbox[i], bbox[i + 1]]
                                    for i in range(5, 15, 2)])
            self.all_landmarks_5.append(landmark)
            self.det_faces.append(bbox[0:5])

        if len(self.det_faces) == 0:
            return 0
        if only_keep_largest:
            h, w, _ = self.input_img.shape
            self.det_faces, largest_idx = get_largest_face(
                self.det_faces, h, w)
            self.all_landmarks_5 = [self.all_landmarks_5[largest_idx]]
        elif only_center_face:
            h, w, _ = self.input_img.shape
            self.det_faces, center_idx = get_center_face(self.det_faces, h, w)
            self.all_landmarks_5 = [self.all_landmarks_5[center_idx]]

        # pad blurry images
        if self.pad_blur:
            self.pad_input_imgs = []
            for landmarks in self.all_landmarks_5:
                # get landmarks
                eye_left = landmarks[0, :]
                eye_right = landmarks[1, :]
                eye_avg = (eye_left + eye_right) * 0.5
                mouth_avg = (landmarks[3, :] + landmarks[4, :]) * 0.5
                eye_to_eye = eye_right - eye_left
                eye_to_mouth = mouth_avg - eye_avg

                # Get the oriented crop rectangle
                # x: half width of the oriented crop rectangle
                x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
                #  - np.flipud(eye_to_mouth) * [-1, 1]: rotate 90 clockwise
                # norm with the hypotenuse: get the direction
                x /= np.hypot(*x)  # get the hypotenuse of a right triangle
                rect_scale = 1.5
                x *= max(np.hypot(*eye_to_eye) * 2.0 * rect_scale,
                         np.hypot(*eye_to_mouth) * 1.8 * rect_scale)
                # y: half height of the oriented crop rectangle
                y = np.flipud(x) * [-1, 1]

                # c: center
                c = eye_avg + eye_to_mouth * 0.1
                # quad: (left_top, left_bottom, right_bottom, right_top)
                quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
                # qsize: side length of the square
                qsize = np.hypot(*x) * 2
                border = max(int(np.rint(qsize * 0.1)), 3)

                # get pad
                # pad: (width_left, height_top, width_right, height_bottom)
                pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                       int(np.ceil(max(quad[:, 1]))))
                pad = [
                    max(-pad[0] + border, 1),
                    max(-pad[1] + border, 1),
                    max(pad[2] - self.input_img.shape[0] + border, 1),
                    max(pad[3] - self.input_img.shape[1] + border, 1)
                ]

                if max(pad) > 1:
                    # pad image
                    pad_img = np.pad(
                        self.input_img, ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
                    # modify landmark coords
                    landmarks[:, 0] += pad[0]
                    landmarks[:, 1] += pad[1]
                    # blur pad images
                    h, w, _ = pad_img.shape
                    y, x, _ = np.ogrid[:h, :w, :1]
                    mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0],
                                                       np.float32(w - 1 - x) / pad[2]),
                                      1.0 - np.minimum(np.float32(y) / pad[1],
                                                       np.float32(h - 1 - y) / pad[3]))
                    blur = int(qsize * blur_ratio)
                    if blur % 2 == 0:
                        blur += 1
                    blur_img = cv2.boxFilter(pad_img, 0, ksize=(blur, blur))
                    # blur_img = cv2.GaussianBlur(pad_img, (blur, blur), 0)

                    pad_img = pad_img.astype('float32')
                    pad_img += (blur_img - pad_img) * \
                        np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
                    pad_img += (np.median(pad_img, axis=(0, 1)) -
                                pad_img) * np.clip(mask, 0.0, 1.0)
                    pad_img = np.clip(pad_img, 0, 255)  # float32, [0, 255]
                    self.pad_input_imgs.append(pad_img)
                else:
                    self.pad_input_imgs.append(np.copy(self.input_img))

        return len(self.all_landmarks_5)

    def align_warp_face(self, save_cropped_path=None, border_mode='constant'):
        """Align and warp faces with face template.
        """
        if self.pad_blur:
            assert len(self.pad_input_imgs) == len(
                self.all_landmarks_5), f'Mismatched samples: {len(self.pad_input_imgs)} and {len(self.all_landmarks_5)}'
        for idx, landmark in enumerate(self.all_landmarks_5):
            # use 5 landmarks to get affine matrix
            # use cv2.LMEDS method for the equivalence to skimage transform
            # ref: https://blog.csdn.net/yichxi/article/details/115827338
            affine_matrix = cv2.estimateAffinePartial2D(
                landmark, self.face_template, method=cv2.LMEDS
            )[0]
            self.affine_matrices.append(affine_matrix)
            # warp and crop faces
            if border_mode == 'constant':
                border_mode = cv2.BORDER_CONSTANT
            elif border_mode == 'reflect101':
                border_mode = cv2.BORDER_REFLECT101
            elif border_mode == 'reflect':
                border_mode = cv2.BORDER_REFLECT
            if self.pad_blur:
                input_img = self.pad_input_imgs[idx]
            else:
                input_img = self.input_img
            # pdb.set_trace()
            cropped_face = cv2.warpAffine(
                input_img,
                affine_matrix,
                self.face_size,
                borderMode=border_mode,
                borderValue=(135, 133, 132),
            )  # gray
            self.cropped_faces.append(cropped_face)
            # save the cropped face
            if save_cropped_path is not None:
                path = os.path.splitext(save_cropped_path)[0]
                save_path = f'{path}_{idx:02d}.{self.save_ext}'
                imwrite(cropped_face, save_path)

    def get_inverse_affine(self, save_inverse_affine_path=None):
        """Get inverse affine matrix."""
        for idx, affine_matrix in enumerate(self.affine_matrices):
            inverse_affine = cv2.invertAffineTransform(affine_matrix)
            inverse_affine *= self.upscale_factor
            self.inverse_affine_matrices.append(inverse_affine)
            # save inverse affine matrices
            if save_inverse_affine_path is not None:
                path, _ = os.path.splitext(save_inverse_affine_path)
                save_path = f'{path}_{idx:02d}.pth'
                torch.save(inverse_affine, save_path)

    def add_restored_face(self, restored_face, input_face=None):
        if self.is_gray:
            # convert img into grayscale
            restored_face = bgr2gray(restored_face)
            if input_face is not None:
                restored_face = adain_npy(
                    restored_face, input_face)  # transfer the color
        self.restored_faces.append(restored_face)

    def paste_faces_to_input_image(
        self, save_path=None, upsample_img=None, draw_box=False, face_upsampler=None
    ):
        h, w, _ = self.input_img.shape
        h_up, w_up = int(h * self.upscale_factor), int(w * self.upscale_factor)

        if upsample_img is None:
            # simply resize the background
            # upsample_img = cv2.resize(self.input_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4)
            upsample_img = cv2.resize(
                self.input_img, (w_up, h_up), interpolation=cv2.INTER_LINEAR
            )
        else:
            upsample_img = cv2.resize(
                upsample_img, (w_up, h_up), interpolation=cv2.INTER_LANCZOS4
            )

        assert len(self.restored_faces) == len(
            self.inverse_affine_matrices), ('length of restored_faces and affine_matrices are different.')

        inv_mask_borders = []
        for restored_face, inverse_affine in zip(
            self.restored_faces, self.inverse_affine_matrices
        ):
            if face_upsampler is not None:
                restored_face = face_upsampler.enhance(
                    restored_face, outscale=self.upscale_factor
                )[0]
                inverse_affine /= self.upscale_factor
                inverse_affine[:, 2] *= self.upscale_factor
                face_size = (
                    self.face_size[0] * self.upscale_factor,
                    self.face_size[1] * self.upscale_factor,
                )
            else:
                # Add an offset to inverse affine matrix, for more precise back alignment
                if self.upscale_factor > 1:
                    extra_offset = 0.5 * self.upscale_factor
                else:
                    extra_offset = 0
                inverse_affine[:, 2] += extra_offset
                face_size = self.face_size
            inv_restored = cv2.warpAffine(
                restored_face, inverse_affine, (w_up, h_up))

            # always use square mask
            mask = np.ones(face_size, dtype=np.float32)
            inv_mask = cv2.warpAffine(mask, inverse_affine, (w_up, h_up))
            # remove the black borders
            inv_mask_erosion = cv2.erode(
                inv_mask,
                np.ones(
                    (int(2 * self.upscale_factor), int(2 * self.upscale_factor)),
                    np.uint8,
                ),
            )
            pasted_face = inv_mask_erosion[:, :, None] * inv_restored
            total_face_area = np.sum(inv_mask_erosion)  # // 3
            # add border
            if draw_box:
                h, w = face_size
                mask_border = np.ones((h, w, 3), dtype=np.float32)
                border = int(1400 / np.sqrt(total_face_area))
                mask_border[border : h - border, border : w - border, :] = 0
                inv_mask_border = cv2.warpAffine(
                    mask_border, inverse_affine, (w_up, h_up)
                )
                inv_mask_borders.append(inv_mask_border)
            # compute the fusion edge based on the area of face
            w_edge = int(total_face_area**0.5) // 20
            erosion_radius = w_edge * 2
            inv_mask_center = cv2.erode(
                inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8)
            )
            blur_size = w_edge * 2
            inv_soft_mask = cv2.GaussianBlur(
                inv_mask_center, (blur_size + 1, blur_size + 1), 0
            )
            if len(upsample_img.shape) == 2:  # upsample_img is gray image
                upsample_img = upsample_img[:, :, None]
            inv_soft_mask = inv_soft_mask[:, :, None]

            # cv2.imwrite("inv_soft_mask_1.png", (255 * inv_soft_mask).astype(np.uint8))

            # parse mask
            if self.use_parse:
                # inference
                face_input = cv2.resize(
                    restored_face, (512, 512), interpolation=cv2.INTER_LINEAR)
                face_input = img2tensor(face_input.astype(
                    'float32') / 255., bgr2rgb=True, float32=True)
                normalize(face_input, (0.5, 0.5, 0.5),
                          (0.5, 0.5, 0.5), inplace=True)
                face_input = torch.unsqueeze(face_input, 0).to(self.device)
                with torch.no_grad():
                    out = self.face_parse(face_input)[0]
                out = out.argmax(dim=1).squeeze().cpu().numpy()

                parse_mask = np.zeros(out.shape)
                MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255,
                                 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]
                for idx, color in enumerate(MASK_COLORMAP):
                    parse_mask[out == idx] = color
                #  blur the mask
                parse_mask = cv2.GaussianBlur(parse_mask, (101, 101), 11)
                parse_mask = cv2.GaussianBlur(parse_mask, (101, 101), 11)
                # remove the black borders
                thres = 10
                parse_mask[:thres, :] = 0
                parse_mask[-thres:, :] = 0
                parse_mask[:, :thres] = 0
                parse_mask[:, -thres:] = 0
                parse_mask = parse_mask / 255.0

                parse_mask = cv2.resize(parse_mask, face_size)
                parse_mask = cv2.warpAffine(
                    parse_mask, inverse_affine, (w_up, h_up), flags=3
                )
                inv_soft_parse_mask = parse_mask[:, :, None]
                # pasted_face = inv_restored
                fuse_mask = (inv_soft_parse_mask < inv_soft_mask).astype('int')
                inv_soft_mask = inv_soft_parse_mask * fuse_mask + inv_soft_mask * (1 - fuse_mask)

            # cv2.imwrite("z_inv_soft_mask.png", (255 * inv_soft_mask).astype(np.uint8))
            # cv2.imwrite("z_1-inv_soft_mask.png", (255 * (1 - inv_soft_mask)).astype(np.uint8))
            # cv2.imwrite("z_upsample_img.png", upsample_img.astype(np.uint8))
            # cv2.imwrite("z_pasted_face.png", pasted_face.astype(np.uint8))

            # alpha channel
            if len(upsample_img.shape) == 3 and upsample_img.shape[2] == 4:
                alpha = upsample_img[:, :, 3:]
                upsample_img = inv_soft_mask * pasted_face + \
                    (1 - inv_soft_mask) * upsample_img[:, :, 0:3]
                upsample_img = np.concatenate((upsample_img, alpha), axis=2)
            else:
                upsample_img = inv_soft_mask * pasted_face + \
                    (1 - inv_soft_mask) * upsample_img

            # cv2.imwrite("z_merged.png", upsample_img.astype(np.uint8))
            # import time
            # time.sleep(100)

        if np.max(upsample_img) > 256:  # 16-bit image
            upsample_img = upsample_img.astype(np.uint16)
        else:
            upsample_img = upsample_img.astype(np.uint8)

        # draw bounding box
        if draw_box:
            # upsample_input_img = cv2.resize(input_img, (w_up, h_up))
            img_color = np.ones([*upsample_img.shape], dtype=np.float32)
            img_color[:, :, 0] = 0
            img_color[:, :, 1] = 255
            img_color[:, :, 2] = 0
            for inv_mask_border in inv_mask_borders:
                upsample_img = inv_mask_border * img_color + \
                    (1 - inv_mask_border) * upsample_img
                # upsample_input_img = inv_mask_border * img_color + (1 - inv_mask_border) * upsample_input_img

        if save_path is not None:
            path = os.path.splitext(save_path)[0]
            save_path = f'{path}.{self.save_ext}'
            imwrite(upsample_img, save_path)
        return upsample_img

    def clean_all(self):
        self.all_landmarks_5 = []
        self.restored_faces = []
        self.affine_matrices = []
        self.cropped_faces = []
        self.inverse_affine_matrices = []
        self.det_faces = []
        self.pad_input_imgs = []


class FaceAligner(object):
    def __init__(self,
                 upscale_factor,
                 face_size=512,
                 crop_ratio=(1, 1),
                 det_model='retinaface_resnet50',
                 save_ext='png',
                 template_3points=False,
                 pad_blur=False,
                 use_parse=False,
                 device=None):
        self.template_3points = template_3points  # improve robustness
        self.upscale_factor = int(upscale_factor)
        # the cropped face ratio based on the square face
        self.crop_ratio = crop_ratio  # (h, w)
        assert (self.crop_ratio[0] >= 1 and self.crop_ratio[1]
                >= 1), 'crop ration only supports >=1'
        self.face_size = (
            int(face_size * self.crop_ratio[1]), int(face_size * self.crop_ratio[0]))
        self.det_model = det_model

        if self.det_model == 'dlib':
            # standard 5 landmarks for FFHQ faces with 1024 x 1024
            self.face_template = np.array([[686.77227723, 488.62376238], [586.77227723, 493.59405941],
                                           [337.91089109, 488.38613861], [
                                               437.95049505, 493.51485149],
                                           [513.58415842, 678.5049505]])
            self.face_template = self.face_template / (1024 // face_size)
        elif self.template_3points:
            self.face_template = np.array([[192, 240], [319, 240], [257, 371]])
        else:
            # standard 5 landmarks for FFHQ faces with 512 x 512
            # facexlib
            self.face_template = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                                           [201.26117, 371.41043], [313.08905, 371.15118]])

            # dlib: left_eye: 36:41  right_eye: 42:47  nose: 30,32,33,34  left mouth corner: 48  right mouth corner: 54
            # self.face_template = np.array([[193.65928, 242.98541], [318.32558, 243.06108], [255.67984, 328.82894],
            #                                 [198.22603, 372.82502], [313.91018, 372.75659]])

        self.face_template = self.face_template * (face_size / 512.0)
        if self.crop_ratio[0] > 1:
            self.face_template[:, 1] += face_size * \
                (self.crop_ratio[0] - 1) / 2
        if self.crop_ratio[1] > 1:
            self.face_template[:, 0] += face_size * \
                (self.crop_ratio[1] - 1) / 2
        self.save_ext = save_ext
        self.pad_blur = pad_blur
        if self.pad_blur is True:
            self.template_3points = False

        self.all_landmarks_5 = []
        self.det_faces = []
        self.affine_matrices = []
        self.inverse_affine_matrices = []
        self.cropped_faces = []
        self.restored_faces = []
        self.pad_input_imgs = []

        if device is None:
            # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.device = get_device()
        else:
            self.device = device

    def set_image(self, img):
        self.input_img = img

    def align_pair_face(self, img_lq, img_gt, landmarks):
        img_lq = (img_lq[:, :, ::-1] * 255).round().astype(np.uint8)
        img_gt = (img_gt[:, :, ::-1] * 255).round().astype(np.uint8)

        self.set_image(img_gt)
        img_lq, img_gt = self.align_warp_face(img_lq, img_gt, landmarks)
        img_lq = img_lq[:, :, ::-1] / 255.0
        img_gt = img_gt[:, :, ::-1] / 255.0
        return img_lq, img_gt

    def align_single_face(self, img, landmarks, border_mode='constant'):
        """Align and warp faces with face template.
           Suppose input images are Numpy array, (h, w, c), BGR, uint8, [0, 255]
        """
        # warp and crop faces
        if border_mode == 'constant':
            border_mode = cv2.BORDER_CONSTANT
        elif border_mode == 'reflect101':
            border_mode = cv2.BORDER_REFLECT101
        elif border_mode == 'reflect':
            border_mode = cv2.BORDER_REFLECT

        img = (img[:, :, ::-1] * 255).round().astype(np.uint8)

        affine_matrix = cv2.estimateAffinePartial2D(
            landmarks, self.face_template, method=cv2.LMEDS)[0]
        img = cv2.warpAffine(
            img, affine_matrix, img.shape[0:2], borderMode=border_mode, borderValue=(135, 133, 132))  # gray
        img = img[:, :, ::-1] / 255.0
        return img

    def align_warp_face(self, img_lq, img_gt, landmarks, border_mode='constant'):
        """Align and warp faces with face template.
           Suppose input images are Numpy array, (h, w, c), BGR, uint8, [0, 255]
        """
        # use 5 landmarks to get affine matrix
        # use cv2.LMEDS method for the equivalence to skimage transform
        # ref: https://blog.csdn.net/yichxi/article/details/115827338
        scale = img_gt.shape[0] / img_lq.shape[0]
        # warp and crop faces
        if border_mode == 'constant':
            border_mode = cv2.BORDER_CONSTANT
        elif border_mode == 'reflect101':
            border_mode = cv2.BORDER_REFLECT101
        elif border_mode == 'reflect':
            border_mode = cv2.BORDER_REFLECT

        affine_matrix = cv2.estimateAffinePartial2D(
            landmarks, self.face_template, method=cv2.LMEDS)[0]
        img_gt = cv2.warpAffine(
            img_gt, affine_matrix, img_gt.shape[0:2], borderMode=border_mode, borderValue=(135, 133, 132))  # gray

        affine_matrix = cv2.estimateAffinePartial2D(
            landmarks / scale, self.face_template / scale, method=cv2.LMEDS)[0]
        img_lq = cv2.warpAffine(
            img_lq, affine_matrix, img_lq.shape[0:2], borderMode=border_mode, borderValue=(135, 133, 132))  # gray

        return img_lq, img_gt

    def clean_all(self):
        self.all_landmarks_5 = []
        self.restored_faces = []
        self.affine_matrices = []
        self.cropped_faces = []
        self.inverse_affine_matrices = []
        self.det_faces = []
        self.pad_input_imgs = []
