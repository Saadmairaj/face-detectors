import os
import cv2
import dlib
import requests
import numpy as np
import os.path as osp
from tqdm import tqdm


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """Resizes the given image with only one value either width or 
    height while also keeping the image proportion. If both the width 
    and height are None, then return the original image.

    Author: Adrian Rosebrock :: jrosebr1/imutils

    Args:
        image (np.array): Image to be resized
        width (int, optional): Width of the output image. Defaults to None.
        height (int, optional): Height of the output image. Defaults to None.
        inter (Any, optional): To shrink an image, it will generally look 
            best with #INTER_AREA interpolation, whereas to enlarge an image, 
            it will generally look best with c#INTER_CUBIC (slow) or #INTER_LINEAR 
            (faster but still looks OK).. Defaults to cv2.INTER_AREA.

    Returns:
        np.array: Resized output image
    """
    if width is None and height is None:
        return image

    dim = None
    (h, w) = image.shape[:2]

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def find_faces(img, net, confidence, resize, mean, scalefactor,
               crop=False, swapRB=True, transpose=False):
    """Caffemodel face detector constructor.

    Args:
        img: Input images (all with 1-, 3- or 4-channels).
        net (cv2.dnn.readNet): Takes the cv2::dnn:readNet instance.
        confidence (float, optional): Confidence score is used to refrain from making 
            predictions when it is not above a sufficient threshold. Defaults to 0.5.
        resize (list[int], optional): Spatial size for output image. Default is (300, 300).
        mean (tuple, optional): scalar with mean values which are subtracted from channels. 
            Values are intended to be in (mean-R, mean-G, mean-B) order if image has BGR 
            ordering and swapRB is true. Defaults to (104.0, 177.0, 123.0).
        scalefactor (float, optional): Multiplier for images values. Defaults to 1.0.
        crop (bool, optional): Flag which indicates whether image will be cropped after 
            resize or not. Defaults to False.
        swapRB (bool, optional): Flag which indicates that swap first and last channels 
            in 3-channel image is necessary. Defaults to False.
        transpose (bool, optional): Tranpose image. Defaults to False.
    """
    h, w = img.shape[:2]
    resize_img = cv2.resize(img, resize)
    if transpose:
        # only for onnx model
        resize_img = resize_img[:, ::-1].transpose()

    blob = cv2.dnn.blobFromImage(
        image=resize_img, scalefactor=scalefactor,
        size=resize, crop=crop, mean=mean,
        swapRB=swapRB)
    net.setInput(blob)
    res = net.forward()
    faces = []
    for i in range(res.shape[2]):
        face = {}
        face['confidence'] = res[0, 0, i, 2]
        if face['confidence'] > confidence:
            box = res[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            face['bbox'] = [y, x1, y1, x]
            faces.append(face)
    return faces


def _rect_to_css(rect):
    """Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order

    Args:
        rect (dlib.rectangle): a dlib 'rect' object

    Returns
        List: a plain tuple representation of the rect in (top, right, bottom, left) order"""

    return rect.top(), rect.right(), rect.bottom(), rect.left()


def area_of(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Perform hard non-maximum-supression to filter out boxes with iou greater
    than threshold

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
        picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
    """Select boxes that contain human faces

    Args:
        width: original image width
        height: original image height
        confidences (N, 2): confidence array
        boxes (N, 4): boxes array in corner-form
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
        boxes (k, 4): an array of boxes kept
        labels (k): an array of labels for each boxes kept
        probs (k): an array of probabilities for each boxes being in corresponding labels
    """
    picked_labels = []
    picked_box_probs = []

    boxes = boxes[0]
    confidences = confidences[0]
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]

        if probs.shape[0] == 0:
            continue

        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate(
            [subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
                             iou_threshold=iou_threshold,
                             top_k=top_k
                             )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])

    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])

    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


def download_model(link, models_path=None, block_size=1024):
    """Downloads pre-trained models from the given link 
    that are saved in "models" folder in the base directory 
    of the package"""
    try:
        base_dir = osp.abspath(osp.dirname(osp.dirname(__file__)))
        models_path = models_path or osp.join(base_dir, "models")
        name = osp.join(models_path, link.split('/')[-1])

        if not osp.exists(models_path):
            os.makedirs(models_path)

        response = requests.get(link, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))

        print(f"\nDownloading file: \t{link.split('/')[-1]}")
        print(f"Block size: \t\t{block_size}")
        print(f"Total size (bytes): \t{total_size_in_bytes}")
        print(f"Downloading link: \t{link}")

        progress_bar = tqdm(total=total_size_in_bytes,
                            unit='iB', unit_scale=True)
        with open(name, 'wb') as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)

        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            raise RuntimeError("Incomplete download")
        return osp.abspath(name)

    except Exception as e:
        raise RuntimeError(f"Something went wrong. {e}")


class FaceDetectorModels:
    """
    Internal models handler class.
    Stores path to the face detector models.
    Use method absolute_path(..) to get the absolute 
    path to the required model. If the model doesn't 
    exists then this handler downloads the required model.
    """

    # caffe face detection
    caffe_model_ssd_300x300 = "res10_300x300_ssd_iter_140000.caffemodel"
    caffe_config_ssd_300x300 = "res10_300x300_ssd_iter_140000.prototxt"

    # onnx face detector
    ultra_light_320_face_detector = "ultra_light_320.onnx"
    ultra_light_640_face_detector = "ultra_light_640.onnx"

    # cnn dlib
    cnn_mmod_face_detector = "mmod_human_face_detector.dat"

    # pose models
    landmarks_5 = "shape_predictor_5_face_landmarks.dat"
    landmarks_68 = "shape_predictor_68_face_landmarks_GTX.dat"

    @staticmethod
    def absolute_path(name):
        """Get the absolute path to the given model or the config"""
        of = osp.join("models", name)
        model_path = osp.abspath(
            osp.join(
                osp.abspath(osp.dirname(osp.dirname(__file__))), of
            )
        )
        if not osp.exists(model_path):
            link = f"https://github.com/Saadmairaj/models/releases/download/v0.0.1/{name}"
            return download_model(link)

        return model_path


class BaseModel:
    """
    Base model class for face detector
    """

    NAME = None

    def _setup(self):
        """Internal function, do not call directly.

        Setup the model with the configurations"""
        raise NotImplementedError(
            "Do not initialise BaseModel class directly. "
            "Inherit the class into a class.")

    def _scale_back(self, faces, image=None, original_size=None, scale=None):
        """Internal function.

        Convert a tuple in (top, right, bottom, left) 
        order to a dlib `rect` object or get simple list of faces"""
        if not isinstance(faces, (tuple, list)):
            faces = [faces]

        if faces and isinstance(faces[0], dlib.rectangle):
            faces = [_rect_to_css(rect) for rect in faces]

        scale = scale or self.scale
        scale_ = round(1 / scale)

        if image is not None and original_size is not None:
            h, w = image.shape[:2]
            if original_size is not None:
                scale_ = int(original_size[1] / h)

        scale_ = scale or scale_

        rects = [
            dict(
                bbox=(
                    round(face['bbox'][0] * (1 / scale_)),  # right
                    round(face['bbox'][1] * (1 / scale_)),  # bottom
                    round(face['bbox'][2] * (1 / scale_)),  # left
                    round(face['bbox'][3] * (1 / scale_))   # top
                ),
                confidence=face['confidence'],
            ) for face in faces
        ]
        return rects

    def _prep_image(self, image, resize_width=None, scale=None):
        """Internal function. 

        Prepares the image before detection"""
        scale = scale or self.scale


        if self.convert_color is not None:
            image = cv2.cvtColor(image, self.convert_color)
        if resize_width is not None:
            return resize(image, width=resize_width)
        return cv2.resize(image, (0, 0), fx=scale, fy=scale)

    def detect_faces(self, image):
        """Detects faces in the given image

        Args:
            image: Give numpy.array image
        Return:
            List[Any, List]: List of faces coordinates with coordinates"""
        raise NotImplementedError(
            "Do not initialise BaseModel class directly. "
            "Inherit the class into a class.")

    def detect_faces_keypoints(self, image, get_all=False):
        """Detects faces with keypoints of mouth, eyes, nose. 

        Detecting faces along with keypoints can be a bit 
        slower compared to just detecting faces so if keypoints 
        is not what you need use only detect_face.

        Args:
            image: Give numpy.array image
            get_all (bool): If true then returns all the keypoints of the face

        Return:
            List[Any, List]: List of faces coordinates with coordinates and 
                main keypoints of the faces"""
        if getattr(self, 'landmarks_predictor', None) is None:
            self.landmarks_predictor = dlib.shape_predictor(
                FaceDetectorModels.absolute_path(FaceDetectorModels.landmarks_68))

        faces: list = self.detect_faces(image)

        for index, face in enumerate(faces.copy()):
            right, bottom, left, top = face['bbox']
            rect = dlib.rectangle(top, right, bottom, left)
            landmarks = self.landmarks_predictor(image, rect)

            if get_all:
                keypoints = {p: (landmarks.part(p).x, landmarks.part(p).y)
                             for p in range(68)}
            else:
                eye_left_x = landmarks.part(36).x + round(
                    (landmarks.part(39).x - landmarks.part(36).x) / 2)
                eye_left_y = landmarks.part(38).y + (
                    landmarks.part(40).y - landmarks.part(38).y)

                eye_right_x = landmarks.part(42).x + round(
                    (landmarks.part(45).x - landmarks.part(42).x) / 2)
                eye_right_y = landmarks.part(44).y + (
                    landmarks.part(46).y - landmarks.part(44).y)

                keypoints = {
                    "left_eye": (eye_left_x, eye_left_y),
                    "right_eye": (eye_right_x, eye_right_y),
                    "nose": (landmarks.part(30).x, landmarks.part(30).y),
                    "mouth_left": (landmarks.part(48).x, landmarks.part(48).y),
                    "mouth_right": (landmarks.part(54).x, landmarks.part(54).y),
                    "chin": (landmarks.part(8).x, landmarks.part(8).y)
                }

            faces[index].update(
                {"keypoints": keypoints, "bbox": _rect_to_css(rect)})

        return faces

    def __str__(self):
        if self.NAME is None:
            raise NotImplementedError(
                "Do not initialise BaseModel class directly. "
                "Inherit the class into a class.")
        return f"<Face Detectors: {self.__class__.__name__}>"

    def __repr__(self):
        if self.NAME is None:
            raise NotImplementedError(
                "Do not initialise BaseModel class directly. "
                "Inherit the class into a class.")
        dict_str = ""
        for k, v in self.__dict__.items():
            dict_str += f"{k}={v}, "
        return f"<{self.__class__.__name__}> ({dict_str.rstrip(', ')})"
