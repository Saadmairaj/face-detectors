import cv2
import numpy as np
import onnxruntime as ort
from face_detectors.detectors.utils import BaseModel, predict, FaceDetectorModels


class UltralightDetector(BaseModel):
    """
    Ultra light onnx model class

    Use Ultralight320Detector or Ultralight640Detector 
    instead using this class
    """

    def __init__(self, model_size, confidence=0.7, mean=[127, 127, 127],
                 convert_color=cv2.COLOR_BGR2RGB, **kw):
        """Ultra light constructor. 
        Initialize  Ultralight320Detector or Ultralight640Detector 
        instead using this class instead.

        Args:
            model_size (Union[str, int]): Take 320 or 640 as model size argument.
            confidence (float, optional): Confidence score is used to refrain from making 
                predictions when it is not above a sufficient threshold. Defaults to 0.7.
            mean (list[int], optional): Metric used to measure the performance of models doing
                detection tasks. Defaults to [127, 127, 127].
            convert_color ([type], optional): Takes OpenCV COLOR codes to convert the images. 
                Defaults to cv2.COLOR_BGR2RGB.
            scale (float, optional): Scales the image for faster output (No need to set 
                this manually). Default is 1.
            cache (bool, optional): It uses same model for all the created sessions. 
                Default is True
        """
        self.mean = mean
        self.confidence = confidence
        self.scale = kw.get('scale', 1)
        self.convert_color = convert_color

        assert str(model_size) in ('320', '640'), "Invalid model size "
        self.resize = (320, 240) if model_size in (320, '320') else (640, 480)

        self._setup(**kw)

    def _setup(self, **kw):
        """Internal function, do not call directly.

        Setup the model with the configurations"""
        assert self.resize in [(320, 240), (640, 480)], "Invalid model resize"

        self.ort_session, self.input_name = self._face_detection_onnx_session(
            cache=kw.get('cache', True))

        assert self.ort_session != None, "Ort session cannot be None, setup failed"
        assert self.input_name != None, "Ort input name cannot be None, setup failed"

        self._setup = True
        return self

    def detect_faces(self, image):
        """Detects faces in the given image

        Args:
            image: Give numpy.array image

        Return:
            List[Any, List]: List of faces coordinates"""
        assert getattr(
            self, '_setup', None) is not None, "The model is not setup."

        image = self._prep_image(image)
        h, w, _ = image.shape

        img = cv2.resize(image, self.resize)
        img_mean = np.array(self.mean)
        img = (img - img_mean) / 128
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        confidences, boxes = self.ort_session.run(None, {self.input_name: img})
        faces, labels, confidences = predict(
            w, h, confidences, boxes, self.confidence)

        pad = 0
        faces = [{"bbox": [face[1] - pad, face[0] - pad, face[3] + pad, face[2] + pad],
                  "confidence": conf} for face, conf in zip(faces, confidences)]
        return self._scale_back(faces=faces, scale=self.scale)

    def _face_detection_onnx_session(self, model_size, cache=True):
        """Internal function. Don't call directly.

        Returns onnx session for model 320 or 640 models

        Args:
            model_size (str): Which model size to select
            cache (bool): Wheather to create a new session for every 
                detector or cache the previous one. By default is True"""

        raise NotImplementedError('Do not class this class directly. Use "Ultralight320Detector" '
                                  'or "Ultralight640Detector" instead.')


class Ultralight320Detector(UltralightDetector):
    """
    Ultra light 320x240 onnx model class.

    This model has good execution time with impressive accuracy even on CPU.
    It resizes the target images into 320z240 res which gives fast and decent results
    """

    ort_session_320 = None
    input_name_320 = None
    NAME = "Ultra Light Detector 320px"

    def __init__(self, confidence=0.7, mean=[127, 127, 127],
                 convert_color=cv2.COLOR_BGR2RGB, **kw):
        """Ultralight320Detector constructor

        Args:
            confidence (float, optional): Confidence score is used to refrain from making 
                predictions when it is not above a sufficient threshold. Defaults to 0.7.
            mean (list[int], optional): Metric used to measure the performance of models doing
                detection tasks. Defaults to [127, 127, 127].
            convert_color (int, optional): Takes OpenCV COLOR codes to convert the images. 
                Defaults to cv2.COLOR_BGR2RGB.
            scale (float, optional): Scales the image for faster output (No need to set 
                this manually). Default is 1.
            cache (bool, optional): It uses same model for all the created sessions. 
                Default is True
        """
        super().__init__(320, confidence=confidence,
                         mean=mean, convert_color=convert_color, **kw)

    def _face_detection_onnx_session(self, cache=True):
        """Internal function. Don't call directly.

        Returns onnx session for model 320 model

        Args:
            model_size (str): Which model size to select
            cache (bool): Wheather to create a new session for every 
                detector or cache the previous one. By default is True"""

        if not cache:
            ort_session = ort.InferenceSession(FaceDetectorModels.absolute_path(
                FaceDetectorModels.ultra_light_320_face_detector))
            input_name = ort_session.get_inputs()[0].name
            return ort_session, input_name

        if self.ort_session_320 is None and self.input_name_320 is None:
            self.ort_session_320, self.input_name_320 = self._face_detection_onnx_session(
                False)
        return self.ort_session_320, self.input_name_320


class Ultralight640Detector(UltralightDetector):
    """
    Ultra light 640x480 onnx model class.

    This model has good execution time with impressive accuracy even on CPU.
    It resizes the target images into 640x480 res which is more accuracy 
    then Ultralight320Detector but a little slow.
    """

    ort_session_640 = None
    input_name_640 = None
    NAME = "Ultra Light Detector 640px"

    def __init__(self, confidence=0.7, mean=[127, 127, 127],
                 convert_color=cv2.COLOR_BGR2RGB, **kw):
        """Ultralight640Detector constructor

        Args:
            confidence (float, optional): Confidence score is used to refrain from making 
                predictions when it is not above a sufficient threshold. Defaults to 0.7.
            mean (list[int], optional): Metric used to measure the performance of models doing
                detection tasks. Defaults to [127, 127, 127].
            convert_color (int, optional): Takes OpenCV COLOR codes to convert the images. 
                Defaults to cv2.COLOR_BGR2RGB.
            scale (float, optional): Scales the image for faster output (No need to set 
                this manually). Default is 1.
            cache (bool, optional): It uses same model for all the created sessions. 
                Default is True
        """
        super().__init__(640, confidence=confidence,
                         mean=mean, convert_color=convert_color, **kw)

    def _face_detection_onnx_session(self, cache=True):
        """Internal function. Don't call directly.

        Returns onnx session for 640 model

        Args:
            model_size (str): Which model size to select
            cache (bool): Wheather to create a new session for every 
                detector or cache the previous one. By default is True"""

        if not cache:
            ort_session = ort.InferenceSession(FaceDetectorModels.absolute_path(
                FaceDetectorModels.ultra_light_640_face_detector))
            input_name = ort_session.get_inputs()[0].name
            return ort_session, input_name

        if self.ort_session_640 is None and self.input_name_640 is None:
            self.ort_session_640, self.input_name_640 = self._face_detection_onnx_session(
                False)
        return self.ort_session_640, self.input_name_640
