import cv2
from face_detectors.detectors.utils import BaseModel, find_faces, FaceDetectorModels


class CaffemodelDetector(BaseModel):
    """
    Caffe model dnn readnet class 
    Caffe is a deep learning framework made with 
    expression, speed, and modularity in mind
    """

    NAME = "Caffemodel Detector"
    model_path = None
    config_path = None

    def __init__(self, confidence=0.5, mean=(104.0, 177.0, 123.0), scalefactor=1.0,
                 crop=False, swapRB=False, transpose=False, convert_color=None, **kw):
        """Caffemodel face detector constructor.

        Args:
            confidence (float, optional): Confidence score is used to refrain from making 
                predictions when it is not above a sufficient threshold. Defaults to 0.5.
            mean (tuple, optional): scalar with mean values which are subtracted from channels. 
                Values are intended to be in (mean-R, mean-G, mean-B) order if image has BGR 
                ordering and swapRB is true. Defaults to (104.0, 177.0, 123.0).
            scalefactor (float, optional): Multiplier for images values. Defaults to 1.0.
            crop (bool, optional): Flag which indicates whether image will be cropped after 
                resize or not. Defaults to False.
            swapRB (bool, optional): Flag which indicates that swap first and last channels 
                in 3-channel image is necessary. Defaults to False.
            transpose (bool, optional): Transpose image. Defaults to False.
            convert_color ([type], optional): Takes OpenCV COLOR codes to convert the images. 
                Defaults to cv2.COLOR_BGR2RGB.
            resize (list[int], optional): Spatial size for output image. Default is (300, 300).
            scale (float, optional): Scales the image for faster output (No need to set
                this manually). Default is 1.
        """
        self.crop = crop
        self.mean = mean
        self.swapRB = swapRB
        self.transpose = transpose
        self.confidence = confidence
        self.scalefactor = scalefactor
        self.scale = kw.pop('scale', 1)
        self.convert_color = convert_color
        self.resize = kw.get("resize", (300, 300))

        self._setup(**kw)

    def _setup(self, **kw):
        """Internal function. Do not call directly

        Setup the model with the configurations"""
        DNN_BACKEND = (cv2.dnn.DNN_BACKEND_CUDA
                       if kw.get('gpu') is True
                       else cv2.dnn.DNN_BACKEND_OPENCV)
        DNN_TARGET = (cv2.dnn.DNN_TARGET_CUDA
                      if kw.get('gpu') is True
                      else cv2.dnn.DNN_TARGET_CPU)

        self.model_path = FaceDetectorModels.absolute_path(
            FaceDetectorModels.caffe_model_ssd_300x300)
        self.config_path = FaceDetectorModels.absolute_path(
            FaceDetectorModels.caffe_config_ssd_300x300)

        assert self.scale != None, "Scale is required"
        assert self.model_path != None, "Model path is empty"
        assert self.config_path != None, "Config path is empty"

        self.net = cv2.dnn.readNet(self.model_path, self.config_path)
        self.net.setPreferableBackend(DNN_BACKEND)
        self.net.setPreferableTarget(DNN_TARGET)

        self._setup = True
        return self

    def detect_faces(self, image):
        """Detects faces in the given image

        Args:
            image: Give numpy.array image
        Return:
            List[Any, List]: List of faces coordinates"""
        assert getattr(self, '_setup', None) != None, "The model is not setup."

        image = self._prep_image(image)
        faces = find_faces(img=image, net=self.net, confidence=self.confidence,
                           resize=self.resize, mean=self.mean, scalefactor=self.scalefactor,
                           crop=self.crop, swapRB=self.swapRB, transpose=self.transpose)

        return self._scale_back(faces, scale=self.scale)
