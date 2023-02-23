from yolov4.tf import YOLOv4
import tensorflow as tf
import time
import cv2
import numpy as np

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


class ObjectDetectorAPI:
    def __init__(self):
        self.yolo = YOLOv4(tiny=True)
        self.yolo.classes = "Yolov4/coco.names"
        self.yolo.make_model()
        self.yolo.load_weights("Yolov4/yolov4-tiny.weights", weights_type="yolo")
    
    def predict(self, image):
        start_time=time.time()
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image = self.yolo.resize_image(img)
        # 0 ~ 255 to 0.0 ~ 1.0
        resized_image = resized_image / 255.
        #input_data == Dim(1, input_size, input_size, channels)
        input_data = resized_image[np.newaxis, ...].astype(np.float32)

        candidates = self.yolo.model.predict(input_data)

        _candidates = []
        result = img.copy()
        for candidate in candidates:
            batch_size = candidate.shape[0]
            grid_size = candidate.shape[1]
            _candidates.append(tf.reshape(candidate, shape=(1, grid_size * grid_size * 3, -1)))
            #candidates == Dim(batch, candidates, (bbox))
            candidates = np.concatenate(_candidates, axis=1)
            #pred_bboxes == Dim(candidates, (x, y, w, h, class_id, prob))
            pred_bboxes = self.yolo.candidates_to_pred_bboxes(candidates[0], iou_threshold=0.35, score_threshold=0.40)
            pred_bboxes = pred_bboxes[~(pred_bboxes==0).all(1)] #https://stackoverflow.com/questions/35673095/python-how-to-eliminate-all-the-zero-rows-from-a-matrix-in-numpy?lq=1
            pred_bboxes = self.yolo.fit_pred_bboxes_to_original(pred_bboxes, img.shape)
            exec_time = time.time() - start_time
            #print("time: {:.2f} ms".format(exec_time * 1000))
            result = self.yolo.draw_bboxes(img, pred_bboxes)
        return result, pred_bboxes
