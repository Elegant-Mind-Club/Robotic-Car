import os
import sys
import time

import copy
import ailia
import cv2
import numpy as np
import socket
import multiprocessing

# import original modules
sys.path.append('../../util')
# logger
from logging import getLogger  # noqa: E402

from image_utils import imread, normalize_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402
from webcamera_utils import get_capture, get_writer
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from image_utils import normalize_image  # noqa
from detector_utils import plot_results  # noqa
from nms_utils import batched_nms
logger = getLogger(__name__)

# Replace with your ESP32's IP address and port
esp32_ip = "192.168.1.106"
esp32_port = 8080

# ======================
# Parameters
# ======================
WEIGHT_v20_PATH = 'midas.onnx'
MODEL_v20_PATH = 'midas.onnx.prototxt'
WEIGHT_v21_PATH = 'midas_v2.1.onnx'
MODEL_v21_PATH = 'midas_v2.1.onnx.prototxt'
WEIGHT_v21_SMALL_PATH = 'midas_v2.1_small.onnx'
MODEL_v21_SMALL_PATH = 'midas_v2.1_small.onnx.prototxt'
REMOTE_PATH_MIDAS = 'https://storage.googleapis.com/ailia-models/midas/'

IMAGE_PATH = 'input.jpg'
SAVE_IMAGE_PATH = 'input_depth.png'
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 384
IMAGE_HEIGHT_SMALL = 256
IMAGE_WIDTH_SMALL = 256
IMAGE_MULTIPLE_OF = 32

WEIGHT_YOLOV9E_PATH = 'yolov9e.onnx'
MODEL_YOLOV9E_PATH = 'yolov9e.onnx.prototxt'
WEIGHT_YOLOV9C_PATH = 'yolov9c.onnx'
MODEL_YOLOV9C_PATH = 'yolov9c.onnx.prototxt'
REMOTE_PATH_YOLO = 'https://storage.googleapis.com/ailia-models/yolov9/'


COCO_CATEGORY = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

THRESHOLD = 0.25
IOU = 0.7
DETECTION_SIZE = 640

# ======================
# Arguemnt Parser Config
# ======================
parser = get_base_parser('MiDaS model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-v21', '--version21', dest='v21', action='store_true',
    help='Use model version 2.1.'
)
parser.add_argument(
    '-t', '--model_type_midas', default='large', choices=('large', 'small'),
    help='model type: large or small. small can be specified only for version 2.1 model.'
)
parser.add_argument(
    '-th', '--threshold',
    default=THRESHOLD, type=float,
    help='The detection threshold for yolo.'
)
parser.add_argument(
    '-iou', '--iou',
    default=IOU, type=float,
    help='The detection iou for yolo.'
)
parser.add_argument(
    '-w', '--write_prediction',
    nargs='?',
    const='txt',
    choices=['txt', 'json'],
    type=str,
    help='Output results to txt or json file.'
)
parser.add_argument(
    '-ds', '--detection_size',
    default=DETECTION_SIZE, type=int,
    help='The detection width and height for yolo.'
)
parser.add_argument(
    '-m', '--model_type_yolo', default='v9e',
    choices=('v9e', 'v9c'),
    help='model type'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)


# ======================
# Secondary Functions
# ======================

def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y

    return y


def scale_boxes(img1_shape, boxes, img0_shape):
    """
    Rescales bounding boxes (in the format of xyxy) from the shape of the image they were originally specified in
    (img1_shape) to the shape of a different image (img0_shape).

    Args:
      img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
      boxes (numpy.ndarray): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
      img0_shape (tuple): the shape of the target image, in the format of (height, width).
    Returns:
      boxes (numpy.ndarray): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain

    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, img0_shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, img0_shape[0])  # y1, y2

    return boxes



# ======================
# Main functions
# ======================


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_h, im_w, _ = img.shape
    size = args.detection_size

    r = min(size / im_h, size / im_w)
    oh, ow = int(round(im_h * r)), int(round(im_w * r))
    if ow != im_w or oh != im_h:
        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)

    dh, dw = size - oh, size - ow
    if True:
        stride = 32
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    # divide padding into 2 sides
    dw /= 2
    dh /= 2

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=(114, 114, 114))  # add border

    # Scale input pixel value to 0 to 1
    img = normalize_image(img, normalize_type='255')
    img = img.transpose((2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img

def post_processing(preds, img, orig_shape):
    conf_thres = args.threshold
    iou_thres = args.iou

    xc = np.max(preds[:, 4:], axis=1) > conf_thres

    none_out = np.zeros((0, 6))

    x = preds[0].T[xc[0]]  # confidence
    if not x.shape[0]:
        return none_out

    box, cls = np.split(x, [4], axis=1)
    box = xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)

    j = np.argmax(cls, axis=1)
    conf = cls[np.arange(len(cls)), j]

    x = np.concatenate((box, conf.reshape(-1, 1), j.reshape(-1, 1)), axis=1)

    # Check shape
    n = x.shape[0]  # number of boxes
    if not n:  # no boxes
        return none_out

    # sort by confidence and remove excess boxes
    max_nms = 30000
    x = x[np.argsort(-x[:, 4])[:max_nms]]

    c = x[:, 5]
    boxes, scores = x[:, :4], x[:, 4]  # boxes, scores

    # Batched NMS
    i = batched_nms(boxes, scores, c, iou_thres)

    max_det = 300
    i = i[:max_det]  # limit detections
    preds = x[i]

    preds[:, :4] = np.round(scale_boxes(img.shape[2:], preds[:, :4], orig_shape))

    return preds

def predict(net, img):
    orig_shape = img.shape

    img = preprocess(img)

    # feedforward
    if not args.onnx:
        output = net.predict([img])
    else:
        # output = net.run(None, {'images': img})
        output = net.run([x.name for x in net.get_outputs()], {net.get_inputs()[0].name: img})

    preds = output[0]

    preds = post_processing(preds, img, orig_shape)

    return preds

person_detected = False
def convert_to_detector_object(preds, im_w, im_h):
    global person_detected
    person_detected = False
    detector_object = []
    for i in range(len(preds)):
        (x1, y1, x2, y2) = preds[i, :4]
        score = float(preds[i, 4])
        cls = int(preds[i, 5])
        if(COCO_CATEGORY[cls] == 'person'):
            person_detected = True
        r = ailia.DetectorObject(
            category=COCO_CATEGORY[cls],
            prob=score,
            x=x1 / im_w,
            y=y1 / im_h,
            w=(x2 - x1) / im_w,
            h=(y2 - y1) / im_h,
        )
        detector_object.append(r)

    return detector_object

def constrain_to_multiple_of(x, min_val=0, max_val=None):
    y = (np.round(x / IMAGE_MULTIPLE_OF) * IMAGE_MULTIPLE_OF).astype(int)
    if max_val is not None and y > max_val:
        y = (np.floor(x / IMAGE_MULTIPLE_OF) * IMAGE_MULTIPLE_OF).astype(int)
    if y < min_val:
        y = (np.ceil(x / IMAGE_MULTIPLE_OF) * IMAGE_MULTIPLE_OF).astype(int)
    return y


def midas_resize(image, target_height, target_width):
    # Resize while keep aspect ratio.
    h, w, _ = image.shape
    scale_height = target_height / h
    scale_width = target_width / w
    if scale_width < scale_height:
        scale_height = scale_width
    else:
        scale_width = scale_height
    new_height = constrain_to_multiple_of(
        scale_height * h, max_val=target_height
    )
    new_width = constrain_to_multiple_of(
        scale_width * w, max_val=target_width
    )

    return cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_CUBIC
    )


def send_command(command, client_socket):
    print(f"Sending command: {command}")
    client_socket.send((command + '\n').encode())
    try:
        ack = client_socket.recv(1024).decode().strip()
        if ack == "ACK":
            print("Acknowledgment received")
        else:
            print("No acknowledgment received")
    except socket.timeout:
        print("Acknowledgment timeout")


def yolo_worker(input_queue, output_queue):
    net_yolo = ailia.Net(MODEL_YOLOV9E_PATH, WEIGHT_YOLOV9E_PATH, env_id=args.env_id)
    while True:
        frame = input_queue.get()
        if frame is None:
            break
        try:
            preds = predict(net_yolo, frame)
            detect_object = convert_to_detector_object(preds, frame.shape[1], frame.shape[0])
            yolo_img = plot_results(detect_object, frame)
            output_queue.put(yolo_img)
        except Exception as e:
            print(f"Error during YOLO inference: {e}")
            output_queue.put(None)

def midas_worker(input_queue, output_queue, h, w):
    net_midas = ailia.Net(MODEL_v20_PATH, WEIGHT_v20_PATH, env_id=args.env_id)
    while True:
        frame = input_queue.get()
        if frame is None:
            break
        try:
            frame_midas_resized = cv2.resize(frame, (w, h))
            resized_img = normalize_image(frame_midas_resized, 'ImageNet').transpose((2, 0, 1))[np.newaxis, :, :, :]

            if resized_img.ndim == 4:
                net_midas.set_input_shape(resized_img.shape)
            
            result = net_midas.predict(resized_img)

            depth_min = result.min()
            depth_max = result.max()
            max_val = (2 ** 16) - 1
            out = max_val * (result - depth_min) / (depth_max - depth_min + np.finfo("float").eps)

            res_img = (out.transpose(1, 2, 0) / 256).astype("uint8")
            res_img = cv2.cvtColor(res_img, cv2.COLOR_GRAY2BGR)
            output_queue.put((res_img, depth_min))
        except Exception as e:
            print(f"Error during MiDaS inference: {e}")
            output_queue.put((None, None))

def recognize_from_video(client_socket):
    capture = cv2.VideoCapture("http://192.168.1.106/stream")

    h, w = (IMAGE_HEIGHT_SMALL, IMAGE_WIDTH_SMALL)

    yolo_input_queue = multiprocessing.Queue()
    yolo_output_queue = multiprocessing.Queue()
    midas_input_queue = multiprocessing.Queue()
    midas_output_queue = multiprocessing.Queue()

    yolo_process = multiprocessing.Process(target=yolo_worker, args=(yolo_input_queue, yolo_output_queue))
    midas_process = multiprocessing.Process(target=midas_worker, args=(midas_input_queue, midas_output_queue, h, w))
    
    yolo_process.start()
    midas_process.start()

    count = 0

    while True:
        ret, frame = capture.read()
        if not ret or (cv2.waitKey(1) & 0xFF == ord('q')):
            break
        
        yolo_input_queue.put(frame)
        midas_input_queue.put(frame)

        yolo_img = yolo_output_queue.get()
        depth_img, depth_min = midas_output_queue.get()

        if yolo_img is not None:
            cv2.imshow('frame', yolo_img)
        if depth_img is not None:
            cv2.imshow('depth', depth_img)
            print(f"{round(depth_min / 34.34, 2)} inches")

        if person_detected:  # Assuming person_detected is a valid function
            send_command("FORWARD -1000 -1000 -1000 -1000", client_socket)
            count = 0
        else:
            send_command("STOP 0 0 0 0", client_socket)
            count += 1
        if count >= 10:
            send_command("TURN_RIGHT 1000 1000 -1000 -1000", client_socket)
            count = 7
        if round(depth_min / 34.34, 2) < 60:
            send_command("STOP 0 0 0 0", client_socket)

    yolo_input_queue.put(None)
    midas_input_queue.put(None)

    yolo_process.join()
    midas_process.join()

    capture.release()
    cv2.destroyAllWindows()

def main():
    # model files check and download
    check_and_download_models(WEIGHT_YOLOV9E_PATH, MODEL_YOLOV9E_PATH, REMOTE_PATH_YOLO)
    # model files check and download
    check_and_download_models(WEIGHT_v21_PATH, MODEL_v21_PATH, REMOTE_PATH_MIDAS)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((esp32_ip, esp32_port))
        client_socket.settimeout(10)  # Increase the timeout for socket operations
    except socket.error as e:
        print(f"Socket error: {e}")
        return
    
    recognize_from_video(client_socket)


if __name__ == '__main__':
    main()