import json
import os
from collections import OrderedDict

import cv2
from libs.draw import Draw
from libs.geometric import InOutCalculator, get_projection_point, get_point
from libs.geometric import get_line
from libs.person_trackers import PersonTrackers, TrackableObject
from libs.validate import validate
from openvino.inference_engine import IENetwork, IECore


class LineCrossing(object):
    results = {"In": 0, "Out": 0}

    def __init__(self):
        config_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        with open(config_file_path) as f:
            cfg = json.load(f)
        validate(cfg)
        self.running = True
        self.videosource = cfg.get("video")
        self.model_modelfile = cfg.get("pedestrian_model_weights")
        self.model_configfile = cfg.get("pedestrian_model_description")
        self.model_modelfile_reid = cfg.get("reidentification_model_weights")
        self.model_configfile_reid = cfg.get("reidentification_model_description")
        self.coords = cfg.get("coords")
        self.results = {}
        # OPENVINO VARS
        self.ov_input_blob = None
        self.out_blob = None
        self.net = None
        self.ov_n = None
        self.ov_c = None
        self.ov_h = None
        self.ov_w = None
        self.ov_input_blob_reid = None
        self.out_blob_reid = None
        self.net_reid = None
        self.ov_n_reid = None
        self.ov_c_reid = None
        self.ov_h_reid = None
        self.ov_w_reid = None
        # PROCESSOR VARS
        self.confidence_threshold = .85
        self.trackers = []
        self.max_disappeared = 90
        self.max_distance = .3
        self.trackers = None
        self.min_w = 99999
        self.max_w = 1

    def load_openvino(self):
        try:
            ie = IECore()
            net = ie.read_network(model=self.model_configfile, weights=self.model_modelfile)
            self.ov_input_blob = next(iter(net.inputs))
            self.out_blob = next(iter(net.outputs))
            self.net = ie.load_network(network=net, num_requests=2, device_name="CPU")
            # Read and pre-process input image
            self.ov_n, self.ov_c, self.ov_h, self.ov_w = net.inputs[self.ov_input_blob].shape
            del net
        except Exception as e:
            raise Exception(f"Load Openvino error:{e}")
        self.load_openvino_reid()

    def load_openvino_reid(self):
        try:
            ie = IECore()
            net = ie.read_network(model=self.model_configfile_reid, weights=self.model_modelfile_reid)
            self.ov_input_blob_reid = next(iter(net.inputs))
            self.out_blob_reid = next(iter(net.outputs))
            self.net_reid = ie.load_network(network=net, num_requests=2, device_name="CPU")
            # Read and pre-process input image
            self.ov_n_reid, self.ov_c_reid, self.ov_h_reid, self.ov_w_reid = net.inputs[self.ov_input_blob_reid].shape
            del net
        except Exception as e:
            raise Exception(f"Load Openvino reidentification error:{e}")

    def config_env(self, frame):
        h, w = frame.shape[:2]
        door_coords = ((int(self.coords[0][0] * w / 100), int(self.coords[0][1] * h / 100)),
                       (int(self.coords[1][0] * w / 100), int(self.coords[1][1] * h / 100)))
        self.door_line = InOutCalculator(door_coords)
        lp = list(self.door_line.line.coords)
        proj = get_projection_point(lp[0], lp[1], .3)
        self.door_line.max_distance = int(proj["line"].length / 2)
        self.trackers = PersonTrackers(OrderedDict(), door_coords, callback_calc)

    def get_frame(self):
        h = w = None
        try:
            cap = cv2.VideoCapture(self.videosource)
        except Exception as e:
            raise Exception(f"Video source error: {e}")

        while self.running:
            has_frame, frame = cap.read()
            if has_frame:
                if w is None or h is None:
                    h, w = frame.shape[:2]
                    print(frame.shape)
                    self.config_env(frame)
                if frame.shape[1] > 2000:
                    frame = cv2.resize(frame, (int(frame.shape[1] * .3), int(frame.shape[0] * .3)))

                elif frame.shape[1] > 1000:
                    frame = cv2.resize(frame, (int(frame.shape[1] * .8), int(frame.shape[0] * .8)))
                yield frame
            else:
                self.running = False
        return None

    def process_frame(self, frame):
        _frame = frame.copy()
        trackers = []

        frame = cv2.resize(frame, (self.ov_w, self.ov_h))
        frame = frame.transpose((2, 0, 1))
        frame = frame.reshape((self.ov_n, self.ov_c, self.ov_h, self.ov_w))

        self.net.start_async(request_id=0, inputs={self.ov_input_blob: frame})

        if self.net.requests[0].wait(-1) == 0:
            res = self.net.requests[0].outputs[self.out_blob]

            frame = _frame
            h, w = frame.shape[:2]
            out = res[0][0]
            for i, detection in enumerate(out):

                confidence = detection[2]
                if confidence > self.confidence_threshold and int(detection[1]) == 1:  # 1 => CLASS Person

                    xmin = int(detection[3] * w)
                    ymin = int(detection[4] * h)
                    xmax = int(detection[5] * w)
                    ymax = int(detection[6] * h)

                    if get_line([[xmin, ymax], [xmax, ymax]]).length < self.min_w:
                        self.min_w = get_line([[xmin, ymax], [xmax, ymax]]).length
                    elif get_line([[xmin, ymax], [xmax, ymax]]).length > self.max_w:
                        self.max_w = get_line([[xmin, ymax], [xmax, ymax]]).length

                    cX = int((xmin + xmax) / 2.0)
                    cY = int(ymax)
                    point = get_point([cX, cY])
                    if not self.door_line.contains(point):
                        continue

                    trackers.append(
                        TrackableObject((xmin, ymin, xmax, ymax), None, (cX, cY))
                    )
                    Draw.point(frame, (cX, cY), "green")

        for tracker in trackers:
            person = frame[tracker.bbox[1]:tracker.bbox[3], tracker.bbox[0]:tracker.bbox[2]]

            try:
                person = cv2.resize(person, (self.ov_w_reid, self.ov_h_reid))
            except cv2.error as e:
                print(f"CV2 RESIZE ERROR: {e}")
                continue

            person = person.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            person = person.reshape((self.ov_n_reid, self.ov_c_reid, self.ov_h_reid, self.ov_w_reid))

            self.net_reid.start_async(request_id=0, inputs={self.ov_input_blob: person})

            if self.net_reid.requests[0].wait(-1) == 0:
                res = self.net_reid.requests[0].outputs[self.out_blob_reid]
                tracker.reid = res

        self.trackers.similarity(trackers)
        # if len(self.trackers.trackers) > 0:
        door = list(self.door_line.line.coords)
        Draw.line(frame, (int(door[0][0]), int(door[0][1]), int(door[1][0]), int(door[1][1])), "yellow", 3)
        Draw.data(frame, LineCrossing.results)
        return frame

    def render(self, frame):
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        frame = cv2.resize(frame, (960, 540))
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            exit()

    def run(self):
        self.load_openvino()
        for frame in self.get_frame():
            frame = self.process_frame(frame)
            self.render(frame)


def callback_calc(line, first, last):
    line_calc = InOutCalculator(line, first)
    r = line_calc.evaluate(last)
    if r == "P":
        print(f"Result Positive({r}) direction  --> In count")
        LineCrossing.results["In"] += 1
    elif r == "N":
        print(f"Result Negative({r}) direction --> Out count")
        LineCrossing.results["Out"] += 1


if __name__ == '__main__':
    try:
        lc = LineCrossing()
        lc.run()
    except Exception as exception:
        print(exception)
