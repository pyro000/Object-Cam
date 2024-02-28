import PySimpleGUI as sg
from imutils.video import WebcamVideoStream
import imutils
import time
import numpy as np
import threading
import cv2
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as tnnf
import multiprocessing
import netifaces
import socket

# For Zip
# import shutil
# from zipfile import ZipFile

sg.theme('black')
sg.user_settings_filename(filename='settings.json', path='.')


class Key:
    def __init__(self):
        self.letter = ''
        self.number = 0

        local_ip = socket.gethostbyname(socket.gethostname())
        for nic in netifaces.interfaces():
            addrs = netifaces.ifaddresses(nic)
            try:
                if len(addrs[netifaces.AF_LINK][0]['addr']) > 0 and local_ip == addrs[netifaces.AF_INET][0]['addr']:
                    self.mac = addrs[netifaces.AF_LINK][0]['addr'].replace(':', '').zfill(12)
            except KeyError:
                pass

        self.key = self.createproductkey()

    def getkey(self):
        return self.key

    def lettertonumber(self, letter):
        self.letter = letter
        if not self.letter.isalpha():
            return self.letter
        for i, j in enumerate(range(97, 123)):
            if self.letter == chr(j):
                return i

    def numbertoletter(self, number):
        self.number = number
        if f'{self.number}'.isalpha():
            return self.number
        else:
            return chr(97 + int(self.number))

    def createproductkey(self):
        mac_c = []
        mac_c1 = []
        mac_c2 = []
        key = ''

        for i in self.mac:
            mac_c.append(i)
            mac_c1.append(self.lettertonumber(i))
            mac_c2.append(self.numbertoletter(i))

        mac_ci = mac_c[::-1]
        mac_ci1 = mac_c1[::-1]
        mac_ci2 = mac_c2[::-1]
        mac_f = [mac_c, mac_c1, mac_c2, mac_ci, mac_ci1, mac_ci2]

        key_ph = [[3, 2, 0], [1, 5, 1], [0, 7, 1], [5, 2, 1], [2, 8, 0], [3, 10, 1], [4, 4, 0], [3, 1, 0], [0, 3, 1],
                  [2, 7, 0], [3, 3, 0], [2, 6, 1], [4, 8, 1], [5, 7, 0], [0, 3, 0], [4, 9, 0], [5, 6, 0], [0, 7, 0],
                  [3, 9, 0], [1, 3, 0], [2, 7, 1], [5, 8, 1], [2, 1, 1], [1, 1, 0], [0, 11, 1], [0, 1, 1], [5, 7, 1],
                  [4, 8, 0], [2, 5, 0], [4, 0, 0], [5, 5, 1], [2, 3, 1], [5, 0, 0], [1, 10, 0], [2, 7, 0], [1, 3, 0],
                  [3, 7, 0], [3, 5, 1], [1, 5, 1], [3, 3, 1]]

        """import random
        key_ph = [
            [
                random.randrange(0, len(mac_f)),
                random.randrange(len(mac_f[-1])),
                random.randrange(0, 2),
            ]
            for _ in range(40)
        ]
        print(key_ph)"""

        for i in key_ph:
            if i[2] == 1:
                if f'{mac_f[i[0]][i[1]]}' != 'i':
                    key += f'{mac_f[i[0]][i[1]]}'.upper()
            else:
                key += f'{mac_f[i[0]][i[1]]}'

        # print(key)
        return key


class DWConvblock(nn.Module):
    # noinspection PyTypeChecker
    def __init__(self, input_channels, output_channels, size):
        super(DWConvblock, self).__init__()
        self.size = size
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.block = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, size, 1, 2, groups=output_channels, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(output_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),

            nn.Conv2d(output_channels, output_channels, size, 1, 2, groups=output_channels, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(output_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class LightFPN(nn.Module):
    # noinspection PyTypeChecker
    def __init__(self, input2_depth, input3_depth, out_depth):
        super(LightFPN, self).__init__()

        self.conv1x1_2 = nn.Sequential(nn.Conv2d(input2_depth, out_depth, 1, 1, 0, bias=False),
                                       nn.BatchNorm2d(out_depth),
                                       nn.ReLU(inplace=True)
                                       )

        self.conv1x1_3 = nn.Sequential(nn.Conv2d(input3_depth, out_depth, 1, 1, 0, bias=False),
                                       nn.BatchNorm2d(out_depth),
                                       nn.ReLU(inplace=True)
                                       )

        self.cls_head_2 = DWConvblock(input2_depth, out_depth, 5)
        self.reg_head_2 = DWConvblock(input2_depth, out_depth, 5)

        self.reg_head_3 = DWConvblock(input3_depth, out_depth, 5)
        self.cls_head_3 = DWConvblock(input3_depth, out_depth, 5)

    def forward(self, c2, c3):
        s3 = self.conv1x1_3(c3)
        cls_3 = self.cls_head_3(s3)
        obj_3 = cls_3
        reg_3 = self.reg_head_3(s3)

        p2 = tnnf.interpolate(c3, scale_factor=2)
        p2 = torch.cat((p2, c2), 1)
        s2 = self.conv1x1_2(p2)
        cls_2 = self.cls_head_2(s2)
        obj_2 = cls_2
        reg_2 = self.reg_head_2(s2)

        return cls_2, obj_2, reg_2, cls_3, obj_3, reg_3


class ShuffleV2Block(nn.Module):
    # noinspection PyTypeChecker
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = self.channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    @staticmethod
    def channel_shuffle(x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]


class ShuffleNetV2(nn.Module):
    # noinspection PyTypeChecker
    def __init__(self, stage_out_channels, load_param):
        super(ShuffleNetV2, self).__init__()

        self.stage_repeats = [4, 8, 4]
        self.stage_out_channels = stage_out_channels

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ["stage2", "stage3", "stage4"]
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            stage_seq = []
            for i in range(numrepeat):
                if i == 0:
                    stage_seq.append(ShuffleV2Block(input_channel, output_channel,
                                                    mid_channels=output_channel // 2, ksize=3, stride=2))
                else:
                    stage_seq.append(ShuffleV2Block(input_channel // 2, output_channel,
                                                    mid_channels=output_channel // 2, ksize=3, stride=1))
                input_channel = output_channel
            setattr(self, stage_names[idxstage], nn.Sequential(*stage_seq))

        if load_param is False:
            self._initialize_weights()
        else:
            print("load param...")

    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        c1 = self.stage2(x)
        c2 = self.stage3(c1)
        c3 = self.stage4(c2)

        return c2, c3

    def _initialize_weights(self):
        print("initialize_weights...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(torch.load("backbone.pth", map_location=device), strict=True)


class Detector(nn.Module):
    # noinspection PyTypeChecker
    def __init__(self, classes, anchor_num, load_param, export_onnx=False):
        super(Detector, self).__init__()
        out_depth = 72
        stage_out_channels = [-1, 24, 48, 96, 192]

        self.export_onnx = export_onnx
        self.backbone = ShuffleNetV2(stage_out_channels, load_param)
        self.fpn = LightFPN(stage_out_channels[-2] + stage_out_channels[-1], stage_out_channels[-1], out_depth)

        self.output_reg_layers = nn.Conv2d(out_depth, 4 * anchor_num, 1, 1, 0, bias=True)
        self.output_obj_layers = nn.Conv2d(out_depth, anchor_num, 1, 1, 0, bias=True)
        self.output_cls_layers = nn.Conv2d(out_depth, classes, 1, 1, 0, bias=True)

    def forward(self, x):
        c2, c3 = self.backbone(x)
        cls_2, obj_2, reg_2, cls_3, obj_3, reg_3 = self.fpn(c2, c3)

        out_reg_2 = self.output_reg_layers(reg_2)
        out_obj_2 = self.output_obj_layers(obj_2)
        out_cls_2 = self.output_cls_layers(cls_2)

        out_reg_3 = self.output_reg_layers(reg_3)
        out_obj_3 = self.output_obj_layers(obj_3)
        out_cls_3 = self.output_cls_layers(cls_3)

        if not self.export_onnx:
            return out_reg_2, out_obj_2, out_cls_2, out_reg_3, out_obj_3, out_cls_3
        out_reg_2 = out_reg_2.sigmoid()
        out_obj_2 = out_obj_2.sigmoid()
        out_cls_2 = tnnf.softmax(out_cls_2, dim=1)

        out_reg_3 = out_reg_3.sigmoid()
        out_obj_3 = out_obj_3.sigmoid()
        out_cls_3 = tnnf.softmax(out_cls_3, dim=1)

        print("export onnx ...")
        return torch.cat((out_reg_2, out_obj_2, out_cls_2), 1).permute(0, 2, 3, 1), torch.cat((out_reg_3,
                                                                                               out_obj_3, out_cls_3),
                                                                                              1).permute(0, 2, 3, 1)


def load_datafile(data_path):
    # 需要配置的超参数
    cfg = {"model_name": None,

           "epochs": None,
           "steps": None,
           "batch_size": None,
           "subdivisions": None,
           "learning_rate": None,

           "pre_weights": None,
           "classes": None,
           "width": None,
           "height": None,
           "anchor_num": None,
           "anchors": None,

           "val": None,
           "train": None,
           "names": None
           }

    assert os.path.exists(data_path), ".data not found."

    # 指定配置项的类型
    list_type_key = ["anchors", "steps"]
    str_type_key = ["model_name", "val", "train", "names", "pre_weights"]
    int_type_key = ["epochs", "batch_size", "classes", "width",
                    "height", "anchor_num", "subdivisions"]
    float_type_key = ["learning_rate"]

    # 加载配置文件
    with open(data_path, 'r') as f:
        for line in f.readlines():
            if line == '\n' or line[0] == "[":
                continue
            data = line.strip().split("=")
            # 配置项类型转换
            if data[0] in cfg:
                if data[0] in int_type_key:
                    cfg[data[0]] = int(data[1])
                elif data[0] in str_type_key:
                    cfg[data[0]] = data[1]
                elif data[0] in float_type_key:
                    cfg[data[0]] = float(data[1])
                elif data[0] in list_type_key:
                    cfg[data[0]] = [float(x) for x in data[1].split(",")]
                else:
                    print("配置文件有错误的配置项")
            else:
                print("%s配置文件里有无效配置项:%s" % (data_path, data))
    return cfg


def make_grid(h, w, cfg, device):
    hv, wv = torch.meshgrid([torch.arange(h), torch.arange(w)])
    return torch.stack((wv, hv), 2).repeat(1, 1, 3).reshape(h, w, cfg["anchor_num"], -1).to(device)


# noinspection PyArgumentList
def handle_preds(preds, cfg, device):
    # 加载anchor配置
    anchors = np.array(cfg["anchors"])
    anchors = torch.from_numpy(anchors.reshape(len(preds) // 3, cfg["anchor_num"], 2)).to(device)

    output_bboxes = []
    # layer_index = [0, 0, 0, 1, 1, 1] # NOT USED

    for i in range(len(preds) // 3):
        bacth_bboxes = []
        reg_preds = preds[i * 3]
        obj_preds = preds[(i * 3) + 1]
        cls_preds = preds[(i * 3) + 2]

        for r, o, c in zip(reg_preds, obj_preds, cls_preds):
            r = r.permute(1, 2, 0)
            r = r.reshape(r.shape[0], r.shape[1], cfg["anchor_num"], -1)

            o = o.permute(1, 2, 0)
            o = o.reshape(o.shape[0], o.shape[1], cfg["anchor_num"], -1)

            c = c.permute(1, 2, 0)
            c = c.reshape(c.shape[0], c.shape[1], 1, c.shape[2])
            c = c.repeat(1, 1, 3, 1)

            anchor_boxes = torch.zeros(r.shape[0], r.shape[1], r.shape[2], r.shape[3] + c.shape[3] + 1)

            # 计算anchor box的cx, cy
            grid = make_grid(r.shape[0], r.shape[1], cfg, device)
            stride = cfg["height"] / r.shape[0]
            anchor_boxes[:, :, :, :2] = ((r[:, :, :, :2].sigmoid() * 2. - 0.5) + grid) * stride

            # 计算anchor box的w, h
            anchors_cfg = anchors[i]
            anchor_boxes[:, :, :, 2:4] = (r[:, :, :, 2:4].sigmoid() * 2) ** 2 * anchors_cfg  # wh

            # 计算obj分数
            anchor_boxes[:, :, :, 4] = o[:, :, :, 0].sigmoid()

            # 计算cls分数
            anchor_boxes[:, :, :, 5:] = tnnf.softmax(c[:, :, :, :], dim=3)

            # torch tensor 转为 numpy array
            anchor_boxes = anchor_boxes.cpu().detach().numpy()
            bacth_bboxes.append(anchor_boxes)

            # n, anchor num, h, w, box => n, (anchor num*h*w), box
        bacth_bboxes = torch.from_numpy(np.array(bacth_bboxes))
        bacth_bboxes = bacth_bboxes.view(bacth_bboxes.shape[0], -1, bacth_bboxes.shape[-1])

        output_bboxes.append(bacth_bboxes)

        # merge
    return torch.cat(output_bboxes, 1)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction, conf_thres=0.3, iou_thres=0.45, classes=None):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    # nc = prediction.shape[2] - 5  # number of classes NOT USED

    # Settings
    # (pixels) minimum and maximum box width and height
    max_wh = 4096
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 1.0  # seconds to quit after
    # multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img) NOT USED

    t = time.time()
    output = [torch.zeros((0, 6), device="cpu")] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[x[..., 4] > conf_thres]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i].detach().cpu()

        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


class Streaming(threading.Thread):
    def __init__(self, window, val, ip, device, model, cfg, id_c):
        threading.Thread.__init__(self, daemon=True)
        self.playing = True
        self.window = window
        self.val = val
        self.ip = ip
        self.stopped = False
        self.vs = self.o = self.output = self.t_fps = None
        self.checker = CheckCam(self.ip)
        self.device = device
        self.model = model
        self.cfg = cfg
        self.t = time.time()
        self.frames = 1
        self.id = id_c

    def run(self):
        motivo = 'MNL/UNKN'
        t_timeout = time.time()
        results = None

        label_names = []
        with open(self.cfg["names"], 'r') as f:
            for line in f.readlines():
                label_names.append(line.strip())

        self.checker.start()
        change_b_img('Connecting', self.window, self.id, d=True)

        while not self.stopped:
            if not self.playing:
                time.sleep(0.005)
                continue

            if self.checker:
                if self.checker.q.empty() and time.time() - t_timeout >= 6:
                    motivo = 'TIMEOUT'
                    self.checker.terminate()
                    self.checker = None
                    self.stopped = True
                else:
                    try:
                        self.checker.q.get_nowait()
                        self.vs = WebcamVideoStream(src=self.ip).start()
                        s_fps = self.vs.stream.get(cv2.CAP_PROP_FPS)
                        self.t_fps = 1 / s_fps
                        self.o = Blob(self.device, self.model, self.cfg)
                        self.o.start()
                        self.window['Connect'].update('Disconnect', disabled=False)
                        for i in range(3):
                            self.window[f'-RA{i + 1}-'].update(disabled=False)
                        self.checker.terminate()
                        self.checker = None
                    except (Exception,):
                        pass

            if self.vs and self.vs.stream.isOpened():
                frame = self.vs.read()
                if frame is None:
                    motivo = 'No Camera'
                    self.stop()
                    continue
                # check to see if the frame should be displayed to our screen
                try:
                    self.o.q_in.get_nowait()
                    results = self.o.q_out.get_nowait() if self.val['o_bool'] else None
                except (Exception,):
                    pass

                try:
                    self.o.q_in.put({'frame': frame, 'playing': self.val['o_bool']})
                except ValueError:
                    pass

                if results is not None:
                    h, w, _ = frame.shape
                    scale_h, scale_w = h / self.cfg["height"], w / self.cfg["width"]

                    for box in results[0]:
                        box = box.tolist()

                        obj_score = box[4]
                        category = label_names[int(box[5])]

                        x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
                        x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        cv2.putText(frame, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)

                fr = imutils.resize(frame, width=425)
                fr_a = imutils.resize(frame, width=850)
                self.output = [fr, fr_a]

        change_b_img(f'Disconnected: {motivo}', self.window, self.id)
        self.window['Connect'].update('Connect')

    def stop(self):
        if self.vs:
            self.vs.stop()
        if self.o:
            self.o.stop()
        self.stopped = True

    def pause(self):
        self.playing = not self.playing


class CheckCam(multiprocessing.Process):
    def __init__(self, ip_a):
        multiprocessing.Process.__init__(self, daemon=True)
        self.ip = ip_a
        self.q = multiprocessing.Queue()

    def run(self):
        cap = cv2.VideoCapture(self.ip)
        ret, _ = cap.read()
        self.q.put(ret)
        self.q.close()
        self.q.join_thread()


class Blob(multiprocessing.Process):
    def __init__(self, device, model, cfg):
        multiprocessing.Process.__init__(self, daemon=True)
        self.model = model
        self.cfg = cfg
        self.device = device
        self.q_in = multiprocessing.Queue()
        self.q_out = multiprocessing.Queue()

        self.in_result = None
        self.playing = True
        self.stopped = False
        self.new = False

    def run(self):
        tx = []

        while not self.stopped:
            try:
                self.in_result = self.q_in.get_nowait()
            except (Exception,):
                time.sleep(0.01)
                continue

            if not self.in_result['playing']:
                time.sleep(0.01)
                continue

            res_img = cv2.resize(self.in_result['frame'], (self.cfg["width"], self.cfg["height"]),
                                 interpolation=cv2.INTER_LINEAR)
            img = res_img.reshape(1, self.cfg["height"], self.cfg["width"], 3)
            img = torch.from_numpy(img.transpose(0, 3, 1, 2))
            img = img.to(self.device).float() / 255.0

            t = time.time()
            preds = self.model(img)
            tf = time.time() - t
            tx.append(tf)
            print(f'Time: {tf:0.4f}, Avg: {sum(tx) / len(tx):0.4f}')

            output = handle_preds(preds, self.cfg, self.device)
            output_boxes = non_max_suppression(output, conf_thres=0.98, iou_thres=0.4)

            try:
                self.q_out.get_nowait()
            except (Exception,):
                pass
            self.q_out.put(output_boxes)

    def stop(self):
        self.stopped = True
        self.q_in.close()
        self.q_in.join_thread()
        self.q_out.close()
        self.q_out.join_thread()
        self.terminate()

    def pause(self):
        self.playing = not self.playing


def change_b_img(text, window, id_c, d=False):
    img = np.zeros((238, 425, 3), dtype="uint8")
    img[:] = (66, 66, 66)
    img_b = np.zeros((475, 850, 3), dtype="uint8")
    img_b[:] = (66, 66, 66)
    cv2.putText(img, f'{id_c}: {text}', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (218, 218, 218), 1, cv2.LINE_AA)
    cv2.putText(img_b, f'{id_c}: {text}', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (218, 218, 218), 1, cv2.LINE_AA)
    window[f'cam{id_c}'].update(data=cv2.imencode('.png', img)[1].tobytes())
    window[f'cam{id_c}s'].update(data=cv2.imencode('.png', img_b)[1].tobytes())
    window['Connect'].update(disabled=d)
    for i in range(3):
        window[f'-RA{i + 1}-'].update(disabled=d)


def get_sel_radio(val, cams):
    for i in range(cams):
        if val[f'-RA{i + 1}-'] is True:
            return i


def set_key():
    layout = [[sg.Text('Clave de Activación:')], [sg.Input(sg.user_settings_get_entry('key', ''), key='KEY')],
              [sg.Button('Guardar')]]

    window = sg.Window('Clave Activación', layout, element_justification='c', finalize=True)
    window.make_modal()

    while True:
        ev, val = window.read(timeout=100)
        if ev == sg.WINDOW_CLOSED:
            window.close()
            break
        elif ev == 'Guardar':
            sg.user_settings_set_entry('key', val['KEY'])
            window.close()
            break


# noinspection PyUnresolvedReferences,PyTypeChecker
def main():
    """with ZipFile('ia000.zip', 'r') as zipObj:
        zipObj.extractall()
    model = torch.hub.load('ia000', 'custom', path='ia000/ia.pt', source='local')
    shutil.rmtree('ia000/')"""

    multiprocessing.freeze_support()

    cfg = load_datafile('pieza.data')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    model.load_state_dict(torch.load('Piezas.pth', map_location=device))
    model.eval()
    k = Key()

    img = np.zeros((238, 425, 3), dtype="uint8")
    img[:] = (66, 66, 66)
    img = [cv2.putText(img.copy(), f'{i + 1}: Disconnected', (5, 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (218, 218, 218), 1, cv2.LINE_AA) for i in range(3)]

    img_s = np.zeros((475, 850, 3), dtype="uint8")
    img_s[:] = (66, 66, 66)
    img_s = [cv2.putText(img_s.copy(), f'{i + 1}: Disconnected', (5, 15),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (218, 218, 218), 1, cv2.LINE_AA) for i in range(3)]

    menu_def = ['&Archivo', ['&Exportar', ['&Excel', '&PDF']]], \
               ['&Opciones', ['&Clave de Producto']]

    tabcams = [[sg.Tab('All',
                       [[sg.Image(data=cv2.imencode('.png', img[i])[1].tobytes(), key=f'cam{i + 1}') for i in range(2)],
                        [sg.Image(data=cv2.imencode('.png', img[2])[1].tobytes(), key='cam3')]
                        ], element_justification='c')
                ]]
    for i in range(len(img)):
        tabcams[0].append(sg.Tab(f'Camera {i + 1}',
                                 [[sg.Image(data=cv2.imencode('.png', img_s[i])[1].tobytes(), key=f'cam{i + 1}s')]],
                                 element_justification='c'))

    cams = sg.user_settings_get_entry('n_cams', 3)
    saved = sg.user_settings_get_entry('saved', [['', '', '', '', ''] for _ in range(cams)])
    streams = [None for _ in range(cams)]
    tab1 = [
        [sg.Frame('Options', layout=[
            [sg.Radio(f'Camera {i + 1}', 1, key=f'-RA{i + 1}-', enable_events=True) for i in range(3)],
            [
                sg.Text('IP:'),
                sg.Input(saved[0][0], key='ip', size=(15, 1)),
                sg.Text('Port:'),
                sg.Input(saved[0][1], key='port', size=(5, 1)),
                sg.Text('Redirect:'),
                sg.Input(saved[0][2], key='red'),
                sg.Text('User: '),
                sg.Input(saved[0][3], key='user', size=(20, 1)),
                sg.Text('Password:'),
                sg.Input(saved[0][4], key='pass', size=(20, 1)),
            ],
            [sg.Button('Connect'), sg.Checkbox('Object Detection', key='o_bool')]])],
        [sg.TabGroup(tabcams, enable_events=True, key='tab_cams')]]

    tab2 = [[sg.Table(values=[], headings=['1', '2', '3'], max_col_width=55, auto_size_columns=False,
                      col_widths=list(map(lambda list_i: len(list_i) + 2, ['1', '2', '3'])), justification='center',
                      key='-TABLE-', num_rows=20)]]

    layout = [[sg.Menu(menu_def, pad=(10, 10))],
              [sg.TabGroup([[sg.Tab('Cameras', tab1, element_justification='c'),
                             sg.Tab('Data', tab2)]])]]

    window = sg.Window('Title', layout, size=(1280, 720), resizable=True, element_justification='c', finalize=True)
    window['-RA1-'].update(value=True)

    while True:
        ev, val = window.read(timeout=5)

        if ev == sg.WINDOW_CLOSED:
            break

        for stream in streams:
            if stream is not None and not stream.stopped:
                stream.frames += 1

        c = get_sel_radio(val, cams)

        if ev == 'Connect' and (not streams[c] or streams[c].stopped is True) and \
                k.key == sg.user_settings_get_entry('key', ''):
            if val['ip'] and val['port'] and val['red']:
                ip = f'http://{val["user"]}:{val["pass"]}@{val["ip"]}:{val["port"]}/{val["red"]}' \
                    if val['user'] and val['pass'] else f'http://{val["ip"]}:{val["port"]}/{val["red"]}'

                saved[c] = [val['ip'], val['port'], val['red'], val['user'], val['pass']]
                sg.user_settings_set_entry('saved', saved)
                stream = Streaming(window, val, ip, device, model, cfg, c + 1)
                stream.start()
                streams[c] = stream

        elif ev == 'Connect' and streams[c] and not streams[c].stopped:
            streams[c].stop()
            streams[c] = None

        elif ev == 'Connect':
            sg.popup_error('[ERROR]', 'Clave de Producto Inválida')

        elif ev in [f'-RA{i + 1}-' for i in range(3)]:
            num = int(ev.replace('-', '').replace('RA', '')) - 1

            if streams[num] and not streams[num].stopped:
                window['Connect'].update('Disconnect')
            else:
                window['Connect'].update('Connect')

            window['ip'].update(saved[num][0])
            window['port'].update(saved[num][1])
            window['red'].update(saved[num][2])
            window['user'].update(saved[num][3])
            window['pass'].update(saved[num][4])

        elif ev == 'Clave de Producto':
            set_key()

        if val['tab_cams'] == 'All':
            for i, stream in enumerate(streams):
                if stream and not stream.stopped and stream.output is not None \
                        and time.time() - stream.t >= stream.t_fps:
                    stream.t = time.time()
                    stream.val = val
                    f = stream.output
                    cv2.putText(f[0], f'fps: {int(100 / stream.frames)}, {stream.frames / stream.t_fps}', (20, 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)
                    window[f'cam{i + 1}'].update(data=cv2.imencode('.png', f[0])[1].tobytes())
                    stream.frames = 0
        else:
            num_t = int(val['tab_cams'].replace('Camera ', '')) - 1
            if streams[num_t] and not streams[num_t].stopped and streams[num_t].output is not None \
                    and time.time() - streams[num_t].t >= streams[num_t].t_fps:
                streams[num_t].t = time.time()
                streams[num_t].val = val
                f = streams[num_t].output
                cv2.putText(f[1], f'fps: {int(100 / streams[num_t].frames)}, '
                                  f'{streams[num_t].frames / streams[num_t].t_fps}', (20, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)
                window[f'cam{num_t + 1}s'].update(data=cv2.imencode('.png', f[1])[1].tobytes())
                streams[num_t].frames = 0

    for stream in streams:
        if stream is not None:
            stream.stop()
    window.close()


if __name__ == '__main__':
    main()
