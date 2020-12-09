import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import pyrealsense2 as rs
import numpy as np
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def midpoint(x1, y1, x2, y2):
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2
    return mx, my


def opencam(exposure_value):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    profile = pipeline.start(config)

    s = profile.get_device().query_sensors()[1]
    s.set_option(rs.option.exposure, exposure_value)

    # Skip 5 first frames to give the Auto-Exposure time to adjust
    for x in range(5):
        pipeline.wait_for_frames()

    # Store next frameset for later processing:
    frameset = pipeline.wait_for_frames()
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()

    # Cleanup:
    pipeline.stop()
    print("Frames Captured")
    color = np.asanyarray(color_frame.get_data())
    plt.rcParams["axes.grid"] = False
    plt.imshow(color)

    colorizer = rs.colorizer()
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    # plt.imshow(colorized_depth)

    # Create alignment primitive with color as its target stream:
    align = rs.align(rs.stream.color)
    frameset = align.process(frameset)

    # Update color and depth frames:
    aligned_depth_frame = frameset.get_depth_frame()
    colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    return color, colorized_depth, aligned_depth_frame, depth_scale


def depth_detector(color, colorized_depth, aligned_depth_frame, depth_scale, xmin, ymin, xmax, ymax):
    # Show the two frames together:
    images = np.hstack((color, colorized_depth))
    # plt.imshow(images)
    # plt.show()
    height, width = color.shape[:2]

    xmin_depth = int(xmin)
    ymin_depth = int(ymin)
    xmax_depth = int(xmax)
    ymax_depth = int(ymax)
    xmin_depth, ymin_depth, xmax_depth, ymax_depth
    cv2.rectangle(colorized_depth, (xmin_depth, ymin_depth),
                  (xmax_depth, ymax_depth), (255, 255, 255), 2)
    plt.imshow(colorized_depth)
    plt.show()

    depth = np.asanyarray(aligned_depth_frame.get_data())
    getdist = np.asanyarray(aligned_depth_frame.get_data())
    # Crop depth data:
    mx, my = midpoint(ymin_depth, xmin_depth, ymax_depth, xmax_depth)
    depth = depth[ymin_depth:ymax_depth, xmin_depth:xmax_depth].astype(float)
    getdist = getdist[int(mx - 5):int(mx + 5), int(my - 5):int(my + 5)].astype(float)
    # Get data scale from the device and convert to meters

    depth = depth * depth_scale
    getdist = getdist * depth_scale
    dist, _, _, _ = cv2.mean(getdist)
    img = np.zeros(shape=[480, 640, 3], dtype=np.uint8)
    for y in range(len(depth)):
        for x in range(len(depth[y])):
            if depth[y][x] < dist + (dist / 10):
                cv2.circle(img, (x + xmin_depth, y + ymin_depth), 0, (255, 255, 255), -1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=2)
    plt.imshow(img)
    plt.show()
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    contour = contours[0]

    rotrect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rotrect)
    box = np.int0(box)
    cv2.drawContours(color, [box], 0, (0, 0, 255), 2)

    print("distance:", dist)
    plt.imshow(depth)
    plt.show()
    return box, depth, dist  # box coordinate ,depth = depth in bounding box, dis to center point


def detect(save_img=True):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None

    ### 3D ###
    color, colorized_depth, aligned_depth_frame, depth_scale = opencam(1200)
    cv2.imwrite('./data/3D/Image_1.jpg', color)

    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = Path(path[i]), '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = Path(path), '', im0s

            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            print(det)
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()


                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        box, depth, dist = depth_detector(color, colorized_depth, aligned_depth_frame, depth_scale,
                                                          int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                        print(dist)
                        print(names[int(cls)])
                        print(float(conf))
                        print(c1, c2)
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow("asd", im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    # cv2.imwrite(save_path, im0)
                    pass
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % save_dir)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train_cig/web/best.pt', help='model.pt path(s)') #yolov5s.pt
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
