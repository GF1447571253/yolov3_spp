import os
import json
import time
import glob
import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from build_utils import img_utils, torch_utils, utils
from models import Darknet
from draw_box_utils import draw_box


def main():
    cfg = "cfg/my_yolov3.cfg"  # 改成生成的.cfg文件
    weights = "weights/yolov3spp-29.pt"  # 改成自己训练好的权重文件
    json_path = "./data/pascal_voc_classes.json"  # json标签文件
    img_path = "test.mp4"
    convert_image_path = "pictures/"
    assert os.path.exists(cfg), "cfg file {} dose not exist.".format(cfg)
    assert os.path.exists(weights), "weights file {} dose not exist.".format(weights)
    assert os.path.exists(json_path), "json file {} dose not exist.".format(json_path)
    assert os.path.exists(img_path), "image file {} dose not exist.".format(img_path)

    json_file = open(json_path, 'r')
    class_dict = json.load(json_file)
    category_index = {v: k for k, v in class_dict.items()}
    cap = cv2.VideoCapture(img_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    input_size = (int(frame_width), int(frame_height))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Darknet(cfg)
    model.load_state_dict(torch.load(weights, map_location=device)["model"])
    model.to(device)

    model.eval()
    with torch.no_grad():
        # init
        img = torch.zeros((1, 3, int(frame_width), int(frame_height)), device=device)
        model(img)
         # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
         #                          改成视频预测                         #
         # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        print("fps：{fps},\t frames {frames}".format(fps=fps, frames=frames))
        for i in range(int(frames) if frames < 20000 else 20000):
            ret, frame = cap.read()
            img_o = frame  # BGR
            assert img_o is not None, "Image Not Found " + img_path

            img = img_utils.letterbox(img_o, new_shape=input_size, auto=True, color=(0, 0, 0))[0]
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device).float()
            img /= 255.0  # scale (0, 255) to (0, 1)
            img = img.unsqueeze(0)  # add batch dimension

            t1 = torch_utils.time_synchronized()
            pred = model(img)[0]  # only get inference result
            t2 = torch_utils.time_synchronized()
            print(t2 - t1)

            pred = utils.non_max_suppression(pred, conf_thres=0.1, iou_thres=0.6, multi_label=True)[0]
            t3 = time.time()
            print(t3 - t2)

            if pred is None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)

                frame.save("pictures/%05d.jpg" % i)
                continue
                #print("No target detected.")
                #exit(0)

            # process detections
            pred[:, :4] = utils.scale_coords(img.shape[2:], pred[:, :4], img_o.shape).round()
            print(pred.shape)

            bboxes = pred[:, :4].detach().cpu().numpy()
            scores = pred[:, 4].detach().cpu().numpy()
            classes = pred[:, 5].detach().cpu().numpy().astype(np.int) + 1
            img_o = draw_box(img_o[:, :, ::-1], bboxes, classes, scores, category_index)
            img_o.save("pictures/%05d.jpg"%i)
        print("frame_width：{frame_width},\t frame_height {frame_height}".format(frame_width=frame_width, frame_height=frame_height))
        videoWriter = cv2.VideoWriter("Test.avi", cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(frame_width), int(frame_height)))
        for img in sorted(glob.glob(convert_image_path + "*.jpg"), key=os.path.getmtime):
            read_img = cv2.imread(img)
            # img_test = cv2.cvtColor(read_img, cv2.COLOR_BGR2RGB)
            # plt.imshow(img_test)
            # plt.show()
            videoWriter.write(read_img)
        videoWriter.release()
if __name__ == "__main__":
    main()
