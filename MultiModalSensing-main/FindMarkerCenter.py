# import os
#
# import setting_GelForce
# import numpy as np
# from lib import find_marker
# import cv2
#
#
# class Processing:
#     def __init__(self, videoinput):
#         total_frame = videoinput.get(cv2.CAP_PROP_FRAME_COUNT)
#         for i in range(int(total_frame)):
#             success, image = videoinput.read()
#             if not i % 24:
#                 image = image[:, 1334:]  # 根据具体视频分辨率修改
#                 image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#                 _, image = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY)  # 127,255 起始阈值，最大阈值，
#                 thresh_up = np.ones((80, 586)) * 255
#                 image[:80, :] = thresh_up  # image(1080,586)
#                 thresh_right = np.ones((1080, 96)) * 255  # 96=586-490
#                 image[:, 490:] = thresh_right  # image(1080,586)
#                 thresh_left = np.ones((1080, 50)) * 255
#                 image[:, :50] = thresh_left  # image(1080,586)
#
#                 k1 = np.ones((12, 12), np.uint8)
#                 # cv2.imshow('thresh_up', image)
#                 # cv2.waitKey()
#                 # cv2.destroyAllWindows()
#                 binary_img = cv2.morphologyEx(image, cv2.MORPH_OPEN, k1)
#                 tmp_img = binary_img[200:, :]
#                 k2 = np.ones((5, 5), np.uint8)
#                 tmp_img = cv2.morphologyEx(tmp_img, cv2.MORPH_CLOSE, k2)
#                 binary_img[200:, :] = tmp_img
#
#                 keypoints_new = self.find_dots(binary_img)
#
#                 draw_image = cv2.drawKeypoints(binary_img, keypoints_new, np.array([]), (0, 0, 255),
#                                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#                 # Show blobs
#                 # cv2.imshow("key_points", draw_image)
#                 # cv2.waitKey(0)
#
#                 keypoints_new = self.sortkeypoints(keypoints_new, draw_image)
#                 length = len(keypoints_new)
#                 if length != 40:
#                     print(length)
#                     print(i)
#                     cv2.namedWindow("key_points", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
#                     # cv.namedWindow("input", 0)
#                     cv2.resizeWindow("key_points", 586, 800)
#                     cv2.imshow("key_points", draw_image)
#                     cv2.waitKey(0)
#
#                 setting_GelForce.init()
#                 self.m = find_marker.Matching(
#                     N_=setting_GelForce.N_,
#                     M_=setting_GelForce.M_,
#                     fps_=setting_GelForce.fps_,
#                     x0_=setting_GelForce.x0_,
#                     y0_=setting_GelForce.y0_,
#                     dx_=setting_GelForce.dx_,
#                     dy_=setting_GelForce.dy_)
#
#                 self.m.init(keypoints_new)
#                 self.m.run()
#                 flow = self.m.get_flow()
#                 # flow = np.array(flow)
#                 # print(flow.shape)
#                 image = np.array(binary_img.copy()).astype(np.float32)
#                 raw_image = self.\
#                     draw_flow(image, flow)  # flow[0]>>x_ref; flow[1]>>y_ref; flow[2]>>x_current; flow[3]>>y_current
#                 # cv2.imwrite('asdasf.png', raw_image)
#                 # cv2.imshow('thresh_up', raw_image)
#                 # cv2.waitKey()
#                 # cv2.destroyAllWindows()
#
#         # Put the function of transfering to binary image here
#         # Don't forget to get rid of the noises using morphological operations
#         # binary_img = ....
#
#     def draw_flow(self, im_cal, flow):
#         # This function is for visualization of drawing centre motion of each marker
#         # Ox is the X coordinate of reference frame, Oy is the Y coordinate of reference frame
#         # Cx is the X coordinate of current frame, Cy is the Y coordinate of current frame
#         # The shape of each is of Row by Column in the form of numpy array
#         Ox, Oy, Cx, Cy, Occupied = flow
#         K = 1
#         # np.save(f"{self.datapath}/Ox.npy", Ox)
#         # np.save(f"{self.datapath}/Oy.npy", Oy)
#         # np.save(f"{self.datapath}/Cx.npy", Cx)
#         # np.save(f"{self.datapath}/Cy.npy", Cy)
#         # np.save(f"{self.datapath}/Occupied.npy", Occupied)
#
#         for i in range(len(Ox)):
#             for j in range(len(Ox[i])):
#                 pt1 = (int(Ox[i][j]), int(Oy[i][j]))
#                 pt2 = (int(Cx[i][j] + K * (Cx[i][j] - Ox[i][j])), int(Cy[i][j] + K * (Cy[i][j] - Oy[i][j])))
#                 color = (255, 255, 0)
#
#                 if Occupied[i][j] <= -1:
#                     color = (127, 127, 255)
#                 # mask2 = cv2.circle(mask2, (x[i], y[i]), 2, tuple(color), thickness=-1)
#                 # mask2 = cv2.circle(mask2, pt2, 4, tuple(color), thickness=-1)
#                 img = cv2.arrowedLine(im_cal, pt1, pt2, color, 2, tipLength=0.2)
#         return img.astype(np.uint8)
#
#     def find_dots(self, binary_image):
#         # down_image = cv2.resize(binary_image, None, fx=2, fy=2)
#         params = cv2.SimpleBlobDetector_Params()
#         # Change thresholds
#         params.thresholdStep = 5  # 5
#         params.minThreshold = 90  # 1
#         params.maxThreshold = 255  # 120
#         params.minDistBetweenBlobs = 50  # 9
#         params.filterByArea = True
#         params.minArea = 210  # 9
#         params.maxArea = 2000
#         params.filterByCircularity = True
#         params.minCircularity = 0.75
#         params.maxCircularity = 1
#         params.filterByConvexity = True
#         params.minConvexity = 0.65  # 斑点的最小凸度
#         params.maxConvexity = 1
#         params.filterByInertia = True
#         params.minInertiaRatio = 0.4
#         # params.minInertiaRatio = 0.1  # 0.5
#         detector = cv2.SimpleBlobDetector_create(params)
#         keypoints = detector.detect(binary_image.astype(np.uint8))
#
#         return keypoints
#
#     def sortkeypoints(self, keypoints, img):
#         x, y, xy = [], [], []
#         # print(f"keypoint size is {len(keypoints)}")
#         for i in range(len(keypoints)):
#             x.append(keypoints[i].pt[0])
#             y.append(keypoints[i].pt[1])
#             # cv2.putText(img, str(i), (int(keypoints[i].pt[0]), int(keypoints[i].pt[1])), cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 4)
#             xy.append((int(keypoints[i].pt[0]), int(keypoints[i].pt[1])))
#         # xy = sorted(xy)
#         # xy = sorted(xy, key=lambda x : x[1])
#         xy = sorted(xy, key=lambda x: [x[0], x[1]])
#         for j in range(len(xy)):
#             cv2.putText(img, str(j), (int(xy[j][0]), int(xy[j][1])), cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 4)
#             cv2.namedWindow("asd", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
#             cv2.resizeWindow("asd", 586, 800)
#             cv2.imshow('asd', img)
#             cv2.waitKey()
#             cv2.destroyAllWindows()
#
#         # xy_array = []
#         # temp = []
#         # for i in range(len(xy)):
#         #     xy_array.append(i)
#         # print("xy_array is", xy)
#         # print("len of xy_array is", len(xy_array))
#         return xy
#
#
# if __name__ == "__main__":
#     print("[INFO] warming up webcam...")
#     video_folder = os.path.join('E:\Orignal_F\PyCharm\pythonProject3', 'train_tactile')
#     video_paths = [os.path.join(video_folder, path) for path in sorted(os.listdir(video_folder))]
#     center = []  # center维度（视频数，每个视频帧数，40个marker）
#     for i, path in enumerate(video_paths):
#         video_name = os.path.join(video_folder, '{:03d}.mp4'.format(i + 1))  # modify the video name here
#         print(video_name)
#         capture = cv2.VideoCapture(video_name)
#         f = Processing(capture)
#
#     # vs = cv2.VideoCapture(1)
#     # vs = cv2.VideoCapture(r'E:\Orignal_F\PyCharm\pythonProject3\train_tactile\015.mp4')
#     # vs.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 640
#     # vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 480
#     # vs.set(cv2.CAP_PROP_FPS, 30)
#     # vs.set(cv2.CAP_PROP_BUFFERSIZE, 3)
#     # time.sleep(2.0)
#
#     # pba = Processing(vs)

import setting_GelForce
import numpy as np
from lib import find_marker
import cv2
import os


class Processing:
    def __init__(self, videoinput):
        # _, self.frame = self.videoinput.read()
        self.frame = videoinput.get(cv2.CAP_PROP_FRAME_COUNT)  # 帧数
        # Put the function of transfering to binary image here
        # Don't forget to get rid of the noises using morphological operations
        # binary_img = ....
        self.result = []
        for i in range(int(self.frame)):
            _, image = capture.read()
            image = image[:, 1334:]  # 根据具体视频分辨率修改 只保留tactile部分
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, image = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY)  # 127,255 起始阈值，最大阈值，
            thresh_up = np.ones((130, 586)) * 255
            image[:130, :] = thresh_up  # image(1080,586)
            thresh_right = np.ones((1080, 86)) * 255  # 86=586-500
            image[:, 500:] = thresh_right  # image(1080,586)
            thresh_left = np.ones((1080, 50)) * 255
            image[:, :50] = thresh_left  # image(1080,586)
            binary_img = image
            keypoints_new = self.find_dots(binary_img)
            draw_image = cv2.drawKeypoints(binary_img, keypoints_new, np.array([]), (0, 0, 255),
                                                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            keypoints_new = self.sortkeypoints(keypoints_new, draw_image)
            self.result.append(keypoints_new)

    def find_dots(self, binary_image):
        # down_image = cv2.resize(binary_image, None, fx=2, fy=2)
        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.thresholdStep = 5  # 5
        params.minThreshold = 90  # 1
        params.maxThreshold = 255  # 120
        params.minDistBetweenBlobs = 8  # 9
        params.filterByArea = True  # 斑点大小控制
        params.minArea = 300
        params.maxArea = 2000
        params.filterByCircularity = False  # 凹形控制
        params.filterByConvexity = False  # 凸形控制
        params.filterByInertia = False  # 圆形控制
        params.minInertiaRatio = 0.1  # 0.5
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(binary_image)
        draw_image = cv2.drawKeypoints(binary_image, keypoints, np.array([]), (0, 0, 255),
                                       cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        length = len(keypoints)
        if length != 40:
            print(f"keypoint size is {len(keypoints)}")
            print(i)
            cv2.namedWindow("key_points", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            # cv.namedWindow("input", 0)
            cv2.resizeWindow("key_points", 586, 800)
            cv2.imshow("key_points", draw_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # Show blobs
        # cv2.namedWindow('key_points', cv2.WINDOW_NORMAL)
        # cv2.imshow("key_points", draw_image)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        return keypoints

    def sortkeypoints(self, keypoints, img):
        x, y, xy = [], [], []
        # print(f"keypoint size is {len(keypoints)}")
        for i in range(len(keypoints)):
            x.append(keypoints[i].pt[0])
            y.append(keypoints[i].pt[1])
            xy.append((int(keypoints[i].pt[0]), int(keypoints[i].pt[1])))
        # xy = sorted(xy)
        # xy = sorted(xy, key=lambda x : x[1])
        xy = sorted(xy, key=lambda x: [x[1], x[0]])
        xy_array = []
        for j in range(len(xy)):
            cv2.putText(img, str(j), (int(xy[j][0]), int(xy[j][1])), cv2.FONT_HERSHEY_COMPLEX, 1.0, (100, 200, 200), 4)
            cv2.namedWindow("asd", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow("asd", 586, 800)
            cv2.imshow('asd', img)
            cv2.waitKey()
            cv2.destroyAllWindows()
            xy_array.append(j)
        # print("xy_array is", xy)
        # print("len of xy_array is", len(xy_array))
        return xy


if __name__ == "__main__":
    video_folder = os.path.join('E:\Orignal_F\PyCharm\pythonProject3', 'test_tactile')
    video_paths = [os.path.join(video_folder, path) for path in sorted(os.listdir(video_folder))]
    # center = []  # center维度（视频数，每个视频帧数，40个marker, 2）
    for i, path in enumerate(video_paths):
        video_name = os.path.join(video_folder, '{:03d}.mp4'.format(i + 1))  # modify the video name here
        print(video_name)
        capture = cv2.VideoCapture(video_name)
        f = Processing(capture)
        all_coordinates = f.result

        break
