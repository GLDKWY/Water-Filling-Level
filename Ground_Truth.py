import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

out_path = 'result.png'  # save current frame path
# display all the columns
pd.set_option('display.max_columns', None)
# display all the rows
pd.set_option('display.max_rows', None)
# set the number of showed value length of column to 100
pd.set_option('max_colwidth', 100)
# set the number of showed columns to 1000
pd.set_option('display.width', 1000)

tactile_mp4 = './train_tactile/005.mp4'  # the path of video to be processed
annotation_path = './Annotations/005.csv'  # the path of the corresponding annotation


def process(thresh):
    # the approximate coordinate of the bottom of target container
    liquid_limit = int((int(Coords1y) + int(Coords2y)) / 2 + 5)
    tactile_limit = 1334  # boundary of the tactile and pouring procedure
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # set the pixel value of left、right and bottom of the target container to 0
    thresh_0_left = np.zeros((height, int(Coords1x)))
    thresh_0_right = np.zeros((height, width - int(Coords2x)))
    thresh_0_bottom = np.zeros((height - liquid_limit - 10, width))
    thresh[:, :int(Coords1x)] = thresh_0_left
    thresh[:, int(Coords2x):] = thresh_0_right
    thresh[liquid_limit + 10:, :] = thresh_0_bottom
    # print(thresh.shape)  # shape is (row,col)
    if len(contours) != 0:
        # exclude the tactile frame and choose a little bit higher than the bottom of the target container as range
        # of interest (in order to keep the radian shape of bottom of target container )
        thresh_up = thresh[: liquid_limit, : tactile_limit]
        cv2.namedWindow("thresh_up", 0)
        cv2.resizeWindow("thresh_up", 640, 480)
        cv2.imshow('thresh_up', thresh_up)
        y_total = thresh_up.sum(axis=1)  # calculate the total number of pixels along the y-axis
        # select liquid part that its width is 20 pixels around the width of target container to avoid liquid in the air
        y = y_total > (int(Coords2x) - int(Coords1x) - 50) * 255
        thresh_up = thresh_up * y.reshape(len(y), 1)
        if (y != 0).argmax(axis=0):
            up_index = (y != 0).argmax(axis=0)  # find the height of the liquid surface
            print(up_index)
        else:
            up_index = liquid_limit
        # fill up the upper part of the rectangle
        thresh_up[up_index:, int(Coords1x):int(Coords2x)] = np.ones(
            (liquid_limit - up_index, int(Coords2x) - int(Coords1x))) * 255
        thresh[: liquid_limit, : tactile_limit] = thresh_up
    return thresh


# calculate water filling area(represented by number of pixels)
def color_area(image):
    # set the red HSV range
    color = [
        ([0, 25, 21], [17, 255, 255])  # red [156, 43, 46], [180, 255, 255]
    ]  # avoid bubbles：[156, 90, 46], [239, 255, 156]approximate rectangle 153，max contour，156
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    area = 0
    for (lower, upper) in color:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((1, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, anchor=(2, 0), iterations=2)
        # output = cv2.bitwise_and(image, image, mask=mask)

        cv2.imwrite(out_path, mask)

        mat_img = cv2.imread(out_path)
        gray = cv2.cvtColor(mat_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        output = process(thresh)
        kernel = np.ones((3, 3), np.uint8)
        output = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel, anchor=(2, 0), iterations=2)

        contours, _ = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # if contours:
        #     c = max(contours, key=cv2.contourArea)  # 找出最大面积的轮廓
        #     x, y, w, h = cv2.boundingRect(c)
        #     cv2.rectangle(mat_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #     area = w * h
        # else:
        #     print("No contours find")

        # mark contours
        if contours:
            c = max(contours, key=cv2.contourArea)  # find the contour that has the max area
            M = cv2.moments(c)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            if cy > 70:
                cv2.drawContours(output, c, -1, (255, 0, 255), 3)
                area = cv2.contourArea(c)
            else:
                print('the founded liquid is not in the glass')
        else:
            print('No contours find')
        # cv2.imshow("images", np.hstack([image, mat_img]))
        cv2.namedWindow("original", 0)
        cv2.resizeWindow("original", 640, 480)
        cv2.namedWindow("Contours", 0)
        cv2.resizeWindow("Contours", 640, 480)
        cv2.imshow('original', image)
        cv2.imshow('Contours', output)
        out.write(output)
    return area


def glass_area(image):
    frame = cv2.GaussianBlur(image, (5, 5), 10)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, threshold_img = cv2.threshold(frame_gray, 100, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(threshold_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
    cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)
    # cv2.imshow("Frame", frame)
    cv2.waitKey(0)
    area = cv2.contourArea(contours[0])
    return area


def left_click(event):
    global Coords1x, Coords1y
    if event.button == 1:
        try:
            Coords1x = int(event.xdata)
            Coords1y = int(event.ydata)
        except:
            Coords1x = event.xdata
            Coords1y = event.ydata
        print("Coordinate of left bottom: ", Coords1x, Coords1y)


def right_click(event):
    global Coords2x, Coords2y
    if event.button == 3:
        try:
            Coords2x = int(event.xdata)
            Coords2y = int(event.ydata)
        except:
            Coords2x = event.xdata
            Coords2y = event.ydata
        print("Coordinate of right bottom: ", Coords2x, Coords2y)


def count(event):
    global glass_square, Coords3x, Coords3y
    if event.button == 2:
        try:
            Coords3x = int(event.xdata)
            Coords3y = int(event.ydata)
        except:
            Coords3x = event.xdata
            Coords3y = event.ydata
        print("Coordinate of right surface: ", Coords3x, Coords3y)
        glass_square = (Coords3x - int(Coords1x)) * (int(Coords2y) - Coords3y)


cap = cv2.VideoCapture(tactile_mp4)
total_frames = cap.get(7)
rate = cap.get(5)  # frame rate
duration = total_frames / rate  # total seconds
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
current_frame = 0

# read first frame to get glass area
success, frame = cap.read()
# frame = cv2.resize(frame, (640, 560), interpolation=cv2.INTER_AREA)
cup_img = frame

Coords1x, Coords1y = 'NA', 'NA'
Coords2x, Coords2y = 'NA', 'NA'
Coords3x, Coords3y = 'NA', 'NA'
glass_square = 0
cup_fig = plt.figure()
plt.imshow(cv2.cvtColor(cup_img, cv2.COLOR_BGR2RGB))
# click the left mouse button to select Coordinate of left bottom
cup_fig.canvas.mpl_connect('button_press_event', left_click)
# click the right mouse button to select Coordinate of right bottom
cup_fig.canvas.mpl_connect('button_press_event', right_click)
# # click the mid mouse button to select Coordinate of right surface
cup_fig.canvas.mpl_connect('button_press_event', count)
plt.show()
print('Total area of the target Container: ', glass_square)
out = cv2.VideoWriter('outpy.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 24, (width, height), False)

times = []
all_liquid_squares = []
all_liquid_percentages = []
while True:
    success, frame = cap.read()
    if success:
        current_frame += 1
        # frame = cv2.resize(frame, (640, 560), interpolation=cv2.INTER_AREA)
        print("current video time: ", current_frame / total_frames * duration, 's')
        times.append(round(float(current_frame / total_frames * duration), 2))
        liquid_square = color_area(frame)
        all_liquid_squares.append(round(float(liquid_square), 2))
        print('Current liquid area: ', liquid_square)
        print('Current liquid area (percentage): ', liquid_square / glass_square * 100, '%')
        all_liquid_percentages.append(round(float(liquid_square / glass_square * 100), 2))
    # if [ESC] key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# do a bit of cleanup
out.release()
cap.release()
cv2.destroyAllWindows()

# df = pd.read_csv('2023_4_24.csv')


# for i in range(len(all_liquid_percentages)):
#     all_liquid_percentages[i] = float(all_liquid_percentages[i])
#     if i == 0:
#         all_liquid_percentages[i] = 0
#     elif all_liquid_percentages[i] > all_liquid_percentages[i - 1] + 10:
#         t = i - 1
#         while 1:
#             if all_liquid_percentages[t] == -1:
#                 t = t - 1
#             else:
#                 break
#         if all_liquid_percentages[i] > all_liquid_percentages[t] + 10:  # 真的比前一个有效值大20%以上
#             all_liquid_percentages[i] = -1  # 把不合理的值填为-1
# # 把-1的位置插值
#
# for i in range(len(all_liquid_percentages)):
#     if all_liquid_percentages[i] == -1:
#         x = []
#         b = i - 1
#         f = i + 1
#         n = 0
#         temp = []
#         while True:
#             if b < 0 or n == 3:
#                 break
#             else:
#                 if all_liquid_percentages[b] == -1:
#                     b = b - 1
#                     continue
#                 else:
#                     temp.append(all_liquid_percentages[b])
#                     x.append(b)
#                     n = n + 1
#                     b = b - 1
#         n = 0
#         while True:
#             if f == len(all_liquid_percentages) or n == 3:
#                 break
#             else:
#                 if all_liquid_percentages[f] == -1:
#                     f = f + 1
#                     continue
#                 else:
#                     temp.append(all_liquid_percentages[f])
#                     x.append(f)
#                     n = n + 1
#                     f = f + 1
#         f_per = lagrange(x, temp)
#         all_liquid_percentages[i] = round(f_per(i), 2)
#         # all_liquid_squares[i]=glass_square*all_liquid_percentages[i]/100

y = savgol_filter(all_liquid_percentages, 37, 3, mode='nearest')
tmp = np.array(all_liquid_percentages)
not_zero_index = (tmp != 0).argmax(axis=0)
y[:not_zero_index] = 0  # set the percentages that before the first non-zero percentage to 0
y[y < 0] = 0  # set the negative percentage to 0

# y_max_index = np.argmax(y)
# y_max = np.max(y)
# set the stable percentage as the final percentage
y_last_index = np.argmax(y >= y[-1])
y[y_last_index:] = y[-1]


# print(all_liquid_percentages)
# print(len(all_liquid_percentages))
# print(y)
# print(len(y))
plt.rcParams['font.sans-serif'] = 'Arial'
plt.plot(all_liquid_percentages, 'grey', label='Original', linewidth=2.5)
plt.plot(y, 'r', label='Filtered', linewidth=2.5)
plt.title('Water Filling Level Estimation Ground Truth', fontsize=15)
plt.xlabel('Frame Index', fontsize=15)
plt.ylabel('Percentage / %', fontsize=15)
plt.xticks(np.linspace(0, 200, 5, endpoint=True))
# plt.yticks(np.linspace(0, 70, 8, endpoint=True))
plt.legend()
plt.grid(linestyle='-.')
plt.show()

dataframe = pd.DataFrame({'time': times, 'square': all_liquid_squares, 'percentage': y})
dataframe.to_csv(annotation_path, index=False, sep=',')

