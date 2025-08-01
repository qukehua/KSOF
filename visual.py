import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import PIL.Image as Image
import matplotlib.animation as animation
import imageio


# human36m_connectivity_dict = [[16, 17, 0], [17, 18, 0], [18, 19, 0], [19, 21, 0], [20, 22, 0],  # 右臂
#                               [0, 1, 0], [1, 2, 0], [2, 3, 0], [3, 4, 0], [4, 5, 0],  # 右腿
#                               [15, 14, 0], [14, 13, 0], [13, 12, 0], [12, 11, 0],  # 躯干
#
#                               [0, 6, 1], [6, 7, 1], [7, 8, 1], [8, 9, 1], [9, 10, 1],  # 左腿
#                               [24, 25, 1], [25, 26, 1], [26, 27, 1], [27, 29, 1], [28, 30, 1]]  # 左臂

human36m_connectivity_dict = [[16, 17, 1], [17, 18, 1], [18, 19, 1], [19, 21, 1], [20, 22, 1],  # 右臂
                              [0, 1, 0], [1, 2, 0], [2, 3, 0], [3, 4, 0], [4, 5, 0],  # 右腿
                              [14, 15, 0], [13, 14, 0], [12, 13, 0], [11, 12, 0],  # 躯干

                              [0, 6, 1], [6, 7, 1], [7, 8, 1], [8, 9, 1], [9, 10, 1],  # 左腿
                              [24, 25, 0], [25, 26, 0], [26, 27, 0], [27, 29, 0], [28, 30, 0]]  # 左臂

human36m_connectivity_truth = [[16, 17, 3], [17, 18, 3], [18, 19, 3], [19, 21, 3], [20, 22, 3],  # 右臂
                               [0, 1, 3], [1, 2, 3], [2, 3, 3], [3, 4, 3], [4, 5, 3],  # 右腿
                               [14, 15, 3], [13, 14, 3], [12, 13, 3], [11, 12, 3],  # 躯干

                               [0, 6, 3], [6, 7, 3], [7, 8, 3], [8, 9, 3], [9, 10, 3],  # 左腿
                               [24, 25, 3], [25, 26, 3], [26, 27, 3], [27, 29, 3], [28, 30, 3]]  # 左臂


# CMU_connectivity_dict = [[29, 30, 0], [30, 31, 0], [31, 32, 0], [33, 34, 0], [34, 35, 0], [36, 37, 0],  # 左臂
#                          [7, 8, 0], [8, 9, 0], [9, 10, 0], [10, 11, 0], [11, 12, 0],  # 左腿
#                          [19, 18, 0], [18, 17, 0], [17, 16, 0], [15, 14, 0], [14, 13, 0],  # 躯干
#                          [1, 2, 1], [2, 3, 1], [3, 4, 1], [4, 5, 1], [5, 6, 1],  # 右腿
#                          [20, 21, 1], [21, 22, 1], [22, 23, 1], [24, 25, 1], [25, 26, 1], [27, 28, 1],  # 右臂
#                          ]


def draw3Dpose(human, pose_3d, ax, lcolor="#3498db", rcolor="#e74c3c", gcolor="#808080", add_labels=False):
    for i in human:
        x, y, z = [np.array([pose_3d[i[0], j], pose_3d[i[1], j]]) for j in range(3)]

        ax.plot(x, y, z, lw=2, c=lcolor if i[2] == 1 else rcolor if i[2] == 0 else gcolor)

    RADIUS = 650  # space around the subject
    xroot, yroot, zroot = pose_3d[11, 0], pose_3d[11, 1], pose_3d[11, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")



def transPNG(img):
    # img = img.convert("RGBA")
    datas = img.getdata()
    newData = list()
    for item in datas:
        if item[0] > 230 and item[1] > 230 and item[2] > 230:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    img.putdata(newData)
    return img



def transPNGbw(img):
    # img = img.convert("RGBA")
    datas = img.getdata()
    newData = list()
    for item in datas:
        if item[0] < 25 and item[1] < 25 and item[2] < 25:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    img.putdata(newData)
    return img


def image_compose(IMAGES_PATH, IMAGE_SIZE, IMAGE_COLUMN, IMAGE_ROW, IMAGE_SAVE_PATH,
                  LEFT_DIST, RIGHT_DIST, UPPER_DIST, LOWER_DIST, OVERLOP):

    if os.path.exists(IMAGE_SAVE_PATH):
        os.remove(IMAGE_SAVE_PATH)
    aft_image = Image.new('RGB',
                          (IMAGE_COLUMN * (IMAGE_SIZE[0] - LEFT_DIST - RIGHT_DIST) - OVERLOP * (IMAGE_COLUMN - 1),
                           IMAGE_ROW * (IMAGE_SIZE[1] - UPPER_DIST - LOWER_DIST)))  
    file_list = os.listdir(IMAGES_PATH)
    file_list.sort(key=lambda x: int(x.split('_')[-1][:-4]))  
    image_names = [name for name in file_list]
 
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
                (IMAGE_SIZE[0], IMAGE_SIZE[1]), Image.ANTIALIAS).crop(
                (LEFT_DIST, UPPER_DIST, IMAGE_SIZE[0] - RIGHT_DIST, IMAGE_SIZE[1] - LOWER_DIST))
            from_image = transPNG(from_image)
            r, g, b, a = from_image.split()
            aft_image.paste(from_image, ((x - 1) * (IMAGE_SIZE[0] - LEFT_DIST - RIGHT_DIST - OVERLOP),
                                         (y - 1) * (IMAGE_SIZE[1] - UPPER_DIST - LOWER_DIST)), mask=a)
            aft_image = transPNGbw(aft_image)
    return aft_image.save(IMAGE_SAVE_PATH) 


def smooth(src, sample_len=20, kernel_size=0):
    """
    data:[bs, 60, 96]
    """
    src_data = src[:, -sample_len:, :].copy()
    smooth_data = src_data.copy()
    for i in range(kernel_size, sample_len):
        smooth_data[:, i] = np.mean(src_data[:, kernel_size:i + 1], axis=1)
    return smooth_data


if __name__ == '__main__':

    path_npy = 'point_npy'
    path_img = 'img_H3.6'
    path_gif = 'gif_H3.6'
    files = os.listdir(path_npy) 
    for file in files:  
        # print(file)

        if not os.path.isdir(file):  
            if file == 'predict_npy':
                human = human36m_connectivity_dict
            if file == 'truth_npy':
                human = human36m_connectivity_truth

            new_path_npy = os.path.join(path_npy, file)
            new_files = os.listdir(new_path_npy)  
            print(new_files)

            new_path_img = os.path.join(path_img, file)  
            new_path_gif = os.path.join(path_gif, file)  
            if not os.path.exists(new_path_img):
                os.makedirs(new_path_img)
            if not os.path.exists(new_path_gif):
                os.makedirs(new_path_gif)

            for new_file in new_files: 
                name_path = os.path.basename(new_file)  
                name = os.path.splitext(name_path)[0] 
                specific_3d_skeleton = np.load(new_path_npy + '/' + name_path, allow_pickle=True)  
                # print(specific_3d_skeleton.shape)

                image = os.path.join(new_path_img, name)  
                gif = os.path.join(new_path_gif, name)  
                if not os.path.exists(image):
                    os.makedirs(image)
                if not os.path.exists(gif):
                    os.makedirs(gif)

                # specific_3d_skeleton = np.load('pred_p3d_20.npy')  # (8, 10, 38, 3)
                # specific_3d_skeleton = specific_3d_skeleton.reshape(specific_3d_skeleton.shape[0], 35, 38, 3)  # (8, 20, 38, 3)
                # specific_3d_skeleton = smooth(specific_3d_skeleton)
                # specific_3d_skeleton = np.load(path_npy,allow_pickle=True)
                # specific_3d_skeleton = specific_3d_skeleton.reshape(specific_3d_skeleton.shape[0], 10, 32,
                #                                                     3)  # (9, 10, 38, 3)
                # print("*********************************")
                # print(specific_3d_skeleton.shape)


                for i in range(specific_3d_skeleton.shape[0]):
                    for j in range(specific_3d_skeleton.shape[1]):
                        fig = plt.figure()
                        ax = fig.add_subplot(111, projection='3d')
                        plt.ion()
                        # print(specific_3d_skeleton[i][j].shape)
                        img_path = os.path.join(image, 'batch_' + str(
                            i))  
                        if not os.path.exists(img_path):
                            os.makedirs(img_path)
                            # ax.lines = []
                        ax.view_init(elev=45, azim=45)  
                        specific_3d_skeleton[i][j] = specific_3d_skeleton[i][j][:, [0, 2, 1]]
                        draw3Dpose(human, specific_3d_skeleton[i][j], ax)
                        plt.axis('off')  
                        plt.savefig(os.path.join(img_path, name + '_' + str(j) + '.png'))  
                        plt.ioff()  
                        plt.close()
                        print('Saving to: ' + img_path)

                for i in range(specific_3d_skeleton.shape[0]):
                    open_path = os.path.join(image,
                                             'batch_' + str(i) + '/')  
                    save_path = os.path.join(gif, 'batch_' + str(i))  
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    path = open_path 
                    size = [640, 480]  
                    column = specific_3d_skeleton.shape[1]  
                    row = 1  
                    save_png = os.path.join(save_path,
                                            name + '.png')  
                    # print(name)
                    # print(save_path)

                    left_dist = 210  
                    right_dist = 180  
                    upper_dist = 90 
                    lower_dist = 80 
                    overlap = 140 
                    image_compose(path, size, column, row, save_png, left_dist, right_dist, upper_dist, lower_dist,
                                  overlap)
                    print('Splicing diagram Saving to: ' + save_png)

                    # save_gif = os.path.join(save_path,
                    #                         name + '.gif') 
                    # plt.cla()
                    # files = os.listdir(open_path)
                    # files.sort(key=lambda x: int(x.split('_')[-1][:-4]))  # .png所以是[:-4]
                    #
                    #
                    # # frames = []
                    # # for file in files:
                    # #     img = Image.open(open_path + os.sep + file)
                    # #     frames.append(img)
                    # # frames[0].save(save_gif, append_images=frames[1:], loop=0, save_all=True, duration=400)
                    #
                    # gif_images = []
                    # for path in files:
                    #     # print(path)
                    #     gif_images.append(imageio.imread(os.path.join(open_path, path)))
                    # imageio.mimsave(save_gif, gif_images, fps=25)
                    # print('Gif Saving to: ' + save_gif)
