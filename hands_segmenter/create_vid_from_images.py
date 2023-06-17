import cv2
import numpy as np
import glob
import os

# frameSize = (500, 500)

dirs = "/data/valid/"

def create_video_dirs(dirs):

    for dir in os.listdir(dirs):
        print(dir)

        for subdir in os.listdir(os.path.join(dirs, dir)):
            if os.path.isfile(os.path.join(dirs, dir, subdir)):
                continue
            print(subdir)
            files = glob.glob(os.path.join(dirs, dir, subdir, "*.jpg"))

            sample = cv2.imread(files[0])

            height, width, _ = sample.shape

            video=cv2.VideoWriter(os.path.join(dirs, dir, subdir + '.avi'),cv2.VideoWriter_fourcc(*'DIVX'),20,(width,height))

            for file in files:
                img = cv2.imread(file)
                video.write(img)

            video.release()


def create_video_single_dir(dir): 

    files = sorted([f for f in os.listdir(dir)])

    sample = cv2.imread(os.path.join(dir, files[0]))

    height, width, _ = sample.shape

    video=cv2.VideoWriter(os.path.join(dir + 'video_1_out.avi'),cv2.VideoWriter_fourcc(*'DIVX'),10,(width,height))

    for file in files:

            if not os.path.isfile(os.path.join(dir, file)):
                continue
            ext = file.split(".")[-1]

            if ext  not in ['jpg', 'png']:
                continue
            print(os.path.join(dir, file))
            img = cv2.imread(os.path.join(dir, file))
            video.write(img)

    video.release()


dir = "/media/apeksha/DATA1/Video-Classification/data/version_12/"
create_video_single_dir(dir)