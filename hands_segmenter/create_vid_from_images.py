import cv2
import numpy as np
import glob
import os


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

    video=cv2.VideoWriter(os.path.join(dir ,'video_2_out.avi'),cv2.VideoWriter_fourcc(*'DIVX'),10,(width,height))

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


dir = ""
create_video_single_dir(dir)