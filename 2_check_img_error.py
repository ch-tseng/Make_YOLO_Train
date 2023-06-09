import cv2
import glob
import os, tqdm
from configparser import ConfigParser
import ast

cfg = ConfigParser()
cfg.read("config.ini",encoding="utf-8")

img_type = ['.jpg', '.jpeg', '.png' ]
project_name = cfg.get("global", "project_name")
baseFolder = cfg.get("global", "baseFolder")
src_folder = os.path.join(baseFolder, project_name, "yolo")
dst_bad_folder = os.path.join(baseFolder, project_name, "bad_imgs")

src_folder = src_folder.replace('\\', '/')
dst_bad_folder = dst_bad_folder.replace('\\', '/')

if not os.path.exists(dst_bad_folder):
    os.makedirs(dst_bad_folder)

lst_imgs = []
for t in tqdm.tqdm(img_type):
    lst_imgs += glob.glob(os.path.join(src_folder, '*'+t.lower()))
    lst_imgs += glob.glob(os.path.join(src_folder, '*'+t.upper()))

print('total images:', len(lst_imgs))

for img_path in tqdm.tqdm(lst_imgs):
    test = cv2.imread(img_path)
    try:
        tshape = test.shape
    except:
        print('error img', img_path)
        bname = os.path.basename(img_path)
        b_name, e_name = os.path.splitext(bname)
        txt_filepath = os.path.join(src_folder, b_name+'.txt')
 
        os.rename(img_path, os.path.join(dst_bad_folder, bname))
        if os.path.exists(txt_filepath):
            os.rename(txt_filepath, os.path.join(dst_bad_folder, b_name+'.txt'))
