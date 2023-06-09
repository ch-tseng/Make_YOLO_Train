import glob, os
import os.path
import time
from shutil import copyfile
import shutil
import cv2
from xml.dom import minidom
from os.path import basename
from tqdm import tqdm
from libThreads import *
from configparser import ConfigParser
import ast

cfg = ConfigParser()
cfg.read("config.ini",encoding="utf-8")
#--------------------------------------------------------------------
#augmentation
color2gry2rgb = False
roate90 = False

multiThread = cfg.getint("global", "multiThread")  #default = 1
project_name = cfg.get("global", "project_name")
xmlFolder = cfg.get("global", "xmlFolder")
imgFolder = cfg.get("global", "imgFolder")
negFolder = cfg.get("global", "negFolder")
baseFolder = cfg.get("global", "baseFolder")
saveYoloPath = os.path.join(baseFolder, project_name, "yolo")
classList = ast.literal_eval(cfg.get("global", "classList"))
img_cp_type = cfg.getint("global", "img_cp_type")  # 0--> copy, 1--> move

#---------------------------------------------------------------------

xmlFolder = xmlFolder.replace("\\", '/')
imgFolder = imgFolder.replace("\\", '/')
negFolder = negFolder.replace("\\", '/')
baseFolder = baseFolder.replace("\\", '/')
saveYoloPath = saveYoloPath.replace("\\", '/')

if not os.path.exists(saveYoloPath):
    os.makedirs(saveYoloPath)

def transferYolo( xmlFilepath, imgFilepath, newname=None):
    global imgFolder

    img_file, img_file_extension = os.path.splitext(imgFilepath)
    img_filename = basename(img_file)

    if(xmlFilepath is not None):
        img = cv2.imread(imgFilepath)
        if color2gry2rgb is True:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.imwrite( os.path.join(saveYoloPath ,'gray_'+img_filename+img_file_extension), gray_3channel)
        if roate90 is True:
            rotate1 = imutils.rotate(img, 90)
            rotate2 = imutils.rotate(img, -90)
            rotate3 = imutils.rotate(img, 180)
            cv2.imwrite( os.path.join(saveYoloPath ,'r90_'+img_filename+img_file_extension), rotate1)
            cv2.imwrite( os.path.join(saveYoloPath ,'r270_'+img_filename+img_file_extension), rotate2)
            cv2.imwrite( os.path.join(saveYoloPath ,'r180_'+img_filename+img_file_extension), rotate3)

        imgShape = img.shape
        img_h = imgShape[0]
        img_w = imgShape[1]

        try:
            labelXML = minidom.parse(xmlFilepath)
        except:
            return

        labelName = []
        labelXmin = []
        labelYmin = []
        labelXmax = []
        labelYmax = []
        totalW = 0
        totalH = 0
        countLabels = 0

        tmpArrays = labelXML.getElementsByTagName("filename")
        for elem in tmpArrays:
            filenameImage = elem.firstChild.data

        tmpArrays = labelXML.getElementsByTagName("name")
        for elem in tmpArrays:
            labelName.append(str(elem.firstChild.data))

        tmpArrays = labelXML.getElementsByTagName("xmin")
        for elem in tmpArrays:
            labelXmin.append(int(elem.firstChild.data))

        tmpArrays = labelXML.getElementsByTagName("ymin")
        for elem in tmpArrays:
            labelYmin.append(int(elem.firstChild.data))

        tmpArrays = labelXML.getElementsByTagName("xmax")
        for elem in tmpArrays:
            labelXmax.append(int(elem.firstChild.data))

        tmpArrays = labelXML.getElementsByTagName("ymax")
        for elem in tmpArrays:
            labelYmax.append(int(elem.firstChild.data))

        yoloFilename = os.path.join(saveYoloPath, img_filename + ".txt")
        #print("writeing to {}".format(yoloFilename))

        with open(yoloFilename, 'a') as the_file:
            i = 0
            for className in labelName:
                if(className in classList):
                    classID = classList[className]
                    x = (labelXmin[i] + (labelXmax[i]-labelXmin[i])/2) * 1.0 / img_w
                    y = (labelYmin[i] + (labelYmax[i]-labelYmin[i])/2) * 1.0 / img_h
                    w = (labelXmax[i]-labelXmin[i]) * 1.0 / img_w
                    h = (labelYmax[i]-labelYmin[i]) * 1.0 / img_h

                    the_file.write(str(classID) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n')
                    i += 1

        if color2gry2rgb is True:
            copyfile(yoloFilename, os.path.join(saveYoloPath, 'gray_'+img_filename + ".txt"))

    else:
        yoloFilename = os.path.join(saveYoloPath , newname + ".txt")
        #print("writeing negative file to {}".format(yoloFilename))

        with open(yoloFilename, 'a') as the_file:
            the_file.write('')

    the_file.close()

def pos2yolo(files):
    for file in tqdm(files):
        filename, file_extension = os.path.splitext(file)
        file_extension = file_extension.lower()

        if(file_extension == ".jpg" or file_extension==".png" or file_extension==".jpeg" or file_extension==".bmp"):
            imgfile = os.path.join(imgFolder, file)
            xmlfile = os.path.join(xmlFolder ,filename + ".xml")

            if(os.path.isfile(xmlfile)):
                transferYolo( xmlfile, imgfile)
                try:
                    img = cv2.imread(imgfile)
                    test = img.shape
                except:
                    print('cannot read', imgfile)
                    return

                cv2.imwrite(os.path.join(saveYoloPath ,file), img)

                if img_cp_type == 1:
                    #shutil.move(imgfile, os.path.join(saveYoloPath ,file))
                    os.remove(imgfile)

def neg2yolo(files):
    for file in tqdm(files):
        filename, file_extension = os.path.splitext(file)
        file_extension = file_extension.lower()
        imgfile = os.path.join(negFolder ,file)

        if(file_extension == ".jpg" or file_extension==".png" or file_extension==".jpeg" or file_extension==".bmp"):
            nid  = time.time()
            nfilename = 'neg_' + str(nid).replace(".","")

            if color2gry2rgb is True:
                img = cv2.imread(imgfile)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                gray_file = os.path.join(saveYoloPath ,'gray_'+nfilename+file_extension)
                cv2.imwrite( gray_file, gray_3channel)
                transferYolo( None, gray_file, 'gray_' + nfilename)

            transferYolo( None, imgfile, nfilename)

            if img_cp_type == 0:
                copyfile(imgfile, os.path.join(saveYoloPath ,nfilename + file_extension))
            else:
                shutil.move(imgfile, os.path.join(saveYoloPath ,nfilename + file_extension))


#---------------------------------------------------------------
fileCount = 0

print("[Step 1/2] Transfrt all labeled images to yolo format.")
allfiles = os.listdir(imgFolder)
run_jobs(allfiles, multiThread, pos2yolo, debug=False)

print("[Step 2/2] Transfrt all negative images to yolo format.")
if(os.path.exists(negFolder)):
    allfiles = os.listdir(negFolder)
    print('neg len', len(allfiles))
    run_jobs(allfiles, multiThread, neg2yolo, debug=False)
