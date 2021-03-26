import glob, os
import os.path
import time
from shutil import copyfile
import cv2
from xml.dom import minidom
from os.path import basename
from tqdm import tqdm

#--------------------------------------------------------------------
color2gry2rgb = True
xmlFolder = "/WORK1/dataset/sale_crowndHuman/dataset_v20210324/labels"
imgFolder = "/WORK1/dataset/sale_crowndHuman/dataset_v20210324/images"
#negFolder = ""
negFolder = "/WORK1/dataset/sale_crowndHuman/dataset_v20210324/negatives"
saveYoloPath = "/WORK1/dataset/sale_crowndHuman/dataset_v20210324/yolo/"
classList = { "person_vbox":0, "person_head":1 }

#---------------------------------------------------------------------

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

        imgShape = img.shape
        img_h = imgShape[0]
        img_w = imgShape[1]

        labelXML = minidom.parse(xmlFilepath)
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

#---------------------------------------------------------------
fileCount = 0

print("[Step 1/2] Transfrt all labeled images to yolo format.")
for file in tqdm(os.listdir(imgFolder)):
    filename, file_extension = os.path.splitext(file)
    file_extension = file_extension.lower()

    if(file_extension == ".jpg" or file_extension==".png" or file_extension==".jpeg" or file_extension==".bmp"):
        imgfile = os.path.join(imgFolder, file)
        xmlfile = os.path.join(xmlFolder ,filename + ".xml")

        if(os.path.isfile(xmlfile)):
            #print("id:{}".format(fileCount))
            #print("processing {}".format(imgfile))
            #print("processing {}".format(xmlfile))
            fileCount += 1

            transferYolo( xmlfile, imgfile)
            copyfile(imgfile, os.path.join(saveYoloPath ,file))
            #if color2gry2rgb is True:
            #    cv2.imwrite( os.path.join(saveYoloPath ,'gray_'+file), gray_3channel)

print("[Step 2/2] Transfrt all negative images to yolo format.")
nid = 0
if(os.path.exists(negFolder)):
    for file in tqdm(os.listdir(negFolder)):
        filename, file_extension = os.path.splitext(file)
        file_extension = file_extension.lower()
        imgfile = os.path.join(negFolder ,file)

        if(file_extension == ".jpg" or file_extension==".png" or file_extension==".jpeg" or file_extension==".bmp"):
            nid += 1
            nfilename = 'neg_' + str(nid).zfill(6)
            transferYolo( None, imgfile, nfilename)
            copyfile(imgfile, os.path.join(saveYoloPath ,nfilename + file_extension))

