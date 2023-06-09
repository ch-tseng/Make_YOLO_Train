import random
import glob, os, sys
import os.path
from tqdm import tqdm
from configparser import ConfigParser
import ast

cfg = ConfigParser()
cfg.read("config.ini",encoding="utf-8")

#---------------------------------------------------------
testRatio = 0.2
project_name = cfg.get("global", "project_name")
classList = ast.literal_eval(cfg.get("global", "classList"))
baseFolder = cfg.get("global", "baseFolder")

saveYoloPath = os.path.join(baseFolder, project_name, "yolo")
cfgFolder = os.path.join(baseFolder, project_name, "cfg_train")
weights_save = os.path.join(baseFolder, project_name, "weights")

#----------------------------------------------------
baseFolder = baseFolder.replace("\\", '/')
saveYoloPath = saveYoloPath.replace("\\", '/')
cfgFolder = cfgFolder.replace("\\", '/')
weights_save = weights_save.replace("\\", '/')

# make image list -------------------------------------
fileList = []
#outputFile = os.path.join(cfgFolder,"img_list.txt")

if not os.path.exists(saveYoloPath):
    print("There is no such folder for ", saveYoloPath)
    sys.exit()

if not os.path.exists(weights_save):
    os.makedirs(weights_save)

if not os.path.exists(cfgFolder):
    os.makedirs(cfgFolder)

print("[Step 1/4] Find images in {}".format(saveYoloPath))
for file in tqdm(os.listdir(saveYoloPath)):
    filename, file_extension = os.path.splitext(file)
    file_extension = file_extension.lower()

    if file_extension in ['.jpeg', '.jpg', '.png', '.bmp']:
        if os.path.exists(os.path.join(saveYoloPath, filename + '.txt')):
            fileList.append(os.path.join(saveYoloPath, file))

trainCount = len(fileList)

# train test dataset -------------------------------------
outputTrainFile = os.path.join(cfgFolder,"train.txt")
outputTestFile = os.path.join(cfgFolder ,"test.txt")

testCount = int(len(fileList) * testRatio)
trainCount = len(fileList) - testCount

a = range(len(fileList))
test_data = random.sample(a, testCount)
train_data = [x for x in a if x not in test_data]

print("[Step 2/4] Generate train.txt, {} images total.".format(len(train_data)))
with open(outputTrainFile, 'w') as the_file:
    for i in tqdm(train_data):
        the_file.write(fileList[i] + "\n")
the_file.close()

print("[Step 3/4] Generate test.txt, {} images total.".format(len(test_data)))
with open(outputTestFile, 'w') as the_file:
    for i in tqdm(test_data):
        the_file.write(fileList[i] + "\n")
the_file.close()

# make config -----------------------------------------
cfg_obj_names = "obj.names"
cfg_obj_data = "obj.data"
cfg_fastestdet = "config.yaml"
classes = len(classList)

#pathCFG = os.path.join(cfgFolder, "weights")
#if not os.path.exists(pathCFG):
#    os.makedirs(pathCFG)

path_weights_save = os.path.join( weights_save, project_name, 'darknet' )
if not os.path.exists(path_weights_save):
    os.makedirs(path_weights_save)

with open(os.path.join(cfgFolder, cfg_obj_data), 'w') as the_file:
    the_file.write("classes= " + str(classes) + "\n")
    the_file.write("train  = " + os.path.join(cfgFolder ,"train.txt") + "\n")
    the_file.write("valid  = " + os.path.join(cfgFolder ,"test.txt") + "\n")
    the_file.write("names = " + os.path.join(cfgFolder ,"obj.names") + "\n")
    the_file.write("backup = " + path_weights_save + "/")
the_file.close()

with open(os.path.join(cfgFolder ,cfg_obj_names), 'w') as the_file:
    for className in classList:
        the_file.write(className + "\n")
the_file.close()

#for yolo fastestdet
with open("cfg/fastestdet/config.yaml", "r") as f:
    lines = f.readlines()

nlines = []
for line in lines:
    line = line.replace("{TRAIN}", os.path.join(cfgFolder ,"train.txt"))
    line = line.replace("{VAL}", os.path.join(cfgFolder ,"test.txt"))
    line = line.replace("{NAMES}", os.path.join(cfgFolder ,"obj.names"))
    line = line.replace("{NC}", str(len(classList)))
    nlines.append(line)

with open(os.path.join(cfgFolder ,"fastestdet_config.yaml"), 'w') as the_file:
    for line in nlines:
        the_file.write(line)

print("[Step 4/4] obj.names,obj.data generated.")
