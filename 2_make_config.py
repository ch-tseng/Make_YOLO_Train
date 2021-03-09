import random
import glob, os, sys
import os.path
from tqdm import tqdm

#---------------------------------------------------------
testRatio = 0.2
classList = { "forklift":0 }
saveYoloPath = "/WORK1/MyProjects/for_Sale/forklift/dataset/yolo"
cfgFolder = "/WORK1/MyProjects/for_Sale/forklift/train_models/yolo_models/"
darknet_home = "/home/chtseng/frameworks/darknet.v4/"
#--------------------------------------------------------

# make image list -------------------------------------
fileList = []
#outputFile = os.path.join(cfgFolder,"img_list.txt")

if not os.path.exists(saveYoloPath):
    print("There is no such folder for ", saveYoloPath)
    sys.exit()

if not os.path.exists(cfgFolder):
    os.makedirs(cfgFolder)

print("[Step 1/4] Find images in {}".format(saveYoloPath))
for file in tqdm(os.listdir(saveYoloPath)):
    filename, file_extension = os.path.splitext(file)
    file_extension = file_extension.lower()

    if(file_extension == ".txt"):
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
classes = len(classList)

pathCFG = os.path.join(cfgFolder, "weights")
if not os.path.exists(pathCFG):
    os.makedirs(pathCFG)

with open(os.path.join(cfgFolder, cfg_obj_data), 'w') as the_file:
    the_file.write("classes= " + str(classes) + "\n")
    the_file.write("train  = " + os.path.join(cfgFolder ,"train.txt") + "\n")
    the_file.write("valid  = " + os.path.join(cfgFolder ,"test.txt") + "\n")
    the_file.write("names = " + os.path.join(cfgFolder ,"obj.names") + "\n")
    the_file.write("backup = " + os.path.join(cfgFolder ,"weights") + "/")
the_file.close()

with open(os.path.join(cfgFolder ,cfg_obj_names), 'w') as the_file:
    for className in classList:
        the_file.write(className + "\n")
the_file.close()

print("[Step 4/4] obj.names,obj.data generated.")
