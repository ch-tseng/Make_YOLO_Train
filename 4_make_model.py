import os,sys

classList = { "0":0, "1":1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8, "9":9, \
              "A":10, "B":11, "C":12, "D":13, "E":14, "F":15, "G":16 ,"H":17, "I":18, "J":19, \
              "K":20, "L":21, "M":22, "N":23, "O":24, "P":25, "Q":26, "R":27, "S":28, "T":29, \
              "U":30, "V":31, "W":32, "X":33, "Y":34, "Z":35, "plate":36 }

cfgFolder = "/WORKING/modelSale/ritchie_plates_chars/dataset/aug_1020/cfg_train"

'''
classList = { "balaclava_ski_mask":0, "eyeglasses":1, "face_no_mask":2, "face_other_covering":3, "face_shield":4, \
              "face_with_mask":5, "face_with_mask_incorrect":6, "gas_mask":7, "goggles":8, "hair_net":9, "hat":10, \
              "helmet":11, "hijab_niqab":12, "hood":13, "mask_colorful":14, "mask_surgical":15, "other":16, \
              "scarf_bandana":17, "sunglasses":18, "turban":19 }
'''
dark_home = "/home/chtseng/frameworks/darknet"
yolofastest_home = "/home/chtseng/frameworks/darknet"
yolov5_home = "/home/chtseng/frameworks/yolov5"

yolo_config = {
    '416': "6, 20,  11, 20,  10, 30,  16, 25,  15, 36,  27, 36,  64, 49, 101, 45, 110, 71",
    '512': "8, 22,  11, 35,  15, 27,  17, 41,  24, 33,  28, 51,  78, 59, 124, 56, 135, 88",
    '608': "10, 26,  13, 41,  18, 32,  20, 49,  29, 39,  34, 60,  92, 70, 147, 66, 160,104",
    '640': "11, 29,  15, 44,  21, 34,  23, 52,  38, 54,  92, 79, 135, 56, 155, 85, 182,122",
    '960': "15, 41,  20, 65,  29, 50,  31, 77,  46, 61,  53, 95, 146,111, 232,105, 253,165",
    '1280': "16, 45,  22, 80,  35, 54,  32, 79,  37,103,  51, 78,  55,116,  93,115, 202,119, 217,184, 320,141, 371,225",
    '1536': "24, 65,  33,104,  46, 80,  50,123,  73, 98,  85,152, 233,177, 371,168, 405,263"
}

yolotiny_config = {
    '320': "6, 15,   8, 23,  12, 19,  15, 28,  52, 36,  84, 46",
    '416': "8, 19,  10, 30,  15, 26,  20, 37,  68, 47, 110, 60",
}

#---------------------------------------------------------------------

classNum = len(classList)
filterNum = (classNum + 5) * 3

#default size:
#    yolov3: 608, yolov3-tiny:416, yolov4:608, yolov4-tiny:416, yolo-fastest:320,
#    yolo-fastest-xl:320, yolov4x-mish:640, yolov4-csp:512, yolov4-cspx-p7:1536
#    [CFG FILE, PRE-TRAINED WEIGHTS, SIZE, BATCH, DIVISION-BATCH]
cfgs = {
    "yolov3": ["cfg/yolov3.cfg", "pretrained/darknet53.conv.74", 608, 64, 32],
    "yolov3-tiny": ["cfg/yolov3-tiny.cfg", "pretrained/yolov3-tiny.conv.15", 416, 70, 2],
    "yolov3-spp": ["cfg/yolov3-spp.cfg", "pretrained/yolov3-spp.weights", 608, 64, 32],
    "yolov4": ["cfg/yolov4.cfg", "pretrained/yolov4.conv.137", 608, 64, 64],
    "yolov4-tiny": ["cfg/yolov4-tiny.cfg", "pretrained/yolov4-tiny.conv.29", 416, 72, 1],
    "yolo-fastest": ["cfg/yolo-fastest-1.1.cfg", "pretrained/yolo-fastest.conv.109", 320, 160, 2],
    "yolo-fastest-xl": ["cfg/yolo-fastest-1.1-xl.cfg", "pretrained/yolo-fastest-xl.conv.109", 320, 120, 2],
    "yolov4x-mish": ["cfg/yolov4x-mish.cfg", "pretrained/yolov4x-mish.conv.166", 640, 64, 64],
    "yolov4-csp": ["cfg/yolov4-csp.cfg", "pretrained/yolov4-csp.conv.142", 512, 64, 64],
    "yolov4-cspx-p7": ["cfg/cspx-p7-mish.cfg", "pretrained/cspx-p7-mish_hp.344.conv", 1536, 64, 64],
    "yolov5n": ["cfg/yolov5n.yaml", "yolov5n.pt", 640, 128, 1],
    "yolov5s": ["cfg/yolov5s.yaml", "yolov5s.pt", 640, 64, 1],
    "yolov5m": ["cfg/yolov5m.yaml", "yolov5m.pt", 640, 24, 1],
    "yolov5l": ["cfg/yolov5l.yaml", "yolov5l.pt", 640, 12, 1],
    "yolov5x": ["cfg/yolov5x.yaml", "yolov5x.pt", 640, 8, 1],
    "yolov5n-p6": ["cfg/yolov5n6.yaml", "yolov5n6.pt", 1280, 64, 1],
    "yolov5s-p6": ["cfg/yolov5s6.yaml", "yolov5s6.pt", 1280, 48, 1],
    "yolov5m-p6": ["cfg/yolov5m6.yaml", "yolov5m6.pt", 1280, 32, 1],
    "yolov5l-p6": ["cfg/yolov5l6.yaml", "yolov5l6.pt", 1280, 24, 1],
    "yolov5x-p6": ["cfg/yolov5x6.yaml", "yolov5x6.pt", 1280, 12, 1],
}

pwd = os.getcwd()

#make dataset yaml for YOLOV5
with open('cfg/data_yolov5.yaml') as file:
    dataset_content = file.read()
file.close

class_txt = '['
for i, cname in enumerate(classList):
    class_txt += "'{}'".format(cname)
    if i<(len(classList)-1): class_txt += ', '
class_txt += ']'

dataset_content = dataset_content.replace("{TRAIN_LIST}", os.path.join(cfgFolder,'train.txt'))
dataset_content = dataset_content.replace("{TEST_LIST}", os.path.join(cfgFolder,'test.txt'))
dataset_content = dataset_content.replace("{CLASSES}", str(classNum))
dataset_content = dataset_content.replace("{CLASS_LIST}", class_txt)

file = open(os.path.join(cfgFolder, 'ds_yolov5.yaml'), "w")
file.write(dataset_content)
file.close
#---- end

tfile = open( os.path.join(cfgFolder, 'train_cmd.txt'), 'w')

for cfg_name in cfgs:
    if(cfg_name[:6] == 'yolov5'):
        anchors = yolo_config[str(cfgs[cfg_name][2])]
        anch_list = anchors.split(',')

        anchors1, anchors2, anchors3, anchors4 = "", "", "", ""
        for a in range(0,6):
            anchors1 += anch_list[a]
            if a<6: anchors1 += ','
        for a in range(6,12):
            anchors2 += anch_list[a]
            if a<12: anchors2 += ','
        for a in range(12,18):
            anchors3 += anch_list[a]
            if a<18: anchors3 += ','
        if len(anch_list) > 18:
            for a in range(18,24):
                anchors4 += anch_list[a]
                if a<24: anchors4 += ','

        #file_updated = file_content.replace("{CLASSES}", str(classNum))
        #file_updated = file_updated.replace("{ANCHOR1}", str(anchors1))
        #file_updated = file_updated.replace("{ANCHOR2}", str(anchors2))
        #file_updated = file_updated.replace("{ANCHOR3}", str(anchors3))
        #if len(anch_list) >= 18:
        #    file_updated = file_updated.replace("{ANCHOR4}", str(anchors4))

        #cfg_file = cfg_name + '.yaml'

        exec_cmd = " cd {}\n python train.py \\\n    --data {} \\\n    --imgsz {} \\\n    --batch {} \\\n    --epochs 300 \\\n    --weights {}".format( \
            yolov5_home, os.path.join(cfgFolder, 'ds_yolov5.yaml'), cfgs[cfg_name][2], \
            cfgs[cfg_name][3],cfgs[cfg_name][1])


    else:
        with open(cfgs[cfg_name][0]) as file:
            file_content = file.read()
        file.close

        batch = cfgs[cfg_name][3]
        div = cfgs[cfg_name][4]

        if(cfg_name in ["yolov3-tiny", "yolov4-tiny", "yolo-fastest", "yolo-fastest-xl"]):
            anch = yolotiny_config[str(cfgs[cfg_name][2])]

        else:
            anch = yolo_config[str(cfgs[cfg_name][2])]

        file_updated = file_content.replace("{BATCH}", str(batch))
        file_updated = file_updated.replace("{SUBDIVISIONS}", str(div))
        file_updated = file_updated.replace("{SIZE}", str(cfgs[cfg_name][2]))
        file_updated = file_updated.replace("{FILTERS}", str(filterNum))
        file_updated = file_updated.replace("{CLASSES}", str(classNum))
        file_updated = file_updated.replace("{ANCHORS}", anch)

        cfg_file = cfg_name +'.cfg'

        if (cfg_name in ["yolo-fastest", "yolo-fastest-xl"]):
            exec_cmd = "{}/darknet detector train \\\n    {} \\\n    {} \\\n    {} \\\n    -dont_show \\\n    -mjpeg_port 8090 \\\n    -clear \\\n    -gpus 0".format(\
                yolofastest_home, os.path.join(cfgFolder,'obj.data'), os.path.join(cfgFolder,cfg_name+'.cfg'), os.path.join(pwd,cfgs[cfg_name][1]))

        else:
            exec_cmd = "{}/darknet detector train \\\n    {} \\\n    {} \\\n    {} \\\n    -dont_show \\\n    -mjpeg_port 8090 \\\n    -clear \\\n    -gpus 0".format(\
                dark_home, os.path.join(cfgFolder,'obj.data'), os.path.join(cfgFolder,cfg_name+'.cfg'), os.path.join(pwd,cfgs[cfg_name][1]))


    file = open(os.path.join(cfgFolder, cfg_file), "w")
    file.write(file_updated)
    file.close


    print("-----------------------------------------")
    print(" Command for training {} model".format(cfg_name.upper()))
    print("-----------------------------------------")
    print(exec_cmd)
    print('')

    tfile.write("---------------------------------------------------------------------\n")
    tfile.write(" [{} model] \n".format(cfg_name.upper()))
    tfile.write(exec_cmd + '\n\n')

tfile.close()
