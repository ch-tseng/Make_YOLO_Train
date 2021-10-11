import os,sys

#classList = { "D00":0, "D10":1, "D20":2, "D21":3, "D30":4, "D31":5, "D40":6, "D41":7, "D42":8, "D99":9 }
classList = { "D20":0, "D21":1 }
cfgFolder = "/WORKING/ROAD_D20_D21/dataset/aug_1010/cfg_train"

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
    '416': "22, 19,  32, 55,  62, 30,  74, 72,  48,134, 194, 65, 113,130, 141,222, 285,194",
    '512': "27, 23,  40, 65,  79, 36,  97, 86,  57,157, 130,162, 262,103, 173,267, 351,271",
    '608': "32, 28,  46, 81,  90, 43, 109,105,  70,197, 284, 95, 165,191, 208,326, 419,281",
    '640': "26, 17,  46, 23,  45, 40,  77, 37,  77, 73, 125, 59, 146,104, 186,160, 298,204",
    '960': "50, 44,  73,127, 142, 68, 171,165, 110,309, 448,149, 260,301, 326,513, 658,447",
    '1280': "57, 52, 125, 79,  73,171, 278,104, 164,184, 139,410, 268,274, 664,184, 292,549, 450,394, 494,724, 980,593",
    '1536': "81, 70, 119,203, 232,110, 283,262, 177,487, 413,486, 768,265, 527,817, 1079,757"
}

yolotiny_config = {
    '320': "15,  9,  27, 15,  41, 26,  63, 42,  85, 69, 139, 99",
    '416': "19, 12,  35, 19,  53, 34,  82, 55, 110, 90, 181,128",
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
    "yolov5s": ["cfg/yolov5s.yaml", "pretrained/yolov5s.pt", 640, 42, 1],
    "yolov5m": ["cfg/yolov5m.yaml", "pretrained/yolov5m.pt", 640, 24, 1],
    "yolov5l": ["cfg/yolov5l.yaml", "pretrained/yolov5l.pt", 640, 12, 1],
    "yolov5x": ["cfg/yolov5x.yaml", "pretrained/yolov5x.pt", 640, 8, 1],
    "yolov5s-p6": ["cfg/yolov5s6.yaml", "pretrained/yolov5s6.pt", 1280, 48, 1],
    "yolov5m-p6": ["cfg/yolov5m6.yaml", "pretrained/yolov5m6.pt", 1280, 32, 1],
    "yolov5l-p6": ["cfg/yolov5l6.yaml", "pretrained/yolov5l6.pt", 1280, 24, 1],
    "yolov5x-p6": ["cfg/yolov5x6.yaml", "pretrained/yolov5x6.pt", 1280, 12, 1],
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
    print('test', cfg_name)
    with open(cfgs[cfg_name][0]) as file:
        file_content = file.read()
    file.close

    if(cfg_name[:6] == 'yolov5'):
        print("TEST", str(cfgs[cfg_name][2]))
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

        file_updated = file_content.replace("{CLASSES}", str(classNum))
        file_updated = file_updated.replace("{ANCHOR1}", str(anchors1))
        file_updated = file_updated.replace("{ANCHOR2}", str(anchors2))
        file_updated = file_updated.replace("{ANCHOR3}", str(anchors3))
        if len(anch_list) >= 18:
            file_updated = file_updated.replace("{ANCHOR4}", str(anchors4))

        cfg_file = cfg_name + '.yaml'

        exec_cmd = " cd {}\n python train.py \\\n    --data {} \\\n    --cfg {} \\\n    --batch {} \\\n    --epochs 300 \\\n    --noautoanchor    \\\n    --weights {}".format( \
            yolov5_home, os.path.join(cfgFolder, 'ds_yolov5.yaml'), os.path.join(cfgFolder,cfg_name+'.yaml'), \
            cfgs[cfg_name][3],os.path.join(pwd,cfgs[cfg_name][1]))


    else:
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
