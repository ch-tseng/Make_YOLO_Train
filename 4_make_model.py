import os,sys

classList = { "D00":0, "D10":1, "D20":2, "D21":3, "D30":4, "D31":5, "D40":6, "D41":7, "D42":8, "D99":9 }
cfgFolder = "/WORKING/WORKS/ROAD_Defects/dataset/aug_20211112/cfg_train"

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
    '320_6': "19, 16,  38, 38,  47, 90,  97, 53,  97,132, 180,170",
    '416_6': "24, 21,  49, 49,  61,116, 127, 68, 126,172, 233,221",
    '320_9': "17, 14,  25, 42,  47, 23,  58, 55,  38,101, 149, 51,  87,102, 112,172, 226,156",
    '416_9': "22, 19,  33, 53,  65, 30,  79, 70,  47,126, 107,133, 211, 83, 144,218, 286,226",
    '512_9': "27, 23,  40, 65,  78, 36,  97, 85,  59,154, 131,162, 260,101, 174,265, 341,282",
    '608_9': "31, 28,  47, 80,  89, 44, 110,104,  72,193, 282, 97, 165,194, 212,327, 430,297",
    '640_9': "34, 29,  50, 83,  98, 46, 119,109,  74,199, 170,203, 321,116, 220,337, 439,341",
    '960_9': "43, 44,  93, 63,  84,148, 194, 98, 117,314, 205,214, 494,211, 286,413, 520,558",
    '1280_9': "67, 58, 100,162, 199, 92, 243,216, 146,387, 329,410, 649,256, 442,671, 879,695",
    '640_12': "28, 26,  62, 39,  37, 84, 136, 53,  81, 92,  71,202, 135,136, 321, 91, 146,271, 228,200, 249,366, 489,302",
    '960_12': "42, 38,  91, 58,  53,126, 114,137, 201, 78, 111,317, 189,190, 440,150, 253,309, 302,505, 516,332, 562,614",
    '1280_12': "56, 52, 122, 79,  73,170, 271,103, 163,183, 143,403, 270,271, 640,182, 291,541, 455,399, 498,733, 978,604",
    '1536_12': "65, 66, 140, 89, 117,209, 269,136, 257,287, 154,484, 687,190, 433,391, 328,622, 854,479, 550,794, 1027,956"
}

#yolotiny_config = {
#    '320': "7, 17,  11, 23,  20, 37,  53, 39,  85, 48, 170,105",
#    '416': "9, 22,  15, 30,  27, 48,  68, 51, 110, 62, 221,137",
#}

#---------------------------------------------------------------------

classNum = len(classList)
filterNum = (classNum + 5) * 3

#default size:
#    yolov3: 608, yolov3-tiny:416, yolov4:608, yolov4-tiny:416, yolo-fastest:320,
#    yolo-fastest-xl:320, yolov4x-mish:640, yolov4-csp:512, yolov4-cspx-p7:1536
#    [CFG FILE, PRE-TRAINED WEIGHTS, SIZE, BATCH, DIVISION-BATCH]
cfgs = {
    "yolov3": ["cfg/yolov3.cfg", "pretrained/darknet53.conv.74", '608_9', 64, 32],
    "yolov3-tiny": ["cfg/yolov3-tiny.cfg", "pretrained/yolov3-tiny.conv.15", '416_6', 70, 2],
    "yolov3-spp": ["cfg/yolov3-spp.cfg", "pretrained/yolov3-spp.weights", '608_9', 64, 32],
    "yolov4": ["cfg/yolov4.cfg", "pretrained/yolov4.conv.137", '608_9', 64, 64],
    "yolov4-tiny": ["cfg/yolov4-tiny.cfg", "pretrained/yolov4-tiny.conv.29", '416_6', 72, 1],
    "yolo-fastest": ["cfg/yolo-fastest-1.1.cfg", "pretrained/yolo-fastest.conv.109", '320_6', 160, 2],
    "yolo-fastest-xl": ["cfg/yolo-fastest-1.1-xl.cfg", "pretrained/yolo-fastest-xl.conv.109", '320_6', 120, 2],
    "yolov4x-mish": ["cfg/yolov4x-mish.cfg", "pretrained/yolov4x-mish.conv.166", '640_9', 64, 64],
    "yolov4-csp": ["cfg/yolov4-csp.cfg", "pretrained/yolov4-csp.conv.142", '512_9', 64, 64],
    "yolov4-cspx-p7": ["cfg/cspx-p7-mish.cfg", "pretrained/cspx-p7-mish_hp.344.conv", '1536_12', 64, 64],
    "yolov5n": ["cfg/yolov5n.yaml", "yolov5n.pt", '640_9', 128, 1],
    "yolov5s": ["cfg/yolov5s.yaml", "yolov5s.pt", '640_9', 64, 1],
    "yolov5m": ["cfg/yolov5m.yaml", "yolov5m.pt", '640_9', 24, 1],
    "yolov5l": ["cfg/yolov5l.yaml", "yolov5l.pt", '640_9', 12, 1],
    "yolov5x": ["cfg/yolov5x.yaml", "yolov5x.pt", '640_9', 8, 1],
    "yolov5s-p6": ["cfg/yolov5s6.yaml", "yolov5s6.pt", '1280_12', 48, 1],
    "yolov5m-p6": ["cfg/yolov5m6.yaml", "yolov5m6.pt", '1280_12', 32, 1],
    "yolov5l-p6": ["cfg/yolov5l6.yaml", "yolov5l6.pt", '1280_12', 24, 1],
    "yolov5x-p6": ["cfg/yolov5x6.yaml", "yolov5x6.pt", '1280_12', 12, 1],
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
        anchors = yolo_config[cfgs[cfg_name][2]]
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

        with open(cfgs[cfg_name][0]) as file:
            file_content = file.read()
        file.close

        #file_content = open( cfgs[cfg_name][0], 'w')

        file_updated = file_content.replace("{CLASSES}", str(classNum))
        file_updated = file_updated.replace("{ANCHOR1}", str(anchors1))
        file_updated = file_updated.replace("{ANCHOR2}", str(anchors2))
        file_updated = file_updated.replace("{ANCHOR3}", str(anchors3))
        if len(anch_list) >= 18:
            file_updated = file_updated.replace("{ANCHOR4}", str(anchors4))

        cfg_file = cfg_name + '.yaml'

        exec_cmd = " cd {}\n python train.py \\\n    --data {} \\\n    --imgsz {} \\\n    --batch {} \\\n    --epochs 300 \\\n    --weights {}".format( \
            yolov5_home, os.path.join(cfgFolder, 'ds_yolov5.yaml'), cfgs[cfg_name][2].split('_')[0], \
            cfgs[cfg_name][3],cfgs[cfg_name][1])


    else:
        with open(cfgs[cfg_name][0]) as file:
            file_content = file.read()
        file.close

        batch = cfgs[cfg_name][3]
        div = cfgs[cfg_name][4]

        #if(cfg_name in ["yolov3-tiny", "yolov4-tiny", "yolo-fastest", "yolo-fastest-xl"]):
        #    anch = yolotiny_config[str(cfgs[cfg_name][2])]

        #else:
        anch = yolo_config[str(cfgs[cfg_name][2])]

        file_updated = file_content.replace("{BATCH}", str(batch))
        file_updated = file_updated.replace("{SUBDIVISIONS}", str(div))
        file_updated = file_updated.replace("{SIZE}", str(cfgs[cfg_name][2].split('_')[0]))
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
