import os,sys

project_name = "M2022_CrowdedHuman"
classList = { "person_head":0, "person_vbox":1 }

cfgFolder = "/data/ai_models/training/{}/cfg_train/".format(project_name)
weights_save = "/data/ai_models/training/{}/weights/".format(project_name)

'''
classList = { 'D00':0, 'D10':1, 'D20':2, 'D21':3, 'D30':4 ,'D31':5, 'D40':6, 'D41':7, 'D42':8, 'D99':9 }
cfgFolder = "/WORKING/WORKS/road_defects_2022_04/aug_20220408/cfg_train"

classList = { "balaclava_ski_mask":0, "eyeglasses":1, "face_no_mask":2, "face_other_covering":3, "face_shield":4, \
              "face_with_mask":5, "face_with_mask_incorrect":6, "gas_mask":7, "goggles":8, "hair_net":9, "hat":10, \
              "helmet":11, "hijab_niqab":12, "hood":13, "mask_colorful":14, "mask_surgical":15, "other":16, \
              "scarf_bandana":17, "sunglasses":18, "turban":19 }
'''
dark_home = "/home/chtseng/frameworks/darknet"
yolofastest_home = "/home/chtseng/frameworks/darknet"
yolov5_home = "/data/ai_models/frameworks/yolov5"
yolov7_home = "/data/ai_models/frameworks/yolov7"
yolor_home = "/home/chtseng/frameworks/yolor"
current_path = os.getcwd()

cfgs = {
    "yolov7": [os.path.join(current_path,"cfg/yolov7/yolov7.yaml"), 'data/hyp.scratch.p5.yaml', '640_9', 32, 8, 3],
    "yolov7x": [os.path.join(current_path,"cfg/yolov7/yolov7x.yaml"), 'data/hyp.scratch.p5.yaml', '640_9', 32, 8, 3],
    "yolov7w6": [os.path.join(current_path,"cfg/yolov7/yolov7w6.yaml"), 'data/hyp.scratch.p6.yaml', '1280_12', 12, 8, 4],
    "yolov7-tiny": [os.path.join(current_path,"cfg/yolov7/yolov7-tiny.yaml"), 'data/hyp.scratch.tiny.yaml', '640_9', 32, 1, 3],
    "yolov7e6": [os.path.join(current_path,"cfg/yolov7/yolov7e6.yaml"), 'data/hyp.scratch.p6.yaml', '1280_12', 6, 8, 4],
    "yolov7d6": [os.path.join(current_path,"cfg/yolov7/yolov7d6.yaml"), 'data/hyp.scratch.p6.yaml', '1280_12', 5, 8, 4],
    "yolov4": ["cfg/yolov4/yolov4.cfg", "pretrained/yolov4/yolov4.conv.137", '608_9', 64, 64, 3],
    "yolov4-tiny": ["cfg/yolov4/yolov4-tiny.cfg", "pretrained/yolov4/yolov4-tiny.conv.29", '416_6', 72, 1, 3],
    "yolo-fastest": ["cfg/yolo-fastest/yolo-fastest-1.1.cfg", "pretrained/yolo-fastest/yolo-fastest.conv.109", '320_6', 160, 2, 3],
    "yolo-fastest-xl": ["cfg/yolo-fastest/yolo-fastest-1.1-xl.cfg", "pretrained/yolo-fastest/yolo-fastest-xl.conv.109", '320_6', 120, 2, 3],
    "yolov5n": ["cfg/yolov5/yolov5n.yaml", "yolov5n.pt", '640_9', 128, 1, 3],
    "yolov5s": ["cfg/yolov5/yolov5s.yaml", "yolov5s.pt", '640_9', 64, 1, 3],
    "yolov5x-p6": ["cfg/yolov5/yolov5x6.yaml", "yolov5x6.pt", '1280_12', 12, 1, 3],
}

yolo_config = {
    '320_6': "3,  5,  11, 16,  21, 44,  88, 42,  38,108, 106,188",
    '416_6': "5,  6,  14, 21,  27, 58, 114, 55,  49,141, 137,245",
    '320_9': "9, 12,  18, 23,  28, 46,  51, 68,  82, 84,  60,189, 142,113, 110,231, 238,232",
    '416_9': "11, 14,  21, 28,  42, 42,  28, 71,  63, 92, 105,110, 184,152, 103,279, 276,307",
    '512_9': "15, 19,  29, 37,  45, 74,  81,110, 131,134,  96,302, 227,181, 177,370, 381,371",
    '608_9': "5,  7,  15, 21,  23, 58,  55, 36,  47,113,  72,217, 188, 99, 129,357, 326,350",
    '640_9': "6,  7,  16, 22,  24, 53,  71, 47,  40,119,  78,178, 244,114, 112,346, 281,427",
    '960_9': "28, 34,  50, 72,  98,102,  81,194, 179,193, 225,295, 428,344, 242,657, 643,707",
    '1280_9': "37, 47,  73, 92, 112,184, 204,274, 328,336, 241,754, 567,451, 442,925, 952,929",
    '640_12': "17, 21,  30, 39,  55, 56,  38,101,  85,105,  90,188, 149,141, 190,223, 127,417, 335,221, 236,463, 486,470",
    '960_12': "25, 31,  45, 59,  82, 85,  57,152, 128,157, 135,283, 224,212, 285,334, 191,625, 503,332, 354,695, 729,705",
    '1280_12': "10, 12,  25, 35,  36, 90,  75, 53,  68,154, 194,114,  95,286, 222,310, 138,541, 637,250, 274,749, 613,905",
    '1536_12': "40, 50,  71, 94, 132,135,  92,243, 204,251, 216,452, 358,339, 456,534, 306,1001, 804,532, 566,1111, 1166,1128"
}

#yolotiny_config = {
#    '320': "7, 17,  11, 23,  20, 37,  53, 39,  85, 48, 170,105",
#    '416': "9, 22,  15, 30,  27, 48,  68, 51, 110, 62, 221,137",
#}

#---------------------------------------------------------------------

classNum = len(classList)
#filterNum = (classNum + 5) * 3

#    yolov3: 608, yolov3-tiny:416, yolov4:608, yolov4-tiny:416, yolo-fastest:320,
#    yolo-fastest-xl:320, yolov4x-mish:640, yolov4-csp:512, yolov4-cspx-p7:1536
#    [CFG FILE, PRE-TRAINED WEIGHTS, SIZE, BATCH, DIVISION-BATCH, MASKS]
cfgs_total = {
    "yolov7": [os.path.join(current_path,"cfg/yolov7/yolov7.yaml"), os.path.join(current_path, 'cfg/yolov7/hyp.scratch.p5.yaml'), '640_9', 32, 8, 3],
    "yolov7x": [os.path.join(current_path,"cfg/yolov7/yolov7x.yaml"), os.path.join(current_path,'cfg/yolov7/hyp.scratch.p5.yaml'), '640_9', 32, 8, 3],
    "yolov7w6": [os.path.join(current_path,"cfg/yolov7/yolov7w6.yaml"), os.path.join(current_path,'cfg/yolov7/hyp.scratch.p6.yaml'), '1280_12', 32, 8, 4],
    "yolov7-tiny": [os.path.join(current_path,"cfg/yolov7/yolov7-tiny.yaml"), os.path.join(current_path,'cfg/yolov7/hyp.scratch.tiny.yaml'), '640_9', 32, 1, 3],
    "yolov7e6": [os.path.join(current_path,"cfg/yolov7/yolov7e6.yaml"), os.path.join(current_path,'cfg/yolov7/hyp.scratch.p6.yaml'), '1280_12', 32, 8, 4],
    "yolov7e6e": [os.path.join(current_path,"cfg/yolov7/yolov7e6e.yaml"), os.path.join(current_path,'cfg/yolov7/hyp.scratch.p6.yaml'), '1280_12', 32, 8, 4],
    "yolov7d6": [os.path.join(current_path,"cfg/yolov7/yolov7d6.yaml"), os.path.join(current_path,'cfg/yolov7/hyp.scratch.p6.yaml'), '1280_12', 32, 8, 4],
    "yolov3": ["cfg/yolov3/yolov3.cfg", "pretrained/yolov3/darknet53.conv.74", '608_9', 64, 32, 3],
    "yolov3-tiny": ["cfg/yolov3/yolov3-tiny.cfg", "pretrained/yolov3/yolov3-tiny.conv.15", '416_6', 66, 2, 3],
    "yolov3-spp": ["cfg/yolov3/yolov3-spp.cfg", "pretrained/yolov3/yolov3-spp.weights", '608_9', 64, 32, 3],
    "yolov4": ["cfg/yolov4/yolov4.cfg", "pretrained/yolov4/yolov4.conv.137", '608_9', 64, 64, 3],
    "yolov4-tiny": ["cfg/yolov4/yolov4-tiny.cfg", "pretrained/yolov4/yolov4-tiny.conv.29", '416_6', 72, 1, 3],
    "yolo-fastest": ["cfg/yolo-fastest/yolo-fastest-1.1.cfg", "pretrained/yolo-fastest/yolo-fastest.conv.109", '320_6', 160, 2, 3],
    "yolo-fastest-xl": ["cfg/yolo-fastest/yolo-fastest-1.1-xl.cfg", "pretrained/yolo-fastest/yolo-fastest-xl.conv.109", '320_6', 120, 2, 3],
    "yolov4x-mish": ["cfg/yolov4/yolov4x-mish.cfg", '', '640_9', 64, 64, 3],
    "yolov4-csp": ["cfg/yolov4/yolov4-csp.cfg", '', '640_9', 64, 64, 3],
    "yolov4-p5": ["cfg/yolov4/yolov4-p5.cfg", 'pretrained/yolov4-p5.conv.232', '896_12', 64, 64, 4],
    "yolov4-p6": ["cfg/yolov4/yolov4-p6.cfg", 'pretrained/yolov4-p6.conv.289', '1280_12', 64, 64, 4],
    "yolov5n": ["cfg/yolov5/yolov5n.yaml", "yolov5n.pt", '640_9', 128, 1, 3],
    "yolov5s": ["cfg/yolov5/yolov5s.yaml", "yolov5s.pt", '640_9', 64, 1, 3],
    "yolov5m": ["cfg/yolov5/yolov5m.yaml", "yolov5m.pt", '640_9', 24, 1, 3],
    "yolov5l": ["cfg/yolov5/yolov5l.yaml", "yolov5l.pt", '640_9', 12, 1, 3],
    "yolov5x": ["cfg/yolov5/yolov5x.yaml", "yolov5x.pt", '640_9', 8, 1, 3],
    "yolov5s-p6_640": ["cfg/yolov5/yolov5s6.yaml", "yolov5s6.pt", '640_12', 48, 1, 3],
    "yolov5s-p6_960": ["cfg/yolov5/yolov5s6.yaml", "yolov5s6.pt", '960_12', 48, 1, 3],
    "yolov5s-p6": ["cfg/yolov5/yolov5s6.yaml", "yolov5s6.pt", '1280_12', 48, 1, 3],
    "yolov5m-p6": ["cfg/yolov5/yolov5m6.yaml", "yolov5m6.pt", '1280_12', 32, 1, 3],
    "yolov5l-p6": ["cfg/yolov5/yolov5l6.yaml", "yolov5l6.pt", '1280_12', 24, 1, 3],
    "yolov5x-p6": ["cfg/yolov5/yolov5x6.yaml", "yolov5x6.pt", '1280_12', 12, 1, 3],
    "yolor_csp": ["cfg/yolor/yolor_csp.cfg", "pretrained/yolor/yolor_csp.pt", "640_9", 64, 16, 3],
    "yolor_csp_star": ["cfg/yolor/yolor_csp.cfg", "pretrained/yolor/yolor_csp_star.pt", "640_9", 64, 16, 3],
    "yolor_csp_x": ["cfg/yolor/yolor_csp_x.cfg", "pretrained/yolor/yolor_csp_x.pt", "640_9", 66, 22, 3],
    "yolor_csp_x_star": ["cfg/yolor/yolor_csp_x.cfg", "pretrained/yolor/yolor_csp_x_star.pt", "640_9", 66, 33, 3],
    "yolor_p6": ["cfg/yolor/yolor_p6.cfg", "pretrained/yolor/yolor_p6.pt", "1280_12", 66, 66, 3],
    "yolor_w6":  ["cfg/yolor/yolor_w6.cfg", "pretrained/yolor/yolor_w6.pt", "1280_12", 66, 66, 3],
    "yolor_yolov4_csp":  ["cfg/yolor/yolov4_csp.cfg", '', "640_9", 66, 33, 3],
    "yolor_yolov4_csp_x":  ["cfg/yolor/yolov4_csp_x.cfg", '', "640_9", 66, 66, 3],
    "yolor_yolov4_p6": ["cfg/yolor/yolov4_p6.cfg", '', "1280_16", 66,66, 4],
    "yolor_yolov4_p7": ["cfg/yolor/yolov4_p7.cfg", '', "1536_20", 66,66, 4]
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

#make dataset yaml for YOLOV7
with open('cfg/data_yolov7.yaml') as file:
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

file = open(os.path.join(cfgFolder, 'ds_yolov7.yaml'), "w")
file.write(dataset_content)
file.close

#---- end

#make dataset yaml for YOLOR
with open('cfg/yolor_data.yaml') as file:
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

file = open(os.path.join(cfgFolder, 'ds_yolor.yaml'), "w")
file.write(dataset_content)
file.close
#---- end


tfile = open( os.path.join(cfgFolder, 'train_cmd.txt'), 'w')

for cfg_name in cfgs:
    if(cfg_name[:6] in ['yolov5', 'yolov7'] ):
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
        path_project = os.path.join( weights_save, project_name )
        path_project_name = os.path.join(path_project, cfg_name)

        if cfg_name[:6] == 'yolov5':
            exec_cmd = " cd {}\n $(which python) train.py \\\n    --data {} \\\n    --imgsz {} \\\n    --batch {} \\\n    --epochs 300 \\\n    --project {} \\\n    --name {}_ \\\n    --device {} \\\n    --weights {}".format( \
                yolov5_home, os.path.join(cfgFolder, 'ds_yolov5.yaml'), cfgs[cfg_name][2].split('_')[0], \
                cfgs[cfg_name][3], path_project, path_project_name, '{GPU}', cfgs[cfg_name][1])

        elif cfg_name[:6] == 'yolov7':
            if '6' in cfg_name:
                trainfile = 'train_aux.py'
            else:
                trainfile = 'train.py'

            exec_cmd = " cd {}\n $(which python) {} \\\n   --workers {} --device {} --batch-size {}\\\n --data {}\\\n --img {} {}\\\n --cfg {}\\\n --weights '' --project {}\\\n --name {}_ \\\n --hyp {}".format( \
                yolov7_home, trainfile, cfgs[cfg_name][4] , '{GPU}', cfgs[cfg_name][3], \
                os.path.join(cfgFolder, 'ds_yolov7.yaml'), cfgs[cfg_name][2].split('_')[0], \
                cfgs[cfg_name][2].split('_')[0], os.path.join(cfgFolder, cfg_file), path_project, cfg_name, cfgs[cfg_name][1] )
        #exec_cmd += " --freeze 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14"

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
        filterNum = (classNum + 5) * cfgs[cfg_name][5]

        file_updated = file_content.replace("{BATCH}", str(batch))
        file_updated = file_updated.replace("{SUBDIVISIONS}", str(div))
        file_updated = file_updated.replace("{SIZE}", str(cfgs[cfg_name][2].split('_')[0]))
        file_updated = file_updated.replace("{FILTERS}", str(filterNum))
        file_updated = file_updated.replace("{CLASSES}", str(classNum))
        file_updated = file_updated.replace("{ANCHORS}", anch)

        cfg_file = cfg_name +'.cfg'

        if (cfg_name in ["yolo-fastest", "yolo-fastest-xl"]):
            exec_cmd = "{}/darknet detector train \\\n    {} \\\n    {} \\\n    {} \\\n    -dont_show \\\n    -mjpeg_port {} \\\n    -clear \\\n    -gpus {}".format(\
                yolofastest_home, os.path.join(cfgFolder,'obj.data'), os.path.join(cfgFolder,cfg_name+'.cfg'), os.path.join(pwd,cfgs[cfg_name][1]), '{DARKNET_PORT}', '{GPU}')

        elif(cfg_name[:5] == 'yolor'):
            if cfgs[cfg_name][2].split('_')[0] in ['640','512']:
                hyp_file = os.path.join(pwd, 'cfg', 'yolor', 'hyp.scratch.640.yaml')
            elif cfgs[cfg_name][2].split('_')[0] == '1280':
                hyp_file = os.path.join(pwd, 'cfg', 'yolor', 'hyp.scratch.1280.yaml')

            if cfgs[cfg_name][1] != '':
                weights_file = os.path.join(pwd,cfgs[cfg_name][1])
            else:
                weights_file = ''

            exec_cmd = "cd {}\\\n $(which python) train.py --batch-size {} --img {} {} --data {} --cfg {} --weights '{}' --device {} --name {} --hyp {} --epochs {}".format(\
                yolor_home, cfgs[cfg_name][3], cfgs[cfg_name][2].split('_')[0], cfgs[cfg_name][2].split('_')[0], os.path.join(cfgFolder, 'ds_yolor.yaml'), \
                os.path.join(cfgFolder,cfg_name+'.cfg'), weights_file, '{GPU}', cfg_name, hyp_file, 300)

        else:
            exec_cmd = "{}/darknet detector train \\\n    {} \\\n    {} \\\n    {} \\\n    -dont_show \\\n    -mjpeg_port {} \\\n    -clear \\\n    -gpus {}".format(\
                dark_home, os.path.join(cfgFolder,'obj.data'), os.path.join(cfgFolder,cfg_name+'.cfg'), os.path.join(pwd,cfgs[cfg_name][1]), '{DARKNET_PORT}', '{GPU}' )


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
