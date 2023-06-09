import shutil
import os,sys
from configparser import ConfigParser
import ast

cfg = ConfigParser()
cfg.read("config.ini",encoding="utf-8")

project_name = cfg.get("global", "project_name")
baseFolder = cfg.get("global", "baseFolder")
classList = ast.literal_eval(cfg.get("global", "classList"))
yolo_config = ast.literal_eval(cfg.get("models", "yolo_anchors"))

cfgFolder = os.path.join(baseFolder, project_name, "cfg_train")
weights_save = os.path.join(baseFolder, project_name, "weights")

'''
classList = { 'D00':0, 'D10':1, 'D20':2, 'D21':3, 'D30':4 ,'D31':5, 'D40':6, 'D41':7, 'D42':8, 'D99':9 }
cfgFolder = "/WORKING/WORKS/road_defects_2022_04/aug_20220408/cfg_train"

classList = { "balaclava_ski_mask":0, "eyeglasses":1, "face_no_mask":2, "face_other_covering":3, "face_shield":4, \
              "face_with_mask":5, "face_with_mask_incorrect":6, "gas_mask":7, "goggles":8, "hair_net":9, "hat":10, \
              "helmet":11, "hijab_niqab":12, "hood":13, "mask_colorful":14, "mask_surgical":15, "other":16, \
              "scarf_bandana":17, "sunglasses":18, "turban":19 }
'''

dark_home = cfg.get("yoloPath", "yolov4_home")
yolofastest_home = cfg.get("yoloPath", "yolofastest_home")
yolov5_home = cfg.get("yoloPath", "yolov5_home")
yolov7_home = cfg.get("yoloPath", "yolov7_home")
yolor_home = cfg.get("yoloPath", "yolor_home")
fastestdet_home = cfg.get("yoloPath", "fastestdet_home")
current_path = os.getcwd()

baseFolder = baseFolder.replace("\\", '/')
dark_home = dark_home.replace("\\", '/')
yolofastest_home = yolofastest_home.replace("\\", '/')
yolov5_home = yolov5_home.replace("\\", '/')
yolov7_home = yolov7_home.replace("\\", '/')
yolor_home = yolor_home.replace("\\", '/')
fastestdet_home = fastestdet_home.replace("\\", '/')
cfgFolder = cfgFolder.replace("\\", '/')
weights_save = weights_save.replace("\\", '/')
train_models =  ast.literal_eval(cfg.get("models", "train_models"))

'''
cfgs = {
    "yolov3": ["cfg/yolov3/yolov3.cfg", "pretrained/yolov3/darknet53.conv.74", '608_9', 64, 32, 3],
    "yolov3-tiny": ["cfg/yolov3/yolov3-tiny.cfg", "pretrained/yolov3/yolov3-tiny.conv.15", '416_6', 66, 2, 3],
    "yolov4": ["cfg/yolov4/yolov4.cfg", "pretrained/yolov4/yolov4.conv.137", '608_9', 64, 64, 3],
    "yolov4-tiny": ["cfg/yolov4/yolov4-tiny.cfg", "pretrained/yolov4/yolov4-tiny.conv.29", '416_6', 72, 1, 3],
    "yolov5s": ["cfg/yolov5/yolov5s.yaml", "yolov5s.pt", '640_9', 64, 1, 3],
    "yolov5x6": ["cfg/yolov5/yolov5x6.yaml", "yolov5x6.pt", '1280_12', 12, 1, 3],
    "yolov7": [os.path.join(current_path,"cfg/yolov7/yolov7.yaml"), os.path.join(current_path, 'cfg/yolov7/hyp.scratch.p5.yaml'), '640_9', 32, 8, 3],
    "yolov7-tiny": [os.path.join(current_path,"cfg/yolov7/yolov7-tiny.yaml"), os.path.join(current_path,'cfg/yolov7/hyp.scratch.tiny.yaml'), '640_9', 32, 1, 3]
}
'''
#---------------------------------------------------------------------

classNum = len(classList)
#filterNum = (classNum + 5) * 3

#    yolov3: 608, yolov3-tiny:416, yolov4:608, yolov4-tiny:416, yolo-fastest:320,
#    yolo-fastest-xl:320, yolov4x-mish:640, yolov4-csp:512, yolov4-cspx-p7:1536
#    [CFG FILE, PRE-TRAINED WEIGHTS, SIZE, BATCH, DIVISION-BATCH, MASKS]
cfgs_total = {
    "yolov8n": ['','','640_9',-1,9,3],
    "yolov8s": ['','','640_9',-1,9,3],
    "yolov8m": ['','','640_9',-1,9,3],
    "yolov8l": ['','','640_9',-1,9,3],
    "yolov8x": ['','','640_9',-1,9,3],
    "yolov7": [os.path.join(current_path,"cfg/yolov7/yolov7.yaml"), os.path.join(current_path, 'cfg/yolov7/hyp.scratch.p5.yaml'), '640_9', 32, 8, 3],
    "yolov7x": [os.path.join(current_path,"cfg/yolov7/yolov7x.yaml"), os.path.join(current_path,'cfg/yolov7/hyp.scratch.p5.yaml'), '640_9', 32, 8, 3],
    "yolov7w6": [os.path.join(current_path,"cfg/yolov7/yolov7w6.yaml"), os.path.join(current_path,'cfg/yolov7/hyp.scratch.p6.yaml'), '1280_12', 32, 8, 4],
    "yolov7-tiny": [os.path.join(current_path,"cfg/yolov7/yolov7-tiny.yaml"), os.path.join(current_path,'cfg/yolov7/hyp.scratch.tiny.yaml'), '640_9', 32, 1, 3],
    "yolov7e6": [os.path.join(current_path,"cfg/yolov7/yolov7e6.yaml"), os.path.join(current_path,'cfg/yolov7/hyp.scratch.p6.yaml'), '1280_12', 32, 8, 4],
    "yolov7e6e": [os.path.join(current_path,"cfg/yolov7/yolov7e6e.yaml"), os.path.join(current_path,'cfg/yolov7/hyp.scratch.p6.yaml'), '1280_12', 32, 8, 4],
    "yolov7d6": [os.path.join(current_path,"cfg/yolov7/yolov7d6.yaml"), os.path.join(current_path,'cfg/yolov7/hyp.scratch.p6.yaml'), '1280_12', 32, 8, 3],
    "darknet-yolov7-tiny": [os.path.join(current_path,"cfg/darknet_yolov7/yolov7-tiny.cfg"), 'pretrained/darknet_yolov7/yolov7-tiny.conv.87', '416_9', 96, 1, 3],
    "darknet-yolov7": [os.path.join(current_path,"cfg/darknet_yolov7/yolov7.cfg"), 'pretrained/darknet_yolov7/yolov7.weights', '640_9', 32, 8, 3],
    "darknet-yolov7x": [os.path.join(current_path,"cfg/darknet_yolov7/yolov7x.cfg"), 'pretrained/darknet_yolov7/yolov7x.weights', '640_9', 32, 8, 3],
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
    "yolov5n": ["cfg/yolov5/yolov5n.yaml", "yolov5n.pt", '640_9', -1, 1, 3],
    "yolov5s": ["cfg/yolov5/yolov5s.yaml", "yolov5s.pt", '640_9', -1, 1, 3],
    "yolov5m": ["cfg/yolov5/yolov5m.yaml", "yolov5m.pt", '640_9', -1, 1, 3],
    "yolov5l": ["cfg/yolov5/yolov5l.yaml", "yolov5l.pt", '640_9', -1, 1, 3],
    "yolov5x": ["cfg/yolov5/yolov5x.yaml", "yolov5x.pt", '640_9', -1, 1, 3],
    "yolov5s6_640": ["cfg/yolov5/yolov5s6.yaml", "yolov5s6.pt", '640_12', -1, 1, 3],
    "yolov5s6_960": ["cfg/yolov5/yolov5s6.yaml", "yolov5s6.pt", '960_12', -1, 1, 3],
    "yolov5s6": ["cfg/yolov5/yolov5s6.yaml", "yolov5s6.pt", '1280_12', -1, 1, 3],
    "yolov5m6": ["cfg/yolov5/yolov5m6.yaml", "yolov5m6.pt", '1280_12', -1, 1, 3],
    "yolov5l6": ["cfg/yolov5/yolov5l6.yaml", "yolov5l6.pt", '1280_12', -1, 1, 3],
    "yolov5x6": ["cfg/yolov5/yolov5x6.yaml", "yolov5x6.pt", '1280_12', -1, 1, 3],
    "yolor_csp": ["cfg/yolor/yolor_csp.cfg", "pretrained/yolor/yolor_csp.pt", "640_9", 64, 16, 3],
    "yolor_csp_star": ["cfg/yolor/yolor_csp.cfg", "pretrained/yolor/yolor_csp_star.pt", "640_9", 64, 16, 3],
    "yolor_csp_x": ["cfg/yolor/yolor_csp_x.cfg", "pretrained/yolor/yolor_csp_x.pt", "640_9", 66, 22, 3],
    "yolor_csp_x_star": ["cfg/yolor/yolor_csp_x.cfg", "pretrained/yolor/yolor_csp_x_star.pt", "640_9", 66, 33, 3],
    "yolor_p6": ["cfg/yolor/yolor_p6.cfg", "pretrained/yolor/yolor_p6.pt", "1280_12", 66, 66, 3],
    "yolor_w6":  ["cfg/yolor/yolor_w6.cfg", "pretrained/yolor/yolor_w6.pt", "1280_12", 66, 66, 3],
    "yolor_yolov4_csp":  ["cfg/yolor/yolov4_csp.cfg", '', "640_9", 66, 33, 3],
    "yolor_yolov4_csp_x":  ["cfg/yolor/yolov4_csp_x.cfg", '', "640_9", 66, 66, 3],
    "yolor_yolov4_p6": ["cfg/yolor/yolov4_p6.cfg", '', "1280_16", 66,66, 4],
    "yolor_yolov4_p7": ["cfg/yolor/yolov4_p7.cfg", '', "1536_20", 66,66, 4],
    "sparse_yolov5n": ["cfg/yolov5/yolov5n.yaml", 'zoo:cv/detection/yolov5-n/pytorch/ultralytics/coco/base-none?recipe_type=transfer', '640_9', 300,1,1],
    "sparse_yolov5s": ["cfg/yolov5/yolov5s.yaml", 'zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94?recipe_type=transfer', '640_9', 300,1,1],
    "sparse_yolov5l": ["cfg/yolov5/yolov5l.yaml", 'zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned_quant-aggressive_95?recipe_type=transfer', '640_9', 300,1,1],
    "fastestdet": []
}

cfgs = {}
for m in train_models:
    if m in cfgs_total:
        cfgs.update( { m:cfgs_total[m] } )


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
    print(cfg_name)
    if cfg_name == 'fastestdet':
        exec_cmd = "cd {}\n $(which python) train.py --yaml {}".format(fastestdet_home, os.path.join(cfgFolder, 'fastestdet_config.yaml'))


    elif ('yolov5' in cfg_name) or ('yolov6' in cfg_name) or ('yolov8' in cfg_name) \
            or (('yolov7' in cfg_name) and ('darknet' not in cfg_name)):

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

        if not 'yolov8' in cfg_name:
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
            path_project = weights_save
            path_project_name = os.path.join(weights_save, cfg_name)

        if cfg_name[:6] == 'yolov5':
            exec_cmd = " cd {}\n $(which python) train.py \\\n    --data {} \\\n    --imgsz {} \\\n    --batch {} \\\n    --epochs 300 \\\n    --project {} \\\n    --name {}_ \\\n    --device {} \\\n    --weights {}".format( \
                yolov5_home, os.path.join(cfgFolder, 'ds_yolov5.yaml'), cfgs[cfg_name][2].split('_')[0], \
                cfgs[cfg_name][3], weights_save, path_project_name, '{GPU}', cfgs[cfg_name][1])

        elif cfg_name[:6] == 'yolov8':
            exec_cmd = " cd {}\n yolo task=detect mode=train data={} model={}.pt batch={} epochs=100 imgsz={} device={} workers=4".format( \
                       weights_save, os.path.join(cfgFolder, 'ds_yolov5.yaml'), cfg_name, cfgs[cfg_name][3], cfgs[cfg_name][2].split('_')[0], '{GPU}')

        elif cfg_name[:6] == 'yolov7':
            yolov7_ds_path = os.path.join(cfgFolder, cfg_name)
            if not os.path.exists(yolov7_ds_path):
                os.makedirs(yolov7_ds_path)

            #make dataset yaml for YOLOV7
            with open('cfg/data_yolov7.yaml') as file:
                dataset_content = file.read()
            file.close

            class_txt = '['
            for i, cname in enumerate(classList):
                class_txt += "'{}'".format(cname)
                if i<(len(classList)-1): class_txt += ', '
            class_txt += ']'

            dataset_content = dataset_content.replace("{TRAIN_LIST}", os.path.join(yolov7_ds_path,'train.txt'))
            dataset_content = dataset_content.replace("{TEST_LIST}", os.path.join(yolov7_ds_path,'test.txt'))
            dataset_content = dataset_content.replace("{CLASSES}", str(classNum))
            dataset_content = dataset_content.replace("{CLASS_LIST}", class_txt)

            file = open(os.path.join(yolov7_ds_path, 'ds_yolov7.yaml'), "w")
            file.write(dataset_content)
            file.close

            shutil.copy(os.path.join(cfgFolder,'train.txt'), os.path.join(yolov7_ds_path,'train.txt'))
            shutil.copy(os.path.join(cfgFolder,'test.txt'), os.path.join(yolov7_ds_path,'test.txt'))
            #---- end


            if '6' in cfg_name:
                trainfile = 'train_aux.py'
            else:
                trainfile = 'train.py'

            exec_cmd = " cd {}\n $(which python) {} \\\n   --workers {} --device {} --batch-size {}\\\n --data {}\\\n --img {} {}\\\n --cfg {}\\\n --weights '' --project {}\\\n --name {}_ \\\n --hyp {}".format( \
                yolov7_home, trainfile, cfgs[cfg_name][4] , '{GPU}', cfgs[cfg_name][3], \
                os.path.join(yolov7_ds_path,'ds_yolov7.yaml'), cfgs[cfg_name][2].split('_')[0], \
                cfgs[cfg_name][2].split('_')[0], os.path.join(cfgFolder, cfg_file), weights_save, cfg_name, cfgs[cfg_name][1] )
        #exec_cmd += " --freeze 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14"

        #SparseML for YOLOV5
        elif cfg_name[:14] == 'sparse_yolov5s':
            exec_cmd = "cd {}\n sparseml.yolov5.train \\\n   --epochs {} \\\n   --project {} \\\n   --name {}_ \\\n   --data {} --cfg {} \\\n   --weights {} \\\n   --hyp {} \\\n --recipe {}".format( \
                       yolov5_home, cfgs[cfg_name][3], weights_save, cfg_name,
                       os.path.join(cfgFolder, 'ds_yolov5.yaml'), os.path.join(cfgFolder, cfg_file), \
                       cfgs[cfg_name][1], \
                       'data/hyps/hyp.scratch-high.yaml', 'zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned-aggressive_96' )

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

    if (not cfg_name[:6] == 'yolov8') and cfg_name != 'fastestdet':
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
