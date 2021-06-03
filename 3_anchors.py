import os, subprocess

cfgFolder = "/WORKING/modelSale/forklift/cfg_train"
dark_home = "/home/chtseng/frameworks/darknet.v4"
sizes = [ [9,416], [9,512], [9,608], [9,640], [20,1536], [6,320], [6,416] ]
#sizes = [ [9,608] ]


darknet_path = os.path.join(dark_home, 'darknet')
cmd_file = os.path.join(cfgFolder, 'train_cmd.txt')

with open(cmd_file, 'w') as file:
    for i, size in enumerate(sizes):
        pline = "============================================================================================================="
        print(pline)
        file.write(pline+'\n')

        pline = "[{}x{}, {} boxes]".format(\
            sizes[i][1], sizes[i][1], sizes[i][0])

        print(pline)
        file.write(pline+'\n')

        nmap_out = subprocess.run([darknet_path, 'detector', 'calc_anchors', os.path.join(cfgFolder, 'obj.data'), '-num_of_clusters', \
            str(sizes[i][0]), '-width', str(sizes[i][1]), '-height', str(sizes[i][1])], universal_newlines=False, stdout=subprocess.PIPE)

        nmap_lines = nmap_out.stdout.splitlines()
        ans = ''
        for line in nmap_lines:
            if 'anchors =' in str(line):
                ans = line.decode('ascii')
                print(ans)

        file.write(ans + '\n\n')

