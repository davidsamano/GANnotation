
import numpy as np
import scipy.io
import os
import csv
import cv2

# Important ReadMe:
# path to 300W_LP\landmarks folder is given to the path variable. example usage is given, 
# it should work for all platforms and regardless of paths, since absolute paths are used with OS 

def createDataSet(path, feat='pts_3d', size = 12000):
    out_train = os.path.join(path[:-9],'train')
    out_test = os.path.join(path[:-9],'test')
    out_folder = out_test
    
    if not os.path.isdir(out_train):
        os.mkdir(out_train)
    if not os.path.isdir(out_test):
        os.mkdir(out_test)

    flag = True
    count = 0
    out = np.zeros((68,2))

    for folder in sorted(os.listdir(path)):
        for name in sorted(os.listdir(os.path.join(path, folder))):
            out = np.array(scipy.io.loadmat(os.path.join(path, folder, name))['pts_3d']) #- 1
            np.save(os.path.join(out_folder,name[:-8]), out)
            count += 1
            if count % size == 0 and count != 0 and flag:
                out_folder = out_train
                flag = False
    return

def createCSV(path, feat='pts_3d', size = 12000):
    out_train = os.path.join(path[:-9],'train')
    out_test = os.path.join(path[:-9],'test')
    out_folder = out_test

    if not os.path.isdir(out_train):
        os.mkdir(out_train)
    if not os.path.isdir(out_test):
        os.mkdir(out_test)

    flag = True
    count = 0
    csvfile = open(os.path.join(out_folder,'test.csv'), 'w')
    filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    print('starting test')
    for folder in sorted(os.listdir(path)):
        for name in sorted(os.listdir(os.path.join(path, folder))):
            img_file = os.path.join(path[:-9],folder,name[:-8]+'.jpg')
            kp_file = os.path.join(out_folder,name[:-8]+'.npy')
            filewriter.writerow([img_file, kp_file])
            count += 1
            if count % size == 0 and count != 0 and flag:
                out_folder = out_train
                csvfile.close()
                print('starting train')
                csvfile = open(os.path.join(out_folder,'train.csv'), 'w')
                filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                flag = False
    
    csvfile.close()        
    return

def parseKeyPointMat(path = None):
    out = np.zeros((68,2))
    if path == None:
        return out
    
    with open(path) as f:
        content = f.readlines()

    for i in range(3, len(content)-1):
        a,b = str.split(content[i])
        out[i-3,0] = a
        out[i-3,1] = b

    return out

def createDataSetGAN(path):
    for root, _, files in os.walk(path):
        if not 'annot' in root:
            continue
        print('processing:', root)
        out = np.zeros((len(files),68,2))
        for i,name in enumerate(sorted(files)):
            out[i] = parseKeyPointMat(os.path.join(root,name))

        np.save(os.path.join(root[:-5], root[-9:-6]), out)
    return

def aviSplit300VW(path):
    for root, _, files in os.walk(path):
        vid_file = os.path.join(root,'vid.avi')
        kp_file = os.path.join(root,root[-3:]+'.npy')
        frame_path = os.path.join(root,'frames')
        kp_path = os.path.join(root,'keypoints')

        if os.path.exists(vid_file):
            print('processing:', root)
            if not os.path.isdir(frame_path):
                os.mkdir(frame_path)
            if not os.path.isdir(kp_path):
                os.mkdir(kp_path)
            i = 0
            cap = cv2.VideoCapture(vid_file)
            kp = np.load(kp_file)
            while(cap.isOpened()):
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(os.path.join(frame_path,str(i)+'.jpg'), frame)
                    np.save(os.path.join(kp_path,str(i)), kp[i])
                    i += 1
                else:
                    cap.release()
    return

def createCSVGAN(path, size = 22):
    out_train = os.path.join(path,'train')
    out_test = os.path.join(path,'test')
    out_folder = out_test

    if not os.path.isdir(out_train):
        os.mkdir(out_train)
    if not os.path.isdir(out_test):
        os.mkdir(out_test)

    flag = True
    count = 0
    csvfile = open(os.path.join(out_folder,'test.csv'), 'w')
    filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    print('starting test')
    for folder in sorted(os.listdir(path)):
        if 'test' in folder or 'train' in folder or not os.path.isdir(os.path.join(path,folder)):
            continue
        print(folder)
        frame_path = os.path.join(path,folder,'frames')
        kp_path = os.path.join(path,folder,'keypoints')
        kp_files = sorted(os.listdir(kp_path))
        frame_files = sorted(os.listdir(frame_path))
        for i in range(len(frame_files)):
            filewriter.writerow([os.path.join(frame_path,frame_files[0]), os.path.join(kp_path,kp_files[i]), os.path.join(frame_path,frame_files[i])])

        count += 1
        if count % size == 0 and count != 0 and flag:
            out_folder = out_train
            csvfile.close()
            print('starting train')
            csvfile = open(os.path.join(out_folder,'train.csv'), 'w')
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            flag = False
    csvfile.close()        
    return

# feat = 'pts_3d'
# feat = 'pts_2d'
# path = 'C:\\Users\\isaac\\Documents\\Classwork\\Graduate\\DL\\final\\300W_LP\\landmarks'
# createDataSet(path)
# createCSV(path)

path_300vw = '/Users/david/github/cs7643-unganny-valley/data/300VW_Dataset_2015_12_14'
createDataSetGAN(path_300vw)
aviSplit300VW(path_300vw)
createCSVGAN(path_300vw)