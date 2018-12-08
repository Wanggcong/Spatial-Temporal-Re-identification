import os
from shutil import copyfile
import argparse
import shutil

download_path = "./raw-dataset/DukeMTMC-reID/"

parser = argparse.ArgumentParser(description='prepare')
parser.add_argument('--Market', action='store_true', help='prepare dataset market1501')
parser.add_argument('--Duke', action='store_true', help='prepare dataset Duke-MTMC')
opt = parser.parse_args()

if not os.path.isdir(download_path):
    print('please change the download_path')

if opt.Market:
    save_path = "./dataset/Market1501_prepare/"
else:
    save_path = "./dataset/DukeMTMC_prepare/"

if not os.path.exists(save_path):
    os.makedirs(save_path)
# -----------------------------------------
# query
query_path = download_path + '/query'
query_save_path = save_path + '/query'
if not os.path.exists(query_save_path):
    os.makedirs(query_save_path)

for root, dirs, files in os.walk(query_path, topdown=True):
    for name in files:
        if not name[-3:] == 'jpg':
            continue
        ID = name.split('_')
        src_path = query_path + '/' + name
        dst_path = query_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)

# -----------------------------------------
# gallery
gallery_path = download_path + '/bounding_box_test'
gallery_save_path = save_path + '/gallery'
if not os.path.exists(gallery_save_path):
    os.makedirs(gallery_save_path)

for root, dirs, files in os.walk(gallery_path, topdown=True):
    for name in files:
        if not name[-3:] == 'jpg':
            continue
        ID = name.split('_')
        src_path = gallery_path + '/' + name
        dst_path = gallery_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)

# ---------------------------------------
# train_all
train_path = download_path + '/bounding_box_train'
train_save_path = save_path + '/train_all'
if not os.path.exists(train_save_path):
    os.makedirs(train_save_path)

for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name[-3:] == 'jpg':
            continue
        ID = name.split('_')
        src_path = train_path + '/' + name
        dst_path = train_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)

# ---------------------------------------
# train_val
train_path = download_path + '/bounding_box_train'
train_save_path = save_path + '/train'
val_save_path = save_path + '/val'
if not os.path.exists(train_save_path):
    os.makedirs(train_save_path)
    os.makedirs(val_save_path)

for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name[-3:] == 'jpg':
            continue
        ID = name.split('_')
        src_path = train_path + '/' + name
        dst_path = train_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
            dst_path = val_save_path + '/' + ID[0]  # first image is used as val image
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)


# ================================================================================================
# market1501_rename
# ================================================================================================

def parse_frame(imgname, dict_cam_seq_max={}):
    dict_cam_seq_max = {
        11: 72681, 12: 74546, 13: 74881, 14: 74661, 15: 74891, 16: 54346, 17: 0, 18: 0,
        21: 163691, 22: 164677, 23: 98102, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0,
        31: 161708, 32: 161769, 33: 104469, 34: 0, 35: 0, 36: 0, 37: 0, 38: 0,
        41: 72107, 42: 72373, 43: 74810, 44: 74541, 45: 74910, 46: 50616, 47: 0, 48: 0,
        51: 161095, 52: 161724, 53: 103487, 54: 0, 55: 0, 56: 0, 57: 0, 58: 0,
        61: 87551, 62: 131268, 63: 95817, 64: 30952, 65: 0, 66: 0, 67: 0, 68: 0}
    fid = imgname.strip().split("_")[0]
    cam = int(imgname.strip().split("_")[1][1])
    seq = int(imgname.strip().split("_")[1][3])
    frame = int(imgname.strip().split("_")[2])
    count = imgname.strip().split("_")[-1]
    # print(id)
    # print(cam)  # 1
    # print(seq)  # 2
    # print(frame)
    re = 0
    for i in range(1, seq):
        re = re + dict_cam_seq_max[int(str(cam) + str(i))]
    re = re + frame
    new_name = str(fid) + "_c" + str(cam) + "_f" + '{:0>7}'.format(str(re)) + "_" + count
    # print(new_name)
    return new_name


def gen_train_all_rename():
    path = "./dataset/Market1501_prepare/train_all/"
    folderName = []
    for root, dirs, files in os.walk(path):
        folderName = dirs
        break
    # print(len(folderName))

    for fname in folderName:
        # print(fname)

        if not os.path.exists("./dataset/market_rename/train_all/" + fname):
            os.makedirs("./dataset/market_rename/train_all/" + fname)

        img_names = []
        for root, dirs, files in os.walk(path + fname):
            img_names = files
            break
        # print(img_names)
        # print(len(img_names))
        for imgname in img_names:
            newname = parse_frame(imgname)
            # print(newname)
            srcfile = path + fname + "/" + imgname
            dstfile = "./dataset/market_rename/train_all/" + fname + "/" + newname
            shutil.copyfile(srcfile, dstfile)
            # break  # 测试一个id


def gen_train_rename():
    path = "./dataset/Market1501_prepare/train/"
    folderName = []
    for root, dirs, files in os.walk(path):
        folderName = dirs
        break
    # print(len(folderName))

    for fname in folderName:
        # print(fname)

        if not os.path.exists("./dataset/market_rename/train/" + fname):
            os.makedirs("./dataset/market_rename/train/" + fname)

        img_names = []
        for root, dirs, files in os.walk(path + fname):
            img_names = files
            break
        # print(img_names)
        # print(len(img_names))
        for imgname in img_names:
            newname = parse_frame(imgname)
            # print(newname)
            srcfile = path + fname + "/" + imgname
            dstfile = "./dataset/market_rename/train/" + fname + "/" + newname
            shutil.copyfile(srcfile, dstfile)
            # break  # 测试一个id


def gen_val_rename():
    path = "./dataset/Market1501_prepare/val/"
    folderName = []
    for root, dirs, files in os.walk(path):
        folderName = dirs
        break
    # print(len(folderName))

    for fname in folderName:
        # print(fname)

        if not os.path.exists("./dataset/market_rename/val/" + fname):
            os.makedirs("./dataset/market_rename/val/" + fname)

        img_names = []
        for root, dirs, files in os.walk(path + fname):
            img_names = files
            break
        # print(img_names)
        # print(len(img_names))
        for imgname in img_names:
            newname = parse_frame(imgname)
            # print(newname)
            srcfile = path + fname + "/" + imgname
            dstfile = "./dataset/market_rename/val/" + fname + "/" + newname
            shutil.copyfile(srcfile, dstfile)
            # break  # 测试一个id


def gen_query_rename():
    path = "./dataset/Market1501_prepare/query/"
    folderName = []
    for root, dirs, files in os.walk(path):
        folderName = dirs
        break
    # print(len(folderName))

    for fname in folderName:
        # print(fname)

        if not os.path.exists("./dataset/market_rename/query/" + fname):
            os.makedirs("./dataset/market_rename/query/" + fname)

        img_names = []
        for root, dirs, files in os.walk(path + fname):
            img_names = files
            break
        # print(img_names)
        # print(len(img_names))
        for imgname in img_names:
            newname = parse_frame(imgname)
            # print(newname)
            srcfile = path + fname + "/" + imgname
            dstfile = "./dataset/market_rename/query/" + fname + "/" + newname
            shutil.copyfile(srcfile, dstfile)
            # break  # 测试一个id


def gen_gallery_rename():
    path = "./dataset/Market1501_prepare/gallery/"
    folderName = []
    for root, dirs, files in os.walk(path):
        folderName = dirs
        break
    # print(len(folderName))

    for fname in folderName:
        # print(fname)

        if not os.path.exists("./dataset/market_rename/gallery/" + fname):
            os.makedirs("./dataset/market_rename/gallery/" + fname)

        img_names = []
        for root, dirs, files in os.walk(path + fname):
            img_names = files
            break
        # print(img_names)
        # print(len(img_names))
        for imgname in img_names:
            newname = parse_frame(imgname)
            # print(newname)
            srcfile = path + fname + "/" + imgname
            dstfile = "./dataset/market_rename/gallery/" + fname + "/" + newname
            shutil.copyfile(srcfile, dstfile)
            # break  # 测试一个id


if opt.Market:
    gen_train_all_rename()
    gen_train_rename()
    gen_val_rename()
    gen_query_rename()
    gen_gallery_rename()
    shutil.rmtree("./dataset/Market1501_prepare/")
    print("Done!")
