# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/08 11:45
@Author        : Tianxiaomo
@File          : coco_annotatin.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import json
from collections import defaultdict
from tqdm import tqdm
import os
import cv2

"""hyper parameters"""
json_file_path = 'annotations/instances_train2017.json'
images_dir_path = 'train2017/'
output_path = './json2txt/'

"""load json file"""
name_box_id = defaultdict(list)
id_name = dict()
with open(json_file_path, encoding='utf-8') as f:
    data = json.load(f)

"""generate labels"""
images = data['images']
annotations = data['annotations']
for ant in tqdm(annotations):
    id = ant['image_id']
    # print(id)
    # name = os.path.join(images_dir_path, images[id]['file_name'])
    name = os.path.join(images_dir_path, '{:012d}.jpg'.format(id))
    img = cv2.imread(name)
    img_shape = img.shape
    # print(img_shape)
    cat = ant['category_id']

    if cat >= 1 and cat <= 11:
        cat = cat - 1
    elif cat >= 13 and cat <= 25:
        cat = cat - 2
    elif cat >= 27 and cat <= 28:
        cat = cat - 3
    elif cat >= 31 and cat <= 44:
        cat = cat - 5
    elif cat >= 46 and cat <= 65:
        cat = cat - 6
    elif cat == 67:
        cat = cat - 7
    elif cat == 70:
        cat = cat - 9
    elif cat >= 72 and cat <= 82:
        cat = cat - 10
    elif cat >= 84 and cat <= 90:
        cat = cat - 11

    # name_box_id[name].append([cat, ant['bbox']])
    name_box_id[name] = [cat, ant['bbox']]
    # print("name : ", name)
    # print("name_box_id : ", name_box_id[name])

    """write to txt"""
    n = name.replace(name.split('.')[-1], 'txt')
    # print("n : ", n)
    # print("name_box_id[name] : ", name_box_id[name][:10])
    with open(n, 'a') as f:
        # for j in name_box_id[name]:
            # print(j)

            # img_shape : (height, width, dimension)

        a = name_box_id[name][1][0] # x coordinate
        b = name_box_id[name][1][1] # y coordinate
        c = name_box_id[name][1][2] # width
        d = name_box_id[name][1][3] # height

        # Normalize (0 ~ 1)
        a = (a + c) / (2 * img_shape[1])
        b = (b + d) / (2 * img_shape[0])
        c = c / (2 * img_shape[1])
        d = d / (2 * img_shape[0])
        
        # print(k)
        # print('j[1] : ', j[1][:])
        f.write(str(name_box_id[name][0]) + '\t' + str(a) + '\t' + str(b) + '\t' + str(c) + '\t' + str(d) + '\n')

            #print("write : ", float(i))
            # print("name_box_id[name] : ", j)
            # print("write : ", float(j))
            #f.write(str(j[0][0]))


# for a in tqdm(annotations):
#     with open(output_path, 'w') as f:
#         print("output_path : ", output_path)
#         for key in tqdm(name_box_id.keys()):
#             f.write(key)
#             box_infos = name_box_id[key]
#             for info in box_infos:
#                 x_min = int(info[0][0])
#                 y_min = int(info[0][1])
#                 x_max = x_min + int(info[0][2])
#                 y_max = y_min + int(info[0][3])
#
#                 box_info = " %d,%d,%d,%d,%d" % (
#                     int(info[1]), x_min, y_min, x_max, y_max)
#                 f.write(box_info)
#             f.write('\n')
