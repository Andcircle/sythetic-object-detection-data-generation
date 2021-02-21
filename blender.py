import argparse
import glob
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from math import pi
import random

import transformer

class Blender:
    def __init__(self, object_dir, csv_path):

        csv = open(csv_path)
        csv.readline()
        csv = csv.readlines()

        self.objects = []

        for i, line in enumerate(csv):
            obj = dict()
            obj['file_name'], obj['sign_name'], _ = line.split(',')
            obj_path = os.path.join(object_dir, obj['file_name']+'.png')
            obj['image'] = cv2.imread(obj_path, cv2.IMREAD_UNCHANGED)
            obj['class_id'] = i

            self.objects.append(obj)


    def get_sign_names(self):
        return [obj['sign_name'] for obj in self.objects]


    def blend(self, background_image):
        l=len(self.objects)

        id = random.randrange(0,l)

        transformed_obj_img = transformer.random_transform_image(self.objects[id]['image'])
        oh, ow, _ = transformed_obj_img.shape
        bh, bw, _ = background_image.shape
        h = random.randrange(0,bh-oh)
        w = random.randrange(0,bw-ow)

        src = transformed_obj_img[...,:3]
        alpha = transformed_obj_img[...,3:4].astype('float64') / 255
        tgt = background_image[h:h+oh, w:w+ow]
        background_image[h:h+oh, w:w+ow] = (src * alpha + tgt * (1 - alpha)).astype('int')

        # Using format (x, y) ~ (w, h)
        bbox = np.array([[w, h], [w+ow-1, h+oh-1]])

        return self.objects[id], bbox


def _parse_args():
    parser = argparse.ArgumentParser(description='This script is used to generate random object'
                                                 ' and blend it with target background image', 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--object_dir',
                        help='Objects directory containing object images',
                        default='./image_argb')

    parser.add_argument('--csv_path',
                        help='CSV file of meta information of objects',
                        default='./traffic_sign.csv')

    parser.add_argument('--background_path',
                        help='Test background image',
                        default='./background_jpg/2007_000241.jpg')

    return parser.parse_args()


def main():
    args = _parse_args()
    blender = Blender(args.object_dir, args.csv_path)

    background_image = cv2.imread(args.background_path, cv2.IMREAD_UNCHANGED)

    cv2.imwrite('./test/background.png', background_image)

    for i in range(5):
        _, bbox = blender.blend(background_image)
        transformer.draw_bbox(background_image, bbox)

    cv2.imwrite('./test/background_blended.png', background_image)
    plt.imshow(background_image)
    plt.show()

if __name__=='__main__':
    main()