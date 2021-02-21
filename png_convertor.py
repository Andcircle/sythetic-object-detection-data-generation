import argparse
import glob
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np


def _parse_args():
    parser = argparse.ArgumentParser(description='This script is used to convert RGB jpg image into ARGB png image', 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input_dir',
                        help='input directory containing original images,'
                        'may not have alpha channel.',
                        default='./image_raw')

    parser.add_argument('--output_dir',
                        help='output directory containing processed png images',
                        default='./image_argb')

    return parser.parse_args()

def convert_image(image):
    image_gray=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_alpha=np.ones_like(image_gray) * 255

    row, col = image_gray.shape
    stack=[(0,0),(0,col-1),(row-1,0),(row-1,col-1)]
    moves=[(0,1),(1,0),(0,-1),(-1,0)]

    while stack:
        r,c=stack.pop(0)
        image_alpha[r,c]=0

        for m in moves:
            nr=r+m[0]
            nc=c+m[1]

            if 0<=nr<row and 0<=nc<col and image_gray[nr,nc]>240:
                image_gray[nr,nc]=0
                stack.append((nr, nc))

    #opencv use BGRA format
    image=np.concatenate((image, np.expand_dims(image_alpha,axis=-1)), axis=2)

    return image
    

def convert_dir(input_dir, output_dir):
    input_paths = glob.glob(os.path.join(input_dir,'*.png'))

    count=0
    for input_path in input_paths:
        image=cv2.imread(input_path, -1)
        if image.shape[2]==3:
            count +=1
            image=convert_image(image)
            print('{} images converted'.format(count))

        file_name=input_path.split('/')[-1]
        output_path=os.path.join(output_dir, file_name)
        cv2.imwrite(output_path, image)


def main():
    args = _parse_args()
    convert_dir(args.input_dir, args.output_dir)

if __name__=='__main__':
    main()