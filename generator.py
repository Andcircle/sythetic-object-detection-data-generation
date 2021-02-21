import argparse
import glob
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from math import pi
import random
import xml.etree.ElementTree as et

import blender
import transformer


def _parse_args():
    parser = argparse.ArgumentParser(description='This script is used to generate dataset'
                                                 ' with object images and background images', 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--object_dir',
                        help='Objects directory containing object images',
                        default='./image_argb')

    parser.add_argument('--csv_path',
                        help='CSV file of meta information of objects',
                        default='./traffic_sign.csv')

    parser.add_argument('--background_dir',
                        help='Directory to backgrounds',
                        default='./background_jpg')

    parser.add_argument('--generated_dir',
                        help='Image output directory',
                        default='./generated_jpg')
    
    parser.add_argument('--label_dir',
                        help='Directory to output label xml',
                        default='./label_xml')

    parser.add_argument('--name_path',
                        help='Path to .name file where all the object names are saved',
                        default='./traffic_sign.names')

    return parser.parse_args()


def generate(args):
    bld = blender.Blender(args.object_dir, args.csv_path)
    names = [name + '\n' for name in bld.get_sign_names()]

    with open(args.name_path, 'w') as name_file:
        name_file.writelines(names)
        name_file.close()

    background_paths = glob.glob(os.path.join(args.background_dir, '*.jpg'))

    for i, background_path in enumerate(background_paths):
        background_image = cv2.imread(background_path, cv2.IMREAD_ANYCOLOR)

        background_name = background_path.split('/')[-1]
        h, w, d = background_image.shape

        # Tree structure of sigle frame data, will be saved in xml
        root = et.Element('annotation')
        
        folder = et.SubElement(root, 'folder')
        folder.text = args.generated_dir
        filename = et.SubElement(root, 'filename')
        filename.text = background_name

        size = et.SubElement(root, 'size')
        width = et.SubElement(size, 'width')
        width.text = str(w)
        height = et.SubElement(size, 'height')
        height.text = str(h)
        depth = et.SubElement(size, 'depth')
        depth.text = str(d)

        r = random.randrange(0,4)
        for j in range(r):
            obj, bbox = bld.blend(background_image)

            obj_ndoe = et.SubElement(root, 'object')

            obj_name = et.SubElement(obj_ndoe, 'name')
            obj_name.text = obj['sign_name']
            obj_id = et.SubElement(obj_ndoe, 'class_id')
            obj_id.text = str(obj['class_id'])

            bndbox = et.SubElement(obj_ndoe, 'bndbox')
            xmin = et.SubElement(bndbox, 'xmin')
            xmin.text = str(bbox[0,0])
            xmax = et.SubElement(bndbox, 'xmax')
            xmax.text = str(bbox[1,0])
            ymin = et.SubElement(bndbox, 'ymin')
            ymin.text = str(bbox[0,1])
            ymax = et.SubElement(bndbox, 'ymax')
            ymax.text = str(bbox[1,1])
        
        output_data_path = os.path.join(args.generated_dir, background_name)
        cv2.imwrite(output_data_path, background_image)

        label_name = background_name.split('.')[0] + '.xml'
        output_annotation_path = os.path.join(args.label_dir, label_name)
        tree = et.ElementTree(root) 
      
        with open (output_annotation_path, "wb") as f : 
            tree.write(f)

        if i%100 == 0:
            print('{} frames completed'.format(i))


def parse_xml(xml):
    if not len(xml):
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = parse_xml(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def validate(args):
    data_paths = glob.glob(os.path.join(args.generated_dir, '*.jpg'))
    label_paths = glob.glob(os.path.join(args.label_dir, '*.xml'))

    if len(data_paths)>0 and len(label_paths)>0:
        image = cv2.imread(data_paths[0], cv2.IMREAD_ANYCOLOR)

        with open(label_paths[0], 'r') as f:
            annotation_xml = et.fromstring(f.read())
        
        annotation = parse_xml(annotation_xml)['annotation']
        
        for obj in annotation['object']:
            b = obj['bndbox']
            bbox = np.array([[int(b['xmin']), int(b['ymin'])], [int(b['xmax']), int(b['ymax'])]])
            transformer.draw_bbox(image, bbox)

        plt.imshow(image)
        plt.show()
    else:
        print('Do not have any data in specified folder')   


def main():
    args = _parse_args()
    generate(args)
    # validate(args)


if __name__=='__main__':
    main()