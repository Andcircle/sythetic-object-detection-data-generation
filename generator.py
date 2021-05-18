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
import kmean


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
                        default='./label')

    parser.add_argument('--name_path',
                        help='Path to .name file where all the object names are saved',
                        default='./traffic_sign.names')

    parser.add_argument('--meta_path',
                        help='Path to .meta file where the statistics of current dataset are saved',
                        default='./traffic_sign.meta')

    parser.add_argument('--resize_size',
                        help='The resized input size to yolo model',
                        default=416)

    return parser.parse_args()


def xyxy2xywh(xyxy, image_shape):
    image_h, image_w, _ = image_shape
    xywh=np.zeros(4)
    xywh[0]=(xyxy[0,0] + xyxy[1,0])/2 /image_w
    xywh[1]=(xyxy[0,1] + xyxy[1,1])/2 /image_h
    xywh[2]=(xyxy[1,0] - xyxy[0,0]) /image_w
    xywh[3]=(xyxy[1,1] - xyxy[0,1]) /image_h
    return xywh

def xywh2xyxy(xywh, image_shape):
    image_h, image_w, _ = image_shape
    xyxy=np.zeros((2,2))
    xyxy[0,0]=(xywh[0] - xywh[2]/2) * image_w
    xyxy[0,1]=(xywh[1] - xywh[3]/2) * image_h
    xyxy[1,0]=(xywh[0] + xywh[2]/2) * image_w
    xyxy[1,1]=(xywh[1] + xywh[3]/2) * image_h
    return xyxy


def generate(args):
    bld = blender.Blender(args.object_dir, args.csv_path)
    names = bld.get_sign_names()
    # used for statistics
    class_counter={name:0 for name in names}
    names = [name + '\n' for name in bld.get_sign_names()]

    with open(args.name_path, 'w') as name_file:
        name_file.writelines(names)

    background_paths = glob.glob(os.path.join(args.background_dir, '*.jpg'))

    xml_train_label_dir = os.path.join(args.label_dir+'_xml','train')
    if not os.path.exists(xml_train_label_dir):
        os.makedirs(xml_train_label_dir)

    xml_val_label_dir = os.path.join(args.label_dir+'_xml','val')
    if not os.path.exists(xml_val_label_dir):
        os.makedirs(xml_val_label_dir)

    txt_train_label_dir = os.path.join(args.label_dir+'_txt','train')
    if not os.path.exists(txt_train_label_dir):
        os.makedirs(txt_train_label_dir)

    txt_val_label_dir = os.path.join(args.label_dir+'_txt','val')
    if not os.path.exists(txt_val_label_dir):
        os.makedirs(txt_val_label_dir)

    train_image_dir = os.path.join(args.generated_dir,'train')
    if not os.path.exists(train_image_dir):
        os.makedirs(train_image_dir)

    val_image_dir = os.path.join(args.generated_dir,'val')
    if not os.path.exists(val_image_dir):
        os.makedirs(val_image_dir)

    # used for statistics
    whs = []

    for i, background_path in enumerate(background_paths):
        background_image = cv2.imread(background_path, cv2.IMREAD_ANYCOLOR)
        background_name = background_path.split('/')[-1]

        xml_label_name = background_name.split('.')[0] + '.xml'
        txt_label_name = background_name.split('.')[0] + '.txt'

        # split traing and validation 8 to 2
        if i%10<8:
            xml_label_path = os.path.join(xml_train_label_dir, xml_label_name)
            txt_label_path = os.path.join(txt_train_label_dir, txt_label_name)
            output_data_path = os.path.join(train_image_dir, background_name)
        else:
            xml_label_path = os.path.join(xml_val_label_dir, xml_label_name)
            txt_label_path = os.path.join(txt_val_label_dir, txt_label_name)
            output_data_path = os.path.join(val_image_dir, background_name)

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

        with open(txt_label_path, 'w') as f_txt:
            
            # Each image has 1~5 traffic signs
            r = random.randrange(1,5)
            for j in range(r):
                obj, bbox = bld.blend(background_image)
                if obj is None:
                    continue

                xywh = xyxy2xywh(bbox, background_image.shape)
                obj_line = '{:.3f} {:.3f} {:.3f} {:.3f} {:.3f}\n'.format(obj['class_id'], xywh[0], 
                                                   xywh[1], xywh[2], xywh[3])
                f_txt.write(obj_line)

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

                class_counter[obj['sign_name']]+=1
                
                h, w, _ = background_image.shape
                r = args.resize_size / max(h, w)
                wh=[(bbox[1,0]-bbox[0,0])*r, (bbox[1,1]-bbox[0,1])*r]
                whs.append(wh)
        
        cv2.imwrite(output_data_path, background_image)

        tree = et.ElementTree(root) 
        with open (xml_label_path, "wb") as f_xml : 
            tree.write(f_xml)

        if i%100 == 0:
            print('{} frames completed'.format(i))

    with open(args.meta_path, 'w') as meta:
        meta.write('total images: {}\n'.format(i))
        meta.write('total objects: {}\n'.format(len(whs)))
        avg_wh=np.mean(np.array(whs), axis=1)
        meta.write('average width of objects: {}\n'.format(avg_wh[0]))
        meta.write('average height of objects: {}\n'.format(avg_wh[1]))

        yolo_tiny_anchor, _ = kmean.kmean(whs, 6, 50)
        yolo_tiny_anchor = np.rint(yolo_tiny_anchor).astype('int')
        yolo_anchor, _ = kmean.kmean(whs, 9, 50)
        yolo_anchor = np.rint(yolo_anchor).astype('int')

        def area(x):
            return x[:,0] * x[:,1]
        order = np.argsort(area(yolo_anchor))
        yolo_anchor=yolo_anchor[order]
        order = np.argsort(area(yolo_tiny_anchor))
        yolo_tiny_anchor=yolo_tiny_anchor[order]

        str_yolo_anchor = 'yolo_anchor: \n' + \
            np.array2string(np.rint(yolo_anchor).astype('int')) + '\n'
        meta.write(str_yolo_anchor)

        str_yolo_tiny_anchor = 'yolo_tiny_anchor: \n' + \
            np.array2string(np.rint(yolo_tiny_anchor).astype('int')) + '\n'
        meta.write(str_yolo_tiny_anchor)

        meta.write('\n class counter: \n')
        for name, count in class_counter.items():
            meta.write('{}: {} \n'.format(name, count))


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
    data_paths = glob.glob(os.path.join(args.generated_dir, 'val', '*.jpg'))
    xml_label_dir = os.path.join(args.label_dir+'_xml', 'val')
    txt_label_dir = os.path.join(args.label_dir+'_txt', 'val')

    for data_path in data_paths:
        image = cv2.imread(data_path, cv2.IMREAD_ANYCOLOR)

        name=data_path.split('/')[-1]
        xml_label_path = os.path.join(xml_label_dir, name.replace('jpg','xml'))
        txt_label_path = os.path.join(txt_label_dir, name.replace('jpg','txt'))

        with open(xml_label_path, 'r') as f_xml:
            annotation_xml = et.fromstring(f_xml.read())
        
        annotation = parse_xml(annotation_xml)['annotation']
        
        for obj in annotation['object']:
            b = obj['bndbox']
            bbox = np.array([[int(b['xmin']), int(b['ymin'])], [int(b['xmax']), int(b['ymax'])]])
            transformer.draw_bbox(image, bbox)

            print(bbox)

        boxes = np.loadtxt(txt_label_path).reshape(-1, 5)

        for box in boxes[:,1:]:
            xyxy=xywh2xyxy(box, image.shape)
            transformer.draw_bbox(image, xyxy, (255,0,0),1)

            print(xyxy)


        plt.imshow(image)
        plt.show()   


def main():
    args = _parse_args()
    generate(args)
    # validate(args)


if __name__=='__main__':
    main()