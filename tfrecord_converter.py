import time
import os
import hashlib

from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import lxml.etree
import glob

flags.DEFINE_string('data_dir', './generated_jpg',
                    'path to dataset')
flags.DEFINE_string('label_dir', './label_xml',
                    'path to annotation')
flags.DEFINE_enum('split', 'train', [
                  'train', 'val'], 'specify train or val spit')
flags.DEFINE_string('tfrecord_dir', './tfrecords', 'outpot dataset')
# flags.DEFINE_string('classes', './traffic_sign.names', 'classes file')


def build_example(annotation):
    img_path = os.path.join(
        FLAGS.data_dir, annotation['filename'])
    img_raw = open(img_path, 'rb').read()
    key = hashlib.sha256(img_raw).hexdigest()

    width = int(annotation['size']['width'])
    height = int(annotation['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    # truncated = []
    # views = []
    # difficult_obj = []

    if 'object' in annotation:
        for obj in annotation['object']:
            # difficult = bool(int(obj['difficult']))
            # difficult_obj.append(int(difficult))

            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(int(obj['class_id']))
            # truncated.append(int(obj['truncated']))
            # views.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            annotation['filename'].encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            annotation['filename'].encode('utf8')])),
        'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        # 'image/object/difficult': tf.train.Feature(int64_list=tf.train.Int64List(value=difficult_obj)),
        # 'image/object/truncated': tf.train.Feature(int64_list=tf.train.Int64List(value=truncated)),
        # 'image/object/view': tf.train.Feature(bytes_list=tf.train.BytesList(value=views)),
    }))
    return example


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


def main(_argv):
    # class_map = {name: idx for idx, name in enumerate(
    #     open(FLAGS.classes).read().splitlines())}
    # logging.info("Class mapping loaded: %s", class_map)

    tfrecord_path = os.path.join(FLAGS.tfrecord_dir, 'tfrecord_{}.tfrecord'.format(FLAGS.split))
    writer = tf.io.TFRecordWriter(tfrecord_path)
    label_paths = glob.glob(os.path.join(FLAGS.label_dir, '*.xml'))

    count=0
    for i, label_path in enumerate(label_paths):

        if (FLAGS.split == 'train' and i%10<8) or (FLAGS.split == 'val' and i%10>=8):
            annotation_xml = lxml.etree.fromstring(open(label_path).read())
            annotation = parse_xml(annotation_xml)['annotation']
            tf_example = build_example(annotation)
            writer.write(tf_example.SerializeToString())
            
            count+=1
            if count%100 == 0:
                print('{} data frame completed'.format(count))

    writer.close()
    logging.info("Done")


if __name__ == '__main__':
    app.run(main)
