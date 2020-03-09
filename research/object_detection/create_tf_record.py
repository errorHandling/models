## creating tfRecord 
#create_ead_tf_record.py
"""
python object_detection/dataset_tools/create_pet_tf_record.py \
        --data_dir=/home/user/pet \
        --output_dir=/home/user/pet/output

        python object_detection/dataset_tools/create_ead_tf_record.py \
        --anno_path=/home/user/pet \
        --image_path=\
        --output_path=/home/user/pet/output
"""
import hashlib
import io
import logging
import os
import random
import re
import glob
from PIL import Image
import contextlib2
from lxml import etree
import numpy as np
import PIL.Image
import tensorflow as tf

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags=tf.app.flags
#flags.DEFINE_string('data_dir','','root directory to dataset')
#flags.DEFINE_string('output_dir','','path to directory to output tfRecord')

#FLAGS=flags.FLAGS

#mount drive 

flags = tf.app.flags
flags.DEFINE_string('output_path', '', '/content/drive/My Drive/endoscopy-artefact-detection-_ead_-dataset/trainingData_detection/')
flags.DEFINE_string('anno_path', '', '/content/drive/My Drive/endoscopy-artefact-detection-_ead_-dataset/trainingData_detection/trainingData_detection/')
flags.DEFINE_string('image_path', '', '/content/drive/My Drive/endoscopy-artefact-detection-_ead_-dataset/trainingData_detection/trainingData_detection/')

FLAGS = flags.FLAGS


def create_tf_example(currentName, anno_path, image_path):
    
    currentNameSplit = currentName.split('.')[0]
    currentImageName = currentNameSplit + '.jpg'
	print(currentImageName)
    with tf.gfile.GFile(os.path.join(image_path, '{}'.format(currentImageName)), 'rb') as fid:
        encoded_image_data = fid.read()
    
    image = Image.open(image_path + currentImageName)
    width, height = image.size

    filename = currentNameSplit.encode('utf8')
    image_format = b'jpg'
    
    with open(anno_path + currentName) as file: 
        lines = file.readlines()
         
        xmins = [] 
        xmaxs = [] 
        ymins = [] 
        ymaxs = [] 
        classes_text = []
        classes = [] 
             
        for li in range(len(lines)): 
            print('Lines[li]: {}'.format(lines[li]))
            classID = lines[li].split()[0]
            xmins.append(float(lines[li].split()[1]) / width)
            xmaxs.append(float(lines[li].split()[3]) / width)
            ymins.append(float(lines[li].split()[2]) / height)
            ymaxs.append(float(lines[li].split()[4]) / height)
            classID = float(lines[li].split()[4])
            if int(classID) == 0:
                className = 'specularity'
                classes_text.append(className.encode('utf8'))
                classID = 0
                classes.append(classID)
            elif int(classID) == 1:
                className = 'saturation'                
                classes_text.append(className.encode('utf8'))
                classID = 1
                classes.append(classID)
            elif int(classID) == 2:
                className = 'artifact'  
                classID = 2
                classes_text.append(className.encode('utf8'))
                classes.append(classID)
            elif int(classID)==3:
                className='blur'
                classID = 3
                classes_text.append(className.encode('utf8'))
                classes.append(classID)
            elif int(classID)==4:
                className='contrast'
                classID = 4
                classes_text.append(className.encode('utf8'))
                classes.append(classID)
            elif int(classID)==5:
                className='bubbles'
                classID = 5
                classes_text.append(className.encode('utf8'))
                classes.append(classID)
            elif int(classID)==6:
                className='instrument'
                classID = 6
                classes_text.append(className.encode('utf8'))
                classes.append(classID)
            else:
                print('Error with Image Annotations in {}'. format(currentName))
    
    tf_example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': dataset_util.int64_feature(height),
    'image/width': dataset_util.int64_feature(width),
    'image/filename': dataset_util.bytes_feature(filename),
    'image/source_id': dataset_util.bytes_feature(filename),
    'image/encoded': dataset_util.bytes_feature(encoded_image_data),
    'image/format': dataset_util.bytes_feature(image_format),
    'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
    'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
    'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
    'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
    'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
    'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example
  
  
  
def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    allAnnotationFiles = []
    os.chdir(FLAGS.anno_path)
    for file in sorted(glob.glob("*.{}".format('txt'))):
        allAnnotationFiles.append(file)
  
    for currentName in allAnnotationFiles:

      tf_example = create_tf_example(currentName, FLAGS.anno_path, FLAGS.image_path)
      writer.write(tf_example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    tf.app.run()
