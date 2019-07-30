# =============================================================================
# Ignore WARNING:root:Lossy conversion from float64 to uint8.
# https://stackoverflow.com/questions/52165705/how-to-ignore-root-warnings
# =============================================================================

"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from scipy import misc
from skimage.transform import resize
import imageio.core.util
import imageio
import os
import tensorflow as tf
import numpy as np
import facenet
import modules.detect_face
import random
from time import sleep

# =============================================================================
# Ignore WARNING:root
def silence_imageio_warning(*args, **kwargs):
    pass
imageio.core.util._precision_warn = silence_imageio_warning

print("[INFO] Starting program ...")
sleep(random.random())

# Input iamges: Directory with unaligned images
input_dir_name = "align_input"
dataset        = facenet.get_dataset(input_dir_name)

# Output Images: Directory with aligned face thumbnails
output_dir_name = "align_output"
output_dir      = os.path.expanduser(output_dir_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# add to result
output_dir_name2 = "result"
output_dir2      = os.path.expanduser(output_dir_name2)
if not os.path.exists(output_dir2):
    os.makedirs(output_dir2)

print("[INFO] Creating networks and loading parameters ...")
# Using GPUs|TensorFlow: Allowing GPU memory growth
# https://www.tensorflow.org/guide/using_gpu
with tf.Graph().as_default():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.333
    session = tf.Session(config=config)
    with session.as_default():
        pnet, rnet, onet = modules.detect_face.create_mtcnn(session, None)

minsize     = 20                # minimum size of face
threshold   = [ 0.6, 0.7, 0.7 ] # three steps's threshold
factor      = 0.709             # scale factor
image_size  = 160   # Image size (height, width) in pixels
margin      = 0     # Margin for the crop around the bounding box (height, width) in pixels
# Detect and align multiple faces per image
detect_multiple_faces = False

# Add a random key to the filename to allow alignment using multiple processes
random_key = np.random.randint(0, high=99999)
bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)
    
with open(bounding_boxes_filename, "w") as text_file:
    nrof_images_total = 0
    nrof_successfully_aligned = 0
    for cls in dataset:
        output_class_dir = os.path.join(output_dir, cls.name)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
#==============================================================add to result
        output_class_dir2 = os.path.join(output_dir2, cls.name)
        if not os.path.exists(output_class_dir2):
            os.makedirs(output_class_dir2) 
#==============================================================          
        for image_path in cls.image_paths:
            nrof_images_total += 1
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            output_filename = os.path.join(output_class_dir, filename+'.png')
            print("[INFO] %s ..." % image_path)
            if not os.path.exists(output_filename):
                try:
#                    img = misc.imread(image_path)
                    img = imageio.imread(image_path)
                except (IOError, ValueError, IndexError) as e:
                    errorMessage = '{}: {}'.format(image_path, e)
                    print("[INFO] %s ..." % errorMessage)
                else:
                    if img.ndim<2:
                        print('[INFO] Unable to align ... "%s"' % image_path)
                        text_file.write('%s\n' % (output_filename))
                        continue
                    if img.ndim == 2:
                        img = facenet.to_rgb(img)
                    img = img[:,:,0:3]
                    
                    bounding_boxes, _ = modules.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]
                    if nrof_faces>0:
                        det = bounding_boxes[:,0:4]
                        det_arr = []
                        img_size = np.asarray(img.shape)[0:2]
                        if nrof_faces>1:
                            bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                            img_center = img_size / 2
                            offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                            offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                            index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                            det_arr.append(det[index,:])
                        else:
                            det_arr.append(np.squeeze(det))

                        for i, det in enumerate(det_arr):
                            det = np.squeeze(det)
                            bb = np.zeros(4, dtype=np.int32)
                            bb[0] = np.maximum(det[0]-margin/2, 0)
                            bb[1] = np.maximum(det[1]-margin/2, 0)
                            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
                            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
                            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
#                            scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                            scaled = resize(cropped, (image_size, image_size), order=1, mode='reflect',anti_aliasing=True)                   
                            nrof_successfully_aligned += 1
                            filename_base, file_extension = os.path.splitext(output_filename)
                            if detect_multiple_faces:
                                output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                            else:
                                output_filename_n = "{}{}".format(filename_base, file_extension)
#                            misc.imsave(output_filename_n, scaled)
                            imageio.imwrite(output_filename_n, scaled)                            
                            text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
                    else:
                        print('[INFO] Unable to align ..."%s"' % image_path)
                        text_file.write('%s\n' % (output_filename))
                            
print('[INFO] Total number of images: %d' % nrof_images_total)
print('[INFO] Number of successfully aligned images: %d' % nrof_successfully_aligned)
