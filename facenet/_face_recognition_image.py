"""Performs face alignment and calculates L2 distance between the embeddings of images."""

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
import modules.detect_face
import tensorflow as tf
import numpy as np
import facenet
import imutils
import pickle
import cv2
import os
import shutil 



# =============================================================================
print("[INFO] Starting program ...")
# Path to the data directory containing aligned LFW face patches
pictures = os.listdir(r'images')
for pic in pictures:
  input_data_dir = "align_output" 
# image_files_input: Images to compare
# model_dir: Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file
  image_path_input    = os.path.join("images",pic)#"images/01.jpg"
  image_files_input   = os.path.expanduser(image_path_input)
  model_file          = "lib/20180402-114759/20180402-114759.pb"
  model_dir           = os.path.expanduser(model_file)

  output_dir_name = "result"
  output_dir      = os.path.expanduser(output_dir_name)

# =========================================================================
  src =image_path_input
  dst =r"C:\Users\user\Anaconda3\envs\tensorflow\FaceNet\result"
#移動檔案




# Classifier model file name as a pickle (.pkl) file
# For training this is the output and for classification this is an input
  classifier_filename_dir = "lib/my_classifier.pkl"

  minsize     = 20    # minimum size of face
  threshold   = [0.6, 0.7, 0.7]   # three steps's threshold
  factor      = 0.709             # scale factor
  image_size  = 182   # Image size (height, width) in pixels
  input_image_size = 160
  margin      = 0    # Margin for the crop around the bounding box (height, width) in pixels

  gpu_memory_fraction_max = 0.5  # Upper bound on the amount of GPU memory that will be used by the process

# Create a list of class names
  dataset = facenet.get_dataset(input_data_dir)

  class_names = [cls.name.replace('_', ' ') for cls in dataset]

# =============================================================================
  print('[INFO] Creating networks and loading parameters ...')
  with tf.Graph().as_default():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction_max
    session = tf.Session(config=config)
    with session.as_default():
        pnet, rnet, onet = modules.detect_face.create_mtcnn(session, None)
        
        # Load the model
        print('[INFO] Loading feature extraction model ...')
        facenet.load_model(model_dir)
        
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        
        print('[INFO] Loading classifier model ...')
        # Classifier model file name as a pickle (.pkl) file
        # For training this is the output and for classification this is an input
        classifier_filename_exp = os.path.expanduser(classifier_filename_dir)         
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)
        
        image = cv2.imread(os.path.expanduser(image_files_input))

        
        # check dimensions image
        if image.ndim<2:
            print("[INFO] Exit program ...")
            exit()
        if image.ndim == 2:
            image = facenet.to_rgb(image)
        image = image[:,:,0:3]
        
        # get bounding boxes of faces
        bounding_boxes, _ = modules.detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
        nrof_faces = bounding_boxes.shape[0]    # number of faces
        print('[INFO] Detected face numbers: %d ...' % nrof_faces)
        
        # ===================================================
        if nrof_faces > 0:
            det = bounding_boxes[:,0:4]
            img_size = np.asarray(image.shape)[0:2]  
            
            cropped = []
            aligned = []
            scaled  = []
            bb = np.zeros((nrof_faces,4), dtype=np.int32)
            
            # Process each faces
            for i in range(nrof_faces):
                emb_array = np.zeros((1, embedding_size))
#                det = np.squeeze(det)
#                bb = np.zeros(4, dtype=np.int32)
                
                bb[i][0] = np.maximum(det[i][0]-margin/2, 0)
                bb[i][1] = np.maximum(det[i][1]-margin/2, 0)
                bb[i][2] = np.minimum(det[i][2]+margin/2, img_size[1])
                bb[i][3] = np.minimum(det[i][3]+margin/2, img_size[0])
                
                cropped.append(image[bb[i][1]:bb[i][3],bb[i][0]:bb[i][2],:])
                cropped[i] = facenet.flip(cropped[i], False)
                aligned.append(resize(cropped[i], (image_size, image_size), order=1, mode='reflect',anti_aliasing=True))
                aligned[i]  = cv2.resize(aligned[i], (input_image_size,input_image_size), interpolation=cv2.INTER_CUBIC)
                aligned[i]  = facenet.prewhiten(aligned[i])
                scaled.append(aligned[i].reshape(-1,input_image_size,input_image_size,3))
                
                feed_dict = { images_placeholder:scaled[i], phase_train_placeholder:False }
                emb_array[0,:] = session.run(embeddings, feed_dict=feed_dict)
                
                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                
                # show bounding boxe of face
                cv2.rectangle(image, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                
                
                # show person name
                text_x = bb[i][0]
                text_y = bb[i][1] - 10
                       
                if np.max(predictions[0]) > 0.90:
                    person_name = class_names[best_class_indices[0]]
                    text_color  = (0, 255, 0)   # BGR
                else:
                    person_name = 'Unknown'
                    text_color  = (0, 0, 255)   # BGR
                cv2.putText(image, person_name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, text_color, thickness=1, lineType=2)
                for i in class_names:
                    if(person_name == i):
                        shutil.copy(src, os.path.join(dst, i))
        #end_if

        
        
        
         
        #print(len(class_names))                     
        
        
        image = imutils.resize(image, width=800)
        #cv2.imshow('IP-Camera (Press ESC to exit)',image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()              
# ============================================================================














