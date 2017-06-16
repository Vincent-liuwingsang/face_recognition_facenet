from __future__ import absolute_import
from scipy import misc
import os
import sys
import cv2
import argparse
import facenet
import pickle
import tensorflow as tf
import align.detect_face as adf
import numpy as np

def main(args):
	# parameters for face detection
	minsize = 20 
	threshold = [ 0.6, 0.7, 0.7 ] 
	factor = 0.709
	turn = True
	cam = cv2.VideoCapture(0)
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter('output.avi',fourcc, 30.0, (640,480))

	with tf.Graph().as_default():
		print "Setting GPU options..."        
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
		with sess.as_default():
			print "Initiating models..."
			# load the model
			facenet.load_model(args.model)
			# Get caffemodels for face detection            
			pnet, rnet, onet = adf.create_mtcnn(sess, None)
			# Get input and output tensors
			images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
			embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
			phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
			embedding_size = embeddings.get_shape()[1]

	# load the classifier
	print "Initiating Classifier..."
	classifier_exp = os.path.expanduser(args.classifier_filename)	
	with open(classifier_exp, 'rb') as instream:
		(model, classnames) = pickle.load(instream)
	

	# Turn on camera and start facial detetion and recognition
	print "Start recording..."
	while cam.isOpened():
		_, frame = cam.read()
		if frame.ndim==2:
			frame = facenet.to_rgb(frame)
		frame = frame[:,:,0:3]
		
		# detect and align faces using MTCCN
		bounds,_ = adf.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
		num_of_faces = bounds.shape[0]
		if num_of_faces>0:
			det = bounds[:,0:4]
			img_size = np.asarray(frame.shape)[0:2]

			bb = np.zeros((num_of_faces,4), dtype=np.int32)
			bb[:,0] = np.maximum(det[:,0]-16, 0)
			bb[:,1] = np.maximum(det[:,1]-16, 0)
			bb[:,2] = np.minimum(det[:,2]+16, img_size[1])
			bb[:,3] = np.minimum(det[:,3]+16, img_size[0])

			# crop and scale face(s)
			emb = np.zeros((num_of_faces,embedding_size))
			for i in range(num_of_faces):
				cropped = frame[bb[i,1]:bb[i,3],bb[i,0]:bb[i,2],:]
				scaled=misc.imresize(cropped, (160, 160), interp='bilinear')		
				scaled=facenet.prewhiten(scaled)
				feed_dict = {images_placeholder:[scaled], phase_train_placeholder:False }
				emb[i,:] = sess.run(embeddings, feed_dict=feed_dict)
										
			# highest probability
			predictions = model.predict_proba(emb)
			best_class = np.argmax(predictions, axis=1)
			best_class_prob = predictions[np.arange(len(best_class)), best_class]

			# default threshold: 0.55
			threshold_prob = 0.80				
			
			for i in range(num_of_faces):
				name = "Unknown"
				if best_class_prob[i]>threshold_prob:
					name = classnames[best_class[i]]
				prob = best_class_prob[i]
				cv2.rectangle(frame, (bb[i,0],bb[i,1]), (bb[i,2],bb[i,3]), (0,0,255), 2)
				cv2.putText(frame, name+','+str(round(prob*100,2))+'%', (bb[i,0],bb[i,1]-12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1)
		cv2.imshow('Recording', frame)
		out.write(frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	print "Ended."
	cam.release()
	out.release()
	cv2.destroyAllWindows


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename', 
        help='Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output and for classification this is an input.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
