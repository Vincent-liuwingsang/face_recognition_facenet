import sys
import os
import cv2
import argparse
import time
import face_recognition as fr
import tensorflow as tf


"""
from __future__ import absolute_import
data_dir_split = data_dir.split('/')
sys.path.insert(0,''.join(data_dir_split[:len(data_dir_split)-2])+"/facenet/src/align")
import align.detect_face as df
"""

def main(args):
    data_dir = os.path.dirname(os.path.realpath(__file__))
    subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    dest = data_dir+'/'+args.database_name
    if args.database_name not in subdirs:
	os.mkdir(dest)
    dest+='/'+args.subject_name
    if not os.path.isdir(dest):
	os.mkdir(dest)


    # 0:default camera, 1:external camera, can stubstitue with file_name for video clip   
    cam = cv2.VideoCapture(0)
    count = 0


    print "Recording..."
    while count<args.volume and cam.isOpened():
	_, frame = cam.read()
	frame_small = cv2.resize( frame, (0,0), fx=args.size, fy=args.size)
	if (len(fr.face_locations(frame_small))>0):	
	    serial = ['0']*5
	    serial[len(serial)-len(str(count)):]=str(count)[:]
	    cv2.imwrite(dest+'/'+args.subject_name+'_'+"".join(serial)+'.jpg',frame_small)
	    count+=1
	    time.sleep(0.1)

    cam.release()
    print "Done."
    print "%s images were captured and stored in %s" % (args.volume, dest)
    
        
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('database_name', type=str, help='Indicates the name of the database folder that the photos are going to')
    parser.add_argument('subject_name', type=str, help='Indicates the name of the subject')
    parser.add_argument('--volume', type=int, help='Indicates the number of photos to be taken', default = 50)
    parser.add_argument('--size', type=float, help='factor of resizing', default = 1.0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    start=time.time()
    print "Camera initiating..."
    main(parse_arguments(sys.argv[1:]))
    end=time.time()
    volume = sys.argv[3] if len(sys.argv)>3 else '50'
    print "Average: %f per photo" % ((end-start)/float(volume))
