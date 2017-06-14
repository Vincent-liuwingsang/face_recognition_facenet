import sys
import os
import cv2
import argparse

data_dir = os.path.dirname(os.path.realpath(__file__))

def main(args):
    subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    dest = data_dir+'/'+args.database_name
    if args.database_name not in subdirs:
	os.mkdir(dest)

    
        
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('database_name', type=str, help='Indicates the name of the database folder that the photos are going to')
    parser.add_argument('subject_name', type=str, help='Indicates the name of the subject')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
