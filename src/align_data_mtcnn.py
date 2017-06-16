from __future__ import absolute_import

import sys
import align.align_dataset_mtcnn as admt



def main(args):
    admt.main(args)    

if __name__== '__main__':
    main(admt.parse_arguments(sys.argv[1:]))
