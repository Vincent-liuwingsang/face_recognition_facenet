from __future__ import absolute_import

import sys
import classifier


def main(args):
    classifier.main(args)
    

if __name__== "__main__":
    main(classifier.parse_arguments(sys.argv[1:]))
