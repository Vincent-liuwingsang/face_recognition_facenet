![Demo](https://user-images.githubusercontent.com/16081202/27239793-f2f99966-5304-11e7-8558-e194f4bd5cb9.gif)
This repository contains a complete pipline for real time face recognition. It includes:
 - gathering images from your webcam/video source
 - detection,alignment,resize of face images using Multi-task Cascaded Convolutional networks
 - train models using your own aligned images
 - computing embedings from your images
 - real-time facial recognition through webcam


Usage:
 - To add new data, put python add_new_data.py database_name subject_name <number_of_photos> <size_factor> in command line.
 - To allign data, put python align_data_mtccn.py input_dir output_dir <image_size> <margin> <random_order> <gpu_memory_fraction> in command line.
 - To train your model, put
 - To activate webcam/video for facial recognition: python face_rec.py model_dir classifier_dir

Modification possibility:
 - build raspberry Pi video pipeline(stream data from server) and add voice module for greeting purposes (Alert when no one is home? greet friends and family?)
 - rescale images with better technique to obtain a more accurate representation of faces.

