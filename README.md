AHRCR
====================================================================
This is a repository for my Bachelor's thesis - Arabic Handwritten Characters Recognition using Support Vector Machines and Neural Networks.

# Abstract
This project aims at performing Character Recognition on a dataset of Arabic Handwritten characters. Support Vector Machines and Neural Networks were the models of choice. In general, first the features are extracted from the raw images, transform them to achieve non-linearity and then classify them using either an SVM or a Softmax classiﬁer. Both classical and modern methods were used to perform these tasks. And histogram of oriented gradients and convolutional neural networks used to extract features, Kernels followed by SVM and fully-connected network to perform non-linear classiﬁcation. In total, eight diﬀerent methods applied (depending on the feature extraction and transformation and classiﬁer being used) and report a 93.27% test accuracy.

# Libraries Used
- Python
    + TensorFlow
    + Numpy
    + Pandas
- Matlab
    + LIBSVM 