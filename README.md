The project includes two solutions for a traffic signs recognition. Original dataset is available at https://www.kaggle.com/valentynsichkar/traffic-signs-preprocessed.

The first solution utilizes tensorflow framework to build a scalable Deep Neural Network to recognize 43 different traffic signs. Its accuracy with the cross validation set reaches 90%. Algorithm implements batch normalization, learning rate decay and dropout. For cost function minimization it uses minibatch Adam optimizer.

In the second solution, convolutional layers are used and a model based on VGG-16 architecture is built. Its accuracy with the cross validation set reaches 95%. Algorithm implements batch normalization, learning rate decay and dropout. For cost function minimization it uses minibatch Adam optimizer.

Files:
DNN_architecture.py -> script containing implementation of scallable DNN
CNN_architecture -> script containing implementation of scallable CNN
CNN_CNN_for_trafficsigns.ipynb -> Google Colab notebook showing results of using CNN
