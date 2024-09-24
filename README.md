
# Stellar Classification Model with Tensorflow 2.0 and Utilization of GPU Support
This project demonstrates how to use TensorFlow 2.0 with GPU support to create a Convolutional Neural Network (CNN) model for predicting stellar classifications. We will use the Stellar Classification dataset from Kaggle to build a model that classifies stars into different types based on their features.

# Datasets
Datasetes used is "Stellar Classification Dataset - SDSS17" from fedesoriano. Datasets which will be used contains 100k observations amongst the stars. This dataset is published by SDSS (Sloan Digital Sky Survey) under public domain. You can access the datasets [here](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17/data) in Kaggle

There are several attributes/features, each explains the nature of the observation with additional one atrrbiute named "class" which classifies the class of each object.

* obj_ID = Object Identifier, the unique value that identifies the object in the image catalog used by the CAS
* alpha = Right Ascension angle (at J2000 epoch)
* delta = Declination angle (at J2000 epoch)
* u = Ultraviolet filter in the photometric system
* g = Green filter in the photometric system
* r = Red filter in the photometric system
* i = Near Infrared filter in the photometric system
* z = Infrared filter in the photometric system
* run_ID = Run Number used to identify the specific scan
* rereun_ID = Rerun Number to specify how the image was processed
* cam_col = Camera column to identify the scanline within the run
* field_ID = Field number to identify each field
* spec_obj_ID = Unique ID used for optical spectroscopic objects (this means that 2 different observations with the same spec_obj_ID must share the output class)
* class = object class (galaxy, star or quasar object)
* redshift = redshift value based on the increase in wavelength
* plate = plate ID, identifies each plate in SDSS
* MJD = Modified Julian Date, used to indicate when a given piece of SDSS data was taken
* fiber_ID = fiber ID that identifies the fiber that pointed the light at the focal plane in each observation
Dataset Utilization includes Preprocessing the data by discarding unimportant or disruptive attribute and normalizing the data to achieve better results. Data will be split for Train/Valid/Test purposes with split ratio of 70%/20%/10%.


# Model Architecture
For this project, we use a Convolutional Neural Network (CNN) model to classify stars based on the features provided in the dataset. The key components of the architecture include:

- Input layer
- Convolutional layers (Conv2D)
- MaxPooling layers
- Dense layers for classification
- Output layer with softmax activation for multi-class classification
The input data is reshaped to fit the CNN requirements. The model also utilizzes the early stopping method to recognize signs of Potential Overfitting within the model and prevent such actions by stopping the model training.

# Result
The Project have managed to score an accuracy of ~96% as well as maintaining high recall & precision.

                  precision    recall  f1-score   support

           0           0.97      0.97      0.97     19807
           1           0.94      0.91      0.93      6336
           2           0.96      1.00      0.98      7157

    accuracy                               0.96     33300
    macro avg          0.96      0.96      0.96     33300
    weighted avg       0.96      0.96      0.96     33300

#License
This project/code is publised under MIT License. However datasets that are used is under SDSS Public Domain.

#References
- fedesoriano. (January 2022). Stellar Classification Dataset - SDSS17. Retrieved from https://www.kaggle.com/fedesoriano/stellar-classification-dataset-sdss17.