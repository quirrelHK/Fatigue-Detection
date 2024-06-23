# Fatigue-Detection
This is a personal project that aims to detect levels of fatigue in a person using facial data. 

### Home Screen
![Home Page](https://github.com/quirrelHK/Fatigue-Detection/blob/main/results/home.png)

### Example 1
![Example 1](https://github.com/quirrelHK/Fatigue-Detection/blob/main/results/res1.jpg)

### Example 2-1
![Example 2-1](https://github.com/quirrelHK/Fatigue-Detection/blob/main/results/res2-1.jpg)

### Example 2-2
![Example 2-2](https://github.com/quirrelHK/Fatigue-Detection/blob/main/results/res2-2.jpg)

## Description
Long-term fatigue may develop into chronic syndrome and eventually harm a person's physical and mental health. Some system needs to be designed to assess the tiredness of a person to avoid short-term fatigue to be developed into chronic fatigue. Many systems exist to detect short-term fatigue in a person but most use only eye aperture as a criterion to assess levels of fatigue which might be good for detecting driver drowsiness but, performs poorly in detecting long-term fatigue.

Other facial cues like under-eyes, nose, skin and mouth might also help in detecting fatigue.

## Proposal
In this project, I aim to develop an efficient way to detect fatigue in person. The goal is to detect fatigue in a person using a single image. 
Facial points like eyes, under-eyes, nose, skin and mouth are important for detecting fatigue. We use Dlib to extract the facial ROI and train a classification model on this data.

## Training Data
A custom dataset is used for training the model. Data consist of 403 images for both classes which were collected from a few volunteers, a few images from a couple of online resources are also used in this data. After filtering the dataset, some performance improvements were seen.


## Pre-processing
Dlib is used for extracting ROIs on the face; eye, undereye, nose, jaw and mouth. Data is cleaned before training.


## Training
An ensemble model was created, and 5 CNNs were trained on the cleaned data, each for the region of interest.


## Performance 
The ensemble model resulted in an accuracy of 86% on the test data.

## Conclusion
The model seems to perform well on the dataset. Furthermore, since the dataset is very small for a deep learning based classifier, the model performs not so well on unseen data. There is some over-fitting on the training data.


## Future Work
Increase the quality and quantity of the dataset, maybe some synthetically increase the number of images. Some machine learning based approaches can also be explored.

