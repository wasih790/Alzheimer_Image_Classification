# Alzheimer's Disease Image Classification using CNN Models

## Project Description
The objective of this project is to develop a deep learning model using Convolutional Neural Networks (CNN) to accurately classify brain MRI images of Alzheimer’s disease. The dataset consists of four classes to identify: MildDemented, ModerateDemented, NonDemented, and VeryMildDemented. The model's accurate identification of these classes can assist in the early detection and diagnosis of Alzheimer's disease, contributing to better control and treatment options for affected individuals.


## Business Understanding
### Problem Statement
Alzheimer's disease is a significant and increasing public health crisis with a huge impact on patients, families, and healthcare systems. Early detection of Alzheimer's can lead to better control and treatment, potentially slowing down its progression and enhancing the quality of life for affected individuals.

### Target Audience
The target audience for this project includes healthcare professionals, neurologists, medical researchers, and data scientists working on Alzheimer's disease-related studies. Additionally, medical institutions and hospitals looking to incorporate advanced technology for early diagnosis of Alzheimer's could also benefit from the outcomes of this project.

### Impact on the Real World
If successfully developed and deployed, the CNN model for Alzheimer's disease identification could have several real-world impacts:

- Early Detection: Early detection allows for intervention and treatment, potentially slowing down the advancement of the disease and improving patient outcomes.
- Reduced Healthcare Costs: Early detection and accurate diagnosis can lead to more targeted treatment plans, potentially reducing long-term healthcare costs.
- Research Advancement: The model can assist researchers by providing a tool for large-scale screening and analysis, leading to a deeper understanding of Alzheimer's disease and potential areas for further research.


## Data Understanding
### Source of Raw Data
The raw data for this project was obtained from a public dataset on Hugging Face. The dataset consists of brain MRI images stored in JPEG (.jpg) format, and they are organized into folders based on their respective classes (MildDemented, ModerateDemented, NonDemented, and VeryMildDemented).


## Data Preparation
### Preprocessing Steps
The following preprocessing steps were applied to the data:

- Normalization: Scaling the pixel values of the images to a range between 0 and 1 to improve model performance.
- Data Augmentation: Introducing variations in the data by applying transformations like rotation and flipping, increasing the diversity of the dataset, and improving model generalization.
- Train-Test Split: Dividing the dataset into training and testing sets to evaluate the model's performance on unseen data. But instead of the sklearn method, we used ImageDataGenerator, created an training generator, validation generator, and testing generator. 


## Modeling
### Modeling Techniques
For the image classification problem of identifying Alzheimer's disease stages based on brain MRI images, we used Convolutional Neural Networks (CNNs). Three CNN models were developed from scratch, two transfer learning models (GoogLeNet and ResNet50) were explored, and a Keras Tuner with Hyperband model was developed. 

### Target Variable
The dataset used in this project consists of MRI scans of the brain with four different classes to classify:

- Very Mild Demented
- Moderate Demented
- Mild Demented
- Non-Demented

### Model Evaluation
The primary metrics used to evaluate the models were Accuracy and AUC scores.

#### Models from Scratch
Three Convolutional Neural Network (CNN) models were created from scratch. The details of each model and their accuracies are as follows:

1. Model v1:
- Architecture: Custom CNN
- Accuracy: 0.6930
- AUC: 0.9066

2. Model v2:
- Architecture: Transfer Learning (VGG16-based)
- Accuracy: 0.8912
- AUC: 0.9766

3. Model v3:
- Architecture: Transfer Learning with Regularization (VGG16-based)
- Regularization: Dropout layers (0.2, 0.4, 0.1), TensorBoard was also used to log different method of regularization.
- Accuracy: 0.9341
- AUC: 0.9923

#### Transfer Learning Models
Two transfer learning models were used - GoogLeNet and ResNet50, both implemented using Google Colaboratory on the cloud.

1. GoogLeNet Model:
- Accuracy: 0.7588
- AUC: 0.9388

2. ResNet50 Model:
- Accuracy: 0.6383
- AUC: 0.8801

#### Hyperparameter Tuning with Keras Tuner
- The Keras Tuner library was used to find the optimal set of hyperparameters for the TensorFlow models. Hyperband tuning algorithm was employed in a for loop to create models with 2 to 20 layers, each with 32 to 512 units. The best model achieved an accuracy of 0.8075 and can be found in the 'Keras Tuner' folder.

### Tools/Methodologies
Python Libraries: NumPy, Os, TensorFlow, Keras, Sklearn
Analysis Environment: Python programming language and its libraries on the Google Colaboratory cloud platform


## Folder Structure
The repository contains the following folders:

1. Models From Scratch: Contains the three CNN models developed from scratch with their respective accuracy scores.
2. Transfer Learning Models: Contains the transfer learning models (GoogLeNet and ResNet50) and their accuracy scores.
3. Keras Tuner: Contains the model developed using Keras Tuner with Hyperband tuning algorithm and its accuracy score.
4. Useful EDA: Contains the Umap for the images and the localization of anomalies in images.


## Conclusion
The best-performing model achieved an accuracy of 0.9341 and was developed from scratch with regularization techniques. The CNN models outperformed the transfer learning models (GoogLeNet and ResNet50) in this specific task of Alzheimer's disease image classification. The Keras Tuner model using Hyperband achieved an accuracy of 0.8075, demonstrating the efficacy of hyperparameter tuning.

The project's outcomes have the potential to contribute significantly to the early detection and diagnosis of Alzheimer's disease, leading to improved patient outcomes and reduced healthcare costs. Furthermore, the models and techniques developed can serve as valuable tools for medical professionals and researchers in the field of Alzheimer's disease-related studies.