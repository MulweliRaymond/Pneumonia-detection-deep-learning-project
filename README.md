# Pneumonia-detection-deep-learning-project

Background

Pneumonia is a respiratory illness characterized by inflammation in the lungs caused by various pathogens, including bacteria, viruses, and fungi. Early and accurate diagnosis of pneumonia is crucial for timely treatment and patient recovery. Traditionally, pneumonia diagnosis has relied on clinical symptoms, laboratory tests, and chest X-ray examinations interpreted by radiologists. However, manual interpretation of chest X-rays can be subjective, time-consuming, and prone to variability among radiologists.

In recent years, deep learning techniques, particularly convolutional neural networks (CNNs), have shown remarkable success in various computer vision tasks, including medical image analysis. CNNs have the ability to automatically learn relevant features from raw image data, making them well-suited for pattern recognition tasks such as pneumonia detection on chest X-rays.

This project aims to develop a deep learning model based on a convolutional neural network architecture for automated pneumonia classification from chest X-ray images. The model should be able to accurately distinguish between normal and pneumonia cases, thereby assisting radiologists and healthcare professionals in the diagnosis process.
Aim and Objectives
Aim

The primary aim of this project is to develop, train, and evaluate two distinct Convolutional Neural Network (CNN) models for accurately classifying chest X-ray images as either Pneumonia or Normal. The project seeks to identify the most effective model in terms of accuracy, precision, recall, and F1-score to potentially aid in the timely and accurate diagnosis of pneumonia in clinical settings.
Objectives

    Acquire the chest X-ray dataset from Kaggle, consisting of 4273 pneumonia cases and 1583 normal cases.
    Preprocess the images by resizing them to a consistent resolution and normalizing pixel values to ensure compatibility with CNN input requirements.
    Organize the dataset into training, validation, and test sets to facilitate model training and evaluation.
    Develop and define the architecture for two distinct CNN models with different convolutional layers and structures.
    Train both CNN models using the training dataset and use the validation dataset to tune hyperparameters and prevent overfitting, ensuring the models are well-calibrated.
    Evaluate the performance of both CNN models on the test dataset using various metrics, including accuracy, precision, recall, and F1-score.
    Compare the performance of the two models based on the evaluation metrics and identify the superior model for potential deployment in clinical applications.

Dataset

The dataset used in this study is based on medical data of chest X-ray images provided by Kermany et al. [2018]. The data was obtained from Kaggle and is available at Kaggle Chest X-Ray Pneumonia Dataset. Every chest X-ray image was taken as a standard clinical procedure for the patients. The collection contains photos with a range of resolutions, from 712x439 to 2338x2025. The collection contains 4273 photos of pneumonia cases and 1583 images of normal cases. The dataset has subfolders for each image category (Pneumonia/Normal) and is arranged into three folders (train, test, and validation). In our dataset, 0 represents normal cases, and 1 represents pneumonia cases.
Data Exploration

To gain insights into the composition of our dataset and ensure a proper understanding of its characteristics, we analyzed the class distribution of the chest X-ray images. The figure below illustrates the distribution of pneumonia and normal cases within the original dataset. This distribution highlights the significant class imbalance, with pneumonia cases outnumbering normal cases.

To provide a visual understanding of the dataset, we randomly selected and displayed a sample of images from the combined dataset. The figure below shows four randomly chosen chest X-ray images from the entire dataset, including both normal and pneumonia cases. This visualization helps illustrate the variety and characteristics of the images our models are trained on.

Data Preprocessing

In this section, we outline the steps taken to preprocess the image data, including loading, transforming, and batching the data.
Loading the Dataset

To handle our dataset efficiently, we use the ImageFolder class from the torchvision library. This class is designed for datasets where images are organized in a directory structure, with each subdirectory representing a different class. We load our training, validation, and test datasets by specifying the root directories where these images are stored.
Transforming the Data

Transformations are applied to standardize the input images and to augment the dataset, which helps in improving the model’s robustness and generalization ability. We use a series of transformations provided by the torchvision.transforms module:

    Resizing: All images are resized to a uniform size of 64x64 pixels. This ensures that all input images have the same dimensions, which is a requirement for processing in neural networks.
    Random Horizontal Flip: To augment our dataset and make the model more robust to variations, we randomly flip images horizontally with a probability of 0.5. This means that, on average, half of the images will be flipped during training, effectively doubling the variety of our training data.
    To Tensor: Finally, the images are converted to PyTorch tensors. This transformation normalizes the pixel values of the images to the range [0, 1], which is suitable for input into neural networks.

Combining and Splitting the Dataset

To ensure an even distribution of data and to make the training, validation, and testing phases more robust, we combined our initial training, validation, and test datasets into a single dataset. This approach helps to avoid data leakage and ensures that our model is evaluated on unseen data. The following steps were performed:

    Combining Datasets: We merged the original training data, validation data, and testing data into a single dataset. This step creates a unified dataset from which we can draw our new training, validation, and test subsets.
    Shuffling and Splitting: After combining the datasets, the entire dataset was shuffled to ensure randomness. We then split it into three subsets: 70% for training, 20% for testing, and 10% for validation. This method ensures that each subset is representative of the overall data distribution.
    Creating Subsets: Using the generated indices, we created new subsets for training, validation, and testing. This allows us to feed the data into our model in a structured way.

Creating Data Loaders

Once the datasets are loaded and transformed, we use data loaders to handle the batching, shuffling, and parallel loading of data. The DataLoader class from torch.utils.data is used for this purpose:

    Batch Size: We set a batch size of 16, which determines the number of samples that will be propagated through the network at one time. This size is a compromise between computational efficiency and the stability of the gradient updates.
    Number of Workers: We utilized all available CPU cores to load the data in parallel. This speeds up the data-loading process, especially when working with large datasets.
    Shuffling: For the training dataset, we enabled shuffling to ensure that the model does not learn any order-specific patterns from the data. For validation and test datasets, we disabled shuffling to maintain consistency in evaluation.

By following these preprocessing steps, we ensured that our data is efficiently loaded, uniformly transformed, and appropriately batched. This prepares the dataset for effective training and evaluation of our deep learning models, contributing to better model performance and generalization.
Classification Networks and Algorithms

In this study, we employed two deep convolutional neural network architectures for the image classification task. These architectures have been instrumental in advancing the field of computer vision and have demonstrated remarkable performance on various image recognition tasks.
Model 1: Layers and Hyperparameters

Layers:

    Four Convolutional Layers
    Four Activation Layers with ReLU functions
    Two Pooling Layers (MaxPool2d)
    One Dropout Layer with probability: 0.5
    One Flatten Layer
    One Linear Fully Connected Layer
    One Sigmoid Output Activation

Hyperparameters:

    Convolutional Layer 1: in channels=input channels, out channels=20, kernel size=3, stride=1, padding=1
    Convolutional Layer 2: Input Channels=20, Output Channels=20, Kernel Size=3, Stride=1, Padding=1
    Convolutional Layer 3: Input Channels=20, Output Channels=20, Kernel Size=3, Stride=1, Padding=1
    Convolutional Layer 4: Input Channels=20, Output Channels=20, Kernel Size=3, Stride=1, Padding=1
    MaxPool2d Layer: Kernel Size=2
    Dropout Layer: Probability=0.5
    Fully Connected Layer: Input Features=20 * 16 * 16 and Output Features=1

Model 2: Layers and Hyperparameters

Layers:

    Two Convolutional Layers
    Two Activation Layers with ReLU functions
    Two Pooling Layers (MaxPool2d)
    One Dropout Layer with probability: 0.5
    One Flatten Layer
    Two Linear Fully Connected Layers
    One Sigmoid Output Activation

Hyperparameters:

    Convolutional Layer 1: in channels=input channels, out channels=32, kernel size=5, stride=1, padding=2
    MaxPool2d Layer 1: Kernel Size=2
    Convolutional Layer 2: in channels=32, out channels=64, kernel size=3, stride=1, padding=1
    MaxPool2d Layer 2: Kernel Size=2
    Fully Connected Layer 1: Input Features=64 * 16 * 16 and Output Features=1024
    Fully Connected Layer 2: Input Features=1024 and Output Features=1
    Dropout Layer: Probability=0.5

Model Training and Evaluation

The training and evaluation process was conducted on an NVIDIA Tesla T4 GPU, provided by Google Colab, to leverage GPU acceleration and reduce training time.
Training Process

The training process involved the following key steps:

    Model Initialization: We initialized the CNN models and defined the loss function and optimizer. We used the Binary Cross-Entropy Loss (BCEWithLogitsLoss) for binary classification and the Adam optimizer with a learning rate of 0.001.

    Training Loop: The training loop iterated over the epochs, and within each epoch, the loop iterated over the batches of the training dataset. For each batch, the following operations were performed:
        Forward pass: The input images were passed through the network to obtain the predicted outputs.
        Loss calculation: The loss was calculated between the predicted outputs and the true labels.
        Backward pass: The gradients were computed using backpropagation.
        Parameter update: The model parameters were updated using the optimizer.

    Validation: After each epoch, the model was evaluated on the validation dataset to monitor its performance and detect any signs of overfitting. The validation loss and accuracy were recorded.

    Early Stopping: Early stopping was implemented to prevent overfitting. If the validation loss did not improve for a specified number of epochs (patience), training was stopped.

Evaluation Metrics

The performance of the models was evaluated using the following metrics:

    Accuracy: The ratio of correctly predicted instances to the total instances.
    Precision: The ratio of correctly predicted positive instances to the total predicted positive instances.
    Recall: The ratio of correctly predicted positive instances to the total actual positive instances.
    F1-Score: The harmonic mean of precision and recall.

Confusion Matrix

A confusion matrix was generated to provide a detailed view of the model's performance. The confusion matrix included the following components:

    True Positives (TP): Correctly predicted positive instances.
    True Negatives (TN): Correctly predicted negative instances.
    False Positives (FP): Incorrectly predicted positive instances.
    False Negatives (FN): Incorrectly predicted negative instances.

Training and Validation Curves

To visualize the training process, training and validation loss curves were plotted for each epoch. These curves help identify overfitting and monitor the convergence of the models.

In this study, we developed and evaluated two convolutional neural network models for automated pneumonia detection from chest X-ray images. Both models demonstrated high accuracy, with Model 2 slightly outperforming Model 1 in terms of all evaluation metrics.

The results indicate that deep learning techniques, specifically CNNs, can effectively assist in the diagnosis of pneumonia, potentially reducing the workload of radiologists and improving diagnostic accuracy. Future work could involve exploring more advanced architectures, incorporating more diverse datasets, and deploying the models in clinical settings for real-world evaluation.

This project showcases the potential of deep learning in medical image analysis and highlights the importance of rigorous evaluation and model comparison to identify the best-performing solutions.

References

Kermany, D. S., Goldbaum, M., Cai, W., Valentim, C. C. S., Liang, H., Baxter, S. L., ... & Zhang, K. (2018). Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. Cell, 172(5), 1122-1131.e9. DOI: 10.1016/j.cell.2018.02.010

P. T. Mooney, "Chest X-Ray Images (Pneumonia)," Kaggle, [Online]. Available: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
