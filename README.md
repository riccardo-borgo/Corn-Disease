# Corn-Disease

The aim of this project is to buil models (specifically neural networks) that are able to classify images of corn leaves based on a specific disease.

The different diseases are:
- **Blight**: foliar disease of corn (maize) caused by a parasite. With its characteristic cigar-shaped lesions, this disease can cause significant yield loss in susceptible corn hybrids;
- **Common Rust**: caused by the a fungus and occurs every growing season. It is seldom a concern in hybrid corn. Early symptoms of common rust are chlorotic flecks on the leaf surface;
- **Gray Leaf Spot**: it is a foliar fungal disease that affects maize. GLS is considered one of the most significant yield-limiting diseases of corn worldwide. There are two fungal pathogens that cause GLS. Symptoms seen on corn include leaf lesions, discoloration (chlorosis), and foliar blight;
- **Healthy**: this is not properly a disease but in order to make an all around classification there has been included also the healthy leaves.

After briefly discussing some tecnical aspects about biology let's jump into something more interesting for us.

We started with a folder, divided into subfolders, containing the different leaf images divided according to the disease. The first step was to build an actual dataset.

## First Part - Data Loading

Here we simply created the dataset using a very simple function:

```python
def create_data():
    name = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
    final_images = []
    final_labels = []

    for disease in name:
        folder_path = '/kaggle/input/Corn Images/' + disease
        images = []
        labels = [disease] * len(os.listdir(folder_path))

        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)
            images.append(img)

        final_images.extend(images)
        final_labels.extend(labels)

    return final_images, final_labels
```

## Part 2 - Data Visualization

When dealing with image classification tasks I think that the EDA part is not very heavy, since here we can only show the images and analyze the labels. And, in fact, we done that:

![image](https://github.com/riccardo-borgo/Corn-Disease/assets/51230348/33e3f92e-b645-4003-ad7c-95671117cc23)

![image](https://github.com/riccardo-borgo/Corn-Disease/assets/51230348/667db5fc-1beb-4ca5-96a7-61c29a5ab13e)

An important thing to say about the second plot is that we can clearly see a slight imbalance between the labels: the **Gray Leaf Spot** disease has a tittle less examples comapred to the others. We will address this problem later on.

## Part 3 - Data Preparation

 In this part we addressed a very tricky problem about the number of channels. We saw that vey few images presented 4 channels instead of 3 (for a not known reason) so later on we couldn't proceed with the training of the network. After spending hours trying to find the problem we detect it and convert every image into a 3 chanell image.

 After that we split the dataset into training and validation. We decided to choose a stratified split since the distribution of the labels was not homogenous, even though this problem will be addressed.

 ![image](https://github.com/riccardo-borgo/Corn-Disease/assets/51230348/6d0d7771-c6f5-4f7b-b339-948944331fd8)

 ## Part 4 - Data Augmentation

 The Data Augmentation process is fundamental when dealing with images. This process involves applying some transformations to the original images in order to "create" other examples to train the network. After the process the user can decided to either mantain both the images (Augmented and not) or only use augmented images. We have procedeed with the second option.

In out case the path was:
- Resizing: (224x224)
- Rotation: from -45 to 45 degrees
- Horizontal Flip
- Vertical Flip
- Zoom: from 0 to 50 pixels
- Contrast
- Brightness

All these adjustments has been made through a probability, so if a number chosen randomly was lower than 0.2 then the adjustment was applied

Since we encountered some problems applying an oversampling technique before applying data augmentation we were obliged to oversample augmented images.

We decided to use SMOTE for oversampling and this is the result:

![image](https://github.com/riccardo-borgo/Corn-Disease/assets/51230348/b7692f75-420e-4dcd-b5e7-da21b025dd53)

This is also the result of the Augmentation process:

![image](https://github.com/riccardo-borgo/Corn-Disease/assets/51230348/d79af918-315b-4f1a-a4ad-2ec6bf9d8d6d)

## Part 5 - Data Modelling

The last, but not least, part of the project is the application of neural networks in order to predict, as best as possible, the 4 diseases.

We firstly started with a first proposal of a CNN, that we decided to name: **C-CNN** (Corn-CNN). Here there is the structure:

![image](https://github.com/riccardo-borgo/Corn-Disease/assets/51230348/651ef949-7ec1-45db-bbd8-dd2f8adecc64)

It is a very simple network but after some trials we found out that maybe the task is relatively simple so we don't need a complex model. We added also a callback function in order to keep the epochs high but the system will stop the training when some criteria will be met.

The C-CNN performed quite well:

![image](https://github.com/riccardo-borgo/Corn-Disease/assets/51230348/2cd76f72-cd32-4ed6-bfb6-7f1605f2e04e)

But after checking the performance in the validation set the result was: 
- Loss: 0.803
- Accuracy: 0.805

So we can assert that our model is capable to generalize.

We can now pass to the last subpart, the application of **Transfer Learning**.

Since this is a process that nowadays is becoming more and more spread across all fields of AI and ML we deicided to give it a try. We implemented a **VGGNet16** network with **imagenet** weights, linked with two last layers: a Flatten and a Dense with a Softmax function.

The results of this network are incredible: after only two epochs the accuracy reaches 0.91 on the training set. At the end, the final performances are:
- 0.96 accuracy on the training
- 0.89 accuracy on the validation

It is present a slight overfit but the results are still impressive.






 

