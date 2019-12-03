# Final-Project-Group-11

## Objective
The purpose of this project is to use deep learning to train a network that can measure building density of areas that are currently poorly mapped by using geospatial data and satellite images as input. This model could then potentially be used to measure the growth or expansion of any given area by comparing historical images to current ones. This problem was chosen due to the interest of our team having taken GIS and machine learning courses while in the George Washington University Data Science Masters program.

## Data
There were many potential data sources of satellite images that could be used for this task, including Esri images and opensource platform data such as OpenStreetMap. For training purposes, the scope of the area to train was narrowed to the DC, Maryland, and Virginia region. This provided more than enough data to train our network as a proof of concept, but if needed in the future there are methods to generate more images off the original sources. In addition to unstructured satellite image data, we leveraged structured building coverage maps to help generate building density labels for the images. The building coverage maps we used were generated and open sourced by Microsoft for the entire United States and are available online as geoJSONs. To read more about the techniques they used to generate 125,192,184 building shapes, visit [this Github link](https://github.com/microsoft/USBuildingFootprints).

Building Polygons

## Tools Used
This project leveraged a number of technologies and libraries during the preparation, analysis and presentation phase. The project is presented in the form of an interactive [HTML page](https://dats6203.azurewebsites.net/) using Python Flask, Bootstrap, CSS, and client-side JavaScript. It also leveraged a custom scraping and retrieval framework with Python, GeoPandas, Folium, Selenium and the Requests library for data retrieval and processing. Model training and image augmentation leveraged Tensorflow, Keras and OpenCV computer vision libraries. Source code management leveraged GitHub and Visual Studio Code.

Imagery Distribution

## Network and Framework Selection
We opted to use a Convolutional Neural Network (CNN) due to its strengths in image processing and pattern recognition in addition to experimenting with Multi-Layer Perceptron (MLP) for benchmarking and comparison. While the plan started with a standard CNN, we implemented additional customizations to improve performance. We used the Keras framework to create the CNN network with Tensorflow as the back-end. We took roughly 9,000 training images and augmented them through 90, 180, 270 degree rotations and flips, creating over 50,000 augmented images and then ran them through multiple convolutional layers with fully connected layers with regularization, pooling and dropout to prevent overfitting against the training data. We used model checkpointing against validation loss using mean square error. While we pulled data from across DC, Maryland and Virginia, we upsampled urban areas to get a more even training distribution, and split our training and validation set geographically to prevent contamination from overlapping images.

CNN Loss

## Architecture
For architecture and computing/storage resources, this project leveraged Google Cloud Platform (GCP), Amazon Web Services (AWS) and Microsoft Azure infrastructure for multiple elements including the hosting and integration of the Python Flask. For the Flask app, we used Azure Web Apps which is a fully-managed web hosting platform supporting continuous integration/continuous deployment (CI/CD), application monitoring and metrics. Hosting of image data used GCP storage buckets, while model training and deep learning experimentation leveraged GCP and AWS GPU-backed compute instances.
