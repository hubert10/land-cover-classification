# -*- coding: utf-8 -*-
"""Remote Sensing  Predictables  - Classification V1.03

Created on Nov 27 at Manobi Africa Dakar 16:34:42 2021

@authors: Pierre C. Traore - ICRISAT/ Manobi Africa
          Steven Ndung'u - ICRISAT/ Manobi Africa
          Hubert Kanyamahanga - ICRISAT/ Manobi Africa
          Glorie Wowo - - ICRISAT/ Manobi Africa
"""
################################################################################
########################### Introduction  ######################################
################################################################################

"""
This notebook shows you how to perform supervised classification.The Classifier 
package handles supervised classification using Earth Engine resources. 
The general workflow for classification is:
"""
#####0. Environment Setup
#####1. Collect training data. Assemble features which have a property that
#######stores the known class label and properties storing numeric values for the predictors.
#####2. Instantiate a classifier. Set its parameters if necessary.
#####3. Train the classifier using the training data.
#####4. Classify an image or feature collection.
#####5. Estimate classification errors with independent validation data.
#####6. Generate raster products from of aoi given date windows
#####7. Generate estimated pixel areas per class classified

################################################################################
####################### Install and load packages ##############################
################################################################################

""" if you are working with Colab, this packages will require to be installed every time the 
script is run. If runnning from a local machine this can be commented out/deleted
"""
# Create a virtualenv and install all these packages and then you are ready to go
# pip install geemap,
# pip install ipygee
# pip install geopandas
# pip install js2py
# pip install folium
# pip install rasterio
# pip install tslearn
# pip install earthengine-api

# Import os packages
import os, ee, json, subprocess
import ipygee as ui
from os import path as op
from datetime import datetime

# Pandas modules to interact with spatial data
import geopandas as gpd
import pandas as pd
import numpy as np
from functools import reduce

# Root directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = BASE_DIR + "/"
print(PROJECT_ROOT)

################################################################################
################## Authenticate/Setup GEE, GDRIVE, WP #########################
################################################################################
# Change the path to reflect your environment
output_path = PROJECT_ROOT + "Results"

# Make sure that folder exists
if not op.isdir(output_path):
    os.mkdir(output_path)
# Login into GEE
ee.Authenticate()
ee.Initialize()

# set up your working paths

# Change the file paths to match the file locations in your Gee project.
# These tow files have to be uploaded in the GEE assets folder
study_site_path = "projects/ee-hubertkanye/assets/SRV"
crop_noncrop = "projects/ee-hubertkanye/assets/crop_noncrop_new"

# Select the start and end date of the time series of interest:
start_date = "2020-02-01"
end_date = "2020-06-30"
# Output directory
out_dir = os.path.join(os.path.expanduser("~"), output_path)

################################################################################
####################### Model Building & Preprocessing #########################
################################################################################

# Create an interactive map
# Map = geemap.Map()

# Function to mask clouds S2
def mask_cloud_and_shadows(image):
    quality_band = image.select("QA60")
    cloudmask = 1 << 10
    cirrusmask = 1 << 11
    mask = quality_band.bitwiseAnd(cloudmask).eq(0) and (
        quality_band.bitwiseAnd(cirrusmask).eq(0)
    )
    return image.updateMask(mask).divide(10000)


# Function to calculate vegetative indices layers, additional feature
# to enrich the classification product.


def add_indices(image):
    # selected indices: other veg indices of interest can be added
    ndbi = image.normalizedDifference(["B11", "B8"]).rename(["ndbi"])
    ndvi = image.normalizedDifference(["B8", "B4"]).rename(["ndvi"])
    mndwi = image.normalizedDifference(["B3", "B11"]).rename(["mndwi"])
    bsi = image.expression(
        "(( X + Y ) - (A + B)) /(( X + Y ) + (A + B)) ",
        {
            "X": image.select("B11"),  # swir1
            "Y": image.select("B4"),  # red
            "A": image.select("B8"),  # nir
            "B": image.select("B2"),  # blue
        },
    ).rename("bsi")
    return image.addBands(ndbi).addBands(ndvi).addBands(mndwi).addBands(bsi)


def load_data(study_site_path, crop_noncrop):
    # 1 Load SENTINEL-2 IMAGE COLLECTION
    sent2 = ee.ImageCollection("COPERNICUS/S2_SR")
    # 2 Load file that covers the Upper delta region of the SRV
    studysite = ee.FeatureCollection(study_site_path)
    # 3 Load our training datasets that containts crop and non crop geometry
    crop_noncrop = ee.FeatureCollection(crop_noncrop)
    return studysite, crop_noncrop


def get_composite(start_date, end_date, studysite):
    """Returns the composite"""
    # 4 Add our area of interest to the Map
    # Map.addLayer(studysite.geometry(), {}, "Area of Interest")

    # We will now search for Sentinel 2 imagery, a multispectral satellite
    # with ~10m resolution and repeat coverage every 5 days.
    # Filters will include selecting bands, a date range, and only imagery within a defined Area of Interest (AOI).

    filtered = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterDate(start_date, end_date)
        .filterMetadata("CLOUDY_PIXEL_PERCENTAGE", "less_than", 20)
        .map(mask_cloud_and_shadows)
        .filter(ee.Filter.bounds(studysite.geometry()))
    ).select("B.*", "SCL")
    # Input imagery is a cloud-free Sentinel_2 composite.
    composite = filtered.median().clip(studysite)
    # Append the vegetative indices layers to the composite
    composite = add_indices(composite)
    # Map.addLayer(composite, {"bands": ["B4", "B3", "B2"]}, "filtered_composite")
    # Map.addLayer(crop_noncrop.geometry())
    return composite


def get_class_labels(crop_noncrop):
    """Returns the class labels"""

    crop = crop_noncrop.filter(ee.Filter.eq("landcover", 1))
    noncrop = crop_noncrop.filter(ee.Filter.eq("landcover", 2))
    # Combine the manually trained data of the crops into a reference dataset*/
    class_labels = crop.merge(noncrop)
    # Add our classes to the Map
    # Map.addLayer(class_labels)
    return class_labels


def train_test_split(composite, class_labels):
    """Returns the both the training and testing partitions"""

    # Overlay the point on the image to get training data.
    training = composite.sampleRegions(
        **{"collection": class_labels, "properties": ["landcover"], "scale": 70}
    )
    # random uniforms to the training dataset.
    withRandom = training.randomColumn()

    # We want to reserve some of the data for testing, to avoid overfitting the model.
    split = 0.7  # Roughly 70% training, 30% testing.
    trainingPartition = withRandom.filter(ee.Filter.lt("random", split))
    testingPartition = withRandom.filter(ee.Filter.gte("random", split))
    return trainingPartition, testingPartition


# Define some variables to be used for model training and validation
studysite, crop_noncrop = load_data(study_site_path, crop_noncrop)
composite = get_composite(start_date, end_date, studysite)
class_labels = get_class_labels(crop_noncrop)
train_data, test_data = train_test_split(composite, get_class_labels(crop_noncrop))


def train_the_classifier(params, model, train_data):
    """Returns the classifier model"""

    label = "landcover"
    # Training the classifier: model = smileRandomForest
    classifier = (
        model(**params)
        .setOutputMode("CLASSIFICATION")
        .train(train_data, label, composite.bandNames())
    )
    # Create an RF classifier with custom parameters
    return classifier


def perform_classification(composite):
    # Classify the image.
    classified = composite.classify(
        train_the_classifier(
            rf_init_params, ee.Classifier.smileRandomForest, train_data
        )
    )
    sld_intervals_crop = (
        "<RasterSymbolizer>"
        + '<ColorMap type="intervals" extended="false">'
        + '<ColorMapEntry color="#10d22c" quantity="1" label="Crop"/>'
        + '<ColorMapEntry color="#000000" quantity="2" label="Non Crop"/>'
        + "</ColorMap>"
        + "</RasterSymbolizer>"
    )
    # Add the classified image to the Map
    # Map.addLayer(classified.sldStyle(sld_intervals_crop), {}, "RF Classified Layer")
    # Map.addLayer(classified, {"min": 0, max:3, "palette": ['#37d615', '#3223d6']}, 'RF_classified')
    return classified


# Set initial params for RF classifier
rf_init_params = {
    "numberOfTrees": 150,  # the number of individual decision tree models
    "variablesPerSplit": None,  # the number of features to use per split
    "minLeafPopulation": 1,  # smallest sample size possible per leaf
    "bagFraction": 0.5,  # fraction of data to include for each individual tree model
    "maxNodes": None,  # max number of leafs/nodes per tree
    "seed": 34,
}  # random seed for "random" choices like sampling. Setting this allows others to
# reproduce your exact results even with stochastic parameters


# Define the classifier and the classified
classifier = train_the_classifier(
    rf_init_params, ee.Classifier.smileRandomForest, train_data
)

classified = perform_classification(composite)

################################################################################
############################ Model Evaluation ##################################
################################################################################


def test_the_classifier(classifier, test_data):
    """Returns the confusion matrix"""

    # test_data:  testingPartition
    test = test_data.classify(classifier)
    # Get a confusion matrix representing expected accuracy.
    rf_confusionMatrix = test.errorMatrix("landcover", "classification")
    return rf_confusionMatrix


rf_confusionMatrix = test_the_classifier(classifier, test_data)

# 5. Accuracy Assessment

# More statistics
headers = [
    "Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "F1-score",
    "Kappa statistic",
]

metrics = []


def compute_metrics(conf_matr):
    """Returns a list of six computed metrics"""

    y = np.array(
        [
            [conf_matr.getInfo()[1][1], conf_matr.getInfo()[1][2]],
            [conf_matr.getInfo()[2][1], conf_matr.getInfo()[2][2]],
        ]
    )
    cm = np.asmatrix(y)
    accurracy = conf_matr.accuracy().getInfo()
    metrics.append(accurracy)
    precision = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    metrics.append(precision)
    recall = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    metrics.append(recall)
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    metrics.append(specificity)
    f1 = 2 * (precision * recall) / (precision + recall)
    metrics.append(f1)
    kappa = conf_matr.kappa().getInfo()
    metrics.append(kappa)
    return metrics


data = compute_metrics(test_the_classifier(classifier, test_data))
rf_acc_ass = pd.DataFrame(data=[data], columns=headers)
print(f"Random Forest:\n{rf_acc_ass.head()}\n")

################################################################################
######################### Model Outputs/Exports ################################
################################################################################


def export_accurracy_assessment(out_dir, data, header):
    """Exports the metrics as a csv file to the provided output path"""
    accurracy_assessment = os.path.join(out_dir, "accurracy_assessment.json")
    data = dict(zip(header, data))
    with open(accurracy_assessment, "w") as outfile:
        json.dump(data, outfile)


def export_classified_tiff(classified, studysite, start_date, end_date):
    """Export the image, specifying scale and region."""
    # Warning, This specific export takes at least 1 hour, and 700+ MB
    # GEE folder does not use a path, it uses a unique name to find the export location

    now = datetime.now()  # current date and time
    time = now.strftime("%m%d%Y_%H%M")
    # We will save it to Google Drive for later reuse
    raster_name = op.join(
        out_dir,
        "crop_noncrop_classification_{}_to_{}_{}".format(
            start_date, end_date, str(time)
        ),
    )
    task = ee.batch.Export.image.toDrive(
        **{
            "image": classified,
            "description": "crop_noncrop_classification_{}_to_{}_{}".format(
                start_date, end_date, str(time)
            ),
            "folder": op.basename(output_path),
            "scale": 10,
            "fileNamePrefix": raster_name.split("/")[-1],
            "region": studysite.geometry(),
            "fileFormat": "GeoTIFF",
            "formatOptions": {"cloudOptimized": "true"},
            "maxPixels": 1e12,
        }
    )
    # This task will run in the background even if you close this notebook.
    # You can also check on the status of the task through the Javascript GEE interface
    # https://code.earthengine.google.com
    return task.start()


################################################################################
################### Area calculations (Pixel Based) ############################
################################################################################

# Class Area calculation


def get_item(item):
    areaDict = ee.Dictionary(item)
    classNumber = ee.Number(areaDict.get("classification")).format()
    # The result will be in square meters, this converts them into square kilometers
    area = ee.Number(areaDict.get("sum")).divide(1e6).round()
    return ee.List([classNumber, area])


def classAreaLists(class_areas):
    return class_areas.map(get_item)


def export_class_area(classified, start_date, end_date):
    areaImage = ee.Image.pixelArea().addBands(classified)
    areas = areaImage.reduceRegion(
        **{
            "reducer": ee.Reducer.sum().group(
                **{"groupField": 1, "groupName": "classification",}
            ),
            "geometry": studysite.geometry(),
            "scale": 10,
            "tileScale": 16,  # Higher values of tileScale result in tiles smaller by a factor of tileScale^2 and this won't fit in memory for large image
            "maxPixels": 1e10,
        }
    )
    class_areas = ee.List(areas.get("groups"))
    # Flattens said dictionary so it is readable for us
    result = ee.Dictionary(classAreaLists(class_areas).flatten())
    now = datetime.now()  # current date and time
    time = now.strftime("%m%d%Y_%H%M")
    class_area = os.path.join(out_dir, "class_area_{}.json".format(time))
    fc = ee.FeatureCollection([ee.Feature(None, result)])
    task = ee.batch.Export.table.toDrive(
        **{
            "collection": fc,
            "description": "class_area_coverage_{}_to_{}_{}".format(
                start_date, end_date, str(time)
            ),
            "folder": op.basename(output_path),
            "fileNamePrefix": class_area.split("/")[-1],
            "fileFormat": "CSV",
        }
    )
    return task.start()


# export_class_area(classified)
# Call the functions

# export_accurracy_assessment(out_dir, data, headers)
# export_class_area(classified, start_date,end_date)
# export_classified_tiff(classified, studysite,start_date,end_date)


################################################################################
################################# Predictions ##################################
################################################################################

"""
Here, we want to generate a new product provided new windows for the following year. 
The model above has been trained using images sampled from 2020, we now can use the 
model to predict outputs  for 2021. 
"""
# Create a new window

################################################################################
############################# 2021 Prediction 1 ################################
################################################################################

start_date_new = "2021-02-14"
end_date_new = "2021-03-06"
pred_comp = get_composite(start_date_new, end_date_new, studysite)
print("Prediction between: 14 FEB - 06 MAR")
pred_comp = perform_classification(pred_comp)
export_class_area(pred_comp, start_date_new, end_date_new)
export_classified_tiff(
    pred_comp, studysite, start_date_new, end_date_new,
)


################################################################################
############################# 2021 Prediction 2 ################################
################################################################################

start_date_new = "2021-02-14"
end_date_new = "2021-03-16"
pred_comp1 = get_composite(start_date_new, end_date_new, studysite)
print("Prediction between: 14 FEB - 16 MAR")
pred_comp1 = perform_classification(pred_comp1)
export_class_area(pred_comp1, start_date_new, end_date_new)
export_classified_tiff(pred_comp1, studysite, start_date_new, end_date_new)

################################################################################
############################# 2021 Prediction 3 ################################
################################################################################

start_date_new = "2021-02-01"
end_date_new = "2021-03-03"
pred_comp2 = get_composite(start_date_new, end_date_new, studysite)
print("Prediction between: 01 FEB - 03 MAR")
pred_comp2 = perform_classification(pred_comp2)
export_class_area(pred_comp2, start_date_new, end_date_new)
export_classified_tiff(pred_comp2, studysite, start_date_new, end_date_new)
