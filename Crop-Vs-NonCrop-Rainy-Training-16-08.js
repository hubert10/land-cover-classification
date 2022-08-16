// #############################################################################
// ### Data (sentinel 2)  Preparation and feature extraction                ###
// #############################################################################
 
//Function to mask clouds S2 // 
function maskS2srClouds(data) {
  var qa = data.select('QA60'); 

  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  return data.updateMask(mask).divide(10000);
}

// Filter Sentinel-2 collection for the Rainy planting season for 2021
var filtered = sent2.filterDate("2021-08-01", "2021-11-15")
                .filterMetadata('CLOUDY_PIXEL_PERCENTAGE',
                                'less_than', 20)
                .map(maskS2srClouds)
                .select('B.*','SCL')
                .filterBounds(studysite);

//Exploring image collection and its metadata
print("A Sentinel-2 scene:", filtered);

var composite = filtered.median().clip(studysite); 
print("composite image:", composite);

//Function to calculate vegetative indices layers
var addIndices = function(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename(['ndvi']);
  var ndbi = image.normalizedDifference(['B11', 'B8']).rename(['ndbi']);
  var mndwi = image.normalizedDifference(['B3', 'B11']).rename(['mndwi']); 
  var bsi = image.expression(
      '(( X + Y ) - (A + B)) /(( X + Y ) + (A + B)) ', {
        'X': image.select('B11'), //swir1
        'Y': image.select('B4'),  //red
        'A': image.select('B8'), // nir
        'B': image.select('B2'), // blue
  }).rename('bsi');
  return image.addBands(ndvi).addBands(ndbi).addBands(mndwi).addBands(bsi)
}

//append the vegetative indices layers to the composite
var composite = addIndices(composite);

/*Combine the manually trained data of the crops into a reference dataset*/
// var crop = crop_noncrop_rainy.filter(ee.Filter.eq('landcover', 1));
var crop = crop;
Map.addLayer(crop, {color: 'green'}, 'crop');
print('crop', crop);

// var noncrop = crop_noncrop_rainy.filter(ee.Filter.eq('landcover', 2));
var noncrop = noncrop;
Map.addLayer(noncrop, {color: 'brown'}, 'noncrop');
print('noncrop', noncrop);

/*Combine the manually trained data of the crops into a reference dataset*/
var class_labels = crop.merge(noncrop);
print('class_labels', class_labels);

// Overlay the point on the image to get training data.
var training = composite.sampleRegions({
  collection: class_labels, 
  properties: ['landcover'], 
  scale: 90
});

// #############################################################################
// ###               Modelling                                               ###
// #############################################################################

// random uniforms to the training dataset.
var withRandom = training.randomColumn('random');

// We want to reserve some of the data for testing, to avoid overfitting the model.
var split = 0.7;  // Roughly 70% training, 30% testing.
var trainingPartition = withRandom.filter(ee.Filter.lt('random', split));
var testingPartition = withRandom.filter(ee.Filter.gte('random', split));


print('training partition', trainingPartition.size());
print('testing partition', testingPartition.size());

var init_params = {"numberOfTrees":150,// the number of individual decision tree models
              "variablesPerSplit":null,// the number of features to use per split
              "minLeafPopulation":1,// smallest sample size possible per leaf
              "bagFraction":0.5, // fraction of data to include for each individual tree model
              "maxNodes":null, // max number of leafs/nodes per tree
               "seed":34};// random seed for "random" choices like sampling. Setting this allows others to reproduce your exact results even with stocastic parameters
// Train a classifier.
var classifier = ee.Classifier.smileRandomForest(init_params).train({
  features: trainingPartition,  
  classProperty: 'landcover', 
  inputProperties: composite.bandNames()
});

// Classify the image.
var classified = composite.classify(classifier);

var sld_intervals_crop =
'<RasterSymbolizer>' +
  '<ColorMap type="intervals" extended="false">' +
    '<ColorMapEntry color="#10d22c" quantity="1" label="Crop"/>' + 
    '<ColorMapEntry color="#000000" quantity="2" label="Non Crop"/>' +
    '</ColorMap>' +
'</RasterSymbolizer>';

Map.addLayer(classified.sldStyle(sld_intervals_crop),{}, 'RF Classified Layer');

// #############################################################################
// ###               Model performance Evaluation                           ###
// #############################################################################

//Evaluate the performance of the model.

var test = testingPartition.classify(classifier);

var RF_confusionMatrix = test.errorMatrix('landcover', 'classification');
print('RF_Confusion_Matrix', RF_confusionMatrix);

print('RF_test accuracy', RF_confusionMatrix.accuracy());

// Calculate consumer's accuracy, also known as user's accuracy or
// specificity and the complement of commission error (1 − commission error).
print("Specificity accuracy RF", RF_confusionMatrix.consumersAccuracy());

// Calculate producer's accuracy, also known as sensitivity and the
// compliment of omission error (1 − omission error).
print("Recall or Sensitivity accuracy RF", RF_confusionMatrix.producersAccuracy());

// Calculate kappa statistic.
print('Kappa statistic RF', RF_confusionMatrix.kappa());

// #############################################################################
// ###                 Hyper parameter Tuning                               ###
// #############################################################################

// Tune the numberOfTrees parameter to obtain the optimal number of trees to 
//be used in the classification.
/*
var numTreesList = ee.List.sequence(10, 150, 10);

var accuracies = numTreesList.map(function(numTrees) {
  var classifier = ee.Classifier.smileRandomForest(numTrees)
      .train({
        features: training,
        classProperty: 'landcover',
        inputProperties: composite.bandNames()
      });

  return test
    .classify(classifier)
    .errorMatrix('landcover', 'classification')
    .accuracy();
});

var chart = ui.Chart.array.values({
  array: ee.Array(accuracies),
  axis: 0,
  xLabels: numTreesList
  }).setOptions({
      title: 'Hyperparameter Tuning for the numberOfTrees Parameters',
      vAxis: {title: 'Validation Accuracy'},
      hAxis: {title: 'Number of Tress', gridlines: {count: 15}}
  });
print(chart);
*/
// #############################################################################
// ###                 Output export                                         ###
// #############################################################################


//Export classification output


/*
Export.image.toDrive({  
 image:classified ,  
 description: 'crop_noncrop_classification_may_01122020',  
 scale: 10,  
 folder: 'earthengine',
 region: studysite,  
 fileFormat: 'GeoTIFF',  
 maxPixels: 1e12
});

/*

//Export classification output

Export.image.toDrive({  
 image:May_14 ,  
 description: 'SCL_Classification_03112021',  
 scale: 10,  
 region: studysite,  
 fileFormat: 'GeoTIFF'
});

*/
// #############################################################################
// ###                Add Legend                                             ###
// #############################################################################

var legend = ui.Panel({style: {position: 'middle-right', padding: '8px 15px'}});

var makeRow = function(color, name) {
  var colorBox = ui.Label({
    style: {color: '#ffffff',
      backgroundColor: color,
      padding: '10px',
      margin: '0 0 4px 0',
    }
  });
  var description = ui.Label({
    value: name,
    style: {
      margin: '0px 0 4px 6px',
    }
  }); 
  return ui.Panel({
    widgets: [colorBox, description],
    layout: ui.Panel.Layout.Flow('horizontal')}
)};

var title = ui.Label({
  value: 'Legend',
  style: {fontWeight: 'bold',
    fontSize: '16px',
    margin: '0px 0 4px 0px'}});
    
legend.add(title);
legend.add(makeRow("black",'Non Crop'));
legend.add(makeRow("49fb88",'Crop'));

Map.add(legend);

// #############################################################################
// ###        Extracting the Scene Classification Map                       ###
// #############################################################################

var legend = ui.Panel({style: {position: 'middle-right', padding: '8px 15px'}});

var makeRow = function(color, name) {
  var colorBox = ui.Label({
    style: {color: '#ffffff',
      backgroundColor: color,
      padding: '10px',
      margin: '0 0 4px 0',
    }
  });
  var description = ui.Label({
    value: name,
    style: {
      margin: '0px 0 4px 6px',
    }
  }); 
  return ui.Panel({
    widgets: [colorBox, description],
    layout: ui.Panel.Layout.Flow('horizontal')}
)};

var title = ui.Label({
  value: 'Legend',
  style: {fontWeight: 'bold',
    fontSize: '16px',
    margin: '0px 0 4px 0px'}});
    
legend.add(title);
//#######################################################
// Legend for all 11 layers                            ##
//#######################################################

legend.add(makeRow("#ff0004",'Saturated or defective'));
legend.add(makeRow("#868686",'Dark Area Pixels'));
legend.add(makeRow("#774b0a",'Cloud Shadow'));
legend.add(makeRow("#10d22c",'Vegetation'));
legend.add(makeRow("#ffff52",'Bare Soil'));
legend.add(makeRow("#0000ff",'Water'));
legend.add(makeRow("#818181",'Clouds Low Probability / Unclassifiedl'));
legend.add(makeRow("#c0c0c0",'Clouds Medium Probability'));
legend.add(makeRow("#f1f1f1",'Clouds High Probability'));
legend.add(makeRow("#bac5eb",'Cirrus'));
legend.add(makeRow("#52fff9",'Snow / Ice'));
//legend.add(makeRow("#000000",'Others'));
// Map.add(legend);

// Training Data Generation
      
var Aug_01_1 = ee.Image('COPERNICUS/S2_SR/20210815T113321_20210815T113451_T28QCD')
                .select('B.*','SCL')
                .clip(studysite); 

var Aug_01_2 = ee.Image('COPERNICUS/S2_SR/20210822T112121_20210822T112449_T28QDD')
                .select('B.*','SCL')
                .clip(studysite); 

var Sept_01_1 = ee.Image('COPERNICUS/S2_SR/20210929T113319_20210929T114417_T28QCD')
                .select('B.*','SCL')
                .clip(studysite); 
                
var Sept_01_2 = ee.Image('COPERNICUS/S2_SR/20210921T112121_20210921T112452_T28QDD')
                .select('B.*','SCL')
                .clip(studysite); 

var Oct_01_1 = ee.Image('COPERNICUS/S2_SR/20211014T113321_20211014T113454_T28QCD')
                .select('B.*','SCL')
                .clip(studysite); 
var Oct_01_2 = ee.Image('COPERNICUS/S2_SR/20211021T112121_20211021T112452_T28QDD')
                .select('B.*','SCL')
                .clip(studysite); 

var Nov_01_1 = ee.Image('COPERNICUS/S2_SR/20211103T113321_20211103T113452_T28QCD')
                .select('B.*','SCL')
                .clip(studysite); 
var Nov_01_2 = ee.Image('COPERNICUS/S2_SR/20211113T113331_20211113T113450_T28QDD')
                .select('B.*','SCL')
                .clip(studysite); 
var rgbVis = {
  min: 0.0,
  max: 3000,
  bands: ['B4', 'B3', 'B2'],
};

// We need to add bands we want to visualize with the selected image

Map.addLayer(Aug_01_1, rgbVis, 'Aug 01 1 Image')
Map.addLayer(Aug_01_2, rgbVis, 'Aug 01 2 Image')

Map.addLayer(Sept_01_1, rgbVis, 'Sept 01 1 Image')
Map.addLayer(Sept_01_2, rgbVis, 'Sept 01 2 Image')

Map.addLayer(Oct_01_1, rgbVis, 'Oct 01 1 Image')
Map.addLayer(Oct_01_2, rgbVis, 'Oct 01 2 Image')

Map.addLayer(Nov_01_1, rgbVis, 'Nov 01 1 Image')
Map.addLayer(Nov_01_2, rgbVis, 'Nov 01 2 Image')

//The month of may is the period with highest level of vegetation
//in the Upper delta region.
var scl_image_sept = ee.ImageCollection('COPERNICUS/S2_SR')
                .filterBounds(studysite)
                .filterDate("2021-10-01", "2021-10-30")
                .sort('CLOUDY_PIXEL_PERCENTAGE')
                .select("SCL")
                .median()
                .clip(studysite);
print('SCL Image May: ', scl_image_sept);

//Upper delta region is fully covered by two dates image tiles one in the May2 and May 14
var Sept_2019 = sent2.filterDate("2019-08-15", "2019-11-30")
                .filterMetadata('CLOUDY_PIXEL_PERCENTAGE',
                                'less_than', 20)
                .map(maskS2srClouds)
                .select('B.*','SCL')
                .filterBounds(studysite);

var Sept_2020 = sent2.filterDate("2020-08-01", "2020-11-15")
                .filterMetadata('CLOUDY_PIXEL_PERCENTAGE',
                                'less_than', 20)
                .map(maskS2srClouds)
                .select('B.*','SCL')
                .filterBounds(studysite);

var Sept_2021 = sent2.filterDate("2021-08-01", "2021-11-15")
                .filterMetadata('CLOUDY_PIXEL_PERCENTAGE',
                                'less_than', 20)
                .map(maskS2srClouds)
                .select('B.*','SCL')
                .filterBounds(studysite);

//#######################################################
//        Legend for 11 SCL layers                     ##
//#######################################################

var sld_intervals =
'<RasterSymbolizer>' +
  '<ColorMap type="intervals" extended="false">' +
    '<ColorMapEntry color="#ff0004" quantity="1" label="Saturated or defective"/>' +
    '<ColorMapEntry color="#868686" quantity="2" label="Dark Area Pixels"/>' +
    '<ColorMapEntry color="#774b0a" quantity="3" label="Cloud Shadows"/>' +
    '<ColorMapEntry color="#10d22c" quantity="4" label="Vegetation"/>' + 
    '<ColorMapEntry color="#ffff52" quantity="5" label="Bare Soils"/>' +
    '<ColorMapEntry color="#0000ff" quantity="6" label="Water"/>' +
    '<ColorMapEntry color="#818181" quantity="7" label="Clouds Low Probability / Unclassified"/>' +
    '<ColorMapEntry color="#c0c0c0" quantity="8" label="Clouds Medium Probability"/>' +
    '<ColorMapEntry color="#f1f1f1" quantity="9" label="Clouds High Probability"/>' +
    '<ColorMapEntry color="#bac5eb" quantity="10" label="Cirrus"/>' +
    '<ColorMapEntry color="#52fff9" quantity="11" label="Snow / Ice"/>' +
  '</ColorMap>' +
'</RasterSymbolizer>';
Map.add(legend);

Map.addLayer(scl_image_sept.sldStyle(sld_intervals), {}, 'SCL classification ');

// #############################################################################
// ###              Class Area calculation                                   ### 
// #############################################################################

var classArea = function(classified){
  var areaImage = ee.Image.pixelArea().addBands(
        classified);
  
  var areas = areaImage.reduceRegion({ 
        reducer: ee.Reducer.sum().group({
        groupField: 1,
        groupName: 'classification',
      }),
      geometry: studysite.geometry(),
      scale: 10, 
      // tileScale: 16,  // Higher values of tileScale result in tiles smaller by a factor of tileScale^2 and this won't fit in memory for large image
      maxPixels: 1e8
      }); 
  var classAreas = ee.List(areas.get('groups'));
   
  var classAreaLists = classAreas.map(function(item) { // Function within a function to create a dictionary with the values for every group
    var areaDict = ee.Dictionary(item);
    var classNumber = ee.Number(areaDict.get('classification')).format();
    var area = ee.Number(
      areaDict.get('sum')).divide(1e4).round(); // The result will be in square meters, this converts them into square kilometers
    return ee.List([classNumber, area]);
  });
   
  var result = ee.Dictionary(classAreaLists.flatten()); // Flattens said dictionary so it is readable for us
  
  return(result);
};

//Values in kms for the area of each class:
print('Crop noncrop Areas RF: ', classArea(classified));

var composite_Sept_2019 = Sept_2019.median().clip(studysite); 
var composite_Sept_2020 = Sept_2020.median().clip(studysite); 
var composite_Sept_2021 = Sept_2021.median().clip(studysite); 

var composite_Sept_2019 = addIndices(composite_Sept_2019);
var composite_Sept_2020 = addIndices(composite_Sept_2020);
var composite_Sept_2021 = addIndices(composite_Sept_2021);

// Classify the image.
var Sept_2019_classified = composite_Sept_2019.classify(classifier);
var Sept_2020_classified = composite_Sept_2020.classify(classifier);
var Sept_2021_classified = composite_Sept_2021.classify(classifier);

print('Crop noncrop Sept 2019: ', classArea(Sept_2019_classified));
print('Crop noncrop Sept 2020: ', classArea(Sept_2020_classified));
print('Crop noncrop Sept 2021: ', classArea(Sept_2021_classified));

Map.addLayer(Sept_2019_classified.sldStyle(sld_intervals_crop), {}, 'Crop noncrop Sept 2019 ');
Map.addLayer(Sept_2020_classified.sldStyle(sld_intervals_crop), {}, 'Crop noncrop Sept 2020 ');
Map.addLayer(Sept_2021_classified.sldStyle(sld_intervals_crop), {}, 'Crop noncrop Sept 2021 ');

// // Export the FeatureCollection to a KML file.
// Export.table.toDrive({
//   collection: crop,
//   description:'crop14',
//   folder: 'earthengine',
//   fileFormat: 'SHP'
// });

Export.table.toDrive({
  collection: class_labels,
  description:'crop_noncrop_training_08_16',
  folder: 'earthengine_SCL',
  fileFormat: 'SHP'
});
