import ee
ee.Initialize(project='nodal-seer-478715-m2')
point = ee.Geometry.Point(77.0153, 10.8163)
img = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filterDate('2025-01-01', '2025-01-02').first()
print(img.reduceRegion(ee.Reducer.mean(), point, 5000).getInfo())