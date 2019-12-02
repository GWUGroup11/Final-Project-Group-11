# Install modules and other resources
import os
os.system('apt-get update')
os.system('apt-get install firefox-esr --assume-yes')
os.system('pip install folium')
os.system('pip install numpy')
os.system('pip install geopandas')
os.system('pip install shapely')
os.system('pip install geog')
os.system('pip install selenium')
os.system('pip install Pillow')
os.system('pip install tensorflow')
os.system('pip install keras')
os.system('pip install opencv-python')

# Import modules
import time
import numpy as np
from PIL import Image as Im
import io
from io import BytesIO
import folium
import selenium
from selenium import webdriver
from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
import geopandas as gpd
import geog
from shapely.geometry import box, shape, GeometryCollection, Polygon, mapping, Point
import numpy as np
import tensorflow as tf
from keras.models import load_model
import cv2

# Create and return a geopandas dataframe and bounds list
def makebox(latitude, longitude, radius):
    p = Point([longitude, latitude])
    angles = np.linspace(0, 360, 5)
    polygon = geog.propagate(p, angles, radius)
    box = Polygon(polygon).envelope
    gdf = gpd.GeoDataFrame(geometry=[box], crs='epsg:4326')
    bounds = [[min(gdf.geometry[0].exterior.coords.xy[1]),
               min(gdf.geometry[0].exterior.coords.xy[0])],
              [max(gdf.geometry[0].exterior.coords.xy[1]),
               max(gdf.geometry[0].exterior.coords.xy[0])]]
    return gdf, bounds

# Link to Esri World Imagery service plus attribution
EsriImagery = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
EsriAttribution = "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"

# Make a folium map with tile layers
m = folium.Map(tiles=EsriImagery, attr=EsriAttribution, zoom_control=False)
resolution = 2000
radius = 250
RESOLUTION = 500

# Load deep learning model
model = load_model('/home/site/wwwroot/cnn_500.hdf5')

# Find and tag map image
def create_labeled_image(latitude, longitude, radius, resolution):
    # Create bounding box and bounds
    fn='/home/site/wwwroot/static/tempmap_{latitude}_{longitude}.html'.format(latitude=latitude, longitude=longitude)
    if not os.path.exists(fn):
        c, bounds = makebox(latitude, longitude, radius)
        print(latitude, longitude)
        # Fit folium map to bounding box
        m.fit_bounds(bounds)
        print(bounds)
        # Save temp html file from folium
        m.save(fn)
    fn='tempmap_{latitude}_{longitude}.html'.format(latitude=latitude, longitude=longitude)
    fn2='tempmap_{latitude}_{longitude}.png'.format(latitude=latitude, longitude=longitude)
    print(fn)
    url = 'https://dats6203.azurewebsites.net/static/' + fn
    print(url)
    time.sleep(1)
    # Load Firefox in headless mode to screenshot folium map
    options = Options()
    options.add_argument("--headless")
    try:
        with webdriver.Firefox(executable_path='/home/site/wwwroot/geckodriver', service_log_path='/home/site/wwwroot/templog.log', options=options) as driver:
            driver.set_window_size(2000,2000) 
            driver.get(url)
            time.sleep(3)
            im = Im.open(BytesIO(driver.get_screenshot_as_png()))
    except:
        time.sleep(5)
        try:
            with webdriver.Firefox(executable_path='/home/site/wwwroot/geckodriver', service_log_path='/home/site/wwwroot/templog.log', options=options) as driver:
                driver.set_window_size(2000,2000) 
                driver.get(url)
                time.sleep(5)
                im = Im.open(BytesIO(driver.get_screenshot_as_png()))
        except:pass
    # Get dimensions and calculate cropping range
    width, height = im.size   
    left = (width - resolution/2)/2
    top = (height - resolution/2)/2
    right = (width + resolution/2)/2
    bottom = (height + resolution/2)/2
    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    im.save('/home/site/wwwroot/static/' + fn2)
    url = 'https://dats6203.azurewebsites.net/static/' + fn2
    print(url)

# Start loop
while True:
    time.sleep(3)
    # Read queries
    fileHandle2 = open ( '/home/site/wwwroot/q.txt',"r" )
    lineList2 = fileHandle2.readlines()
    fileHandle2.close()
    for tempstring in lineList2[-10:]:
        if ' ' in tempstring or ',' in tempstring:
            tempstring = tempstring.replace(',',' ').replace('  ',' ').strip()
            try:
                # Extract coordinates from query
                latitude = float(tempstring.split(' ')[0])
                longitude = float(tempstring.split(' ')[1])
                fn='/home/site/wwwroot/static/tempmap_{latitude}_{longitude}.html'.format(latitude=latitude, longitude=longitude)
                fn2='/home/site/wwwroot/static/tempmap_{latitude}_{longitude}.png'.format(latitude=latitude, longitude=longitude)
                fn3 = '/home/site/wwwroot/static/tempmap_{latitude}_{longitude}.txt'.format(latitude=latitude, longitude=longitude)
                if not os.path.exists(fn2):
                    create_labeled_image(latitude, longitude, radius, resolution)
                if not os.path.exists(fn3):
                    fn2='/home/site/wwwroot/static/tempmap_{latitude}_{longitude}.png'.format(latitude=latitude, longitude=longitude)
                    # Read and reshape image
                    im1 = cv2.imread(fn2)
                    im1 = cv2.fastNlMeansDenoisingColored(im1, None, 5, 5, 3, 10) 
                    images = [cv2.resize(im1, dsize=(RESOLUTION, RESOLUTION), interpolation=cv2.INTER_CUBIC)]
                    x_test = np.array(images)
                    # Predict urban density using model
                    density = model.predict(x_test)[0][0]
                    print(density)
                    # Store prediction
                    fileHandle3 = open ( fn3, "w" )
                    fileHandle3.write(str(density))
                    fileHandle3.close()
                    print(fn3)
            except:
                pass
        else:
            pass