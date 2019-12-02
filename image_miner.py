import os
workingdir = os.path.dirname(os.path.realpath(__file__))
if '/' in workingdir:
    workingdir = workingdir + '/'
else:
    workingdir = workingdir + '\\'
print('Checking for files in ' + workingdir)

# Download State Building geoJSONs
for i in ['DistrictofColumbia','Virginia','Maryland']:
    print(i)
    if not os.path.exists(workingdir + i + '.geojson'):
        os.system("cd " + workingdir)
        if not os.path.exists(workingdir + i + '.zip'):
            os.system("wget https://usbuildingdata.blob.core.windows.net/usbuildings-v1-1/" + i + ".zip")
        os.system("unzip " + workingdir + i + ".zip")

# Download basic state boundary shapefile
if not os.path.exists(workingdir + 'states.shp'):
    os.system("cd " + workingdir)
    if not os.path.exists(workingdir + 'states_21basic.zip'):
        os.system("wget https://storage.googleapis.com/dats6203/states_21basic.zip")
    os.system("unzip " + workingdir + "states_21basic.zip")

# Create image folder if it doesn't exist
traindir = workingdir + 'train'
print('Checking for ' + traindir)
if not os.path.exists(traindir):
    os.system("mkdir " + traindir)

if '/' in workingdir:
    traindir = traindir + '/'
else:
    traindir = traindir + '\\'

# Install required tools. Earthpy is unlikely to work without conda.
print('Downloading and installing required modules')
os.system('sudo apt-get update')
os.system('sudo apt-get install firefox-esr --assume-yes')
os.system('pip3 install folium')
os.system('pip3 install numpy')
os.system('pip3 install geopandas')
os.system('pip3 install shapely')
os.system('pip3 install geog')
os.system('pip3 install geojson')
os.system('pip3 install selenium')
os.system('pip3 install Pillow')
os.system('pip3 install opencv-python')
try:
    os.system('pip3 install earthpy')
except:
    os.system('conda install -c conda-forge earthpy')

# Import modules
import geopandas as gpd
import earthpy as et
from earthpy import clip as cl
from io import BytesIO
import folium
import io
import os
import time
import random
import geojson
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from IPython.core.display import Image, display
from PIL import Image as Im
import shapely
import json
import numpy as np
import geog
from shapely.geometry import box, shape, GeometryCollection, Polygon, mapping, Point

# Define function to max a box using a coordinate reference system 
# based on center point/radius (latitude, longitude, meters)
# and return a geopandas dataframe and bounds list
def makebox(latitude, longitude, radius):
    p = Point([latitude, longitude])
    angles = np.linspace(0, 360, 5)
    polygon = geog.propagate(p, angles, radius)
    box = Polygon(polygon).envelope
    gdf = gpd.GeoDataFrame(geometry=[box], crs='epsg:4326')
    bounds = [[min(gdf.geometry[0].exterior.coords.xy[1]),
               min(gdf.geometry[0].exterior.coords.xy[0])],
              [max(gdf.geometry[0].exterior.coords.xy[1]),
               max(gdf.geometry[0].exterior.coords.xy[0])]]
    return gdf, bounds

def create_labeled_image(latitude, longitude, radius, df, resolution):
    # Create bounding box and bounds
    c, bounds = makebox(latitude, longitude, radius)
    print(latitude, longitude)
    try:
        # Clip buildings polygon 
        f_clip = cl.clip_shp(df, c)
        # Calculate area of clipped buildings divided by area of bounding box
        area = round(float(sum(f_clip.area)/c.area),10)
    except:
        area = 0.0
    print('Building coverage ratio: ' + str(area))
    m.fit_bounds(bounds)
    # Save temp html file from folium
    delay=5
    fn='tempmap.html'
    tmpurl='file://{path}{mapfile}'.format(path=workingdir,mapfile=fn)
    m.save(fn)
    # Load map tile and create screenshot
    browser.get(tmpurl)
    time.sleep(delay)
    im = Im.open(BytesIO(browser.get_screenshot_as_png()))
    os.remove(fn)
    # Get dimensions and calculate cropping range
    width, height = im.size   
    left = (width - resolution/2)/2
    top = (height - resolution/2)/2
    right = (width + resolution/2)/2
    bottom = (height + resolution/2)/2
    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    ifname = '{}{}_{}_{}_{}.png'.format(traindir,radius,area,latitude,longitude)
    print(ifname)
    im.save(ifname)

# Link to Esri World Imagery service plus attribution
EsriImagery = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
EsriAttribution = "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
# Make a map with tile layers covering a specific bounding box
m = folium.Map(tiles=EsriImagery, attr=EsriAttribution, zoom_control=False)
    
resolution = 1000
# Set options for Firefox private headless browser
firefox_options = Options()
firefox_options.add_argument("--headless")
browser = webdriver.Firefox(executable_path=workingdir+'geckodriver', service_log_path=workingdir+'templog.log', options=firefox_options)
browser.set_window_size(resolution*2, resolution*2) 

# Open JSON file of buildings
df = gpd.read_file(workingdir + "DistrictofColumbia.geojson")
#df = gpd.read_file(workingdir + "Virginia.geojson")
#df = gpd.read_file(workingdir + "Maryland.geojson")

# Load state boundaries and create mask to prevent mapping points on edge of state
usa = gpd.read_file(workingdir + 'states.shp')
selected_state = usa[usa['STATE_ABBR'] == 'DC'].buffer(-.01)
state_bounds = gpd.GeoDataFrame(geometry=selected_state, crs='epsg:4326').reset_index()['geometry'][0]

# Define image parameters
resolution = 2000
radius = 250

# Calculate lat/long range for sampling
l0 =[]
l1 =[]
for i in gpd.GeoDataFrame(geometry=selected_state, crs='epsg:4326').reset_index().geometry:
    l1 += [i.bounds[1],i.bounds[3]]
    l0 += [i.bounds[0],i.bounds[2]]
latrange = min(l0), max(l0)
longrange = min(l1), max(l1)

# Iterate to generate X random labeled images
for i in range(20000):
    print(i)
    latitude = round(random.uniform(latrange[0],latrange[1]),8)
    longitude = round(random.uniform(longrange[0],longrange[1]),8)
    c, bounds = makebox(latitude, longitude, radius)
    if c['geometry'][0].intersects(state_bounds):
        try:create_labeled_image(latitude, longitude, radius, df, resolution)
        except:
            print('Error creating image')
            pass

browser.quit()