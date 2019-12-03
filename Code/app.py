# Import modules
from flask import Flask, escape, request, render_template
from flask_caching import Cache
import os
from glob import glob

# Configure caching
cache = Cache(config={'CACHE_TYPE': 'simple'})

# Initialize app and cache
app = Flask(__name__)
cache.init_app(app)

@app.route('/')
@app.route('/home')
@app.route('/index.html')
@cache.cached(timeout=600)
def index():
    return render_template('index.html')

@app.route('/about')
@cache.cached(timeout=600)
def about():
    return render_template('about.html')

@app.errorhandler(404)
@cache.cached(timeout=600)
def page_not_found(e):
    return render_template('404.html'), 404

@app.route('/search/')
def showsearch():
    q = request.args.get("q", "")
    picurl = 'blank.png'
    density = ''
    refresher = ''
    if ' ' in q or ',' in q:
        tempstring = q.replace(',',' ').replace('  ',' ').strip()
        try:
            latitude = float(tempstring.split(' ')[0])
            longitude = float(tempstring.split(' ')[1])
            print(latitude,longitude)
            f = open("/home/site/wwwroot/q.txt","a+")
            f.write('\n' + str(latitude) + ' ' + str(longitude))
            f.close()
            # Load map image and density
            if os.path.exists('/home/site/wwwroot/static/tempmap_' + str(latitude) + '_' + str(longitude) + '.png') and os.path.exists('/home/site/wwwroot/static/tempmap_' + str(latitude) + '_' + str(longitude) + '.txt'):
                picurl = 'tempmap_' + str(latitude) + '_' + str(longitude) + '.png'
                # Load and score density
                density = float(open ( '/home/site/wwwroot/static/tempmap_' + str(latitude) + '_' + str(longitude) + '.txt',"r" ).read())
                if density > .3:
                    density = str(density) + ' (Dense Urban)'
                elif density > .15:
                    density = str(density) + ' (Urban)'
                elif density > .1:
                    density = str(density) + ' (Light Urban)'
                elif density > .05:
                    density = str(density) + ' (Suburban)'
                elif density > .025:
                    density = str(density) + ' (Light Suburban)'
                elif density > .015:
                    density = str(density) + ' (Rural)'
                else:
                    density = str(density) + ' (Uninhabited)'
                density = 'Density: ' + density
            else:
                picurl = 'spinner.gif'
                density = 'Searching...'
                refresher = '<meta http-equiv="refresh" content="10" >'
        except:
            pass
    return render_template('search.html', picurl=picurl, q=q, density=density, refresher=refresher)

@app.route('/history')
def showhistory():
    # Load density file statistics and plot table
    densities = glob('/home/site/wwwroot/static/tempmap_*.txt')
    results = []
    for item in densities[:1000]:
        try:
            latitude = item.split('_')[1]
            longitude = item.split('_')[2].split('.txt')[0]
            density = float(open(item,'r').read())
            result = {}
            result['latitude'] = latitude
            result['longitude'] = longitude
            result['density_val'] = density
            result['density'] = '<a href="../search/?q=' + latitude + '+' + longitude + '">' + str(density) + '</a>'
            results += [result]
        except:
            pass
    results = sorted(results, key = lambda i: i['density_val'],reverse=True) 
    return render_template('history.html', results=results)