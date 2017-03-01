import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from math import radians, cos, sin, asin, sqrt
import pickle

def haversine(lon1, lat1, lon2, lat2):
    # DD to Radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # Haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6367 * c

    return km

def getBoundaries(locations):
    maxLonLat = np.amax(locations, axis = 0)
    minLonLat = np.amin(locations, axis = 0)
    meanLonLat = np.mean(locations, axis = 0)

    return maxLonLat, minLonLat, meanLonLat

def createMap(locations, species):
    """
    Creates a map from an array of locations (Lon-Lat) and a matching
    array of species (taxon-name)
    """
    locations = locations.astype(float)
    # Get species
    uniqueSpecies = np.unique(species)
    numSpecies = uniqueSpecies.size
    numPlot = 100 + numSpecies * 10
    print("Number of species: " + str(numSpecies))

    # Create plot
    fig = plt.figure()

    # Species 1
    ax1 = fig.add_subplot(numPlot + 1)
    ax1.set_title(uniqueSpecies[0])
    s1 = np.where(species == uniqueSpecies[0])
    maxLonLat, minLonLat, meanLonLat = getBoundaries(locations[s1])
    m = Basemap(projection='stere',
            lat_0=meanLonLat[1],
            lon_0=meanLonLat[0],
            llcrnrlat=minLonLat[1] - 0.1,
            llcrnrlon=minLonLat[0] - 0.1,
            urcrnrlat=maxLonLat[1] + 0.1,
            urcrnrlon=maxLonLat[0] + 0.1)
    m.etopo()

    for location in locations[s1]:
        x, y = m(location[0], location[1])
        m.plot(x, y,
                'o',
                color='Indigo',
                markersize=4)

    if (numSpecies > 1):
        # Species 2
        ax2 = fig.add_subplot(numPlot + 2)
        ax2.set_title(uniqueSpecies[1])
        s2 = np.where(species == uniqueSpecies[1])
        maxLonLat, minLonLat, meanLonLat = getBoundaries(locations[s2])
        m = Basemap(projection='stere',
                lat_0=meanLonLat[1],
                lon_0=meanLonLat[0],
                llcrnrlat=minLonLat[1] - 0.1,
                llcrnrlon=minLonLat[0] - 0.1,
                urcrnrlat=maxLonLat[1] + 0.1,
                urcrnrlon=maxLonLat[0] + 0.1)
        m.etopo()

        for location in locations[s2]:
            x, y = m(location[0], location[1])
            m.plot(x, y,
                    'o',
                    color='Indigo',
                    markersize=4)

    if (numSpecies > 2):
        # Species 3
        ax3 = fig.add_subplot(numPlot + 3)
        ax3.set_title(uniqueSpecies[2])
        s3 = np.where(species == uniqueSpecies[2])
        maxLonLat, minLonLat, meanLonLat = getBoundaries(locations[s3])
        m = Basemap(projection='stere',
                lat_0=meanLonLat[1],
                lon_0=meanLonLat[0],
                llcrnrlat=minLonLat[1] - 0.1,
                llcrnrlon=minLonLat[0] - 0.1,
                urcrnrlat=maxLonLat[1] + 0.1,
                urcrnrlon=maxLonLat[0] + 0.1)
        m.etopo()

        for location in locations[s3]:
            x, y = m(location[0], location[1])
            m.plot(x, y,
                    'o',
                    color='Indigo',
                    markersize=4)

    plt.show()

# Test
#samples = pickle.load(open("samples.pkl", "rb"))
#createMap(samples[:10000, 3:5], samples[:10000, 5])
