import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.basemap import Basemap
import pickle

def getBoundaries(locations):
    maxLonLat = np.amax(locations, axis = 0)
    minLonLat = np.amin(locations, axis = 0)
    meanLonLat = np.mean(locations, axis = 0)

    return maxLonLat, minLonLat, meanLonLat

def getBasemap(boundaries):
    maxLonLat, minLonLat, meanLonLat = boundaries
    m = Basemap(projection='stere',
            lat_0=meanLonLat[1],
            lon_0=meanLonLat[0],
            llcrnrlat=minLonLat[1] - 0.1,
            llcrnrlon=minLonLat[0] - 0.1,
            urcrnrlat=maxLonLat[1] + 0.1,
            urcrnrlon=maxLonLat[0] + 0.1,
            resolution=None)
    m.shadedrelief()

    return m

def getOverviewBasemap():
    m = Basemap(projection='kav7',lon_0=0,resolution=None)
    m.shadedrelief()

    return m

def createMap(locations, species, getOverview=False):
    """
    Creates a map from an array of locations (Lon-Lat) and a matching
    array of species (taxon-name)
    """
    colors = ['red', 'green', 'blue']
    ySplit = 2 if getOverview == True else 1
    locations = locations.astype(float)
    # Get species
    uniqueSpecies = np.unique(species)
    numSpecies = uniqueSpecies.size
    # numPlot = 200 + numSpecies * 10
    print("Number of species: " + str(numSpecies))

    # Create plot
    fig = plt.figure()
    gs = gridspec.GridSpec(ySplit, numSpecies)
    means = []

    # Species 1
    ax1 = fig.add_subplot(gs[ySplit-1, 0])
    ax1.set_title(uniqueSpecies[0])
    s1 = np.where(species == uniqueSpecies[0])
    boundaries = getBoundaries(locations[s1])
    mean1 = boundaries[2]
    means.append(mean1)
    m = getBasemap(boundaries)

    for location in locations[s1]:
        x, y = m(location[0], location[1])
        m.plot(x, y,
                'o',
                color=colors[0],
                markersize=4)

    if (numSpecies > 1):
        # Species 2
        ax2 = fig.add_subplot(gs[ySplit-1, 1])
        ax2.set_title(uniqueSpecies[1])
        s2 = np.where(species == uniqueSpecies[1])
        boundaries = getBoundaries(locations[s2])
        mean2 = boundaries[2]
        means.append(mean2)
        m = getBasemap(boundaries)

        for location in locations[s2]:
            x, y = m(location[0], location[1])
            m.plot(x, y,
                    'o',
                    color=colors[1],
                    markersize=4)

    if (numSpecies > 2):
        # Species 3
        ax3 = fig.add_subplot(gs[ySplit-1, 2])
        ax3.set_title(uniqueSpecies[2])
        s3 = np.where(species == uniqueSpecies[2])
        boundaries = getBoundaries(locations[s3])
        mean3 = boundaries[2]
        means.append(mean3)
        m = getBasemap(boundaries)

        for location in locations[s3]:
            x, y = m(location[0], location[1])
            m.plot(x, y,
                    'o',
                    color=colors[3],
                    markersize=4)

    if (getOverview):
        overview = fig.add_subplot(gs[0, :])
        overview.set_title("Overview")
        meanLocations = np.reshape(means, (numSpecies, 2))
        boundaries = getBoundaries(meanLocations)
        m = getOverviewBasemap()

        colorIndex = 0
        for location in meanLocations:
            x, y = m(location[0], location[1])
            m.plot(x, y,
                    'o',
                    mec='none',
                    color=colors[colorIndex],
                    alpha=0.5,
                    markersize=6)
            colorIndex += 1

    plt.show()

# Test
samples = pickle.load(open("samples.pkl", "rb"))
createMap(samples[:5000, 3:5], samples[:5000, 5], True)
