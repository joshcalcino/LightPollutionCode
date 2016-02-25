import sys
from mpl_toolkits import basemap
import os
import numpy as np
import glob
np.set_printoptions(threshold=np.nan)
import time
import astropy.units as un
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz, EarthLocation, ICRS
import astropy.constants as const
from skyb import integrand
import georasters as gr
import subprocess

from osgeo import osr, gdal
from gdalconst import *
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
import matplotlib as mpl
import numpy.ma as ma
from scipy import integrate

class Conversions(object):
    """ time_obs = the time the object was observed
        UTC_offset = the timezone offset from UTC
        loc_obs = an array containing the infromation about
        the observers location on the Earth [Latitude, Longitude, Heigh above sea level]
        obj = the objects location [RA, DEC]
        obspar = observation parameters [observation wavelength, pressure,
        ground temperature, relative huminitity]
        
        Will probably add/remove input variables based on feedback.
        Most of these functions are based off of the work Maria did, so I am not 100%
        certain on their necessity.
        """
    def __init__(self, time_obs, UTC_offset, loc_obs, obj, obspar):
        self.time_obs = time_obs
        self.UTC_offset = UTC_offset
        self.loc_obs = loc_obs
        self.obj = obj
        self.obspar = obspar
    
    def getTime(self): # get the time of the observation with the UTC offset, might not be necessary
        time = Time(self.time_obs, format='iso', scale='utc') - self.UTC_offset*un.hour
        return time
    
    def convertLatLong(self): # convert the latitude and longitude coordinates to something more useable
        loc_obs = self.loc_obs
        lat = loc_obs[0]
        long = loc_obs[1]
        # convert to annoying astropy units
        long = sum(np.array(long) * (un.deg, un.arcmin, un.arcsec))
        if lat[0] < 0:
            lat = -sum(abs(np.array(lat)) * (un.deg, un.arcmin, un.arcsec))
        else:
            lat = sum(np.array(lat)) * (un.deg, un.arcmin, un.arcsec)
        return lat, long
    
    def getEarthLoc(self):# get the Earth location based off of the latitude, longitude and height
        loc_obs = self.loc_obs
        latlong = self.convertLatLong()
        lat = latlong[0]
        long = latlong[1]
        elev = loc_obs[2]
        Earthloc = EarthLocation(lat=lat, lon=long, height=elev*un.m)
        return Earthloc
    
    def convertRaDec(self):
        obj = self.obj
        ra = obj[0]
        dec = obj[1]
        ra = sum(np.array(ra) * (un.deg, un.arcmin, un.arcsec) * 15)
        if dec[0] < 0:
            dec = -sum(abs(np.array(dec))*(un.deg,un.arcmin,un.arcsec))
        else:
            dec = sum(np.array(dec)*(un.deg,un.arcmin,un.arcsec))
        coord = SkyCoord(ra=ra, dec=dec, frame='icrs')
        return coord
    
    def convertAltAz(self): # convert the RA, DEC into ALT, AZ since it is needed for the sky brightness equations
        time = self.getTime()
        obspar = self.obspar
        Earthloc = self.getEarthLoc()
        coord = self.convertRaDec()
        obswav = obspar[0]
        pressure = obspar[1]
        ground_temp = obspar[2]
        rel_hum = obspar[3]
        coord_AltAz = coord.transform_to(AltAz(obstime=time, location=Earthloc,
                                               obswl = obswav*un.nm,
                                               pressure=pressure,
                                               temperature=ground_temp,
                                               relative_humidity=rel_hum
                                               )
                                         ) # given the time of the observation, atmospheric conditions etc
        return coord_AltAz # what are the alt, az coordinates of the object in the sky
    
    def getelev(self):
        return self.loc_obs[2]

class LoadMaps(Conversions): # this class inherits from the Conversions class since it uses all of the functions
    '''
        This class is concerned with loading in the light pollution, elevation, and eventually
        the aerosol raster data. When the light pollution data is loaded it in, it is sent
        to createMapSlice to have the appropriate chunck centred on the observer taken out.
        Then, a circular mask is generated and projected onto this slice.
        The height map is generated from combining a number of tiles which depend on the
        observers location and the size of their horizon. Since the resolution of the height
        data is much higher than the light, we resample it to a lower resolution.
    '''
    lightfilename = 'zF16_20100111_20110731_rad_v4_avg_vis.tif' # filename of your light pollution tif file
    dist = 24.
    changeMapRes = True
    
    def __init__(self, time_obs, UTC_offset, loc_obs, obj, obspar):
        self.time_obs = time_obs
        self.UTC_offset = UTC_offset
        self.loc_obs = loc_obs
        self.obj = obj
        self.obspar = obspar
        self.lightmap = self.loadFinalLightMap()
        self.heightmap = self.loadFinalHeightMap()
        self.lightlength = list(self.lightmap.shape)

    def loadFinalLightMap(self): # load in the light data
        latstr, longstr = self.getObsLatLongStr()
        diststr = str(self.dist)
        filename = 'LM-'+latstr+'-'+longstr+'-'+diststr+'km.tif'
        print 'Attempting to load light pollution data..'
        if os.path.isfile(filename): # have we created this horizon distance for this observer?
            print 'Pre-existing light map found'
            if self.changeMapRes:
                info = self.getGeotFromFile(filename)
                pixelsize = info[1]*2 # change the pixel size here..
                filename = self.changeMapResolution(filename, pixelsize)
            lightmap = gdal.Open(filename, GA_ReadOnly)
            return self.applyMask(np.array(lightmap.GetRasterBand(1).ReadAsArray()), filename)
        else: # if not, create it and save it
            print 'Generating light pollution map for given location..'
            self.createMapSlice(self.lightfilename, filename[:-4])
            if self.changeMapRes:
                info = self.getGeotFromFile(filename)
                pixelsize = info[1]*2 # change the pixel size here..
                filename = self.changeMapResolution(filename, pixelsize)
            lightmap = gdal.Open(filename, GA_ReadOnly)
            return self.applyMask(np.array(lightmap.GetRasterBand(1).ReadAsArray()), filename)

    def loadFinalHeightMap(self): # load in the elevation data
        latstr, longstr = self.getObsLatLongStr()
        diststr = str(self.dist)
        filename = 'HM-'+latstr+'-'+longstr+'-'+diststr+'km.tif'
        print 'Attempting to load height map data..'
        if os.path.isfile(filename): # does one exist already?
            print 'Pre-existing height map found'
            '''
                below is a way to change the resolution of the map, although it is not
                the most elegant way, it should be pretty straight forward as to how
                it is done
            '''
            if self.changeMapRes:
                info = self.getGeotFromFile(filename)
                pixelsize = info[1]*2 # change the pixel size here..
                filename = self.changeMapResolution(filename, pixelsize)
            heightmap = gdal.Open(filename, GA_ReadOnly)
            return self.applyMask(np.array(heightmap.GetRasterBand(1).ReadAsArray()), filename)
        else: # if not, create is, might take a while
            print 'Generating height map for given location, this may take a few minutes..'
            self.stitchheightmap()
            self.createMapSlice('tmpHM.tif', 'tmpHM2')
            os.system('rm tmpHM.tif')
            self.alignHeightwithLight(filename)
            if self.changeMapRes: # change map resolution?
                info = self.getGeotFromFile(filename)
                pixelsize = info[1]*2 # change the pixel size here..
                filename = self.changeMapResolution(filename, pixelsize)
            heightmap = gdal.Open(filename, GA_ReadOnly)
            return self.applyMask(np.array(heightmap.GetRasterBand(1).ReadAsArray()), filename)

    def loadAersols(self):
        pass # need to add this in later
    
    def changeMapResolution(self, inputfilename, pixelsize):
        '''
            Since gdal wont overwrite an existing file, we need to create a new one.
            The pixel size is in units which the raster file is saved as. In our case I think
            that will be degrees. If you call the function getGeotFromFile(filename),
            you will get the current pixel size of the raster.
        '''
        res = str(pixelsize)
        outputfilename = inputfilename[:-4] + 'px' + res + '.tif'
        resample_method = 'cubic'
        os.system('gdalwarp '+ inputfilename + ' ' + outputfilename + ' -r ' + resample_method+' -tr '+res+' '+res)
        return outputfilename
        
    
    def getlatlongstrings(self): # this is here for naming the files
        lat_min, lat_max = self.getLatMinMax()
        long_min, long_max = self.getLongMinMax()
        lat_min, lat_max = int(np.floor(lat_min/un.deg)), int(np.ceil(lat_max/un.deg))
        long_min, long_max = int(np.floor(long_min/un.deg)), int(np.ceil(long_max/un.deg))
        if lat_min < 0:
            lat_minstr = 'S' + str(np.absolute(lat_min))
        else:
            lat_minstr = 'N' + str(lat_min)
        if lat_max < 0:
            lat_maxstr = 'S' + str(np.absolute(lat_max))
        else:
            lat_maxstr = 'N' + str(lat_max)
        if long_min < 0:
            long_minstr = 'W' + str(np.absolute(long_min))
        else:
            long_minstr = 'E' + str(long_min)
        if long_max < 0:
            long_maxstr = 'W' + str(np.absolute(long_max))
        else:
            long_maxstr = 'E' + str(long_max)
        return lat_minstr, lat_maxstr, long_minstr, long_maxstr
    
    def getObsLatLongStr(self): # this is only for the light and height file names
        lat, long = self.convertLatLong()
        lat, long = round(lat/un.deg, 4), round(long/un.deg, 4)
        if lat < 0:
            latstr = 'S' + str(np.absolute(lat))
        else:
            latstr = 'N' + str(lat)
        if long < 0:
            longstr = 'W' + str(np.absolute(long))
        else:
            longstr = 'E' + str(long)
        return latstr, longstr

    def loadHighResElevation(self): # load the high res elevation data
        lat_min, lat_max = self.getLatMinMax()
        long_min, long_max = self.getLongMinMax()
        lat_min, lat_max = int(np.floor(lat_min/un.deg)), int(np.ceil(lat_max/un.deg))
        long_min, long_max = int(np.floor(long_min/un.deg)), int(np.ceil(long_max/un.deg))
        latarray, longarray = range(lat_min, lat_max+1), range(long_min, long_max+1)
        # create an empty dict for all the height map data
        heightmap = {}
        # create a map of the world to check if the location is in the ocean or not.
        # ASTGDEM data does not cover the ocean, for obvious reasons.
        map = basemap.Basemap(
                        projection="merc",
                        resolution="l",
                        area_thresh=10,
                        llcrnrlon=long_min-0.1,
                        llcrnrlat=lat_min-0.1,
                        urcrnrlon=long_max+0.1,
                        urcrnrlat=lat_max+0.1
                        )
        for lat in latarray:
            for long in longarray:
                if lat < 0:
                    latstr = 'S' + str(-lat)
                else:
                    latstr = 'N' + str(lat)
                if long < 0:
                    longstr = 'W' + str(-long)
                else:
                    longstr = 'E' + str(long)
                filename = 'HighResElevData/ASTGTM2_' + latstr + longstr + '/ASTGTM2_' + latstr + longstr +'_dem.tif'
                print 'attempting to load ' + latstr + '-' + longstr + ' data..'
                while True:
                    x, y = map(long, lat)
                    if not map.is_land(x, y) and not os.path.isfile(filename):
                        heightmap[latstr + ' ' + longstr] = None
                        print 'Point is in the ocean, setting to None..'
                        break
                    try:
                        #heightmap[latstr + '-' + longstr] = gdal.Open(filename, GA_ReadOnly)
                        heightmap[latstr + ' ' + longstr] = gr.from_file(filename)
                        ''' 
                        georasters is slower to load the heightmap data by a factor of about 10,
                        even more so when you are combining the height maps.
                        We should aim to replace this with the siutable gdal scripts
                        ''' 
                        break
                    except ValueError:
                        print "Couldn't load high res elevation data, might be files missing"
        return heightmap
        
    def stitchheightmap(self): # stitch all the height maps we loaded, or create them if we didnt
        heightmap = self.loadHighResElevation()
        if None in heightmap.values():
            # find a height map that isn't None and get its info
            for keys in heightmap:
                if isinstance(heightmap[keys], gr.GeoRaster):
                    key = keys
            xysize = heightmap[key].shape
            sealevel = np.zeros(xysize)
            # getting the geo info
            GeoT = list(heightmap[key].geot)
            for keys, values in heightmap.items(): # for all the maps over the water, create them
                if values is None:
                    lat, long = keys.split()
                    # get rid of the front letter and replace it with the correct sign
                    if lat[0] == 'S':
                        lat = -float(lat[1:]) + 1. - (1 - 0.999861111111112)
                        print lat
                    else:
                        lat = float(lat[1:]) + 1. - (1 - 0.999861111111112)
                    if long[0] == 'W':
                        long = -float(long[1:]) - (1-0.9998611111111)
                    else:
                        long = float(long[1:]) - (1-0.9998611111111)
                    geot = GeoT
                    geot[0], geot[3] = long, lat # not sure why you need to plus one
                    heightmap[keys] = gr.GeoRaster(sealevel, geot, projection=heightmap[key].projection)
        i=0
        print 'combining height maps'
        for item in heightmap: # combine all the height maps, THIS TAKES TOO LONG
            i+=1
            if i==1:
                combineddata = heightmap[item]
            else:
                print heightmap[item]
                print item
                combineddata = combineddata.union(heightmap[item])
        combineddata.to_tiff('tmpHM')
        return combineddata

    # The following method translates given pixel locations into latitude/longitude locations on a given GEOTIF
    # INPUTS: geotifAddr - The file location of the GEOTIF
    #      pixelPairs - The pixel pairings to be translated in the form [[x1,y1],[x2,y2]]
    # OUTPUT: The lat/lon translation of the pixel pairings in the form [[lat1,lon1],[lat2,lon2]]
    # NOTE:   This method does not take into account pixel size and assumes a high enough 
    #	  image resolution for pixel size to be insignificant
    def pixelToLatLon(self, map, pixelPairs):
	    # Get a geo-transform of the dataset
	    gt = map.GetGeoTransform()
	    # Create a spatial reference object for the dataset
	    srs = osr.SpatialReference()
	    srs.ImportFromWkt(map.GetProjection())
	    # Set up the coordinate transformation object
	    srsLatLong = srs.CloneGeogCS()
	    ct = osr.CoordinateTransformation(srs,srsLatLong)
	    # Go through all the point pairs and translate them to pixel pairings
	    latLonPairs = []
	    for point in pixelPairs:
		    # Translate the pixel pairs into untranslated points
		    ulon = point[0]*gt[1]+gt[0]
		    ulat = point[1]*gt[5]+gt[3]
		    # Transform the points to the space
		    (lon,lat,holder) = ct.TransformPoint(ulon,ulat)
		    # Add the point to our return array
		    latLonPairs.append([lat,lon])
	    return latLonPairs
        
    def getHorizonDeg(self): # get the distance to the horizon in degrees
        dist = self.dist
        R = const.R_earth.to('m')
        horizon_dist = dist * 1000. * un.m
        horizon_deg = 180. * un.deg * horizon_dist/(np.pi * R)
        return horizon_deg
    
    def getLatMinMax(self): # minimum and maximum latitude values of the mask
        horizon_deg = self.getHorizonDeg()
        lat = self.convertLatLong()[0]
        lat_min = lat - horizon_deg
        lat_max = lat + horizon_deg
        return lat_min, lat_max
    
    def getLongMinMax(self): # minimum and maximum longitude values of the mask
        horizon_deg = self.getHorizonDeg()
        long = self.convertLatLong()[1]
        long_min = long - horizon_deg
        long_max = long + horizon_deg
        return long_min, long_max

    def alignHeightwithLight(self, outputfilename):
        '''
            Since the light pollution and elevation data have different resolutions,
            we have to make them the same.
        '''
        NDV, xsize, ysize, geotlight, Projection, DataType = gr.get_geo_info(self.lightfilename)
        xres, yres = str(geotlight[1]), str(geotlight[5])
        resample_method = 'cubic'
        os.system('gdalwarp '+ 'tmpHM2.tif' + ' ' + outputfilename + ' -r ' + resample_method+' -tr '+xres+' '+yres)
        os.system('rm tmpHM2.tif')
        
    def createMapSlice(self, mapfilename, outputfilename): # create a slice out of the data
        '''
            We are creating a slice of the data and saving it because we don't want to have
            to load the gigabyte of data every time we want to run this code
        '''
        map = gdal.Open(mapfilename, GA_ReadOnly)
        width = map.RasterXSize
        height = map.RasterYSize
        bands = map.RasterCount
        horizon_deg = self.getHorizonDeg()
        # Get georeference info
        geot = map.GetGeoTransform()
        xOrigin, yOrigin, xOffset, yOffset = self.getMapInfo(geot)
        # Get the data
        data = np.array(map.GetRasterBand(1).ReadAsArray())
        # Make a slice of the data
        xmin, xmax = xOffset[0], xOffset[-1]
        ymin, ymax = yOffset[0], yOffset[-1]
        data_slice = data[ymin:ymax,xmin:xmax]
        new = self.pixelToLatLon(map, [[xmin, ymax]])
        newlat, newlong = new[0][0], new[0][1]
        geot = (newlong, geot[1], geot[2], newlat, geot[4], geot[5])
        NDV, xsize, ysize, GeoT, Projection, DataType = gr.get_geo_info(mapfilename)
        mapslice = gr.GeoRaster(data_slice, geot, projection=Projection, nodata_value=0)
        mapslice.to_tiff(outputfilename)

    def getGeotFromFile(self, filename):
        NDV, xsize, ysize, Geot, Projection, DataType = gr.get_geo_info(filename)
        return Geot
    
    def getMapInfo(self, geot): # just getting some stuff necessary for the mask and mapslice
        lat_min, lat_max = self.getLatMinMax()
        long_min, long_max = self.getLongMinMax()
        xOrigin, yOrigin = geot[0], geot[3]
        pixelsize = geot[1]
        horizon_deg = self.getHorizonDeg()
        n = horizon_deg*2/(pixelsize*un.deg) # resolution
        xValues = np.linspace(long_min/un.deg,long_max/un.deg,n)
        yValues = np.linspace(lat_min/un.deg,lat_max/un.deg,n)
        xOffset = (xValues - xOrigin) / pixelsize
        yOffset = (yValues - yOrigin) / -pixelsize
        yOffset = np.fliplr([yOffset])[0]
        return xOrigin, yOrigin, xOffset, yOffset
    
    def applyMask(self, mapdata, filename):
        geot = self.getGeotFromFile(filename)
        xOrigin, yOrigin, xOffset, yOffset = self.getMapInfo(geot)
        pixelsize = geot[1]
        maplength = len(mapdata)
        lat, long = self.convertLatLong()
        x_obs, y_obs = (long/un.deg - xOrigin)/pixelsize, (lat/un.deg - yOrigin)/-pixelsize
        # define mask
        p = maplength
        x = np.resize(xOffset,(p,1))
        y = np.resize(yOffset,(p,))
        radius = np.ceil(maplength/2.0)
        mask = (x-x_obs)**2 + (y-y_obs)**2 >= radius**2
        datashape = list(mapdata.shape)
        maskshape = list(mask.shape)
        if maskshape[0] - datashape[0] == 1:
            mask = np.delete(mask, -1, 0)
        elif datashape[0] - maskshape[0] == 1:
            data = np.delete(mapdata, -1, 0)
        if maskshape[1] - datashape[1] == 1:
            mask = np.delete(mask, 0, -1)
        elif datashape[1] - maskshape[1] == 1:
            data = np.delete(mapdata, 0, -1)
        # Apply mask
        mx = ma.masked_array(mapdata, mask)
        return mx
    
    def getMap(self): # some functions which im not entirely sure is necessary
        geot = self.getGeotFromFile(self.lightfilename)
        return geot

class SkyBrightness(object): # the guts of the program, should probably be written in C for speed improvements
    N_m = 2.55e19
    sigma_R = 4.6e-27
    c = 0.104
    N_a = N_m
    sigma_a = sigma_R
    gamma = 1./3
    '''
        I guess that these functions need to be set up such that they are
        getting a scalar value as their input. Then can use np.vectorize to
        do the entire map. Might consider writing this section in C/C++ for
        speed improvements.
        
        I have drawn a diagram which shows the geometry of the problem, and have
        documented how each of the values are derived geometrically
        
        See the additional document on the github page which explains the model in more detail
        '''
    def __init__(self, time_obs, UTC_offset, loc_obs, obj, obspar):
        self.time_obs = time_obs
        self.UTC_offset = UTC_offset
        self.loc_obs = loc_obs
        self.obj = obj
        self.obspar = obspar
        self.MapData = LoadMaps(self.time_obs, self.UTC_offset, self.loc_obs, self.obj, self.obspar)
        self.upwardInt = self.getUpwardInt()

    def getLightMap(self): # getting the light map, making sure there are no negative values
        M = self.MapData.lightmap
        M[M < 0] = 0
        M[np.isnan(M)] = 0 # or any nans
        return M
    
    def getUpwardInt(self): # load the light map and convert its values into something useable
        '''
            I am not completely certain of the units for this section, and thus all the values
            are likely to be proportionately off the real values. This is fine if we are
            only interested in fitting the light pollution gradient over a field though.
        '''
        M = self.MapData.lightmap
        map = self.MapData.getMap()
        pixelWidth = map[1]
        pixelHeight = map[5]
        R = const.R_earth.to('cm')
        deg_to_cm = pixelWidth*un.deg*np.pi*R/(180*un.deg)
        area = np.asarray((pixelWidth*deg_to_cm)*(abs(pixelHeight)*deg_to_cm))
        I_up = area * M
        return I_up
    
    def getz(self, alt): # the angle from zenith of the observer's L.O.S
        if alt < 0:
            z = -90 - alt
        else:
            z = 90 - alt
        return z
    
    def getA(self): # A is the altitude of the observer
        A = self.MapData.getelev()/ 1.0e3
        return A
    
    def vectorizeIntegral(self, H, D, Dy, N_a, N_m, sigma_a, sigma_R, c, A, z, az):
        '''
            I think it will be worthwhile putting this section in cython as well.
            We may also want to find a faster integration method, as this is the bottleneck
            in the code.
        '''
        l = self.MapData.lightlength
        ans = np.zeros((l[0],l[1]))
        tmp = 0
        I_up = self.upwardInt
        for i in range(l[0]):
            for j in range(l[1]):
                if I_up[i][j] == 0:
                    continue
                else:
                    #print H[i][j], D[i][j], Dy[i][j], N_a, N_m, sigma_a, sigma_R, c, A, z, az
                    ans[i][j], tmp = integrate.quad(integrand, 0.,
                    np.inf, args = (H[i][j], D[i][j], Dy[i][j], N_a, N_m, sigma_a, sigma_R,
                    c, A, z, az))
        return ans

    def getskybrightness(self, z, az):
        N_a = self.N_a
        N_m = self.N_m
        sigma_a = self.sigma_a
        sigma_R = self.sigma_R
        c = self.c
        A = self.getA()
        D = self.getD()
        Dx, Dy = self.getDxDy()
        H = self.getH()
        aa = time.time()
        vectorizedIntegral = self.vectorizeIntegral(H, D, Dy, 
        N_a, N_m, sigma_a, sigma_R, c, A, z, az)
        I_up = self.upwardInt
        bb = time.time()
        print(bb-aa)
        print 'time printed'
        b = np.pi * N_m * sigma_R * np.exp(-c * H) * I_up * vectorizedIntegral
        print np.sum(b)
        return b

    def loopsky(self, alt, az):
        '''
            Here we are looping over arrays of altitude and azimuth angles to simulate
            an image of the light pollution of the sky.
        '''
        I = np.zeros((len(alt), len(az)))
        count = 0
        for i in range(len(alt)):
            z = self.getz(alt[i])
            print z
            for j in range(len(az)):
                I[i][j] = np.sum(self.getskybrightness(z, az[j]))
                count += 1
                print 'count', count
        return I

    def getdxdy(self):
        '''
            This function will generate the x, y positions over the entire grid, which are
            needed to work out the angles. See the document on github for more info
        '''
        l = self.MapData.lightlength
        if l[0] > l[1]:
            centre_coord = int(l[1]/2)
        else:
            centre_coord = int(l[0]/2)
        dx, dy = np.meshgrid(range(l[1]), range(l[0]))
        dx, dy = dx - centre_coord, dy - centre_coord
        return dx, dy

    def getDxDy(self):
        dx, dy = self.getdxdy()
        horizon_dist = self.MapData.dist # just slapping this value in for now
        p = np.sqrt( np.min(self.MapData.lightlength) )
        scale_factor = 2*horizon_dist/p # convert pixels to m
        sf = scale_factor
        return dx*sf, dy*sf

    def getD(self):
        dx, dy = self.getDxDy()
        D = np.sqrt(dx**2 + dy**2) + 0.001
        D[ self.MapData.lightmap == 0 ] = 0
        return D
    
    def getH(self):
        '''
            Gathering the height map data and removing any nans and setting any negative
            values to zero.
        '''
        H = self.MapData.heightmap
        H[H>0] = 0
        H[np.isnan(H)] = 0
        return H

# Time of observation:
time_obs = '2000-01-01 23:30:00' # Local time 2000-MM-DD hh:mm:ss (YEAR SHOULD CORRESPOND TO JULIAN EPOCH (J2000))
UTC_offset = 10 # Based on location and time of year

# Location of observer: Remember minus sign for S and W.
#lat_obs = [-34,13,10.9] # latitude [degrees,arcminutes,arcseconds]. Map range: [-65,75]
#long_obs = [150,58,20.7] # longitude [degrees,arcminutes,arcseconds]. Map range: [-180,180]
lat_obs = [-31,18,58.4] # latitude [degrees,arcminutes,arcseconds]. Map range: [-65,75]
long_obs = [149,2,32] # longitude [degrees,arcminutes,arcseconds]. Map range: [-180,180]
elev_obs = 1130 # elevation [m]

# Location of astronomical object:
obj_RA = [19,44,56.6] # right ascension [hours,minutes,seconds]
obj_dec = [-14,47,21] # declination [degrees,arcminutes,arcseconds]

# Observing wavelength [nm]
wav_obs = 600

# Refraction corrections for AltAz frame (can use default)
pressure = 0.0 # Atmospheric pressure
ground_temp = 0.0 # Ground-level temperature [C]
relative_humidity = 0.0
loc_obs = [lat_obs, long_obs, elev_obs]
obj = [obj_RA, obj_dec]

obspar = [wav_obs, pressure, ground_temp, relative_humidity]

L = SkyBrightness(time_obs, UTC_offset, loc_obs, obj, obspar)
alt = np.linspace(0, 90, 60)
az = np.linspace(0, 360, 100)
I = L.loopsky(alt, az)
np.save('Icoona', I)

my_cmap = mpl.cm.get_cmap('inferno')
ax = plt.contourf(az, alt, np.log10(I), 1000, cmap = my_cmap)
plt.colorbar(ax)
plt.show()

