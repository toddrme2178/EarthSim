import os
from colorcet import cm_n
import cartopy.crs as ccrs
import pandas as pd
import numpy as np

from earthsim.io import read_3dm_mesh, read_mesh2d

import geoviews as gv
import holoviews as hv
import datashader as ds
from holoviews.operation.datashader import datashade, rasterize


class adhViz():
    """ AdH visualization class"""
    def __init__(self):
        self.tris = None          # mesh elements
        self.verts = None         # mesh nodes
        self.mesh_points = None   # mesh nodes as gv.Points (z as vdim)
        self.tiles = None         # background tiles for display
        self.results = {}         # results dict
        self.projection = None    # crs projection

    def load_mesh(self, fpath, skiprows=1):
        """ method to load an AdH *.3dm mesh from file and set into self """
        # read the mesh, set the verts and elements into the adh object
        self.tris, self.verts = read_3dm_mesh(fpath, skiprows=skiprows)

        self.set_mesh_points()

    def set_mesh_points(self):
        """Method to set the mesh points as gv points (projected if necessary)"""
        # ensure that verts exist
        if self.verts is None:
            raise IOError('Vertices must be set before setting mesh points')

        # if projection is not available
        if self.projection is None:
            # set the points into the adh object
            self.mesh_points = gv.operation.project_points(gv.Points(self.verts, vdims=['z']))
        # if projection is available
        else:
            # set the points into the adh object
            self.mesh_points = gv.operation.project_points(gv.Points(self.verts, vdims=['z'], crs=self.projection))

    def view_elements(self):
        """ Method to display the mesh as wireframe elements"""
        # create trimesh
        tri_mesh = gv.TriMesh((self.tris, self.mesh_points))
        
        # if background tiles have been loaded
        if self.tiles is not None:
            return self.tiles * tri_mesh.edgepaths
        # if no background tiles
        else:
            return tri_mesh.edgepaths
        
    def view_bathy(self):
        """ Method to display the mesh as continuous color contours"""
        # create trimesh
        tri_mesh = gv.TriMesh((self.tris, self.mesh_points))
        
        # if background tiles have been loaded
        if self.tiles is not None:
            return (self.tiles * datashade(tri_mesh, aggregator=ds.mean('z')))
        # if no background tiles
        else:
            return datashade(tri_mesh, aggregator=ds.mean('z'))
    
    def load_tiles(self, source=None):
        """ Method to load background tiles"""
        # default source is open street maps
        if source is None:
            self.tiles = gv.WMTS('https://maps.wikimedia.org/osm-intl/{Z}/{X}/{Y}@2x.png')
        # load specific source
        else:
            self.tiles = gv.WMTS(source) # untested
            
    def create_animation(self, label):
        """ Method to create holoviews dynamic map meshes (1D data only)"""
        
        # function for dynamic map call
        def time_mesh(time):
            # add this time step's data as a vdim under the provided label
            depth_points = self.mesh_points.add_dimension(label, 0, self.results[label][time].values[:, 0], vdim=True)
            # return a trimesh with this data
            return gv.TriMesh((self.tris, depth_points), crs=ccrs.GOOGLE_MERCATOR)
    
        # create dynamic map, resort according to time
        meshes = hv.DynamicMap(time_mesh, kdims='Time').redim.values(Time=sorted(self.results[label].keys()))
        
        # return the dynamic map object
        return meshes
    
    def read_data(self, fpath):
        """ Method to read AdH results data from *.dat file (1D data only)"""
        # get the label from the filename (ensures consistent labeling across within AdH versions)
        label = os.path.splitext(os.path.split(fpath)[1])[0].split('_')[-1].lower()
        # store the results in a dict
        self.results[label] = read_mesh2d(fpath)
        
    def view_animation(self, label, disp_range=None, colorbar=False, colormap=None):
        """ High level method to display an animation of 'label' variable from self"""
        # create the animation meshes
        meshes = self.create_animation(label)
        
        # no display range provided
        if disp_range is None:
            # set default
            disp_range = (-0.3, 0.3)  # todo switch to the range of data
            
        # no colormap provided
        if colormap is None:
            # set default
            colormap = 'rainbow_r'
        
        # if background tiles have been loaded
        if self.tiles is not None:
            return(self.tiles * rasterize(meshes, aggregator=ds.mean(label)).redim.range(
                **{label: disp_range}).options(colorbar=colorbar, cmap=cm_n[colormap]))
        # if no background tiles
        else:
            return rasterize(meshes, aggregator=ds.mean(label)).redim.range(**{label: disp_range})


def reproject_point(points, coord_sys):
    """Method to reproject a map point to a new coordinate system

    Parameters
    ----------
    points - list
        List of PointDraw stream dataframes
    coord_sys - obj
        Destination CCRS coordinate system

    Returns
    -------
    reprojected - list
        List of dataframes with reprojected Latitude, Longtitude, and z values

    """

    # todo figure out if this can be combined into one method with reproject_poly
    # todo there is bokeh/holoviews bug that overwrites the dict labels (https://github.com/ioam/holoviews/issues/2650)
    reprojected = []

    # # reproject the point from Mercator (bokeh output) to the original coordinate system
    # try:
    #     for point in points:
    #         point_reproject = coord_sys.transform_points(ccrs.GOOGLE_MERCATOR, point['Longitude'].values,
    #                                                      point['Latitude'].values)
    #
    #         # put into dataframe, store in list
    #         reprojected.append(pd.DataFrame(data=point_reproject, columns=['Longitude', 'Latitude', 'z']))
    #
    # except:
    #     for point in points:
    #         point_reproject = coord_sys.transform_points(ccrs.GOOGLE_MERCATOR, point['xs'].values, point['ys'].values)
    #
    #         # put into dataframe, store in list
    #         reprojected.append(pd.DataFrame(data=point_reproject, columns=['Longitude', 'Latitude', 'z']))

    point_reproject = coord_sys.transform_points(ccrs.GOOGLE_MERCATOR, points['Longitude'].values,
                                                 points['Latitude'].values)

    # put into dataframe, store in list
    reprojected.append(pd.DataFrame(data=point_reproject, columns=['Longitude', 'Latitude', 'z']))

    # return user_point
    return reprojected


def reproject_path(polys, coord_sys):
    """Method to reproject a map path to a new coordinate system

    Parameters
    ----------
    polys - list
        List of PolyDraw stream dataframes
    coord_sys - obj
        Destination CCRS coordinate system

    Returns
    -------
    reprojected - list
        List of dataframes with reprojected Latitude, Longtitude, and z values

    """
    reprojected = []
    # reproject the line from Mercator (bokeh output) to the original coordinate system
    # bokeh tools always output mercator
    # todo there is bokeh/holoviews bug that overwrites the dict labels (https://github.com/ioam/holoviews/issues/2650)
    # this is wrapped in try/except until that is resolved.
    try:
        for poly in polys:
            path_reproject = coord_sys.transform_points(ccrs.GOOGLE_MERCATOR,
                                                        poly['Longitude'].values, poly['Latitude'].values)
            # put into dataframe, store in list
            reprojected.append(pd.DataFrame(data=path_reproject, columns=['Longitude', 'Latitude', 'z']))

    except:
        for poly in polys:
            path_reproject = coord_sys.transform_points(ccrs.GOOGLE_MERCATOR, poly['xs'].values, poly['ys'].values)
            # put into dataframe, store in list
            reprojected.append(pd.DataFrame(data=path_reproject, columns=['Longitude', 'Latitude', 'z']))

    return reprojected


def colors_and_levels(levels, colors, clip=None, N=255):
    intervals = np.diff(levels)
    cmin, cmax = min(levels), max(levels)
    interval = cmax-cmin
    cmap = []
    for intv, c in zip(intervals, colors):
        cmap += [c]*int(N*(intv/interval))
    if clip is not None:
        clmin, clmax = clip
        lidx = int(N*((cmin-clmin)/interval))
        uidx = int(N*((cmax-clmax)/interval))
        cmap = cmap[lidx:N-uidx]
    return cmap