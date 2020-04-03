"""
"""

# ------------------------------------------------------------------------------
# All import definitions go here:
import ctypes
import math
import os
import numpy as np
import torch
from libtiff import libtiff_ctypes
from libtiff.libtiff_ctypes import TIFF
from libtiff.libtiff_ctypes import libtiff as _tiflib

# ------------------------------------------------------------------------------
# TIFF Exception Handling Class Definition
# ------------------------------------------------------------------------------
class BigTiffException(Exception):
    pass

################################################################################

# ------------------------------------------------------------------------------
# Image Spatial Mapping Class Definition
# ------------------------------------------------------------------------------
class SpatialMap:
    """
    SPATIALMAP - Spatial Reference Mapping object that maps image indices to    spanning world coordinates.

    Parameters:

        ImageSize
        XLimits
        YLimits
        PixelSpacing


    Methods:

        [X,Y] = image2Spatial(R,C)
        [R,C] = spatial2Image(X,Y)
    """

    def __init__(self, imageSize, xlimits, ylimits):
        
        self.ImageSize = imageSize
        self.XLimits   = xlimits
        self.YLimits   = ylimits

        # Compute pixel spacing:
        self.PixelSpacing = np.concatenate([np.diff(xlimits), np.diff(ylimits)])
        self.PixelSpacing = self.PixelSpacing / np.flip(imageSize)

    def image2Spatial(self, indices):
        # Converts the image ROW-by-COL indices to world coordinates.
        coordinates = np.multiply(np.flip(indices) - 0.5, self.PixelSpacing)
        coordinates += np.array([self.XLimits[0], self.YLimits[0]])

        # Bound the coordinate to world extents:
        coordinates[...,0] = np.clip(coordinates[...,0], \
                                   self.XLimits[0], self.XLimits[1])
        coordinates[...,1] = np.clip(coordinates[...,1], \
                                   self.YLimits[0], self.YLimits[1])

        # Done, return:
        return coordinates
    
    def spatial2Image(self, coordinates):
        # Converts X and Y world coordinates to image indices.
        indices = np.subtract(np.flip(coordinates), \
                              np.array([self.XLimits[0], self.YLimits[0]]))
        indices = np.floor( np.divide(indices, self.PixelSpacing) )

        # Clip the indices to image extents:
        indices[...,0] = np.clip(indices[...,0], \
                               self.ImageSize[0], self.ImageSize[1])
        indices[...,1] = np.clip(indices[...,1], \
                               self.ImageSize[0], self.ImageSize[1])

        # Done, return:
        return indices

################################################################################

# ------------------------------------------------------------------------------
# Tile Cache Class Definition
# ------------------------------------------------------------------------------
class LRUCache():
    """Least Recently Used (LRU) cache implementation."""

    def __init__(self, maxsize):
        return

    def __getitem__(self, key):
        return None

    def __setitem__(self, key, value):
        return

################################################################################

# ------------------------------------------------------------------------------
# TIFF I/O Adapter Class Definition
# ------------------------------------------------------------------------------
class TIFFAdapter:
    """
    TIFFADAPTER - Adapter with routines to read and write from TIFF files.

    Parameters:

        Source
        Mode
        Metadata
        PixelSpacing

    Methods:
        getTile(level,)
    """

    #---------------------------------------------------------------------------
    def __init__(self, filename, mode='r'):
        # Find the TIFF file specified:
        if not os.path.exists(filename):
            filename = os.path.abspath(filename)

        # Initialize the object:
        self.Source   = filename
        self.Mode     = mode
        self._tiff    = None
        self.Metadata = []

        # Attempt to open the TIFF file and read directory metadata:
        self._open()
        self._readMetadata()
    
    #---------------------------------------------------------------------------
    def __del__(self):
        if not self._tiff is None:
            self._tiff.close()
            self._tiff = None

    #---------------------------------------------------------------------------
    def _open(self):
        # Encode the string and open the TIFF object:
        try:
            byteName = self.Source.encode('utf8')
            self._tiff = libtiff_ctypes.TIFF.open(byteName)
        except:
            print('Unable to open the file!')

    #---------------------------------------------------------------------------
    def _readMetadata(self):
        # Internal function to read metadata tags from TIFF image:

        # Identify all the compatible tiff library metadata tags:
        tags = [key.split('_')[1] for key in \
                    dir(libtiff_ctypes.tiff_h) if key.startswith('')]
        
        # Find the number of directories in the TIFF image:
        self.NumDirectories = _tiflib.TIFFNumberOfDirectories(self._tiff)

        for numDir in range(0, self.NumDirectories):
            # Create a new dictionary for every TIFF directory:
            
            # Set associated TIFF directory:
            _tiflib.TIFFSetDirectory(self._tiff, numDir)

            _dirMeta = {}
            for tag in tags:
                try:
                    value = self._tiff.GetField(tag.lower())
                    if not value is None:
                        _dirMeta[tag] = value
                except NameError:
                    # Pass on errors caused due to an ill-defined tag name.
                    pass
            
            # Add metadata to attribute:
            self.Metadata.append(_dirMeta)
        
        # Reset TIFF directory to first:
        _tiflib.TIFFSetDirectory(self._tiff, 0)

    #---------------------------------------------------------------------------
    def locatePatch(self, dirNum, patchLoc):
        # Returns the [Row Col] index of the TIFF tile containing the patch
        # location PATCHLOC and the tile origin and end pixel indices.
        _tileLength = self.Metadata[dirNum].get('TILELENGTH')
        _tileWidth  = self.Metadata[dirNum].get('TILEWIDTH')
        _tileid     = np.floor(np.divide(patchLoc, [_tileLength, _tileWidth]))
        _tileid     = _tileid.astype(int)
        
        return _tileid
    
    #---------------------------------------------------------------------------
    def readTile(self, dirnum, trow, tcol):
        """
            Reads a single unit of tile from a TIFF file at the specified tile size.
            
            Parameters:
            
                DirNum
                TileRow
                TileCol
        """
        
        # Set associated TIFF directory:
        _tiflib.TIFFSetDirectory(self._tiff, dirnum)
        
        # TODO - Support separate planar config.
        # Assume CONTIG/CHUNKY Planar Configuration:
        _th         = self.Metadata[dirnum].get('TILELENGTH')
        _tw         = self.Metadata[dirnum].get('TILEWIDTH')
        _bps        = self.Metadata[dirnum].get('BITSPERSAMPLE')
        _spp        = self.Metadata[dirnum].get('SAMPLESPERPIXEL')
        _planar     = self.Metadata[dirnum].get('PLANARCONFIG')
        _tileBytes  = _tiflib.TIFFTileSize(self._tiff).value
        _tileOrigin = [int(trow * _tw), int(tcol * _th)]
        
        if _planar is 1: 
            # Contiguously planar configuration, all channels stored in 
            # contiguous memory locations.
            _tileBuffer = ctypes.create_string_buffer(_tileBytes)
            _readSize   = _tiflib.TIFFReadTile(self._tiff, _tileBuffer, \
                            _tileOrigin[1], _tileOrigin[0], 0, 0)
        
            if not _readSize.value - _tileBytes is 0:
                raise BigTiffException(\
                    'Read an unexpected number of bytes from an encoded tile')
        
            # Cast the data read properly to a numpy array:
            _tile       = np.ctypeslib.as_array(ctypes.cast(_tileBuffer, \
                             ctypes.POINTER(ctypes.c_uint16 if _bps is 16       else ctypes.c_uint8)), (_th, _tw, _spp))
        
        elif _planar is 2:
            # Separate planar configurations, each image channel is stored in
            # individual sections of memory, read each plane one by one and 
            # concatenate them.
            channels = []
            for _sample in range(_spp - 1, -1, -1):
                _channelBuffer = ctypes.create_string_buffer(_tileBytes)
                _readSize   = _tiflib.TIFFReadTile(self._tiff, _channelBuffer, \
                                _tileOrigin[1], _tileOrigin[0], 0, int(_sample))
                
                # Error if expected bytes are not read:
                if not _readSize.value - _tileBytes is 0:
                    raise BigTiffException(\
                        'Read an unexpected number of bytes from an encoded tile for channel %d.' % _sample)
                    
                # Append to channels:
                _channel = np.ctypeslib.as_array(ctypes.cast(_channelBuffer, \
                             ctypes.POINTER(ctypes.c_uint16 if _bps is 16       else ctypes.c_uint8)), (_th, _tw))
                
                channels.append(_channel)
                
            # Convert the channels to a stacked image tile:
            _tile = np.stack(channels, axis=2)
        
        # Convert this numpy array to a pytorch tensor on the GPU.
        # NOTE: This does NOT perform data copy
        #_tile = torch.as_tensor(_tile, device=torch.device('cuda'))
        
        # Done.
        return _tile


################################################################################

# ------------------------------------------------------------------------------
# BigTiff class to read from large TIFF files.
# ------------------------------------------------------------------------------
class Bigtiff():
    """
        BIGTIFF - Enables processing large-images by loading selective units/ 
                  blocks of data into memory.
    """

    Version = '1.0'
    
    #---------------------------------------------------------------------------
    def __init__(self, filename):
        # Biftiff object constructor. Performs the following tasks:
        #   1. Create an adapter instance to perform I/O
        #   2. Spatially map all directories to identical world coordinates.
        #   3. Store some important metadata in-house.
        
        # Construct an adapter for I/O
        self._adapter       = TIFFAdapter(filename, mode='r')
        
        # Derive image properties:
        self.ImageSize      = [[meta['IMAGELENGTH'], meta['IMAGEWIDTH']] \
                                for meta in self._adapter.Metadata]
        self.SpatialMapping = self._getSpatialMaps(self.ImageSize)
        self.Datatype       = self._getDatatype()
        self.Colorspace     = self._getColorspace()
        self.PatchSize      = [[2*meta['TILELENGTH'], 2*meta['TILEWIDTH']] \
                                for meta in self._adapter.Metadata]
        self.DirectoryID    = 0
        
        # Identify the total number of blocks in each image:
        self.NumPatches     = np.ceil(np.divide(self.ImageSize, self.PatchSize))
        self.TotalPatches   = np.sum(np.prod(self.NumPatches, 1))
        
        # Create patch list for sequential read:
        self._createPatchList()
        
    #---------------------------------------------------------------------------
    def __del__(self):
        # Delete the adapter instance. Explicitly calling this to avoid
        # dangling pointers.
        if not self._adapter is None:
            del(self._adapter)
    
    #---------------------------------------------------------------------------
    def _getSpatialMaps(self, imageSizes):
        # Computes the spatial extents of the image from provided image sizes.
        _finest       = np.argmax(np.prod(imageSizes, 1))
        _coarsest     = np.argmin(np.prod(imageSizes, 1))
        _worldXLimits = [0.5, imageSizes[_finest][1] + 0.5]
        _worldYLimits = [0.5, imageSizes[_finest][0] + 0.5]
        
        # Construct spatial maps for each image directory.
        _spatialMaps  = [SpatialMap(imsize, _worldXLimits, _worldYLimits) \
                            for imsize in imageSizes]
        return _spatialMaps
    
    #---------------------------------------------------------------------------
    def _getDatatype(self):
        # Identify the pixel datatype:
        datatype = []
        for meta in self._adapter.Metadata:
            if   meta.get('SAMPLEFORMAT') is 1: # Unsigned Integer
                _type  = "UINT"
            elif meta.get('SAMPLEFORMAT') is 2: # Signed Integer
                _type  = "INT"
            elif meta.get('SAMPLEFORMAT') is 3: # Floating Point Decimal
                _type = 'FLOAT'
            else:
                _type = 'NONE'
                
            _type += " (%d-Bit)" % np.unique(meta.get('BITSPERSAMPLE'))
            datatype.append(_type)
        
        if len(set(datatype)) is 1:
            return datatype[0]
        else:
            return datatype
    
    #---------------------------------------------------------------------------
    def _getColorspace(self):
        # Define all possible color types:
        _colors = ["Grayscale (WhiteIsZero)",
                   "Grayscale (BlackIsZero)",
                   "RGB",
                   "Indexed",
                   "Transparency Mask",
                   "CMYK",
                   "YCbCr"]
        
        _photometric = [_colors[meta.get('PHOTOMETRIC')] \
                            for meta in self._adapter.Metadata]
        
        if len(set(_photometric)) is 1:
            return _photometric[0] 
        else:
            return _photometric
    
    #---------------------------------------------------------------------------
    def _createPatchList(self):
        # Create a list of patches to be read from the TIFF directory.
        _numPatches = self.NumPatches[self.DirectoryID]
        _rows, _cols = np.meshgrid(np.arange(0,_numPatches[0]-1), \
                                   np.arange(0,_numPatches[1]-1))
        _rows = _rows.flatten()
        _cols = _cols.flatten()
        
        self.PatchList = np.column_stack((_rows, _cols)).astype(int)
        self._currentPatch = 0
    
    #---------------------------------------------------------------------------
    def setDirectory(self, dirNum):
        # Need this set before all I/O.
        self.DirectoryID = dirNum
    
    #---------------------------------------------------------------------------
    def setPatchSize(self, patchSize):
        # Set the patch size associated with reading the image
        self.PatchSize[self.DirectoryID] = patchSize
    
    #---------------------------------------------------------------------------
    def getPatch(self, r, c):
        """
            Retrieve the patch [r,c] patch of the image from the current directory.
        """
        
        # Find all tiles required to stitch the patch:
        _patchOrigin = np.array([r*self.PatchSize[self.DirectoryID][0],
                                 c*self.PatchSize[self.DirectoryID][1]])
        _patchEnd    = _patchOrigin + self.PatchSize[self.DirectoryID] - 1
        
        # Bound patch locations to image size:
        _patchOrigin = np.clip(_patchOrigin, \
                               [0,0],self.ImageSize[self.DirectoryID])
        _patchEnd    = np.clip(_patchEnd, \
                               [0,0],self.ImageSize[self.DirectoryID])
        
        # Compute out patch size:
        _patchSize   = _patchEnd - _patchOrigin + 1
        
        
        # Locate the patches in TIFF tiles:
        _originID    = self._adapter.locatePatch(self.DirectoryID, _patchOrigin)
        _endID       = self._adapter.locatePatch(self.DirectoryID, _patchEnd)
        
        # Get all tiles and construct:
        _patch    = []
        
        for _row in range(_originID[0], _endID[0] + 1):
            
            # Iterate row-wise over all required tiles:
            _rowOfTiles = []
            
            for _col in range(_originID[1], _endID[1] + 1):
                # Construct a row of all column tiles
                _tile = self._adapter.readTile(self.DirectoryID, _row, _col)
                _rowOfTiles.append(_tile)
                
            # Concatenate the columns into a single strip and add to patch
            _strip = np.concatenate(_rowOfTiles, axis=1)
            _patch.append(_strip)
            
        # Now, concatenate all strips to form a single patch:
        _patch = np.concatenate(_patch, axis=0)
        _readSize = np.array(_patch.shape)
        
        if not np.array_equal(_patchSize, _readSize[1:2]):
            _patch = _patch[0:_patchSize[0], 0:_patchSize[1]]
               
        return _patch
    
    #---------------------------------------------------------------------------
    def read(self):
        """
            Sequentially read all set of patches from the image.
        """
        # Get current patch id:
        _patchIndex = self.PatchList[self._currentPatch]
        _patch      = self.getPatch(_patchIndex[0], _patchIndex[1])
        
        # Increment current patch id:
        self._currentPatch += 1
        
        return _patch
    
################################################################################

from PIL import Image
import time

btif = Bigtiff('data/tumor_separate.tif')
btif.setPatchSize([1800, 1900])
for n in range(0,3):
    start = time.time()
    patch = btif.read()
    end = time.time()
    print(end - start)
    img   = Image.fromarray(patch)
    img.show()