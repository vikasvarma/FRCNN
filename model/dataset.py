
# Import definitions:
import torch
import numpy as np

# ------------------------------------------------------------------------------
# BigTiff Dataset class to sequentially load batches of data from BigTIFF files.
# ------------------------------------------------------------------------------
class BigtiffDataset(torch.utils.data.Dataset):
    """
        Dataset class for sequential patch loading.
    """
    
    # --------------------------------------------------------------------------
    def __init__(self, images, samples=None):
        """
            Constructor
        """
        super(BigtiffDataset, self).__init__()
        
        self.BigTIFFs = images
        if samples is None:
            self.Samples = self.createSamples()
        else:
            self.Samples = samples
    
    # --------------------------------------------------------------------------
    def createSamples(self):
        """
            Default sampling policy to sample contiguous patches from specified bigTIFFs at current directory.
        """
        _samples = []
        
        for imgID in range(len(self.BigTIFFs)):
            _btif = self.BigTIFFs[imgID]
            
            # Sample at a stride of patch size, incomplete patches at image
            # borders are excluded.
            _rows = np.arange(0, _btif.ImageSize[_btif.DirectoryID, 1], 
                                 _btif.PatchSize[_btif.DirectoryID, 1])
            _cols = np.arange(0, _btif.ImageSize[_btif.DirectoryID, 1], 
                                 _btif.PatchSize[_btif.DirectoryID, 1])
            
            # Create sampling locations:
            _prows, _pcols = np.meshgrid(_rows, _cols)
            _pcord = np.column_stack((_prows.flatten(), 
                                      _pcols.flatten())).astype(int)
            
            for pid in range(_pcord.shape(0)):
                _samples = np.append(_samples, 
                                    (imgID, _pcord[pid,0], _pcord[pid,1]))
        
        # Done.
        return _samples
    
    # --------------------------------------------------------------------------
    def define_samples(self, sampling_policy):
        self.Samples = sampling_policy
    
    # --------------------------------------------------------------------------
    def __getitem__(self, index):
        """
            Returns a single image patch (sample)
        """
        
        # Get sample patch at specified index:
        sample_def = self.Samples[index]
        btif       = self.BigTIFFs[sample_def[0]]
        row_cord   = sample_def[1]
        col_cord   = sample_def[2]
        patch      = btif.getPatch(row_cord, col_cord)
        
        return patch
# ______________________________________________________________________________