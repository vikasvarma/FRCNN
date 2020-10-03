import os
from .bigtiff import Bigtiff, SpatialMap, BigtiffDataset
import json
import torch
import math
import numpy as np

class STACCarribeanDataset(BigtiffDataset):
    """
        Pytorch Dataset for Open AI Carribean Challenge
        Loads TIFF patches and corresponding bounding boxes and labels.
    """
    
    _root   = r".\openai-challenge\data"
    _config = {
        "colombia" : ["borde_rural", "borde_soacha"],
        "guatemala": ["mixco_1_and_ebenezer", "mixco_3"],
        # "st_lucia" : ["casteries", "dennery", "gros_islet"]
        "st_lucia" : ["dennery"]
    }
    _classes = ["concrete_cement", 
                "healthy_metal",
                "incomplete",       
                "irregular_metal", 
                "other"]
    
    # --------------------------------------------------------------------------
    def __init__(self, dataset="colombia", train=True, transform=None):
        
        # Set internal parameters:
        self.Train     = train
        self.Name      = dataset
        self.Transform = transform
        
        # Load images and instantiate super class:
        images = self.load_images()
        self.GroundTruth = self.load_groundtruth(images)
        
        # Assign patch sizes to bigtiffs based on maximum ROI size:
        self.PatchSize = self.set_patchsize(images)
        stride = [int(p/2) for p in self.PatchSize]
        
        # Sampling policy:
        sampleSet = self.createSamples(images, stride)
        super(STACCarribeanDataset, self).__init__(images, samples=sampleSet)
        
        # Filter samples to the ones that only contain ROIs:
        sampleSet = self.filterSamples(sampleSet)
        self._set_sampling_scheme_(sampleSet)
    
    # --------------------------------------------------------------------------
    def get_classes(self):
        return self._classes
        
    # --------------------------------------------------------------------------
    def __getitem__(self, index):
        # Overloading super class method to only return the first 3 (RGB) 
        # channels of the patch.
        
        patch, rois = super().__getitem__(index)
        patch = patch[:,:,:-1]
        
        # Transform the image if applicable:
        if not self.Transform is None:
            # Get patch origin to convert ROI coordinates to patch coordinates.
            patch_origin = self.Samples[index, 1:]
            patch_origin = np.tile(patch_origin, 2)
            rois[:,:4] = rois[:,:4] - patch_origin
            
            # Transform:
            outputs = self.Transform({'image': patch, 'ROI': rois})
            patch   = outputs['image']
            rois    = outputs['ROI']
            
            """
            # Convert them back to global image coordinates:
            if torch.is_tensor(rois):
                rois = rois + torch.from_numpy(patch_origin)
            elif type(rois) is numpy.ndarray:
                rois = rois + patch_origin
            else:
                raise TypeError('Must be numpy array or torch tensor.')
            """
        
        return patch, rois
        
    # --------------------------------------------------------------------------
    def __getrois__(self, index):
        
        # Get the sample patch at specified index:
        sample = self.Samples[index]
        
        # From the list of available ROIs, identify the ones that lie inside 
        # this patch:
        regions       = list(self.BigTIFFs.keys())
        sample_region = regions[sample[0]]
        btif          = self.BigTIFFs[sample_region]
        patch_origin  = sample[1:]
        patch_end     = patch_origin + btif.PatchSize[btif.DirectoryID] - 1
        
        rois = self.GroundTruth[sample_region]['ROI']
        
        if rois.size != 0:
            inside = ((rois[:,0] >= patch_origin[0]) &
                      (rois[:,2] <= patch_end[0]) &
                      (rois[:,1] >= patch_origin[1]) &
                      (rois[:,3] <= patch_end[1]))
            rois = rois[inside, :]
        
        return rois
    
    # --------------------------------------------------------------------------
    def collate(self, batch):
        """Collation function to be used with data loaders"""
        
        images   = []
        roi_size = 5 if self.Train else 4
        rois     = torch.zeros((len(batch), 20, roi_size), dtype=torch.float32)
        rois     = rois.to(batch[0][1].device)
        
        for _b in range(len(batch)):
            # Accumulate patches:
            images.append(batch[_b][0].to(torch.float32))
            
            # Accumulate ROI:
            """
            image_num = torch.Tensor([_b]).expand(batch[_b][1].size(0))
            image_num = image_num.type(batch[_b][1].dtype).view(-1,1)
            image_num = image_num.to(batch[_b][1].device)
            _roi      = torch.cat([image_num, batch[_b][1]], dim=1)
            rois      = torch.cat([rois, _roi], dim=0)
            """
            num_boxes   = batch[_b][1].size(0)
            rois[_b,:num_boxes,:] = batch[_b][1]
            
        # Stack outputs and return
        batch = [torch.stack(images, dim=0), rois]
        return batch
    
    # --------------------------------------------------------------------------
    def load_images(self):
        """
            Use the Colombia dataset:
        """
        
        # Create BigTiff objects for dataset images:
        images = {}
        for dataset in self.Name:
            for region in self._config[dataset]:
                reg_dir = os.path.join(self._root, dataset, region)
                imgfile = '{0}_ortho-cog.tif'.format(region)
                imgfile = os.path.join(reg_dir, imgfile)
                btif = Bigtiff(imgfile)
                
                # Load extents in world coordinates and setup the spatial maps:
                meta_json = '{0}-imagery.json'.format(region)
                meta_json = os.path.join(reg_dir, meta_json)
                with open(meta_json, 'r') as fid:
                    _meta = fid.read()
                metadata = json.loads(_meta)
                _xmin, _ymin, _xmax, _ymax = metadata['bbox']
                
                # Create spatial mapping for each image level:
                refmaps = [SpatialMap(imsize, [_xmin, _xmax], [_ymin, _ymax]) 
                                for imsize in btif.ImageSize]
                btif.setSpatialMap(refmaps)
                
                images[region] = btif
        
        return images
    
    # --------------------------------------------------------------------------
    def load_groundtruth(self, images):
        """
            Parse JSON
        """
        groundtruth = {}
        
        for dataset in self.Name:
            for region in self._config[dataset]:
                # Identify the JSON file that holds ground truth:
                if self.Train:
                    _json = 'train-{0}.geojson'.format(region)
                else:
                    _json = 'test-{0}.geojson'.format(region)

                reg_dir = os.path.join(self._root, dataset, region)
                _json = os.path.join(reg_dir, _json)
                
                # Decode the JSON:
                with open(_json, 'r') as fid:
                    json_data = fid.read()
                _jsonDict = json.loads(json_data)
                features  = _jsonDict["features"]
                
                # Concatenate all features:
                roi_sz  = (len(features),5) if self.Train else (len(features),4)
                rois    = np.zeros(roi_sz, dtype=np.float)
                roi_ids = []
                
                for n, roi in enumerate(features):
                    roi_id        = roi['id']
                    coordinates   = np.array(roi['geometry']['coordinates'])
                    
                    if len(coordinates.shape) != 3:
                        # NOTE: Some data points have ROI coodinates provided as
                        #       two disjoint sets of ROI. For this flatten the 
                        #       list before creating the array.
                        coordinates = coordinates[0][0]

                    coordinates = np.reshape(coordinates, (-1,2))
                    
                    # NOTE: Some ROIs are polygonal due to the shape of the 
                    #       roof not being quadrilateral, these data points are 
                    #       aggregated to an enclosing quadrilateral as the 
                    #       network is only capable of detecting bounding boxes.
                    _top_left  = np.amin(coordinates, axis=0)
                    _bot_right = np.amax(coordinates, axis=0)
                    
                    # Append the roi label info:
                    bbox = np.append(_top_left, _bot_right)
                    if self.Train:
                        material  = roi['properties']['roof_material']
                        cls_label = np.array(self._classes.index(material))
                        bbox      = np.append(bbox, cls_label)
                    
                    rois[n,:]  = bbox
                    roi_ids   += [roi_id]

                # Rearrange the labels into [y1 x2 y2 x2] format:
                if self.Train:
                    rois = rois[:,[1,0,3,2,4]]
                else:
                    rois = rois[:,[1,0,3,2]]
                
                # Convert the ROI coordinates from latitude and longitude 
                # coordinates to image row-col.
                btif   = images[region]
                refmap = btif.SpatialMapping[btif.DirectoryID]
                
                # Correct Y coordinate notation and set the origin to top-left:
                # NOTE: Doing this will flip the ymin and ymax coordinates as they 
                #       would now correspond to opposite corners.
                rois[:,[2,0]] = refmap.YLimits[1] - \
                                    (rois[:,[0,2]] - refmap.YLimits[0])
                
                rois[:,[0,1]] = refmap.spatial2Image(rois[:,[1,0]])
                rois[:,[2,3]] = refmap.spatial2Image(rois[:,[3,2]])
                
                # Add to dictionary:
                groundtruth[region] = {"ID": roi_ids, "ROI": rois}
        
        return groundtruth
    
    # --------------------------------------------------------------------------
    def set_patchsize(self, images):
        # Strategy: For each image, obtain the maximum possible ROI height and 
        #           width. The global patch size (the size of network input) is 
        #           set to the nearest (ceil) power of 2 for all images in the 
        #           dataset.
        
        # Compute maximum size:
        _psize = 0;
        for region in images:
            btif   = images[region]
            roi    = self.GroundTruth[region]['ROI']
            height = roi[:,2] - roi[:,0] + 1
            width  = roi[:,3] - roi[:,1] + 1
            _psize = max(_psize, 
                         np.maximum(np.amax(height), np.amax(width)).item())
            
        # Adjust the patch size to nearest (ceil) power of two and update the 
        # bigTIFFs.
        _psize = np.power(2, np.floor(np.log2(_psize))).astype(np.int).item()
        
        for region in images:
            btif = images[region]
            btif.PatchSize[btif.DirectoryID] = [_psize, _psize]
        
        return [_psize, _psize]
    
    # --------------------------------------------------------------------------
    def count_classes(self, index=None):
        """Compute the class count of ROIs for each sample."""
        
        if index is None:
            index = np.arange(self.Samples.shape[0])
        elif isinstance(index, int):
            index = [index]
        
        count = np.zeros((len(index), len(self._classes)), dtype=np.int)
        for _ind in range(len(index)):
            rois = self.__getrois__(index[_ind])
            count[_ind, :] = np.bincount(rois[:,4].astype(np.int), 
                                         minlength=len(self._classes))
            
        return count
    
    # --------------------------------------------------------------------------
    def balance_classes(self, classids):
        """Balance ROI instances across the dataset
        
            Arguments:
                ClassIDs - Define the set of classes that should be considered while sample balancing. Helps ignore labels that are inconsequential.
        """
        
        # Get ROI class counts for each sample patch:
        samples    = self.SampleID
        counts     = self.count_classes(samples)
        counts     = counts[:, classids]
        totalcount = np.sum(counts, axis=0)
        
        # Find the class with minimum and maximum total count:
        c_min = np.argmin(totalcount)
        c_max = np.argmax(totalcount)
        
        # Class balancing is performed as long as the min-max class ratio is 
        # not within 50%.
        #
        # Balancing Algorithm:
        #   * Randomly sample from samples with non-zero min-class ROI counts 
        #     and zero maximum class ROIs.
        #   * Simulaneously, randomly sample a subset of max-class only samples 
        #     to be removed from the dataset. This levels the field from both 
        #     directions.
        class_ratio = totalcount[c_min] / totalcount[c_max]
        while (class_ratio < 0.5) & (len(samples) < 3*5000):
            # Find samples with maximum min-max class ratio:
            N = np.sum((counts[:,c_min] > 0) & (counts[:,c_max] == 0))
            M = int(0.5*N)
            
            # Min-class samples to add:
            min_sample = np.nonzero((counts[:,c_min]>0) & (counts[:,c_max]==0))
            min_sample = min_sample[0] # Unfold tuple
            min_sample = min_sample[np.random.randint(0, len(min_sample)-1, N)]
            
            # Max-class samples to remove:
            max_sample = np.nonzero((counts[:,c_min]==0) & (counts[:,c_max]>0))
            max_sample = max_sample[0] # Unfold tuple
            max_sample = max_sample[np.random.randint(0, len(max_sample)-1, M)]
            max_sample = np.unique(max_sample)
            
            # Construct new sample set:
            min_sample = samples[min_sample]
            samples    = np.append(np.delete(samples, max_sample), min_sample)
            
            # Recompute total count and min-max class ratio:
            counts      = self.count_classes(samples)[:, classids]
            totalcount  = np.sum(counts, axis=0)
            c_min       = np.argmin(totalcount)
            c_max       = np.argmax(totalcount)
            class_ratio = totalcount[c_min] / totalcount[c_max]
            
        # Done, balanced, update samples:
        balancedset   = self.Samples[samples,:]
        self._set_sampling_scheme_(balancedset)

    # --------------------------------------------------------------------------
    def get_max_rois(self):
        """Find the maximum number of ROIs per batch sample in the dataset"""
        
        maxsize = 0
        for index in self.SampleID:
            rois = self.__getrois__(index);
            maxsize = max(maxsize, rois.shape[0])
        
        return maxsize
# ______________________________________________________________________________


def partition(worker_id):
    """Worker Initialization Function for parallel batch loading."""
    
    worker_info = torch.utils.data.get_worker_info()
    dataset     = worker_info.dataset
    
    # Re-create BigTIFF objects that turned stale after serialization:
    for region in dataset.BigTIFFs:
        imgfile   = dataset.BigTIFFs[region].Source
        dirID     = dataset.BigTIFFs[region].DirectoryID
        patchSize = dataset.BigTIFFs[region].PatchSize[dirID]
        
        dataset.BigTIFFs[region] = Bigtiff(imgfile)
        dataset.BigTIFFs[region].setDirectory(dirID)
        dataset.BigTIFFs[region].setPatchSize(patchSize)
    
    # configure the dataset to only process the split workload
    per_worker  = int(math.ceil(dataset.SampleID.shape[0] /
                      float(worker_info.num_workers) ))
    
    sampleStart = worker_id * per_worker
    sampleEnd   = sampleStart + per_worker
    dataset.SampleID = dataset.SampleID[sampleStart:sampleEnd]