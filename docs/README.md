# VGGT-SALAD SLAM

This code implements an incremental 3D reconstruction of urban environments using the models of my project [Visual Place Recognition](https://github.com/L4rralde/Visual_Place_Recognition).


## TODO
- [ ] Check if parallelizing vggt's preprocessing function adds a gain in time. This should be done with a predefined shared tensor and torch.multiprocessing. Torch.cat and torch.stack creates copies. When predfining the tensor you just need to allocate the data in its right index.
- [ ] Check if making the model's ready for share memory allows to split the main processing function into tow fold: One for views encoding and another for sequence predictions. To do this, my models must be nn.Modules.
- [ ] Saving to disk is a bottleneck. Maybe you can parallelize this operation. Or do not store patch tokens and preprocessed images. Just compute them when needed. Only store descriptors. In such a case, there's no need to move patch tokens and preprocessed images to cpu.
- [ ] Add code to display pointclouds and camera poses in realtime. Just display the last prediction.