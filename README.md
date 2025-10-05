## What about 
<p align=justify>
The scripts contained in this repository were developed for the "Deep Learning & Robot Perception" exam at the "Università degli Studi di Perugia".  
The task consisted of performing depth estimation from an input dataset generated in a virtual environment built with Unreal Engine.
</p>

## The Model
<p align=justify>
The proposed neural network follows an <b>encoder–decoder</b> architecture entirely implemented in PyTorch.  
The <b>encoder</b> is composed of five convolutional blocks, each including convolutional layers, batch normalization, ReLU activation, and max pooling operations, progressively extracting hierarchical spatial features from the RGB input image.  
The <b>decoder</b> reconstructs the depth map through three transposed convolutional layers that progressively upsample the encoded features to recover the spatial resolution and produce the final depth prediction.
</p>

<div align=center>

![](.png/net_depth_estimation.jpg)

</div>

## Performance
The network is evaluated using two metrics:
- **RMSE** (Root Mean Squared Error)  
- **SSIM** (Structural Similarity Index)

<p align=justify>
While both metrics are used to assess performance, SSIM is particularly relevant in this context as it measures pixel-level structural similarity between the predicted and ground-truth depth maps.  
The model is trained using a combined loss function that integrates L1 loss and the square root of the MSE loss, giving greater emphasis to large depth errors.  
A fixed weighting factor is applied to balance the two components, encouraging both global accuracy and pixel-level smoothness.
</p>

