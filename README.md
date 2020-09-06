# PyTorch-to-caffe-Converter-GUI :arrows_counterclockwise:


## Description:
- A conversion tool that can be used in linux environment to convert pytorch model to caffe model
- functions:
    - Can convert PyTorch model to caffe model correctly and without loss of accuracy
    - Provide basic caffe verification function, able to verify the converted model
    - Able to give pop-up error reminders for currently incompatible layers
    - You can choose the path of the pytorch model and the save path of the converted caffe model
    
- Supported Version:(Note that other versions have not been tested yet. You can try it out for yourself!)
    - PyTorch -v1.4.0
    - caffe - v1.0.0  https://github.com/BVLC/caffe
- Supported Layer:
    - conv2d
    - conv2d_transpose
    - linear
    - max_pool2d
    - avgpool_2d
    - adaptive_avgpool2d
    - relu
    - leaky_relu
    - batch_norm
    - dropout
    - softmax
    
- Supported Operations: view(flatten), cat and other common torch operations.
 

## How to use it:
1. Open file convert_gui.py
2. Import the .py file where the model definition is located (note that the name of the imported file is the name of the file when the model was saved)
3. Import the .pth file of the model (note that the entire model is saved)
4. Choose the save location after conversion(Where .protxt and .caffemodel will be saved)
5. Set the input information
6. Click the Convert button to start conversion
7. Browse the detailed conversion process on the right
8. convert error verify whether the conversion is successful

## Example:
### Convert darknet53:

### Other tested model: 
- alexnet
- inception
- resnet
- darknet53
- mobilenetV2
- vggnet

## References:
- https://github.com/xxradon/PytorchToCaffe
- https://github.com/hahnyuan/nn_tools
