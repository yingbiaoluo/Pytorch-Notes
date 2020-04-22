# Pytorch CuDNN设置

## 1、什么是CUDA

> CUDA® is a parallel computing platform and programming model developed by NVIDIA for general computing on graphical processing units (GPUs). With CUDA, developers are able to dramatically speed up computing applications by harnessing the power of GPUs.    (来源[Nvidia官网](https://developer.nvidia.com/cuda-zone))

CUDA (ComputeUnified Device Architecture)，是显卡厂商NVIDIA推出的运算平台。 CUDA是一种由NVIDIA推出的通用并行计算架构，该架构使GPU能够解决复杂的计算问题。

## 2、什么是cuDNN

>  The NVIDIA CUDA® Deep Neural Network library (cuDNN) is a GPU-accelerated library of primitives for [deep neural networks](https://developer.nvidia.com/deep-learning). cuDNN provides highly tuned implementations for standard routines such as forward and backward convolution, pooling, normalization, and activation layers.
>
> Deep learning researchers and framework developers worldwide rely on cuDNN for high-performance GPU acceleration. It allows them to focus on training neural networks and developing software applications rather than spending time on low-level GPU performance tuning. cuDNN accelerates widely used deep learning frameworks, including [Caffe](http://caffe.berkeleyvision.org/),[Caffe2](https://www.caffe2.ai/), [Chainer](https://chainer.org/), [Keras](https://keras.io/),[MATLAB](https://www.mathworks.com/solutions/deep-learning.html), [MxNet](https://mxnet.incubator.apache.org/), [TensorFlow](https://www.tensorflow.org/), and [PyTorch](http://pytorch.org/). These deep learning frameworks, with cuDNN integrated, are used to develop applications such as [conversational AI](https://developer.nvidia.com/conversational-ai), computer vision and recommenders. Access NVIDIA optimized deep learning framework containers from NGC.   (来源[Nvidia官网介绍](https://developer.nvidia.com/cudnn))

NVIDIA cuDNN是用于深度神经网络的GPU加速库。它强调性能、易用性和低内存开销。NVIDIA cuDNN可以集成到更高级别的机器学习框架中，如谷歌的Tensorflow、加州大学伯克利分校的流行caffe软件。简单的插入式设计可以让开发人员专注于设计和实现神经网络模型，而不是简单调整性能，同时还可以在GPU上实现高性能现代并行计算。

CuDNN使用非确定性算法，并且可以使用`torch.backends.cudnn.enabled = False`来进行禁用。

如果设置为`torch.backends.cudnn.enabled =True`，说明设置为使用非确定性算法。

然后再设置：  `torch.backends.cudnn.benchmark = True`，那么CuDNN使用的非确定性算法就会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。

一般来讲，应该遵循以下准则：

1. 如果网络的输入数据维度或类型上变化不大，设置 torch.backends.cudnn.benchmark = True 可以增加运行效率；
2. 如果网络的输入数据在每次 iteration 都变化的话，会导致 cnDNN 每次都会去寻找一遍最优配置，这样**反而会降低**运行效率



代码加载程序前：

```python
# get device and set cuDNN
if config.USE_GPU and torch.cuda.is_available():
    device = torch.device("cuda:{}".format(config.GPUID))
    print('Using GPU: ', torch.cuda.get_device_name(0))
    # cudnn
    cudnn.enabled = config.CUDNN.ENABLED  # True设置为使用非确定性算法
    cudnn.benchmark = config.CUDNN.BENCHMARK
    # 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
    # 当计算图不会改变的时候（每次输入形状相同，模型不改变）的情况下可以提高性能，反之则降低性能
    print('Using cudnn.benchmark')
    cudnn.deterministic = config.CUDNN.DETERMINISTIC  # 卷积操作使用确定性算法，可复线
else:
    device = torch.device("cpu:0")
    print('Warning! Using CPU.')
```



一些小技巧：

1. 有时候可能是因为每次迭代都会引入点临时变量，会导致训练速度越来越慢，基本呈线性增长。开发人员还不清楚原因，但如果周期性的使用`torch.cuda.empty_cache()`的话就可以解决这个问题。这个命令是清除没用的临时变量的。torch.cuda.emptyCache()释放PyTorch的缓存分配器中的缓存内存块。当与其他进程共享GPU时特别有用。