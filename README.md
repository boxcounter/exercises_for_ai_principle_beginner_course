# Course

Vedios: [《小白也能听懂的人工智能原理》- Bilibili](https://www.bilibili.com/cheese/play/ep6173)

* lesson 01：一元一次函数感知器：如何描述直觉
* lesson 02：方差代价函数：知错
* lesson 03：梯度下降和反向传播：能改（上）
* lesson 04：梯度下降和反向传播：能改（下）
* lesson 05：激活函数：给机器注入灵魂
* lesson 06：隐藏层：神经网络为什么working
* lesson 07：高维空间：机器如何面对越来越复杂的问题
* lesson 08：初识Keras：轻松完成神经网络模型搭建
* lesson 09：深度学习：神奇的DeepLearning
* lesson 10：卷积神经网络：打破图像识别的瓶颈
* lesson 11：卷积神经网络：图像识别实战
* lesson 12：循环：序列依赖问题
* lesson 13：LSTM网络：自然语言处理实践

# Anaconda

## TensorFlow
conda create -n bilibili-ai-course-tensorflow
conda activate bilibili-ai-course-tensorflow

conda install -c conda-forge numpy matplotlib=3.5.2 tqdm tensorflow jieba
conda install -c pytorch pytorch 

conda deactivate

## Pytorch
conda create -n bilibili-ai-course-pytorch
conda activate bilibili-ai-course-pytorch

conda install -c conda-forge numpy matplotlib=3.5.2 tqdm jieba onnx
conda install -c pytorch pytorch 

conda deactivate