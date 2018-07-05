## 基于tensorflow的手写体分类模型  
   
#### 本项目以mnist手写体作为数据集，基于·tensorflow·来构建一个多分类模型。
   
#### 数据集是28*28像素的手写体图片，对应0~9十个数字。  
       
#### 构建神经网络  
    
单批次训练数据为batch_size=100     
不加隐藏层，直接输入层到输出层，测试集准确率为87%   
添加复杂隐藏层两层（2000个神经元、1000个神经元）进行训练，不加dropout，训练三十一次（写本文档时，才跑到第6次，测试集准确率96%，训练集准确率99%，有过拟合现象）
  
将dropout分别设置成0.6和0.7，前者在第十二次训练后， 训练准确率92.79%，测试准确率92.81%，基本没有过拟合  
后者第十二次训练后，训练准确率94.69%，测试准确率94.19&，也基本没有过拟合。但是准确度提高了2%（dropout只是0.1的差别）。

尝试不同优化器eg：AdamOptimizier，学习率从GradientDescent的0.1改为0.01，训练速度加快非常多（尝试过学习率不变，准确度非常低）。  
在不同的优化器里，SGD,Momentunm、NAG、Adagrad、Adadelta、Rmsprop中，SGD速度最慢，而且无法逃离鞍点。其余的（Adadelta速度最快）速度都很快，皆可逃离鞍点。  

经验总结，训练速度越快的优化器，它的学习率应该设置0.01以下，防止准确率过低。

##### 为何我们依旧使用SGD最多？因为我们并不是只追求速度，准确率才是王道。