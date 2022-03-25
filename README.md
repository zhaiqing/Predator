# Predator
point cloud registration  
本项目只是对Predator的解释，代码中为了方便理解有很多注释，源码与https://github.com/overlappredator/OverlapPredator 一致  
实验记录  
运行环境是在服务器上跑的，需要Linux环境，还有使用MinkowskiEngine的Predator
### 实验环境
- Python 3.8.5, PyTorch 1.7.1, CUDA 11.2, gcc 9.3.0, GeForce RTX 3090/GeForce GTX 1080Ti
![image](https://user-images.githubusercontent.com/33917900/160037849-0dc08a47-8fa1-49bb-b20f-ea5239fb8f73.png)
P',Q' 是提取出的superpoints，首先通过降采样得到superpoints，这些superpoints认为是一个个patch，再使用一个DGCNN去扩大它的感受野，让它看到更多的global context来描述点云（平滑），之后要让两个patch进行比较（使用Transformer），最后又加了一个DGCNN，使之平滑（周围的点在overlap中，大概率它也在overlap中，使用DGCNN来更新了superpoints的feature map）  
测试结果如下：
![image](https://user-images.githubusercontent.com/33917900/160038190-f7d9740b-f350-4b58-8987-efd4b1ae4e12.png)
### 整体流程  
**Encoder模块：** 对原始点云进行采样得到superpoints P',Q'以及相关特征X<sup>P'</sup>,X<sup>Q'</sup>  
**图卷积神经网络：** 在连接两个特征编码之前，我们首先使用图神经网络（GNN）进一步聚合和加强它们各自的上下文关系。首先，利用k-NN方法将P′和Q'的 superpoints连接成欧几里德空间中的图，通过GNN生成特征X<sup>GNN</sup>  
**cross-attention模块：** 关于潜在的重叠区域的知识只能通过混合两个点云的信息来获得。为此，我们采用了基于消息传递公式的cross-attention模块，得到协同上下文特征X<sup>CA</sup>  
**点的重叠分数：** 上述使用上下文信息的更新是针对每个重叠点单独进行的，而不考虑每个点云内的本地上下文。因此，我们使用另一个GNN在交叉注意块之后显式地更新本地上下文，该GNN具有与上述相同的架构和基础图，但有单独的参数θ。这将产生最终的潜在特征编码F<sup>P'</sup>和F<sup>Q'</sup>，这些特征以线性方式投影以获得重叠分数o<sup>P'</sup>和o<sup>Q'</sup>,可以解释为某个superpoint位于重叠区域的概率,此外，可以计算重叠点之间的软对应，并根据对应权重预测重叠点 p<sub>i</sub>'  的交叉重叠分数。其对应关系Q′ 位于重叠区域的概率![](http://latex.codecogs.com/svg.latex?\tilde{o}_{i}^{P'})为![image](https://user-images.githubusercontent.com/33917900/160043458-2f90969a-2598-4b93-949e-222520a27ca5.png)  
**Decoder模块：**  
![image](https://user-images.githubusercontent.com/33917900/160043560-57d58e12-893c-4a04-8ba8-a79f1d18e69d.png)  
从条件特征F<sup>P'</sup>开始，用重叠分数o<sup>P'</sup>、![](http://latex.codecogs.com/svg.latex?\tilde{o}_{i}^{P'})连接它们，并输出每点特征描述符<img src="https://latex.codecogs.com/svg.image?F^{p}&space;\in&space;\mathbb{R}^{N&space;\times&space;32}" title="https://latex.codecogs.com/svg.image?F^{p} \in \mathbb{R}^{N \times 32}" />，重新定义每个点overlap 和 matchability score <img src="https://latex.codecogs.com/svg.image?o^{P},&space;m^{P}&space;\in&space;\mathbb{R}^{N}" title="https://latex.codecogs.com/svg.image?o^{P}, m^{P} \in \mathbb{R}^{N}" /> 。可匹配性可以被视为一种“条件显著性”，它量化了一个点被正确匹配的可能性。  
**RANSAC：** 最后的<img src="https://latex.codecogs.com/svg.image?scores^{P}&space;=o^{P}\times&space;m^{P}&space;&space;" title="https://latex.codecogs.com/svg.image?scores^{P} =o^{P}\times m^{P} " />(Q的计算与P类似) 得到的，最后通过ransac算法将裁剪后的src和tgt，以及两个的特征F<sup>P</sup>和F<sup>Q</sup>，还有scores输入到RANSAC函数中得到最终的变换R和t。

### 代码解析
**1. 图卷积神经网络(DGCNN)**  
![image](https://user-images.githubusercontent.com/33917900/160038818-317a1c36-a1d0-4e5d-b5ac-d702433903fd.png)
![image](https://user-images.githubusercontent.com/33917900/160038840-50a3a1d7-0206-4e11-a519-387851cb4562.png)
**2. self-attention**  
![image](https://user-images.githubusercontent.com/33917900/160038969-f35f1d87-5fa4-499b-98c2-9f696a71ab54.png)
![image](https://user-images.githubusercontent.com/33917900/160038995-6b94eec5-e3ea-4287-97df-51463cfc7a1d.png)
![image](https://user-images.githubusercontent.com/33917900/160039018-cefdd282-6c66-4aaf-a524-4a9e3163e4fe.png)
**3. cross-attention(得到协同上下文信息)**  
![image](https://user-images.githubusercontent.com/33917900/160039259-e40d9ad0-3171-4640-babe-aa14f0bea740.png)
![image](https://user-images.githubusercontent.com/33917900/160039268-be56cf13-6dec-440e-a1ea-2a3b9cc20d02.png)
![image](https://user-images.githubusercontent.com/33917900/160039270-878a7bc5-ba11-4997-a462-693f9b0abd81.png)
![image](https://user-images.githubusercontent.com/33917900/160039287-ab21c61a-d449-4bdc-8cb3-9bfe1bd321d7.png)
![image](https://user-images.githubusercontent.com/33917900/160039299-d657cd88-9275-4477-bb7e-cdf789280c35.png)
![image](https://user-images.githubusercontent.com/33917900/160039323-ebbd55a6-710f-4147-968a-6e031911f527.png)
![image](https://user-images.githubusercontent.com/33917900/160039340-ee07d35e-093d-41f7-8895-a5af8156ec3f.png)  
**4.GCN**  
![image](https://user-images.githubusercontent.com/33917900/160039558-1a89fb24-f534-4125-94d5-2905a6997bbb.png)  
### 损失函数  
可以参考下面这篇文章
https://blog.csdn.net/weixin_45095281/article/details/120920303
