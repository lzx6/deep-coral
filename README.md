# Deep Coral



# environment need
- pytorch 1.0
- torchvision 0.2.1
- CUDA 9.1
- python 3
# network framwork
![deep coral.png](https://upload-images.jianshu.io/upload_images/16293451-328e07c23bb9234f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
### coral loss：
$L_{CORAL}=\frac{1}{4d^2}||C_S-C_T||_F^2$
### log coral loss：
$L_{log}= \frac{1}{4d^2}||log(C_S)-log(C_T)||_F^2$
$=\frac{1}{4d^2}||Udiag(log(\lambda_1),...,log(\lambda_d))U^T-Vdiag(log(\mu_1),...,log(\mu_d))V^T||_F^2$





# results
![acc_new.png](https://upload-images.jianshu.io/upload_images/16293451-2dcd07bf3cc9ee64.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![loss.png](https://upload-images.jianshu.io/upload_images/16293451-e501f48919602d11.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
# notice
the accuracy can not achieve the result of official DeepCoral due to  the finetune Alexnet model from pytorch !


# 参考

1. [deep coral][https://arxiv.org/abs/1607.01719v1]
2. https://github.com/SSARCandy/DeepCORAL
3. [log coral loss][https://arxiv.org/pdf/1705.08180.pdf]
