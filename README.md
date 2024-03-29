# spectral_bias
This repo is a supplementary material for [author's blog post (Japanese)](https://qiita.com/ShotaDeguchi/items/86f7cceedccb4a29970c). Through a few simple examples, we investigate [spectral bias](https://arxiv.org/abs/1806.08734) spotted in [DNN function approximation](https://doi.org/10.1016/0893-6080(89)90020-8). 

## Example
The following are examples of DNN fitting. One can see DNN approximations (blue dashed lines) learns low-frequency signals first, then gradually shift to high-frequency region. 

|Problem 1 (periodic sinusoidal signal)|Problem 2 (finite Fourier expansion of square-wave)|
|:---:|:---:|
|<img src="https://user-images.githubusercontent.com/49257696/167547117-c0409a2f-8293-4c13-bbfd-ce9378d647bd.gif">|<img src="https://user-images.githubusercontent.com/49257696/167547128-52924d1f-70b1-4477-a159-73a656ab926c.gif">|

## Dependencies
|Library / Package|Version|
|:---:|:---:|
|keras|2.8.0|
|matplotlib|3.5.1|
|numpy|1.22.1|
|pandas|1.4.0|
|python|3.8.10|
|scipy|1.7.3|
|tensorflow|2.8.0|

## References
[1] [author's blog post](https://qiita.com/ShotaDeguchi/items/86f7cceedccb4a29970c). 
<br>
[2] Rahaman, N., Baratin, ., Arpit, ., Draxler, F., Lin, M., Hamprecht, F., Bengio, Y., Courville, A.: On the Spectral Bias of Neural Networks, International Conference on Machine Learning, 2019 ([paper](https://arxiv.org/abs/1806.08734)). 
<br>
[3] Hornik, K., Stinchcombe, M., White, H.: Multilayer feedforward networks are universal approximators, *Neural Networks*, Vol. 2, No. 5, pp. Pages 359-366, 1989. ([paper](https://doi.org/10.1016/0893-6080(89)90020-8))
