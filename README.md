# AnyTSR++: Prompt-Oriented Any-Scale Thermal Super-Resolution for Unmanned Aerial Vehicle 

### MengyuanLi, Changhong Fu*, Ziyu Lu, Zijie Zhang, Haobo Zuo, Liangliang Yao
\* Corresponding author.

## Abstract
Thermal imaging significantly augments the operational capabilities of intelligent unmanned aerial vehicles (UAVs) in complex environments. However, due to the limited resolution of onboard thermal sensors, thermal images captured by UAV suffer from insufficient detail and blurred object boundaries, thereby limiting their practicality. Although super-resolution (SR) provides a promising solution to this issue, existing any-scale SR methods adopt identical feature representations across all scales, lacking the ability to adaptively adjust features according to varying scale requirements, leading to suboptimal SR results. This issue becomes more pronounced in asymmetric scale SR, where the resolution differs significantly along different directions. To address these limitations, a novel prompt-oriented any-scale thermal SR method (AnyTSR++) is proposed for UAV. Specifically, a new image encoder is introduced to explicitly assign any-scale prompt, enabling more precise and adaptive feature representation. Furthermore, an innovative any-scale upsampler is designed by refining the coordinate offset and the local feature ensemble, enhancing spatial awareness and reducing artifacts. Additionally, a novel dataset (UAV-TSR++) comprising 24,000 images covering both land and water surface scenes is constructed to facilitate the community to conduct thermal SR research. Experimental results demonstrate that AnyTSR++ consistently outperforms state-of-the-art methods across both symmetric and asymmetric scaling factors, producing higher-resolution images with greater accuracy and more fine-grained details.


## About Code
### 1. Environment setup
This code has been tested on Ubuntu 22.04, Python 3.10.15, Pytorch 1.13.0, CUDA 11.7.
Please install related libraries before running this code: 
```bash
pip install -r requirements.txt
```

### 2. Test

```bash 
python demo.py                                
```
The testing result will be saved in the `output` directory.

### 3. Contact
If you have any questions, please contact me.

Mengyuan Li

Email: [mengyuanli@tongji.edu.cn](mengyuanli@tongji.edu.cn)


## References 

```

```
