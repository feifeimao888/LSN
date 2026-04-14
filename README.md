# LSN
Lightweight Segmentation Network for Road Landslides in High-Resolution Satellite Images
<img width="2726" height="1090" alt="image" src="https://github.com/user-attachments/assets/ba0ad3e6-47dd-4a48-8fef-9cb95bb5d965" />

Road landslides are global geological hazards, and timely detection of them is crucial for post-disaster management and the prevention and control of secondary disasters. Although High-Resolution Optical Satellite Imagery (HROSI) has the advantage of large-scale detection coverage, its computational efficiency issues restrict the real-time performance of landslide detection. Meanwhile, existing models can hardly balance the requirements of high accuracy and lightweight deployment in complex terrain. To address these problems, this study proposes a road Lightweight Landslide Segmentation Network (LSN) that highlights lightweight design while taking into account high detection accuracy. LSN adopts an efficient asymmetric encoder-decoder architecture. An Attentive Atrous Pyramid Module (AAPM) is designed in the Encoder, which realizes efficient aggregation of multi-scale features through a dynamic feature routing mechanism and depthwise separable atrous convolution, and effectively avoids the surge of channel dimensions. In the Decoder, an Adaptive High-Frequency Injection module is designed to sharpen the topological boundaries of landslides, and is combined with the Semantic-Guided Calibration Fusion module to achieve precise alignment between edge details and deep semantics. In addition, this paper constructs the first HROSI road landslide dataset containing 1100 images. The experimental results show that, with 512×512-pixel input, LSN has only 2.90M parameters, achieves a mean Intersection over Union of 88.39%, and realizes an ultra-fast inference speed of 33 ms per image on the test hardware. Its comprehensive segmentation performance and inference energy efficiency are superior to those of existing advanced models. The dataset and code are publicly available at: https://github.com/feifeimao888/LSN.

<img width="1470" height="1070" alt="image" src="https://github.com/user-attachments/assets/e98679e7-b810-40f9-bdb5-cc88cc5f74d4" />
<img width="1654" height="840" alt="image" src="https://github.com/user-attachments/assets/c0f64575-c0e5-438d-b5f3-bd372dda8ca1" />


The landslide dataset used in the paper has been published, and the LSN code will be published when the paper is accepted.

Status Date   : Apr 03, 2026. 
Current Status: Under Review
