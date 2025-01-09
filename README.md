# DL-Foundation-Models-for-Weather-Prediction
<div align="center">
<img src="https://github.com/JimengShi/AI-Models-Weather-and-Climate/blob/main/figs/weather_forecast.jpg" alt="taxonomy" width="600"/> 
</div>

## Contents
- [Predictive Models](#predictive-learning)
  - [General-Purpose Large Models](#general-predictive-learning)
    - [Transformer](#general-predictive-learning-transformer)
    - [GNN](#general-predictive-learning-gnn)
    - [PhysicsAI](#general-predictive-learning-physicsai)
  - [Domain-Specific Models](#domain-predictive-learning)
    - [Transformer](#domain-predictive-learning-transformer)
    - [GNN](#domain-predictive-learning-gnn)
    - [RNN&CNN](#domain-predictive-learning-rnn&cnn)
    - [Mamba](#domain-predictive-learning-mamba)
    - [PhysicsAI](#domain-predictive-learning-physicsai)
- [Generative Models](#generative-learning)
  - [General-Purpose Large Models](#general-generative-learning)
    - [Diffusion Models](#general-generative-learning-diffusion)
  - [Domain-Specific Models](#domain-generative-learning)
    - [Diffusion Models](#domain-generative-learning-diffusion)
    - [GANs](#domain-generative-learning-gan)
- [Foundation Models](#pretraining-finetuning)
- [Applications](#application)
  - [Precipitation](#percipitation)
  - [Air Quality](#air-quality)
  - [Sea Surface Temperature](#sst)
  - [Flood](flood)
  - [Drought](#drought)
  - [Tropical Storms/Cyclones and Hurricanes](#tropical-storms)
  - [Wildfire](#wildfire)
 


## Taxonomy and Review
**Taxonomy:**
<div align="center">
<img src="https://github.com/JimengShi/AI-Models-Weather-and-Climate/blob/main/figs/frameworks.jpg" alt="taxonomy" width="800"/> 
</div>

**Comprehensive review:**
<div align="center">
<img src="https://github.com/JimengShi/AI-Models-Weather-and-Climate/blob/main/figs/review.jpg" alt="review" width="800"/> 
</div>

<!--------------------------------------  Predictive Models  --------------------------------------> 
<h2 id="predictive-learning">👉👉 Predictive Models</h2>

<h3 id="general-predictive-learning">General-Purpose Large Models</h3>
<h4 id="general-predictive-learning-transformer">Transformer</h4>

- **(FourCastNet)** _FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators_ ```arXiv 2022```     
[[Paper](https://arxiv.org/abs/2202.11214)] [[Code](https://github.com/NVlabs/FourCastNet)]

- **(FuXi)** _FuXi: a cascade machine learning forecasting system for 15-day global weather forecast_ ```Nature 2023```     
[[Paper](https://www.nature.com/articles/s41612-023-00512-1)] [[Code](https://drive.google.com/drive/folders/1NhrcpkWS6MHzEs3i_lsIaZsADjBrICYV)]

- **(FengWu)** _FengWu: Pushing the Skillful Global Medium-range Weather Forecast beyond 10 Days Lead_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2304.02948)] [[Code](https://github.com/OpenEarthLab/FengWu)]

- **(FengWu-4DVar)** _FengWu-4DVar: Coupling the Data-driven Weather Forecasting Model with 4D Variational Assimilation_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2312.12455)] [[Code](https://github.com/OpenEarthLab/FengWu-4DVar)]

- **(SwinVRNN)** _SwinVRNN: A Data-Driven Ensemble Forecasting Model via Learned Distribution Perturbation_ ```JAMES 2023```     
[[Paper](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022MS003211)] [[Code](https://github.com/tpys/wwprediction)]

- **(SwinRDM)** _SwinRDM: Integrate SwinRNN with Diffusion Model towards High-Resolution and High-Quality Weather Forecasting_ ```AAAI 2023```     
[[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/25105)] 

- **(Pangu-Weather)** _Accurate medium-range global weather forecasting with 3D neural networks_ ```Nature 2023```    
[[Paper](https://www.nature.com/articles/s41586-023-06185-3)] [[Code](https://github.com/198808xc/Pangu-Weather)]

- **(Stormer)** _Scaling transformer neural networks for skillful and reliable medium-range weather forecasting_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2312.03876)] [[Code](https://github.com/tung-nd/stormer)]

- **(HEAL-ViT)** _HEAL-ViT: Vision Transformers on a spherical mesh for medium-range weather forecasting_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2403.17016)] 

- **(TianXing)** _linear complexity transformer model with explicit attention decay for global weather forecasting_ ```Advances in Atmospheric Sciences```     
[[Paper](https://link.springer.com/article/10.1007/s00376-024-3313-9)] [[Code](https://link.springer.com/article/10.1007/s00376-024-3313-9)]


<h4 id="general-predictive-learning-gnn">GNN</h4>

- **(GraphCast)** _Graphcast: Learning skillful medium-range global weather forecasting_ ```Science 2023```     
[[Paper](https://arxiv.org/abs/2212.12794)] [[Code](https://github.com/openclimatefix/graph_weather)]

- **(GnnWeather)** _Forecasting Global Weather with Graph Neural Networks_ ```arXiv 2022```     
[[Paper](https://arxiv.org/abs/2202.07575)] [[Code](https://rkeisler.github.io/graph_weather/)]

- **(AIFS)** _AIFS -- ECMWF's data-driven forecasting system_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2406.01465)]

- **(GraphDOP)** _GraphDOP: Towards skilful data-driven medium-range weather forecasts learnt and initialised directly from observations_ ```arXiv 2024```     
[[Paper](https://www.arxiv.org/abs/2412.15687)] 



<h4 id="general-predictive-learning-physicsai">PhysicsAI</h4>

- **(ClimODE)** _ClimODE: Climate and Weather Forecasting with Physics-informed Neural ODEs_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2404.10024)] [[Code](https://github.com/Aalto-QuML/ClimODE)]

- **(WeatherODE)** _Mitigating Time Discretization Challenges with WeatherODE: A Sandwich Physics-Driven Neural ODE for Weather Forecasting_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2410.06560)] [[Code](https://github.com/DAMO-DI-ML/WeatherODE)]

- **(NeuralGCM)** _Neural general circulation models for weather and climate_ ```Nature 2024```     
[[Paper](https://www.nature.com/articles/s41586-024-07744-y)] [[Code](https://github.com/neuralgcm/neuralgcm)]

- **(Conformer)** _STC-ViT: Spatio Temporal Continuous Vision Transformer for Weather Forecasting_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2402.17966)]



<h3 id="domain-predictive-learning">Domain-Specific Models</h3>
<h4 id="domain-predictive-learning-transformer">Transformer</h4>

- **(SwinUnet)** _Spatiotemporal vision transformer for short time weather forecasting_ ```IEEE BigData 2022```     
[[Paper](https://ieeexplore.ieee.org/document/9671442)] [[Code](https://github.com/bojesomo/Weather4Cast2021-SwinUNet3D)] 

- **(Earthformer)** _Earthformer: Exploring Space-Time Transformers for Earth System Forecasting_ ```NeurIPS 2022```     
[[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a2affd71d15e8fedffe18d0219f4837a-Abstract-Conference.html)] [[Code](https://proceedings.neurips.cc/paper_files/paper/2022/file/a2affd71d15e8fedffe18d0219f4837a-Supplemental-Conference.zip)] 

- **(Rainformer)** _Rainformer: Features Extraction Balanced Network for Radar-Based Precipitation Nowcasting_ ```IEEE Geoscience and Remote Sensing Letters 2022```     
[[Paper](https://ieeexplore.ieee.org/abstract/document/9743916)] [[Code](https://github.com/Zjut-MultimediaPlus/Rainformer)]

- **(PFformer)** _PFformer: A Time-Series Forecasting Model for Short-Term Precipitation Forecasting_ ```IEEE Access 2024```     
[[Paper](https://ieeexplore.ieee.org/abstract/document/10678751)] 

- **(OMG-HD)** _OMG-HD: A high-resolution ai weather model for end-to-end forecasts from observations_ ```arXiv 2024```     
[[Paper](https://arxiv.org/pdf/2412.18239)]

- **(U-STN)** _Towards physics-inspired data-driven weather forecasting: integrating data assimilation with a deep spatial-transformer-based u-net in a case study with era5_ ```Geoscientific Model Development 2024```     
[[Paper](https://arxiv.org/pdf/2412.18239)]  [[Code](https://zenodo.org/records/6112374)]


<h4 id="domain-predictive-learning-gnn">GNN</h4>

- **(HiSTGNN)** _HiSTGNN: Hierarchical spatio-temporal graph neural network for weather forecasting_ ```Information Sciences 2023```     
[[Paper](https://www.sciencedirect.com/science/article/pii/S0020025523011659)] [[Code](https://github.com/mb-Ma/HiSTGNN)]

- **(w-GNN)** _Coupling Physical Factors for Precipitation Forecast in China With Graph Neural Network_ ```AGU 2024```     
[[Paper](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023GL106676)] 

- **(WeatherGNN)** _WeatherGNN: Exploiting Meteo- and Spatial-Dependencies for Local Numerical Weather Prediction Bias-Correction_ ```IJCAI 2024```     
[[Paper](https://www.ijcai.org/proceedings/2024/0269.pdf)] 

- **(MPNNs)** _Multi-modal graph neural networks for localized off-grid weather forecasting_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2410.12938)] [[Code](https://github.com/Earth-Intelligence-Lab/LocalizedWeatherGNN/)]



<h4 id="domain-predictive-learning-rnn&cnn">RNN&CNN</h4>

- **(MetNet)** _MetNet: A Neural Weather Model for Precipitation Forecasting_ ```arXiv 2020```     
[[Paper](https://arxiv.org/abs/2003.12140)] [[Code](https://drive.google.com/drive/folders/1X4ggyAdvkGcLGYaKUvygb0aCIZCBoVuW)]

- **(MetNet-3)** _Deep Learning for Day Forecasts from Sparse Observations_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2306.06079)] 

- **(PredRNN)** _PredRNN: A Recurrent Neural Network for Spatiotemporal Predictive Learning_ ``` IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE 2023```     
[[Paper](https://ieeexplore.ieee.org/abstract/document/9749915)] [[Code](https://github.com/thuml/predrnn-pytorch)]

- **(MM-RNN)** _MM-RNN: A Multimodal RNN for Precipitation Nowcasting_ ```IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING 2023```     
[[Paper](https://ieeexplore.ieee.org/abstract/document/10092888)] 

- **(ConvLSTM)** _Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting_ ```NeurIPS 2015```     
[[Paper](https://proceedings.neurips.cc/paper/2015/hash/07563a3fe3bbe7e3ba84431ad9d055af-Abstract.html)] [[Code](https://proceedings.neurips.cc/paper_files/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Supplemental.zip)]



<h4 id="domain-predictive-learning-mamba">Mamba</h4>

- **(MetMamba)** _MetMamba: Regional Weather Forecasting with Spatial-Temporal Mamba Model_ ```arXiv 2024```
[[Paper](https://arxiv.org/abs/2408.06400)] 

- **(MambaDS)** _MambaDS: Near-Surface Meteorological Field Downscaling With Topography Constrained Selective State-Space Modeling_ ```IEEE Transactions on Geoscience and Remote Sensing 2024```     
[[Paper](https://ieeexplore.ieee.org/abstract/document/10752514)] 



<h4 id="domain-predictive-learning-physicsai">PhysicsAI</h4>

- **(NowcastNet)** _Skilful nowcasting of extreme precipitation with NowcastNet_ ```Nature 2023```     
[[Paper](https://www.nature.com/articles/s41586-023-06184-4)] [[Code](https://codeocean.com/capsule/3935105/tree/v1)]

- **(PhysDL)** _Deep learning for physical processes: incorporating prior scientific knowledge_ ```IOPScience 2019```     
[[Paper](https://iopscience.iop.org/article/10.1088/1742-5468/ab3195/meta#jstatab3195s4)] 

- **(PhyDNet)** _Disentangling Physical Dynamics From Unknown Factors for Unsupervised Video Prediction_ ```CVPR 2020```     
[[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Le_Guen_Disentangling_Physical_Dynamics_From_Unknown_Factors_for_Unsupervised_Video_Prediction_CVPR_2020_paper.html)] [[Code](https://github.com/vincent-leguen/PhyDNet)]

- **(DeepPhysiNet)** _DeepPhysiNet: Bridging Deep Learning and Atmospheric Physics for Accurate and Continuous Weather Modeling_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2401.04125)] [[Code](https://github.com/flyakon/DeepPhysiNet)]




<!-------------------------------------- Generative Models  --------------------------------------> 
<h2 id="generative-learning">👉👉 Generative Models</h2>
<h3 id="general-generative-learning">General-Purpose Large Models</h3>
<h4 id="general-generative-learning-diffusion">Diffusion Models</h4>

- **(GenCast)** _GenCast: Diffusion-based ensemble forecasting for medium-range weather_ ```Nature 2024```     
[[Paper](https://www.nature.com/articles/s41586-024-08252-9)] [[Code](https://github.com/google-deepmind/graphcast)]

- **(CoDiCast)** _CoDiCast: Conditional Diffusion Model for Weather Prediction with Uncertainty Quantification_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2409.05975)] [[Code](https://github.com/JimengShi/CoDiCast)]

- **(SEEDs)** _SEEDS: Emulation of Weather Forecast Ensembles with Diffusion Models_ ```Science Advances 2024```     
[[Paper](https://www.science.org/doi/10.1126/sciadv.adk4489)] [[Code](https://github.com/google-research/google-research/tree/master/seeds)] 

- **(ContinuousEnsCast)** _Continuous Ensemble Weather Forecasting with Diffusion models_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2410.05431)] [[Code](https://github.com/martinandrae/Continuous-Ensemble-Forecasting)]



<h3 id="domain-generative-learning">Domain-Specific Models</h3>
<h4 id="domain-generative-learning-diffusion">Diffusion Models</h4>

- **(LDMRain)** _Latent diffusion models for generative precipitation nowcast-ing with accurate uncertainty quantification_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2304.12891)] [[Code](https://github.com/MeteoSwiss/ldcast)] 

- **(PreDiff)** _PreDiff: Precipitation Nowcasting with Latent Diffusion Models_ ```NeurIPS 2023```     
[[Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f82ba6a6b981fbbecf5f2ee5de7db39c-Abstract-Conference.html)] [[Code](https://proceedings.neurips.cc/paper_files/paper/2023/file/f82ba6a6b981fbbecf5f2ee5de7db39c-Supplemental-Conference.zip)]

- **(CasCast)** _CasCast: Skillful High-resolution Precipitation Nowcasting via Cascaded Modelling_ ```ICML 2024```     
[[Paper](https://arxiv.org/abs/2402.04290)] [[Code](https://github.com/OpenEarthLab/CasCast)]

- **(SRNDiff)** _SRNDiff: Short-term Rainfall Nowcasting with Condition Diffusion Model_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2402.13737)] [[Code](https://github.com/ybu-lxd/SRNDiff)]

- **(DiffCast)** _DiffCast: A Unified Framework via Residual Diffusion for Precipitation Nowcasting_ ```CVPR 2024```     
[[Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_DiffCast_A_Unified_Framework_via_Residual_Diffusion_for_Precipitation_Nowcasting_CVPR_2024_paper.html)] [[Code](https://github.com/DeminYu98/DiffCast)]

- **(GEDRain)** _Precipitation nowcasting with generative diffusion models_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2308.06733)] [[Code](https://github.com/fmerizzi/Precipitation-nowcasting-with-generative-diffusion-models)]



<h4 id="domain-generative-learning-gan">GANs</h4>

- **(GANRain)** _Skillful precipitation nowcasting using deep generative models of radar_ ```Nature 2021```     
[[Paper](https://www.nature.com/articles/s41586-021-03854-z)] [[Code](https://github.com/openclimatefix/skillful_nowcasting)] 

- **(MultiScaleGAN)** _Experimental Study on Generative Adversarial Network for Precipitation Nowcasting_ ```IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING 2022```     
[[Paper](https://ieeexplore.ieee.org/abstract/document/9780397)] [[Code](https://github.com/luochuyao/MultiScaleGAN)]

- **(STGM)** _Physical-Dynamic-Driven AI-Synthetic Precipitation Nowcasting Using Task-Segmented Generative Model_ ```AGU 2023```     
[[Paper](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023GL106084)] [[Code](https://zenodo.org/records/8380856)]

- **(PCT-CycleGAN)** _PCT-CycleGAN: Paired Complementary Temporal Cycle-Consistent Adversarial Networks for Radar-Based Precipitation Nowcasting_ ```ACM 2023```     
[[Paper](https://dl.acm.org/doi/abs/10.1145/3583780.3615006)] 




<!-------------------------------------- Foundation Models  --------------------------------------> 
<h2 id="pretraining-finetuning">👉👉 Foundation Models</h2>

- **(ClimaX)** _ClimaX: A foundation model for weather and climate_ ```ICML 2023```     
[[Paper](https://arxiv.org/abs/2301.10343)] [[Code](https://github.com/microsoft/ClimaX)]

- **(W-MAE)** _W-MAE: Pre-trained weather model with masked autoencoder for multi-variable weather forecasting_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2304.08754)] [[Code](https://github.com/Gufrannn/W-MAE)]

- **(Aurora)** _Aurora: A foundation model of the atmosphere._ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2405.13063)] [[Code](https://github.com/microsoft/aurora)]

- **(Prithvi WxC)** _Prithvi WxC: Foundation Model for Weather and Climate_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2409.13598)] [[Code](https://github.com/NASA-IMPACT/Prithvi-WxC)]

- **(AtmosArena)** _AtmosArena: Benchmarking Foundation Models for Atmospheric Sciences_ ```NeurIPS 2024 Workshop FM4Science```     
[[Paper](https://openreview.net/forum?id=cUucUH9y0s)] [[Code](https://github.com/tung-nd/atmos-arena?tab=readme-ov-file)]



<h2 id="application">Applications</h2>


<h3 id="precipitation"> Precipitation</h3>

- Deep Learning and the Weather Forecasting Problem: Precipitation Nowcasting. [[Paper](https://onlinelibrary.wiley.com/doi/abs/10.1002/9781119646181.ch15)]
- PRISMA: A systematic quantitative review on the performance of some of the recent short-term rainfall forecasting techniques. [[Paper](https://iwaponline.com/jwcc/article/13/8/3004/89806/A-systematic-quantitative-review-on-the)]
- Deep Learning Techniques in Extreme Weather Events: A Review [[Paper](https://arxiv.org/abs/2308.10995)]
- Analysis, characterization, prediction, and attribution of extreme atmospheric events with machine learning and deep learning techniques: a review 
 [[Paper](https://link.springer.com/article/10.1007/s00704-023-04571-5)]
- Deep learning for precipitation nowcasting: A survey from the perspective of time series forecasting. [[Paper](https://arxiv.org/abs/2406.04867)]
- Precipitation Nowcasting with Satellite Imagery. [[Paper](https://dl.acm.org/doi/abs/10.1145/3292500.3330762?casa_token=vc2v--4ntikAAAAA:qGIsv_DIQ1kZUgVo9EWssxBQKfMy1Zoo71em1L99Jo1ZtIp8IWylp1k4Cq5FccKergeXF_H3VOeCJw)]
- RainNet v1.0: a convolutional neural network for radar-based precipitation nowcasting. [[Paper](https://gmd.copernicus.org/articles/13/2631/2020/gmd-13-2631-2020.html)]
- Convective Precipitation Nowcasting Using U-Net Model. [[Paper](https://ieeexplore.ieee.org/abstract/document/9508500?casa_token=RKG0XnUfTgsAAAAA:0IaAEdjBy9ipavhcKrelhw-9Ey2bdACh297f60nlMqhrt0fMfWTvDgna4u9FtRWmrjtHhdWBzH8)]
- NowCasting-Nets: Representation Learning to Mitigate Latency Gap of Satellite Precipitation Products Using Convolutional and Recurrent Neural Networks. [[Paper](https://ieeexplore.ieee.org/abstract/document/9732949?casa_token=nPuhy3vPFUMAAAAA:9MV3G2pjrCABmbj8SClabYlrwmWAmlFitucMEqkwSM-Ke0xugAEBNqgTPd-86jVSKwCtQ68wOXk)]
- Domain Generalization Strategy to Train Classifiers Robust to Spatial-Temporal Shift. [[Paper](https://arxiv.org/abs/2212.02968)]
- Region-Conditioned Orthogonal 3D U-Net for Weather4Cast Competition. [[Paper](https://arxiv.org/abs/2212.02059)]
- Skilful nowcasting of extreme precipitation with NowcastNet. [[Paper](https://www.nature.com/articles/s41586-023-06184-4)]
- Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting. [[Paper](https://proceedings.neurips.cc/paper/2015/hash/07563a3fe3bbe7e3ba84431ad9d055af-Abstract.html)]
- PredRNN: Recurrent Neural Networks for Predictive Learning using Spatiotemporal LSTMs. [[Paper](https://proceedings.neurips.cc/paper/2017/hash/e5f6ad6ce374177eef023bf5d0c018b6-Abstract.html?ref=https://githubhelp.com)]
- Nowformer: A Locally Enhanced Temporal Learner for Precipitation Nowcasting. [[Paper](https://s3.us-east-1.amazonaws.com/climate-change-ai/papers/neurips2022/80/paper.pdf)]
- Earthformer: Exploring Space-Time Transformers for Earth System Forecasting. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a2affd71d15e8fedffe18d0219f4837a-Abstract-Conference.html)]
- Rainformer: Features Extraction Balanced Network for Radar-Based Precipitation Nowcasting. [[Paper](https://ieeexplore.ieee.org/abstract/document/9743916?casa_token=wMq7FLlDsqsAAAAA:oY6EF-_gaSRkre6_dtnEwkDiB-Le3Kr6Y32MMXQd2AFVVbK_fY8EWDkNvXfDwwtNeyXo4UW8Ces)]
- The MS-RadarFormer: A Transformer-Based Multi-Scale Deep Learning Model for Radar Echo Extrapolation. [[Paper](https://www.mdpi.com/2072-4292/16/2/274)]
- A Foundation Model for the Earth System. [[Paper](https://arxiv.org/abs/2405.13063)]
- WeatherGFM: Learning A Weather Generalist Foundation Model via In-context Learning. [[Paper](https://arxiv.org/abs/2411.05420)]
- AENN: A GENERATIVE ADVERSARIAL NEURAL NETWORK FOR WEATHER RADAR ECHO EXTRAPOLATION. [[Paper](https://isprs-archives.copernicus.org/articles/XLII-3-W9/89/2019/)]
- MPL-GAN: Toward Realistic Meteorological Predictive Learning Using Conditional GAN. [[Paper](https://ieeexplore.ieee.org/abstract/document/9094665)]
- Skillful Radar-Based Heavy Rainfall Nowcasting Using Task-Segmented Generative Adversarial Network. [[Paper](https://ieeexplore.ieee.org/abstract/document/10182305?casa_token=03L4g8JiYTEAAAAA:gKbB8BVQBKQ2gYm5AnvZcMFXirf2W39tsgobHtPJ9Yu-LvIkDtbxN8guCUVV9zb0__FVyWJlrpM)]
- A Self-Attention Causal LSTM Model for Precipitation Nowcasting. [[Paper](https://ieeexplore.ieee.org/abstract/document/10222152?casa_token=8ytUvFB7P5AAAAAA:PgPPkFfKojwr322LoWATBID8-xvQNIFQR6g_JFftTrV1ZGzCdZ2bTABYvQUWpXfjJJaW6yTcFJI)]
- PCT-CycleGAN: Paired Complementary Temporal Cycle-Consistent Adversarial Networks for Radar-Based Precipitation Nowcasting. [[Paper](https://dl.acm.org/doi/abs/10.1145/3583780.3615006)]
- Precipitation Nowcasting Using Physics Informed Discriminator Generative Models. [[Paper](https://ieeexplore.ieee.org/abstract/document/10715141?casa_token=7gFGZrMbB9AAAAAA:cAr18dEYqnYyiN2cCAo9Ko4IbJCsEl7S88GmwDJuul_8dfjYVU8Y472irOq94SI4JkPfYDQTMdY)]
- GPTCast: a weather language model for precipitation nowcasting. [[Paper](https://arxiv.org/abs/2407.02089)]
- PreDiff: Precipitation Nowcasting with Latent Diffusion Models [[Paper](https://arxiv.org/pdf/2307.10422)]
- Latent diffusion models for generative precipitation nowcasting with accurate uncertainty quantification. [[Paper](https://arxiv.org/abs/2304.12891)]
- DiffCast: A Unified Framework via Residual Diffusion for Precipitation Nowcasting. [[Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_DiffCast_A_Unified_Framework_via_Residual_Diffusion_for_Precipitation_Nowcasting_CVPR_2024_paper.html)]
- CasCast: Skillful High-resolution Precipitation Nowcasting via Cascaded Modelling. [[Paper](https://arxiv.org/abs/2402.04290)]


<h3 id="air-quality"> Air Quality </h3>

- U-Air: when urban air quality inference meets big data. [[Paper](https://dl.acm.org/doi/abs/10.1145/2487575.2488188?casa_token=9V1RKEJDFZIAAAAA:3fuuGdFlRr4ey1P476Ny6BwEu-7WSjM0Welx-dCO4fG5_TVIXuiuSizbjU4Rtlitxrwz6vrpSZr11w)]
- Forecasting of Air Quality Using an Optimized Recurrent Neural Network. [[Paper](https://www.mdpi.com/2227-9717/10/10/2117)]
- Deep Distributed Fusion Network for Air Quality Prediction. [[Paper](https://dl.acm.org/doi/abs/10.1145/3219819.3219822?casa_token=LK4s9pUwvDkAAAAA:7yFVURBsJxlg-62o_LkKqA5Qmh-mO6rrTPzi6hKMM1GOyNSQSk0yhceg8pDzv6raaJtTqlKbsdo3WQ)]
- Time Series Forecasting (TSF) Using Various Deep Learning Models. [[Paper](https://arxiv.org/abs/2204.11115)]
- Group-Aware Graph Neural Network for Nationwide City Air Quality Forecasting. [[Paper](https://dl.acm.org/doi/full/10.1145/3631713?casa_token=zv8LjXMKyKcAAAAA%3AZQYmgVEHyBgQWfiND7Os4eJ_4VCkfKYSFZ9TJK3ECIE7R3tFdURJW_w_jZx35SoRbMCRXrQdg2Dc2Q)]
- PM2.5-GNN: A Domain Knowledge Enhanced Graph Neural Network For PM2.5 Forecasting. [[Paper](https://dl.acm.org/doi/abs/10.1145/3397536.3422208?casa_token=2RRTQZmacw0AAAAA:jpHTPD0Hn2rl1C0L3u1zEwgnZXnfn5WRxFpIOblWiNGD1WQN6YiWESMWa5zN7Sx5IuGcgOhJ-vSDxQ)]
- AirFormer: Predicting Nationwide Air Quality in China with Transformers. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/26676)]
- MGSFformer: A Multi-Granularity Spatiotemporal Fusion Transformer for air quality prediction. [[Paper](https://www.sciencedirect.com/science/article/pii/S1566253524003853?casa_token=G8bt4mXSLWcAAAAA:8yTdn7Ij0KV6UwHikHTTwZGupQbST8VG6Ex3eeEtPYgFwGMwj0U2KoCM-iodTZd1q_VTA2uzjqE)]
- Air Quality Prediction Using the Fractional Gradient-Based Recurrent Neural Network. [[Paper](https://onlinelibrary.wiley.com/doi/full/10.1155/2022/9755422)]
- Air quality prediction using CNN+LSTM-based hybrid deep learning architecture. [[Paper](https://link.springer.com/article/10.1007/s11356-021-16227-w)]


<h3 id="sst"> Sea Surface Temperature </h3>

- ENSO analysis and prediction using deep learning: A review. [[Paper](https://www.sciencedirect.com/science/article/pii/S0925231222014722)]
- Analyzing El Niño–Southern Oscillation Predictability Using Long-Short-Term-Memory Models. [[Paper](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018EA000423)]
- Spatiotemporal Model Based on Deep Learning for ENSO Forecasts. [[Paper](https://www.mdpi.com/2073-4433/12/7/810)]
- Deep learning for multi-year ENSO forecasts. [[Paper](https://www.nature.com/articles/s41586-019-1559-7)]
- Forecasting the Indian Ocean Dipole With Deep Learning Techniques. [[Paper](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021GL094407)]
- Deep Residual Convolutional Neural Network Combining Dropout and Transfer Learning for ENSO Forecasting. [[Paper](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021GL093531)]
- DLENSO: A Deep Learning ENSO Forecasting Model. [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-29911-8_2)]
- Graph Neural Networks for Improved El Niño Forecasting. [[Paper](https://arxiv.org/abs/2012.01598)]
- Transformer for EI Niño-Southern Oscillation Prediction. [[Paper](https://ieeexplore.ieee.org/abstract/document/9504603?casa_token=99pEzfNOyYAAAAAA:Gb_ItwuxuAvu1nQXivM3yyL9ylErw3oIN9DJmT59wNs9RUV-p3KzjvPUx3tnEE3dv1lP1RrJ0TA)]
- A self-attention–based neural network for three-dimensional multivariate modeling and its skillful ENSO predictions. [[Paper](https://www.science.org/doi/full/10.1126/sciadv.adf2827)]
- Spatial-temporal transformer network for multi-year ENSO prediction. [[Paper](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2023.1143499/full)]
- Adaptive Graph Spatial-Temporal Attention Networks for long lead ENSO prediction. [[Paper](https://www.sciencedirect.com/science/article/pii/S0957417424013599?casa_token=HSqO4BVVzDIAAAAA:GfL8nA06lSVxc-_XC3uzdueCfEA2a4kI9tTqQhKtcs-qY8mI0ZLih4Kb7ZSMeu902CdbgIGPNIw)]
- ENSO dataset & comparison of deep learning models for ENSO forecasting. [[Paper](https://link.springer.com/article/10.1007/s12145-024-01295-6)]
- Global Spatiotemporal Graph Attention Network for Sea Surface Temperature Prediction. [[Paper](https://ieeexplore.ieee.org/abstract/document/10056327?casa_token=CO9StPgrpQkAAAAA:32Ou82_YREM8yjfKFCQ--N09jzWEAq3CXQsjNINcsY_umsOPkcTj9pj-4lzPBFIEEGPYuKxCt-4)]
- Physical Knowledge-Enhanced Deep Neural Network for Sea Surface Temperature Prediction. [[Paper](https://ieeexplore.ieee.org/abstract/document/10068549?casa_token=gCpdIk9IltAAAAAA:Flh27ULKiqDLIxSiXGilSO_NMHZJ5kipJshY85CWen3eF1nUk1jONSEGxXuaa-gdL96JmxvmT9I)]
- Explainable deep learning for insights in El Niño and river flows. [[Paper](https://www.nature.com/articles/s41467-023-35968-5)]
- Data-driven multi-step prediction and analysis of monthly rainfall using explainable deep learning. [[Paper](https://www.sciencedirect.com/science/article/pii/S0957417423016627?casa_token=v3lHtMH4K78AAAAA:PDf_e_rs3lu-78NOKSz37TV7G3gD3I0shLVc6xbv2fuNMDhwnKx-PEVezZeR5XUIkXtEGhb2bMs)]
- An Interpretable Deep Learning Approach for Detecting Marine Heatwaves Patterns. [[Paper](https://www.mdpi.com/2076-3417/14/2/601)]
- 3D-Geoformer: A self-attention–based neural network for three-dimensional multivariate modeling and its skillful ENSO predictions. [[Paper](https://www.science.org/doi/10.1126/sciadv.adf2827)]

<h3 id="flood"> Flood </h3>

- Evaluation of artificial intelligence models for flood and drought forecasting in arid and tropical regions. [[Paper](https://www.sciencedirect.com/science/article/pii/S1364815221001791?casa_token=Vc_xmoV6NzkAAAAA:MUqa-UqGzNVB345ag6q4oMXuUxz6kfUI4pZimbZw3XW8G3iXTNQTcZA6TJprKml1rXM4TW4y26U)]
- Flood forecasting with machine learning models in an operational framework. [[Paper](https://hess.copernicus.org/articles/26/4013/2022/hess-26-4013-2022.html)]
- Particle swarm optimization based LSTM networks for water level forecasting: A case study on Bangladesh river network. [[Paper](https://www.sciencedirect.com/science/article/pii/S2590123023000786)]
- Prediction of Flow Based on a CNN-LSTM Combined Deep Learning Approach. [[Paper](https://www.mdpi.com/2073-4441/14/6/993)]
- Designing Deep-Based Learning Flood Forecast Model With ConvLSTM Hybrid Algorithm. [[Paper](https://ieeexplore.ieee.org/abstract/document/9378529)]
- Improving urban flood prediction using LSTM-DeepLabv3+ and Bayesian optimization with spatiotemporal feature fusion. [[Paper](https://www.sciencedirect.com/science/article/pii/S0022169424001379?casa_token=Wz4lAK6f5f0AAAAA:3RGV2Sz9_roQ0QY280kVKVFQHxWPLnqBkPISdPlMVh-TM-jtThoZj7wGjhv0TlLAfGRAvnkBP7c)]
- The Merit of River Network Topology for Neural Flood Forecasting. [[Paper](https://arxiv.org/abs/2405.19836)]
- FloodGNN-GRU: a spatio-temporal graph neural network for flood prediction. [[Paper](https://www.cambridge.org/core/journals/environmental-data-science/article/floodgnngru-a-spatiotemporal-graph-neural-network-for-flood-prediction/93BA1DA8D6ECC93D985656C3BC1EA3DE)]
- Graph Transformer Network for Flood Forecasting with Heterogeneous Covariates. [[Paper](https://arxiv.org/abs/2310.07631)]
- FIDLAR: Forecast-Informed Deep Learning Architecture for Flood Mitigation. [[Paper](https://arxiv.org/abs/2402.13371)]
- Data-driven and knowledge-guided denoising diffusion model for flood forecasting. [[Paper](https://www.sciencedirect.com/science/article/pii/S0957417423034103?casa_token=NUAsUus_qwQAAAAA:VqPjUuJTU_O_BAqwr0uhqu1qYAv2LZwE00gJVd5XYZUJy4I7XdzUCP4By8GLQHbGYGAAue7cVxo)]
- DRUM: Diffusion-based runoff model for probabilistic flood forecasting. [[Paper](https://arxiv.org/abs/2412.11942)]
- Generalizing rapid flood predictions to unseen urban catchments with conditional generative adversarial networks. [[Paper](https://www.sciencedirect.com/science/article/pii/S0022169423002184?casa_token=_ZL7FQaR1M8AAAAA:K0TPEufceOYU359cunDbt4_jn_7QgVkJDLguPS8cvX4ibGraKYk1x39lAkGvK3lxz71O3gZqZ-A)]


<h3 id="drought"> Drought </h3>

- Drought as a natural hazard: concepts and definitions. [[Paper](https://www.taylorfrancis.com/chapters/edit/10.4324/9781315830896-2/drought-natural-hazard-donald-wilhite)]
- Drought predic- tion based on spi and spei with varying timescales using lstm recurrent neural network. [[Paper](https://link.springer.com/article/10.1007/s00500-019-04120-1)]
- Explain- able ai in drought forecasting. [[Paper](https://www.sciencedirect.com/science/article/pii/S2666827021000967)]
- Drought prediction based on feature-based transfer learning and time series imaging. [[Paper](https://ieeexplore.ieee.org/document/9486838)]
- Application of a hybrid arima-lstm model based on the spei for drought forecasting. [[Paper](https://link.springer.com/article/10.1007/s11356-021-15325-z)]
- Evaluation of artificial intelligence models for flood and drought forecasting. [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S1364815221001791)]
- Deep learning oriented satellite remote sensing for drought and prediction in agriculture. [[Paper](https://ieeexplore.ieee.org/document/9691608)]
- Forecasting the propagation from meteorolog- ical to hydrological and agricultural drought in the huaihe river basin with machine learning methods. [[Paper](https://www.researchgate.net/publication/375968830_Forecasting_the_Propagation_from_Meteorological_to_Hydrological_and_Agricultural_Drought_in_the_Huaihe_River_Basin_with_Machine_Learning_Methods)]
- Multivariate time series convo- lutional neural networks for long-term agricultural drought prediction under global warming. [[Paper](https://www.sciencedirect.com/science/article/pii/S0378377424000180)]
- A novel intelligent deep learning predictive model for meteorological drought forecasting. [[Paper](https://link.springer.com/article/10.1007/s12652-022-03701-7)]
- Harnessing deep learning for meteorological drought forecasts in the northern cape, south africa. [[Paper](https://www.researchgate.net/publication/385423967_Harnessing_Deep_Learning_for_Meteorological_Drought_Forecasts_in_the_Northern_Cape_South_Africa)]
- Construction of an integrated drought monitoring model based on deep learning algorithms. [[Paper](https://www.mdpi.com/2072-4292/15/3/667)]
- Drought prediction using artificial intelligence models based on climate data and soil moisture. [[Paper](https://www.nature.com/articles/s41598-024-70406-6)]
- Advanced stacked integration method for forecasting long-term drought severity: Cnn with machine learning models. [[Paper](https://www.researchgate.net/publication/379754606_Advanced_stacked_integration_method_for_forecasting_long-term_drought_severity_CNN_with_machine_learning_models)]
- Multiscale spatiotemporal meteorological drought prediction: A deep learning approach. [[Paper](https://www.sciencedirect.com/science/article/pii/S1674927824000534)]
- Deep learning-oriented c-gan models for vegetative drought prediction on peninsular india. [[Paper](https://ieeexplore.ieee.org/document/10301647)]

 
<h3 id="tropical-storms"> Tropical Storms/Cyclones and Hurricanes </h3>

- Predicting tropical cyclogenesis us- ing a deep learning method from gridded satellite and era5 reanalysis data in the western north pacific basin. [[Paper](https://ieeexplore.ieee.org/document/9399663)]
- Predicting tropical cyclone formation with deep learning. [[Paper](https://journals.ametsoc.org/view/journals/wefo/39/1/WAF-D-23-0103.1.xml)]
- Improvement in forecasting short-term tropical cyclone intensity change and their rapid intensification using deep learning. [[Paper](https://journals.ametsoc.org/view/journals/aies/3/2/AIES-D-23-0052.1.xml)]
- Tropical cyclone track forecasting using fused deep learning from aligned reanalysis data. [[Paper](https://arxiv.org/abs/1910.10566)]
- A novel data-driven tropical cyclone track prediction model based on cnn and gru with multi-dimensional feature selection. [[Paper](https://ieeexplore.ieee.org/document/9085360)]
- Near real-time hurricane rainfall forecasting using convolutional neural network models with integrated multi-satellite retrievals for gpm (imerg) product. [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0169809522000230)]
- Advanced hybrid cnn-bilstm model augmented with ga and ffo for enhanced cyclone intensity forecasting. [[Paper](https://www.sciencedirect.com/science/article/pii/S1110016824001984)]
- Forecasting formation of a tropical cyclone using reanalysis data. [[Paper](https://arxiv.org/abs/2212.06149)]
- Predicting landfall’s location and time of a tropical cyclone using reanalysis data. [[Paper](https://arxiv.org/abs/2103.16108)]
- Deep learning for down-scaling tropical cyclone rainfall to hazard-relevant spatial scales. [[Paper](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022JD038163)]
- Tropical cyclone forecast using multitask deep learning framework. [[Paper](https://ieeexplore.ieee.org/document/9634051)]
- Forecasting tropical cyclones with cascaded diffusion models. [[Paper](https://arxiv.org/abs/2310.01690)]
- Advancing storm surge forecasting from scarce observation data: A causal-inference based spatio-temporal graph neural network approach. [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0378383924000607)]

<h3 id="wildfire"> Wildfire </h3>

- Latent Diffusion Model for Quantitative Precipitation Estimation and Forecast at km Scale. [[Paper](https://d197for5662m48.cloudfront.net/documents/publicationstatus/211030/preprint_pdf/76ff62db3b13a23d0922b2d7690df6f6.pdf)]
- 
- GenCast: GenCast: Diffusion-based ensemble forecasting for medium-range weather [[Paper](https://arxiv.org/pdf/2312.15796v1)]
- ChatClimate: Grounding conversational AI in climate science, in Nature Communications Earth & Environment 2023. [[Paper](https://www.nature.com/articles/s43247-023-01084-x)]
- GenCast: Diffusion-based ensemble forecasting for medium-range weather, in arXiv 2023. [Paper](https://arxiv.org/abs/2312.15796)]
- spateGAN: Spatio-Temporal Downscaling of Rainfall Fields Using a cGAN Approach, Earth and Space Science 2023. [[Paper](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023EA002906#:~:text=We%20propose%20spateGAN%2C%20a%20novel,wide%20radar%20observations%20for%20Germany.)]
- Neural General Circulation Models, in arXiv 2023. [[Paper]](https://arxiv.org/abs/2311.07222)
- Accurate Medium-Range Global Weather Forecasting with 3D Neural Networks, in Nature 2023. [paper] [official code]
- ClimaX: A Foundation Model for Weather and Climate, in arXiv 2023. [pape]] [official code]
- GraphCast: Learning Skillful Medium-Range Global Weather Forecasting, in arXiv 2022. [paper] [official code]
- FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operator, in arXiv 2022. [paper] [official code]
- W-MAE: Pre-Trained Weather Model with Masked Autoencoder for Multi-Variable Weather Forecasting, in arXiv 2023. [paper] [official code]
- FengWu: Pushing the Skillful Global Medium-range Weather Forecast beyond 10 Days Lead, in arXiv 2023. [paper]
- FuXi: A cascade machine learning forecasting system for 15-day global weather forecast, in arXiv 2023. [paper] [official code]
- OceanGPT: A Large Language Model for Ocean Science Tasks, in arXiv 2023. [paper] [official code]
- IBMWeatherGen: Stochastic Weather Generator Tool, [[Code](https://github.com/IBM/IBMWeatherGen/)]
- Diffusion Weather, [[Code](https://github.com/openclimatefix/diffusion_weather)]
- Diffusion Model for Time Series, SpatioTemporal and Tabular Data, [[Paper](https://github.com/yyysjz1997/Awesome-TimeSeries-SpatioTemporal-Diffusion-Model#weather)]
- https://github.com/diff-usion/Awesome-Diffusion-Models, [[Paper](https://github.com/diff-usion/Awesome-Diffusion-Models)]
