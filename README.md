# DL-Foundation-Models-for-Weather-Prediction
<div align="center">
<img src="https://github.com/JimengShi/AI-Models-Weather-and-Climate/blob/main/figs/weather_forecast.jpg" alt="taxonomy" width="600"/> 
</div>

## Contents
- [Predictive Models](#predictive-learning)
  - [General-Purpose Large Models](#general-predictive-learning)
    - [Transformer](#general-predictive-learning-transformer)
    - [GNN](#general-predictive-learning-gnn)
  - [Domain-Specific Models](#domain-predictive-learning)
    - [Transformer](#domain-predictive-learning-transformer)
- [Generative Models](#generative-learning)
  - [General-Purpose Large Models](#general-generative-learning)
    - [Diffusion Models](#general-generative-learning-diffusion)
  - [Domain-Specific Models](#domain-generative-learning)
    - [Diffusion Models](#domain-generative-learning-diffusion)
    - [GANs](#domain-generative-learning-gan)
- [Foundation Models](#pretraining-finetuning)
 


## Taxonomy
<div align="center">
<img src="https://github.com/JimengShi/AI-Models-Weather-and-Climate/blob/main/figs/frameworks.jpg" alt="taxonomy" width="800"/> 
</div>

<!-------------------------------------- Predictive Models  --------------------------------------> 
<h2 id="predictive-learning">Predictive Models</h2>

<h3 id="general-predictive-learning">General-Purpose Large Models</h3>
<h4 id="general-predictive-learning-transformer">Transformer</h4>

- **(FourCastNet)** _FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators_ ```arXiv 2022```     
[[Paper](https://arxiv.org/abs/2202.11214)] [[Code](https://github.com/NVlabs/FourCastNet)]

- **(FuXi)** _FuXi: a cascade machine learning forecasting system for 15-day global weather forecast_ ```Nature 2023```     
[[Paper](https://www.nature.com/articles/s41612-023-00512-1)] [[Code](https://drive.google.com/drive/folders/1NhrcpkWS6MHzEs3i_lsIaZsADjBrICYV)]

- **(FengWu)** _FengWu: Pushing the Skillful Global Medium-range Weather Forecast beyond 10 Days Lead_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2304.02948)] 

- **(FengWu-4DVar)** _FengWu-4DVar: Coupling the Data-driven Weather Forecasting Model with 4D Variational Assimilation_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2312.12455)] 

- **(SwinVRNN)** _SwinVRNN: A Data-Driven Ensemble Forecasting Model via Learned Distribution Perturbation_ ```JAMES 2023```     
[[Paper](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022MS003211)] [[Code](https://github.com/tpys/wwprediction)]

- **(SwinRDM)** _SwinRDM: Integrate SwinRNN with Diffusion Model towards High-Resolution and High-Quality Weather Forecasting_ ```AAAI Technical Track on Computer Vision I 2023```     
[[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/25105)] 

- **(Pangu-Weather)** _Accurate medium-range global weather forecasting with 3D neural networks_ ```Nature 2023```    
[[Paper](https://www.nature.com/articles/s41586-023-06185-3)] [[Code](https://github.com/198808xc/Pangu-Weather)]

- **(Stormer)** _Scaling transformer neural networks for skillful and reliable medium-range weather forecasting_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2312.03876)] [[Code](https://github.com/tung-nd/stormer)]

- **(HEAL-ViT)** _HEAL-ViT: Vision Transformers on a spherical mesh for medium-range weather forecasting_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2403.17016)] 



<h4 id="general-predictive-learning-gnn">GNN</h4>

- **(GraphCast)** _Graphcast: Learning skillful medium-range global weather forecasting_ ```Science 2023```     
[[Paper](https://arxiv.org/abs/2212.12794)] [[Code](https://github.com/openclimatefix/graph_weather)]

- **(GnnWeather)** _Forecasting Global Weather with Graph Neural Networks_ ```arXiv 2022```     
[[Paper](https://arxiv.org/abs/2202.07575)] [[Code](https://rkeisler.github.io/graph_weather/)]



<h4 id="general-predictive-learning-gnn">PhysicsAI</h4>

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

- **(SwinUnet)** _Spatiotemporal vision transformer for short time weather forecasting._ ```IEEE BigData 2022```     
[[Paper](https://ieeexplore.ieee.org/document/9671442)] [[Code](https://github.com/bojesomo/Weather4Cast2021-SwinUNet3D)] 

- **(Earthformer)** _Earthformer: Exploring Space-Time Transformers for Earth System Forecasting_ ```NeurIPS 2022```     
[[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a2affd71d15e8fedffe18d0219f4837a-Abstract-Conference.html)] [[Code](https://proceedings.neurips.cc/paper_files/paper/2022/file/a2affd71d15e8fedffe18d0219f4837a-Supplemental-Conference.zip)] 

- **(Rainformer)** _Rainformer: Features Extraction Balanced Network for Radar-Based Precipitation Nowcasting_ ```IEEE Geoscience and Remote Sensing Letters 2022```     
[[Paper](https://ieeexplore.ieee.org/abstract/document/9743916)] [[Code](https://github.com/Zjut-MultimediaPlus/Rainformer)]

- **(Fformer)** _PFformer: A Time-Series Forecasting Model for Short-Term Precipitation Forecasting_ ```IEEE Access 2024```     
[[Paper](https://ieeexplore.ieee.org/abstract/document/10678751)] 



<h4 id="domain-predictive-learning-transformer">GNN</h4>

- **(HiSTGNN)** _HiSTGNN: Hierarchical spatio-temporal graph neural network for weather forecasting_ ```Information Sciences 2023```     
[[Paper](https://www.sciencedirect.com/science/article/pii/S0020025523011659)] [[Code](https://github.com/mb-Ma/HiSTGNN)]

- **(w-GNN)** _Coupling Physical Factors for Precipitation Forecast in China With Graph Neural Network_ ```AGU 2024```     
[[Paper](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023GL106676)] 

- **(WeatherGNN)** _WeatherGNN: Exploiting Meteo- and Spatial-Dependencies for Local Numerical Weather Prediction Bias-Correction_ ```IJCAI 2024```     
[[Paper](https://www.ijcai.org/proceedings/2024/0269.pdf)] 

- **(MPNNs)** _Multi-modal graph neural networks for localized off-grid weather forecasting_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2410.12938)] [[Code](https://github.com/Earth-Intelligence-Lab/LocalizedWeatherGNN/)]



<h4 id="domain-predictive-learning-transformer">RNN&CNN</h4>

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



<h4 id="domain-predictive-learning-transformer">Mamba</h4>

- **(MetMamba)** _MetMamba: Regional Weather Forecasting with Spatial-Temporal Mamba Model_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2408.06400)] 

- **(MambaDS)** _MambaDS: Near-Surface Meteorological Field Downscaling With Topography Constrained Selective State-Space Modeling_ ```IEEE Transactions on Geoscience and Remote Sensing 2024```     
[[Paper](https://ieeexplore.ieee.org/abstract/document/10752514)] 


<h4 id="domain-predictive-learning-transformer">PhysicsAI</h4>

- **(???)** _PFformer: A Time-Series Forecasting Model for Short-Term Precipitation Forecasting_ ```IEEE Access 2024```     
[[Paper](https://ieeexplore.ieee.org/abstract/document/10678751)] [[Code](https://github.com/Zjut-MultimediaPlus/Rainformer)]

- **(???)** _PFformer: A Time-Series Forecasting Model for Short-Term Precipitation Forecasting_ ```IEEE Access 2024```     
[[Paper](https://ieeexplore.ieee.org/abstract/document/10678751)] [[Code](https://github.com/Zjut-MultimediaPlus/Rainformer)]

- **(???)** _PFformer: A Time-Series Forecasting Model for Short-Term Precipitation Forecasting_ ```IEEE Access 2024```     
[[Paper](https://ieeexplore.ieee.org/abstract/document/10678751)] [[Code](https://github.com/Zjut-MultimediaPlus/Rainformer)]

- **(???)** _PFformer: A Time-Series Forecasting Model for Short-Term Precipitation Forecasting_ ```IEEE Access 2024```     
[[Paper](https://ieeexplore.ieee.org/abstract/document/10678751)] [[Code](https://github.com/Zjut-MultimediaPlus/Rainformer)]





<!-------------------------------------- Generative Models  --------------------------------------> 
<h2 id="generative-learning">Generative Models</h2>
<h3 id="general-generative-learning">General-Purpose Large Models</h3>
<h4 id="general-generative-learning-diffusion">Diffusion Models</h4>

- **(GenCast)** _GenCast: Diffusion-based ensemble forecasting for medium-range weather_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2312.15796)] 

- **(???)** _PFformer: A Time-Series Forecasting Model for Short-Term Precipitation Forecasting_ ```IEEE Access 2024```     
[[Paper](https://ieeexplore.ieee.org/abstract/document/10678751)] [[Code](https://github.com/Zjut-MultimediaPlus/Rainformer)]

- **(???)** _PFformer: A Time-Series Forecasting Model for Short-Term Precipitation Forecasting_ ```IEEE Access 2024```     
[[Paper](https://ieeexplore.ieee.org/abstract/document/10678751)] [[Code](https://github.com/Zjut-MultimediaPlus/Rainformer)]

- **(???)** _PFformer: A Time-Series Forecasting Model for Short-Term Precipitation Forecasting_ ```IEEE Access 2024```     
[[Paper](https://ieeexplore.ieee.org/abstract/document/10678751)] [[Code](https://github.com/Zjut-MultimediaPlus/Rainformer)]




<h3 id="domain-generative-learning">Domain-Specific Models</h3>
<h4 id="domain-generative-learning-diffusion">Diffusion Models</h4>

- **(LDMRain)** _Latent diffusion models for generative precipitation nowcast-ing with accurate uncertainty quantification._ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2304.12891)] [[Code](https://github.com/MeteoSwiss/ldcast)] 

- **(???)** _PFformer: A Time-Series Forecasting Model for Short-Term Precipitation Forecasting_ ```IEEE Access 2024```     
[[Paper](https://ieeexplore.ieee.org/abstract/document/10678751)] [[Code](https://github.com/Zjut-MultimediaPlus/Rainformer)]

- **(???)** _PFformer: A Time-Series Forecasting Model for Short-Term Precipitation Forecasting_ ```IEEE Access 2024```     
[[Paper](https://ieeexplore.ieee.org/abstract/document/10678751)] [[Code](https://github.com/Zjut-MultimediaPlus/Rainformer)]

- **(???)** _PFformer: A Time-Series Forecasting Model for Short-Term Precipitation Forecasting_ ```IEEE Access 2024```     
[[Paper](https://ieeexplore.ieee.org/abstract/document/10678751)] [[Code](https://github.com/Zjut-MultimediaPlus/Rainformer)]

- **(???)** _PFformer: A Time-Series Forecasting Model for Short-Term Precipitation Forecasting_ ```IEEE Access 2024```     
[[Paper](https://ieeexplore.ieee.org/abstract/document/10678751)] [[Code](https://github.com/Zjut-MultimediaPlus/Rainformer)]

- **(???)** _PFformer: A Time-Series Forecasting Model for Short-Term Precipitation Forecasting_ ```IEEE Access 2024```     
[[Paper](https://ieeexplore.ieee.org/abstract/document/10678751)] [[Code](https://github.com/Zjut-MultimediaPlus/Rainformer)]



<h4 id="domain-generative-learning-gan">GANs</h4>

- **(GANRain)** _Skillful precipitation nowcasting using deep generative models of radar._ ```Nature 2021```     
[[Paper](https://www.nature.com/articles/s41586-021-03854-z)] [[Code](https://github.com/openclimatefix/skillful_nowcasting)] 

- **(???)** _PFformer: A Time-Series Forecasting Model for Short-Term Precipitation Forecasting_ ```IEEE Access 2024```     
[[Paper](https://ieeexplore.ieee.org/abstract/document/10678751)] [[Code](https://github.com/Zjut-MultimediaPlus/Rainformer)]

- **(???)** _PFformer: A Time-Series Forecasting Model for Short-Term Precipitation Forecasting_ ```IEEE Access 2024```     
[[Paper](https://ieeexplore.ieee.org/abstract/document/10678751)] [[Code](https://github.com/Zjut-MultimediaPlus/Rainformer)]

- **(???)** _PFformer: A Time-Series Forecasting Model for Short-Term Precipitation Forecasting_ ```IEEE Access 2024```     
[[Paper](https://ieeexplore.ieee.org/abstract/document/10678751)] [[Code](https://github.com/Zjut-MultimediaPlus/Rainformer)]




<!-------------------------------------- Foundation Models  --------------------------------------> 
<h2 id="pretraining-finetuning">Foundation Models</h2>

- **(ClimaX)** _ClimaX: A foundation model for weather and climate_ ```ICML 2023```     
[[Paper](https://arxiv.org/abs/2301.10343)] [[Code](https://github.com/microsoft/ClimaX)]

- **(W-MAE)** _W-MAE: Pre-trained weather model with masked autoencoder for multi-variable weather forecasting_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2304.08754)] [[Code](https://github.com/Gufrannn/W-MAE)]

- **(Aurora)** _Aurora: A foundation model of the atmosphere._ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2405.13063)] [[Code](https://github.com/microsoft/aurora)]

- **(Prithvi WxC)** _Prithvi WxC: Foundation Model for Weather and Climate_ ```arXiv 2024```     
[[Paper](https://arxiv.org/abs/2409.13598)] [[Code](https://github.com/NASA-IMPACT/Prithvi-WxC)]




## Applications
- Latent Diffusion Model for Quantitative Precipitation Estimation and Forecast at km Scale. [[paper](https://d197for5662m48.cloudfront.net/documents/publicationstatus/211030/preprint_pdf/76ff62db3b13a23d0922b2d7690df6f6.pdf)]
- PreDiff: Precipitation Nowcasting with Latent Diffusion Models [[paper](https://arxiv.org/pdf/2307.10422)]
- GenCast: GenCast: Diffusion-based ensemble forecasting for medium-range weather [[paper](https://arxiv.org/pdf/2312.15796v1)]
- ChatClimate: Grounding conversational AI in climate science, in Nature Communications Earth & Environment 2023. [[Paper](https://www.nature.com/articles/s43247-023-01084-x)]
- GenCast: Diffusion-based ensemble forecasting for medium-range weather, in arXiv 2023. [Paper](https://arxiv.org/abs/2312.15796)]
- spateGAN: Spatio-Temporal Downscaling of Rainfall Fields Using a cGAN Approach, Earth and Space Science 2023. [[Paper](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023EA002906#:~:text=We%20propose%20spateGAN%2C%20a%20novel,wide%20radar%20observations%20for%20Germany.)]
- Neural General Circulation Models, in arXiv 2023. [[Paper]](https://arxiv.org/abs/2311.07222)
- 3D-Geoformer: A self-attention–based neural network for three-dimensional multivariate modeling and its skillful ENSO predictions, in Science Advances, 2023. [[Paper](https://www.science.org/doi/10.1126/sciadv.adf2827)]
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
