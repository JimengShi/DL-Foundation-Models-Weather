# DL-Foundation-Models-for-Weather-Prediction
Add Figures 1 & 2 from the paper here.
<div align="center">
<img src="https://github.com/JimengShi/AI-Models-Weather-and-Climate/blob/main/figs/weather_forecast.jpg" alt="taxonomy" width="600"/> 
</div>

## Contents
- [Predictive Models](#predictive-learning)
  - [General-Purpose Large Models](#general-predictive-learning)
    - [Transformer](#general-predictive-learning-transformer)
      
  - [Domain-Specific Models](#domain-predictive-learning)
    - [Transformer](#domain-predictive-learning-transformer)

      
- [Generative Models](#generative-learning)
  - [General-Purpose Large Models](#general-generative-learning)
    - [Diffusion Models](#general-generative-learning-diffusion)
      
  - [Domain-Specific Models](#domain-generative-learning)
    - [Diffusion Models](#domain-generative-learning-diffusion)

      
- [Foundation Models](#pretraining-finetuning)
 


## Taxonomy
<div align="center">
<img src="https://github.com/JimengShi/AI-Models-Weather-and-Climate/blob/main/figs/frameworks.jpg" alt="taxonomy" width="800"/> 
</div>




<h2 id="predictive-learning">Predictive Models</h2>

<h3 id="general-predictive-learning">General-Purpose Large Models</h3>
<h4 id="general-predictive-learning-transformer">Transformer</h4>

- **(FourCastNet)** _FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators_ ```arXiv 2022```     
[[Paper](https://arxiv.org/abs/2202.11214)] [[Code](https://github.com/NVlabs/FourCastNet)] 


<h3 id="domain-predictive-learning">Domain-Specific Models</h3>
<h4 id="domain-predictive-learning-transformer">Transformer</h4>

- **(SwinUnet)** _Spatiotemporal vision transformer for short time weather forecasting._ ```IEEE BigData 2022```     
[[Paper](https://ieeexplore.ieee.org/document/9671442)] [[Code](https://github.com/bojesomo/Weather4Cast2021-SwinUNet3D)] 






<h2 id="generative-learning">Generative Models</h2>

<h3 id="general-generative-learning">General-Purpose Large Models</h3>
<h4 id="general-generative-learning-diffusion">Diffusion Models</h4>

- **(GenCast)** _GenCast: Diffusion-based ensemble forecasting for medium-range weather_ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2312.15796)] 


<h3 id="domain-generative-learning">Domain-Specific Models</h3>
<h4 id="domain-generative-learning-diffusion">Diffusion Models</h4>

- **(LDMRain)** _Latent diffusion models for generative precipitation nowcast-ing with accurate uncertainty quantification._ ```arXiv 2023```     
[[Paper](https://arxiv.org/abs/2304.12891)] [[Code](https://github.com/MeteoSwiss/ldcast)] 






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
