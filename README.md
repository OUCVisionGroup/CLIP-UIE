<div align="center">

<h1>Underwater Image Enhancement by Diffusion Model with Customized CLIP-Classifier</h1>
<div>
    <h4 align="center">
        <a href="https://oucvisiongroup.github.io/CLIP-UIE.html/" target='_blank'>[Project Page]</a>•
        <a href="" target='_blank'>[arXiv]</a> 
    </h4>
</div>

<table>
    <tr>
        <td><center><img src="images/overflow.jpg" height="300">
            
The preparation for the pre-trained model. (a) Randomly select template A from the template pool (underwater domain). Then,  the Color Transfer module, guided by template A, degrades an in-air natural image from INaturalist into underwater domain, constructing paired datasets for training image-to-image diffusion model. (b) The image-to-image diffusion model SR3 is trained to learn the prior knowledge, the mapping from the real underwater degradation domain and the real in-air natural domain, and to generate the corresponding enhancement results under the condition of the input synthetic underwater images produced by Color Transfer.
          </center></td>
</tr>
</table>
</div>

## :desktop_computer: Requirements

- Pytorch >= 1.13.1
- CUDA >= 11.3
- Other required packages in `requirements.txt`
## :running_woman: Inference
## 📦 Models

| Name                       |  Model                       |
|----------------------------|--------------------------------------|
| CLIP-UIE | [Download 🔗](https://drive.google.com/drive/folders/190-6QlKtPKBcG1fxSlXLMKop2exzgGkM?usp=sharing)|
| Learned Prompt | [Download 🔗](https://drive.google.com/drive/folders/1mnvp0sEFbSPCbSqlG-ETYSzmCO-cLTRg?usp=sharing)|

## Testing steps:
- Putting your data into the dataset folder. 
- Download the pre-trained model. Then, put the model in the experiments_supervised folder and change the corresponding paths in config/sr_sr3_32_256_UIEB_SUIM_E_plus_finetune_clip_classifier.json.
- Execute infer_finetune_infer.py to get the inference results in a new folder called experiments.
- More details will be released after the article is accepted.

### Thanks
Our code is based on [SR3](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/tree/master) and [CLIP-LIT](https://github.com/ZhexinLiang/CLIP-LIT). You can refer to their README files and source code for more implementation details. 

## :love_you_gesture: Citation
If you find our work useful for your research, please consider citing the paper:

