# SLIQUE
Release of Self-supervision and Vision-Language supervision Image QUality Evaluator (SLIQUE)

## Usage

### Pre-trained Checkpoints
([Google Drive](https://drive.google.com/file/d/1ojqygcMJMttAi6yb9woaHBvG1Bov4q9z/view?usp=sharing))


### Extracting Image Feature
To obtain SLIQUE features, the following commands can be used. 

````
python demo_feat.py --model_path models/SLIQUE.tar --img_path fig/example.jpg
python demo_feat.py --model_path models/SLIQUE.tar --img_dir fig
````

By default, these features are saved in the 'feat' folder in.npy format. To change the save folder used:

````
python demo_feat.py --feature_save_path feat
````

### Obtaining Scores
After extracting image features, the following commands can be used to get the scores.
````
python demo_score.py --regressor_model_path models/in_the_wild.save 
--models/LIVE.save --feat_path feat/example.jpg.npy
````
## Citation
```
@article{zhou2024vision,
  title={Vision Language Modeling of Content, Distortion and Appearance for Image Quality Assessment},
  author={Zhou, Fei and Huang, Zhicong and Gu, Tianhao and Qiu, Guoping},
  journal={arXiv preprint arXiv:2406.09858},
  year={2024}
}
```