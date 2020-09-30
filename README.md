# Tree-Structured Neural Topic Model
A code for "Tree-Structured Neural Topic Model" in ACL2020

Corresponding paper:
https://www.aclweb.org/anthology/2020.acl-main.73/

Masaru Isonuma, Juncihiro Mori, Danushka Bollegala, and Ichiro Sakata (The University of Tokyo, University of Liverpool)  

### Preprocessing

#### Amazon data (bags and cases)

Download the raw data and put `bags_and_cases.trn` to `data/bags/` from  
https://drive.google.com/uc?id=1Vt_Pnby63OgB1NK-2qwT_K4mryEXMQ-J&export=download  
(The data is distributed in https://github.com/stangelid/oposum)

Run the following script:
```
python preprocess_oposum.py -path_data </path/to/raw/data> -path_output </path/to/preprocessed/data>
```

#### 20 News Groups

Download the raw data and put them to `data/20news/` from  
https://github.com/akashgit/autoencoding_vi_for_topic_models/tree/master/data/20news_clean  
(The data is distributed in https://github.com/akashgit/autoencoding_vi_for_topic_models)


Run the following script:
```
python preprocess_20news.py -path_data </path/to/raw/data> -path_output </path/to/preprocessed/data>
```

### Training

Run the following script:

```
python train.py -gpu <index/of/gpu> -data <bags/or/20news> -path_data </path/to/preprocessed/data> -dir_model <path/to/model/directory> -dir_corpus <path/to/corpus>
```

The trained parameters are saved in `dir_model`.  
The corpus in `dir_corpus` are used for calculating coherence score (NPMI).

### Evaluation

Run the following script:

```
python evaluate.py -gpu <index/of/gpu> -data <bags/or/20news> -path_model <path/to/model/checkpoint>
```

The scores and topic frequent words are displayed in the console.  
You can also use our checkpoint in `model/bags/checkpoint_stable`.  
(Although the scores on this checkpoint slightly differ from the scores in the paper, the difference does not influence the claim of the paper.)  

### Acknowledgement

The module to calculate NPMI (`coherence.py`) is based on the code:  
https://github.com/jhlau/topic_interpretability
