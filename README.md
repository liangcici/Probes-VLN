# Visual-Language Navigation Pretraining via Prompt-based Environmental Self-exploration

This is the code for the [ProbES](https://arxiv.org/abs/2203.04006) paper.

Catalog:
- [x] Generating pretraining dataset
- [ ] Pretraining on generated dataset
- [ ] Finetuning on downstream tasks

## Install Dependencies
1. Python requirements: Need python3.6 or higher and install dependencies:
    ```
    pip install -r requirements.txt
    ```
2. Install [Matterport3D simulator](https://github.com/peteanderson80/Matterport3DSimulator). Notice that this code uses the [old version (v0.1)](https://github.com/peteanderson80/Matterport3DSimulator/tree/v0.1) of the simulator.


## Preparing Dataset
1. Download all of the required data files:
    ```
    python scripts/download-auxiliary-data.py
    wget https://dl.dropbox.com/s/67k2vjgyjqel6og/matterport-ResNet-101-faster-rcnn-genome.lmdb.zip -P data/
    unzip data/matterport-ResNet-101-faster-rcnn-genome.lmdb.zip -d data/
    ```
2. Download pre-computed CLIP features:
    ```
    wget https://nlp.cs.unc.edu/data/vln_clip/features/CLIP-ViT-B-32-views.tsv -P data/img_features/
    ```

3. Generating pretraining dataset:
    ```
    sh scripts/generate_pretrain_data.sh
    ```

## Training
coming soon

## Acknowledgement
The implementation relies on resources from [VLN-BERT](https://github.com/arjunmajum/vln-bert), [Airbert](https://github.com/airbert-vln/airbert) and [CLIP-ViL](https://github.com/clip-vil/CLIP-ViL). We thank the original authors for their open-sourcing.


## Reference
If you find this code useful, please consider citing.
```
@article{liang2022visual,
  title={Visual-Language Navigation Pretraining via Prompt-based Environmental Self-exploration},
  author={Liang, Xiwen and Zhu, Fengda and Li, Lingling and Xu, Hang and Liang, Xiaodan},
  journal={arXiv preprint arXiv:2203.04006},
  year={2022}
}
```
