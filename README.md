# Spatial-Temporal Person Re-identification

----------
Code for st-ReID(pytorch). We achieve **Rank@1=98.1%, mAP=87.6%** without re-ranking and **Rank@1=98.0%, mAP=95.5%** with re-ranking for market1501.For Duke-MTMC, we achieve **Rank@1=94.4%, mAP=83.9%** without re-ranking and **Rank@1=94.5%, mAP=92.7%** with re-ranking.

## Update and FQA:
- 2020.01.08: If you do not want to re-train a model, you can follow this link. https://github.com/Wanggcong/Spatial-Temporal-Re-identification/issues/26#issuecomment-571905649
- 2019.12.26：a demo figure has been added. I am not sure if it works or not because it was written one years ago. I will update this file in the future.
- 2019.07.28: Models(+RE) (google drive Link:https://drive.google.com/drive/folders/1FIreE0pUGiqLzppzz_f7gHw0kaXZb1kC)
- 2019.07.11: Models (+RE) (baiduyun Link:https://pan.baidu.com/s/1QMp22dVGJvBH45e4XPdeKw  password:dn7b) are released. Note that, for market, slightly different from the results in the paper because we use pytorch 0.4.1 to train these models (mAP is slightly higher than paper while rank-1 is slightly lower than paper). We may reproduce the results by Pytorch 0.3 later.
- 2019.07.11: README.md, python3 prepare --Duke ---> python3 prepare.py --Duke
- 2019.06.02: How to add the spatial-temporal constraint into conventional re-id models? You can replace step 2 and step 3 by your own visual feature represenation.
- 2019.05.31: gen_st_model_market.py, added Line 68~69.


## 1. ST-ReID
### 1.1 model
![](https://i.imgur.com/WYCcBHO.jpg)

### 1.2 result
![](https://i.imgur.com/KubElWp.jpg)
----------

![](https://i.imgur.com/Ul6h45K.jpg)


## 2. rerequisites
- **Pytorch 0.3**
- Python 3.6
- Numpy


## 3. experiment
### Market1501
1. data prepare<br>
   1) change the path of dataset <br>
   2) python3 prepare.py --Market

2. train (appearance feature learning) <br>
python3 train_market.py --PCB --gpu_ids 2 --name ft_ResNet50_pcb_market_e --erasing_p 0.5 --train_all --data_dir "/home/huangpg/st-reid/dataset/market_rename/"

3. test (appearance feature extraction) <br>
python3 test_st_market.py --PCB --gpu_ids 2 --name ft_ResNet50_pcb_market_e --test_dir "/home/huangpg/st-reid/dataset/market_rename/" 

4. generate st model (spatial-temporal distribution) <br>
python3 gen_st_model_market.py --name ft_ResNet50_pcb_market_e --data_dir "/home/huangpg/st-reid/dataset/market_rename/"
5. evaluate (joint metric, you can use your own visual feature or spatial-temporal streams) <br>
python3 evaluate_st.py --name ft_ResNet50_pcb_market_e 

6. re-rank<br>
6.1) python3 gen_rerank_all_scores_mat.py --name ft_ResNet50_pcb_market_e <br>
6.2) python3 evaluate_rerank_market.py --name ft_ResNet50_pcb_market_e


### DukeMTMC-reID
1. data prepare<br>
python3 prepare.py --Duke

2. train (appearance feature learning) <br>
python3 train_duke.py --PCB --gpu_ids 2 --name ft_ResNet50_pcb_duke_e --erasing_p 0.5 --train_all --data_dir "/home/huangpg/st-reid/dataset/DukeMTMC_prepare/"

3. test (appearance feature extraction) <br>
python3 test_st_duke.py --PCB --gpu_ids 2 --name ft_ResNet50_pcb_duke_e --test_dir "/home/huangpg/st-reid/dataset/DukeMTMC_prepare/" 

4. generate st model (spatial-temporal distribution) <br>
python3 gen_st_model_duke.py --name ft_ResNet50_pcb_duke_e  --data_dir "/home/huangpg/st-reid/dataset/DukeMTMC_prepare/"

5. evaluate (joint metric, you can use your own visual feature or spatial-temporal streams) <br>
python3 evaluate_st.py --name ft_ResNet50_pcb_duke_e 

6. re-rank<br>
6.1) python3 gen_rerank_all_scores_mat.py --name ft_ResNet50_pcb_duke_e <br>
6.2) python3 evaluate_rerank_duke.py --name ft_ResNet50_pcb_duke_e

## Citation

If you use this code, please kindly cite it in your paper.

```latex
@article{guangcong2019aaai,
  title={Spatial-Temporal Person Re-identification},
  author={Wang, Guangcong and Lai, Jianhuang and Huang, Peigen and Xie, Xiaohua},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  pages={8933-8940},
  year={2019}
}
```
Paper Link:https://wvvw.aaai.org/ojs/index.php/AAAI/article/view/4921
or https://arxiv.org/abs/1812.03282
## Related Repos

Our codes are mainly based on this [repository](https://github.com/layumi/Person_reID_baseline_pytorch) 
