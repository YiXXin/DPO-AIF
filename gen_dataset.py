import torch
import json
import pdb
from PIL import Image
import io
import datasets
from datasets import Dataset
import numpy as np

scores = torch.load('/data3/yixinf/GenAI800_rerank/t2v_metrics/reranking_results/DALLE3_results/clip-flant5-xxl_DALLE3_800_rerank.pt')
img_path = '/data3/baiqil/github/t2v_metrics/datasets/GenAI_800/reranking/DALLE_rerank'
scores = scores.reshape(-1,9)
def get_byte(path):
    im = Image.open(path)
    buf = io.BytesIO()
    im.save(buf, format='JPEG')
    byte_im = buf.getvalue()
    return byte_im

f = open('/data3/baiqil/github/t2v_metrics/datasets/GenAI_800/reranking/humanRating_800.json')
data = json.load(f)

#'jpg_0' 'jpg_1'
def gen():
    num=0
    for str_i in data.keys():
        # str_i = f"{i:05d}"
        prompt = data[str_i]["prompt"]
        max_diff = -1
        for i in range(1,10):
            for j in range(i+1,10):
                human_score0 = np.array(data[str_i]['models']['DALLE_3'][str(i)]).mean()
                human_score1 = np.array(data[str_i]['models']['DALLE_3'][str(j)]).mean()
                diff = abs(human_score0-human_score1)
                if diff > max_diff:
                    max_diff = diff
                    af_i = f"{i:02d}"
                    af_j = f"{j:02d}"
                    jpg_0 = get_byte(f'{img_path}/{str_i}_{af_i}.jpeg')
                    jpg_1 = get_byte(f'{img_path}/{str_i}_{af_j}.jpeg')
                    img0_path = f'{img_path}/{str_i}_{af_i}.jpeg'
                    img1_path = f'{img_path}/{str_i}_{af_j}.jpeg'
                    score0 = scores[num][i-1]
                    score1 = scores[num][j-1]
                    if score0 > score1:
                        label_0 = 1
                    else:
                        label_0 = 0
                    
        yield {'caption':prompt,'img0_path': img0_path, 'img1_path': img1_path, 'jpg_0':jpg_0,'jpg_1':jpg_1,'label_0':label_0}
        # prompts[str_i] = {}
        # prompts[str_i]['caption'] = prompt
        num+=1
       

ds = Dataset.from_generator(gen)
ds.save_to_disk("/ssd0/kewenwu/dpo_datasets_800_new")

