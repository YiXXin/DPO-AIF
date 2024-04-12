import torch
import json
from ipdb import set_trace
from PIL import Image
import io
import datasets
from datasets import Dataset
import numpy as np

# scores = torch.load('/data3/yixinf/GenAI800_rerank/t2v_metrics/reranking_results/DALLE3_results/clip-flant5-xxl_DALLE3_800_rerank.pt')
# scores = scores.reshape(-1,9)

def get_byte(path):
    im = Image.open(path)
    buf = io.BytesIO()
    im.save(buf, format='JPEG')
    byte_im = buf.getvalue()
    return byte_im

#'jpg_0' 'jpg_1'
def gen_pair():
    f = open('/data3/baiqil/github/t2v_metrics/datasets/GenAI_800/reranking/humanRating_800.json')
    data = json.load(f)
    img_path = '/data3/baiqil/github/t2v_metrics/datasets/GenAI_800/reranking/SDXL_rerank'
    num=0
    for str_i in data.keys():
        # str_i = f"{i:05d}"
        prompt = data[str_i]["prompt"]
        for i in range(1,10):
            for j in range(i+1,10):
                af_i = f"{i:02d}"
                af_j = f"{j:02d}"
                img0_path = f'{img_path}/{str_i}_{af_i}.jpeg'
                img1_path = f'{img_path}/{str_i}_{af_j}.jpeg'
                jpg_0 = get_byte(img0_path)
                jpg_1 = get_byte(img1_path)
                # score0 = scores[num][i-1]
                # score1 = scores[num][j-1]
                # set_trace()
                score0 = np.array(data[str_i]['models']['DALLE_3'][str(i)]).mean()
                score1 = np.array(data[str_i]['models']['DALLE_3'][str(j)]).mean()
                if score0 >= score1:
                    label_0 = 1
                else:
                    label_0 = 0
                yield {'caption':prompt,'img0_path': img0_path, 'img1_path': img1_path, 'jpg_0':jpg_0,'jpg_1':jpg_1,'label_0':label_0}
        # prompts[str_i] = {}
        # prompts[str_i]['caption'] = prompt
        num+=1
       
def gen_best():
    f = open('/data3/baiqil/github/t2v_metrics/datasets/GenAI_800/reranking/humanRating_800.json')
    data = json.load(f)
    img_path = '/data3/baiqil/github/t2v_metrics/datasets/GenAI_800/reranking/SDXL_rerank'
    num=0
    for str_i in data.keys():
        # str_i = f"{i:05d}"
        prompt = data[str_i]["prompt"]
        max_diff = 0
        max_i = 1
        max_j = 2
        max_score0 = np.array(data[str_i]['models']['DALLE_3'][str(max_i)]).mean()
        max_score1 = np.array(data[str_i]['models']['DALLE_3'][str(max_j)]).mean()
        for i in range(1,10):
            for j in range(i+1,10):
                # set_trace()
                score0 = np.array(data[str_i]['models']['DALLE_3'][str(i)]).mean()
                score1 = np.array(data[str_i]['models']['DALLE_3'][str(j)]).mean()
                score_diff = abs(score0-score1)
                if score_diff > max_diff:
                    max_diff = score_diff
                    max_i = i
                    max_j = j
                    max_score0 = score0
                    max_score1 = score1

        af_i = f"{max_i:02d}"
        af_j = f"{max_j:02d}"
        img0_path = f'{img_path}/{str_i}_{af_i}.jpeg'
        img1_path = f'{img_path}/{str_i}_{af_j}.jpeg'
        jpg_0 = get_byte(img0_path)
        jpg_1 = get_byte(img1_path)
        if max_score0 >= max_score1:
            label_0 = 1
        else:
            label_0 = 0
        yield {'caption':prompt,'img0_path': img0_path, 'img1_path': img1_path, 'jpg_0':jpg_0,'jpg_1':jpg_1,'label_0':label_0}
        # prompts[str_i] = {}
        # prompts[str_i]['caption'] = prompt
        num+=1


def main():
    
    # ds = Dataset.from_generator(gen_pair)
    # ds.save_to_disk("/data3/yixinf/dpo_datasets")

    ds = Dataset.from_generator(gen_best)
    ds.save_to_disk("/data3/yixinf/dpo_datasets_800")

if __name__ == '__main__':
    main()

