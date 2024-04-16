import t2v_metrics
from torchvision import transforms
from PIL import Image


class Selector():
    
    def __init__(self, device):
        self.device = device
        self.model = t2v_metrics.VQAScore(model='clip-flant5-xl').to(device)
        
    def score(self, img_paths, prompt):
        scores = self.model(img_paths, prompt)
        scores = scores.squeeze().cpu().numpy().tolist()
        return scores