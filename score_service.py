import torch.nn.functional as F
import torch


class similarityScore:

    # method for calculating cosine distance between two tensors 
    def cosine_score(self, reference, sample):
        self.reference = reference
        self.sample = sample

        cosd = F.cosine_similarity(self.reference, self.sample)
        score = (1 - cosd.item())

        if score < 0.4:
            return score, 'ok'
        else:
            return score, 'no match'


    # method for calculating euclidian distance 
    def euclidian_score(self, reference, sample):
        self.reference = reference
        self.sample = sample 

        score = torch.cdist(self.reference, self.sample, p=2)
        
        return score


    def l2_score(self, reference, sample):
        self.reference = reference
        self.sample = sample

        score = torch.dist(self.reference, self.sample, p=2)
    
        
        return score


    







