import os
import numpy as np
import torch
import math

from torch.utils.data import DataLoader
from model import BiSeNet
from face_dataset import FaceMask
from sklearn.metrics import f1_score


if __name__ == '__main__':

    model_dir = '/home/jihyun/workspace/face_parsing/face-parsing.PyTorch/res/exp'
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.eval()

    data_root = '/home/jihyun/workspace/face_parsing/dataset/CelebAMask-HQ/'
    cropsize = [448, 448]
    n_img_per_gpu = 16
    ds = FaceMask(data_root, cropsize=cropsize, mode='val')
    dl = DataLoader(ds, batch_size=16, shuffle=False, drop_last=True)

    f = open(os.path.join(model_dir, 'f_measure.txt'), 'w+')

    for filename in os.listdir(os.path.join(model_dir, 'cp')):

        if filename.endswith('70000_iter.pth'):

            net.load_state_dict(torch.load(os.path.join(model_dir, 'cp', filename)))

            total_f_score = []
            class_f_score = [[] for i in range(n_classes)]

            with torch.no_grad():
                        
                    for i, sample in enumerate(dl):
                        
                        if i == 5: break
                        if (i+1) % 5 == 0:
                            print('processing {}-th batch...'.format(str(i+1)))

                        im, lb = sample
                        im = im.cuda()
                        lb = lb.cuda()
                        lb = torch.squeeze(lb, 1)
                        out = net(im)[0]

                        pred = out.squeeze(0).argmax(1)
                        pred = pred.reshape(-1).cpu().numpy()
                        lb = lb.reshape(-1).cpu().numpy()

                        total_f_score.append(f1_score(lb, pred, average='weighted'))

                        for c in range(n_classes):
                            c_pred = pred == c
                            c_lb = lb == c
                            class_f_score[c].append(f1_score(c_lb, c_pred, average='binary'))
                            print(class_f_score[c])
                        
                        '''
                        # f-measure implementation
                        fs = []
                        for c in range(n_classes):
                            pred_pos = (pred == c).cpu().numpy()
                            true_pos = (lb == c).cpu().numpy()
                            
                            precision = np.sum(np.logical_and(pred_pos, true_pos)) / np.sum(pred_pos)
                            recall = np.sum(np.logical_and(pred_pos, true_pos)) / np.sum(true_pos)
                            
                            f_measure = (2 * precision * recall) / (precision + recall)
                            print(f_measure)

                            if not math.isnan(f_measure):
                                fs.append(np.sum(true_pos) * f_measure)

                        print(sum(fs) / (n_img_per_gpu * cropsize[0] * cropsize[1]))
                        '''

                    f.write(filename + ' | ')
                    f.write('total: ' + str(sum(total_f_score) / len(total_f_score)) + ' | ')

                    # range()로 인덱싱하기
                    for c in range(n_classes):
                        score = np.take(class_f_score, np.arange(c, len(class_f_score), n_classes))
                        print(np.arange(c, len(class_f_score), n_classes))

                    print(len(class_f_score))

            f.close()

