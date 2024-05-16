"""
predict_model

"""
import torch
from torch.autograd import Variable
import numpy as np
import cv2
import os
import argparse
from models.hat_arch import HAT
from numpy.random import RandomState
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",default = 'demo',type=str)
    parser.add_argument("--input_dir",default = r'../dataset/train/R100H/test/data_r1/', type=str)#100 picture
    parser.add_argument("--output_dir", default = r'./test_results/Rain100H/epoch150/',type=str)
    parser.add_argument("--output_dir1", default = r'./test_results/Rain100L/epoch150_Att/',type=str)
    parser.add_argument("--output_dir2", default = r'./test_results/Rain800/epoch150_100/',type=str)
    args = parser.parse_args()
    return args

def align_to_four(img):
    a_row = int(img.shape[0]/4)*4
    a_col = int(img.shape[1]/4)*4
    img = img[0:a_row, 0:a_col]
    return img




def predict(image,patch = False):

    image = np.array(image, dtype='float32')/255.
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :]
    image = torch.from_numpy(image)
    image = Variable(image)

    bs, c, h, w = image.shape
    if patch == True:
        rand_state = RandomState(66)
        h_patch_size = int(h/1.50)
        w_patch_size = int(w/1.50)

        r = rand_state.randint(0, h - h_patch_size)
        c = rand_state.randint(0, w - w_patch_size)

        image = image[:, :, r: r+h_patch_size, c: c+w_patch_size]

    with torch.no_grad():

        out_gan, Attention1 =  net_H(image.cuda())
        print('out_gan',out_gan.size())
        print('Attention1',Attention1.size())


    out_gan = out_gan.cpu().data
    out_gan = out_gan.numpy()
    out_gan = out_gan.transpose((0, 2, 3, 1))
    out_gan = out_gan[0, :, :, :]*255.
    
    Attention1 = Attention1.cpu()
    Attention1 = Attention1.numpy()
    Attention1 = Attention1.transpose((0, 2, 3, 1))
    Attention1 = Attention1[0, :, :, :]*255.

    return out_gan, Attention1


if __name__ == '__main__':
    args = get_args()


    H_path = r'./checkpoint/derain/HAT_150/hat_para.pth'
    net_H = HAT(overlap_ratio=0,upscale=1,upsampler='pixelshuffle',drop_path_rate=0.0).cuda()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(H_path, map_location=device)
    from collections import OrderedDict
    state_dict_new = OrderedDict()
    for k, v in state_dict.items():

        name = k[7:]
        state_dict_new[name] = v


    net_H.load_state_dict(state_dict_new)
    net_H.eval()

    if args.mode == 'demo':
        input_list = sorted(os.listdir(args.input_dir))
        num = len(input_list)
        for i in range(num):

            img = cv2.imread(args.input_dir + input_list[i])
            img = align_to_four(img)
            out_gan, Attention1 = predict(img,patch = False)

            img_name = input_list[i].split('.')[0]

            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            cv2.imwrite(args.output_dir1 + img_name + 'att.png', Attention1)
            cv2.imwrite(args.output_dir2 + img_name + '_derain.png', out_gan)
            print(",",args.output_dir)
    else:
        print ('Mode Invalid!')
