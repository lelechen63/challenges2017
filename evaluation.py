import numpy as np
from nibabel import load as load_nii
import os
from data_manipulation.metrics import dsc_seg
import argparse
def parse_arguments():
    """Parse arguments from command line"""
    description = "test"
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('--path', '-p',
        default="/media/lele/DATA/brain/Brats17TrainingData/HGG5/",
        help = 'path to the data'
        )
   
    return parser.parse_args()
args = parse_arguments()
def pixel_accuracy(eval_segm, gt_segm):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    sum_n_ii = 0
    sum_t_i  = 0

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i  += np.sum(curr_gt_mask)
 
    if (sum_t_i == 0):
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i

    return pixel_accuracy_

def mean_accuracy(eval_segm, gt_segm):
    '''
    (1/n_cl) sum_i(n_ii/t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    accuracy = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
 
        if (t_i != 0):
            accuracy[i] = n_ii / t_i

    mean_accuracy_ = np.mean(accuracy)
    return mean_accuracy_

def mean_IU(eval_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl   = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
 
        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)
 
    mean_IU_ = np.sum(IU) / n_cl_gt
    return mean_IU_

def frequency_weighted_IU(eval_segm, gt_segm):
    '''
    sum_k(t_k)^(-1) * sum_i((t_i*n_ii)/(t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    frequency_weighted_IU_ = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
 
        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i  = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        frequency_weighted_IU_[i] = (t_i * n_ii) / (t_i + n_ij - n_ii)
 
    sum_k_t_k = get_pixel_area(eval_segm)
    
    frequency_weighted_IU_ = np.sum(frequency_weighted_IU_) / sum_k_t_k
    return frequency_weighted_IU_

'''
Auxiliary functions used during evaluation.
'''
def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]

def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask   = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask

def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl

def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _   = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl

def extract_masks(segm, cl, n_cl):
    h, w  = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks

def segm_size(segm):
    try:
        height = segm.shape[0]
        width  = segm.shape[1]
    except IndexError:
        raise

    return height, width

def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")

'''
Exceptions
'''
class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)



def nii2np(path):
    result = load_nii(path).get_data()
    return result
def pixel_level_evaluation(path = '/media/lele/DATA/brain/Brats17TrainingData/HGG5/'):
    patients = os.listdir(path)
    print len(patients)
    pixel_accuracy_instance = []
    mean_IU_instance = []
    mean_accuracy_insintance = []
    frequency_weighted_IU_instance = []
    for patient in patients:
        if patient[0] != 'B':
            continue
        patient = path + patient  + '/'
        fs = os.listdir(patient)
        if len(fs) == 5:
            continue
        gt_path = patient
        seg_path = patient
        for f in fs:
            if f[-10:-7] == 'est' and '.e4.' in f:
                print f
                seg_path = patient + f

            elif f[-10:-7] =='seg':
                gt_path = gt_path + f

        seg_3d = nii2np(seg_path)
        gt_3d = nii2np(gt_path)
        pas = 0
        mas = 0
        mius = 0
        fwius = 0


        for i in range(seg_3d.shape[2]):
            img_gt = gt_3d[:,:,i]
            img_seg = seg_3d[:,:,i]
            pa = pixel_accuracy(img_seg,img_gt)
            pas += pa
            ma = mean_accuracy(img_seg,img_gt)
            mas += ma

            miu = mean_IU(img_seg,img_gt)
            mius += miu

            fwiu = frequency_weighted_IU(img_seg,img_gt)
            fwius += fwiu
        pas = pas / seg_3d.shape[2]
        mas = mas /seg_3d.shape[2]
        mius = mius/ seg_3d.shape[2]
        fwius = fwius / seg_3d.shape[2]
        pixel_accuracy_instance.append(pas)
        mean_accuracy_insintance.append(mas)
        mean_IU_instance.append(mius)
        frequency_weighted_IU_instance.append(fwius)
    pixel_accuracy_total = 0
    mean_accuracy_total = 0
    mean_IU_total = 0
    frequency_weighted_IU_total = 0
    for i in range(len(pixel_accuracy_instance)):
        pixel_accuracy_total += pixel_accuracy_instance[i]
        print pixel_accuracy_instance[i]
        print mean_accuracy_insintance[i]
        mean_accuracy_total += mean_accuracy_insintance[i]
        print mean_IU_instance[i]
        print frequency_weighted_IU_instance[i]
        mean_IU_total += mean_IU_instance[i]
        frequency_weighted_IU_total += frequency_weighted_IU_instance[i]
    ave_pa = pixel_accuracy_total /len(pixel_accuracy_instance)
    ave_ma = mean_accuracy_total / len(pixel_accuracy_instance)
    ave_miu = mean_IU_total /len(pixel_accuracy_instance)
    ave_fwiu = frequency_weighted_IU_total /len(pixel_accuracy_instance)

    print '=================='
    print ave_pa

    print ave_ma
    print ave_miu
    print ave_fwiu
# pixel_level_evaluation()

def label_level_evaluation(path = '/media/lele/DATA/brain/Brats17TrainingData/HGG5/'):
    patients = os.listdir(path)
    print len(patients)
    pixel_accuracy_instance = []
    mean_IU_instance = []
    mean_accuracy_insintance = []
    frequency_weighted_IU_instance = []
    label0 = 0
    label1 = 0
    label2 = 0
    label4 = 0
    dsc = []
    for patient in patients:
        if patient[0] != 'B':
            continue
        p_name  = patient
        patient = path + patient  + '/'
        fs = os.listdir(patient)
        if len(fs) == 5:
            continue
        gt_path = patient 
        seg_path = patient
        for f in fs:
            if f[-10:-7] == 'est':
                seg_path = seg_path + f

            elif f[-10:-7] =='seg':
                gt_path = gt_path + f
        seg_3d = nii2np(seg_path)
        gt_3d = nii2np(gt_path)
        labels = np.unique(gt_3d.flatten())
        results = (p_name,) + tuple([dsc_seg(gt_3d == l, seg_3d == l) for l in labels])
        text = 'Subject %s DSC: ' + '/'.join(['%f' for _ in labels[1:]])
        # print(text % results)
        if results[4]+ results[2] + results[3]  > 0.1:
            dsc.append(results[1:])
    for i in range(len(dsc)):
        label0 += dsc[i][0]
        label1 += dsc[i][1]
        label2 += dsc[i][2]
        label4 += dsc[i][3]
    label0 = label0 / len(dsc)
    label1 = label1 / len(dsc)
    label2 = label2 / len(dsc)
    label4 = label4 / len(dsc)
    print label0
    print label1
    print label2
    print label4
label_level_evaluation(args.path)