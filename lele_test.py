from nibabel import load as load_nii
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

np.set_printoptions(threshold=np.nan)


def get_mask_voxels(mask):
    indices = np.stack(np.nonzero(mask), axis=1)
    indices = [tuple(idx) for idx in indices]
    return indices

def test():
	path = '/media/lele/DATA/brain/Brats17TrainingData/HGG_test/Brats17_2013_10_1/'
	gt = path + 'Brats17_2013_10_1_seg.nii.gz'
	roi_nii = load_nii(gt)
	roi = roi_nii.get_data().astype(dtype=np.bool)
	centers = get_mask_voxels(roi)
	test_samples = np.count_nonzero(roi)
	image = np.zeros_like(roi).astype(dtype=np.uint8)
	# print image
	print test_samples
	print image.shape

#test()
def test_net():
	path = '/media/lele/DATA/brain/Brats17TrainingData/HGG5/Brats17_TCIA_335_1/deep-brats17.D500.f.p13.c3c3c3c3c3.n32n32n32n32n32.d256.e4.pad_valid.test.nii.gz'
	gt = '/media/lele/DATA/brain/Brats17TrainingData/HGG5/Brats17_TCIA_335_1/Brats17_TCIA_335_1_seg.nii.gz'
	output = load_nii(path).get_data()
	gt = load_nii(gt).get_data()
	print output.shape
	print np.count_nonzero(gt  == 1)
	print np.count_nonzero(output == 1)
	print np.count_nonzero(gt  == 2)
	print np.count_nonzero(output == 2)
	print np.count_nonzero(gt  == 4)
	print np.count_nonzero(output == 4)


	gt1 = gt[:,:,90]
	plt.imshow(gt1, cmap='Set2')
	plt.show()


	output1 = output[:,:,90]
	plt.imshow(output1,cmap='Set2')
	plt.show()

test_net()