import h5py
import numpy as np
from PIL import Image


def loading_data(path):
    print '******************************************************'
    print 'dataset:{0}'.format(path)
    print '******************************************************'
    file = h5py.File(path)
    images = file['images'][:].transpose(0,3,2,1)
    labels = file['LAll'][:].transpose(1,0)
    tags = file['YAll'][:].transpose(1,0)
    file.close()
    return images, tags, labels


def split_data(images, tags, labels, QUERY_SIZE, TRAINING_SIZE, DATABASE_SIZE):

	X = {}
	index_all = np.random.permutation(QUERY_SIZE+DATABASE_SIZE)
	ind_Q = index_all[0:QUERY_SIZE]
	ind_T = index_all[QUERY_SIZE:TRAINING_SIZE + QUERY_SIZE]
	ind_R = index_all[QUERY_SIZE:DATABASE_SIZE + QUERY_SIZE]

	X['query'] = images[ind_Q, :, :, :]
	X['train'] = images[ind_T, :, :, :]
	X['retrieval'] = images[ind_R, :, :, :]

	Y = {}
	Y['query'] = tags[ind_Q, :]
	Y['train'] = tags[ind_T, :]
	Y['retrieval'] = tags[ind_R, :]

	L = {}
	L['query'] = labels[ind_Q, :]
	L['train'] = labels[ind_T, :]
	L['retrieval'] = labels[ind_R, :]
	return X, Y, L


class LoadData():
    def __init__(self, path, dataset='NUS-WIDE'):
        if 'NUS-WIDE' in dataset:
            seg = 'NUS-WIDE/'
            wepath = '/home/libsource/Cross_Modal_Datasets/NUS-WIDE/'
        else:
            print "Dataset:{0}".format(dataset)
        print '******************************************************'
        print 					'dataset:{0}'.format(path)
        print '******************************************************'

    def loadimg(self, pathList):
        crop_size = 224
        ImgSelect = np.ndarray([len(pathList), crop_size, crop_size, 3])
        count = 0
        for path in pathList:
            img = Image.open(path)
            xsize, ysize = img.size
            seldim = min(xsize, ysize)
            rate = 224.0 / seldim
            img = img.resize((int(xsize * rate), int(ysize * rate)))
            nxsize, nysize = img.size
            box = (nxsize / 2.0 - 112, nysize / 2.0 - 112, nxsize / 2.0 + 112, nysize / 2.0 + 112)
            img = img.crop(box)
            img = img.convert("RGB")
            img = img.resize((224, 224))
            img = np.array(img)
            if img.shape[2] != 3:
                print 'This image is not a rgb picture: {0}'.format(path)
                print 'The shape of this image is {0}'.format(img.shape)
                ImgSelect[count, :, :, :] = img[:, :, 0:3]
                count += 1
            else:
                ImgSelect[count, :, :, :] = img
                count += 1

        return ImgSelect