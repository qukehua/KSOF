import os
import sys

sys.path.append(os.path.abspath('./'))
from utils import h36motion3d as datasets
from model import GCNMLP as model
from utils.h36m_opt1 import Options
from utils import util
from utils import log

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
import torch.optim as optim

# import matplotlib.pyplot as plt
# import seaborn as sns
import math
from tqdm import tqdm


def main(opt):
	lr_now = opt.lr_now
	start_epoch = 1
	# opt.is_eval = True
	print('>>> create models')
	net_pred = model.GCN_TCN(opt=opt)

	net_pred.to(opt.cuda_idx)


	optimizer = optim.Adam(filter(lambda x: x.requires_grad, net_pred.parameters()), lr=opt.lr_now)
	optimizer = optim.Adam(filter(lambda x: x.requires_grad, net_pred.parameters()), lr=opt.lr_now)
	print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))
	# print(123 + "1")

	if opt.is_load or opt.is_eval:  # load ckpt
		if opt.is_eval:
			model_path_len = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
		else:
			model_path_len = './{}/ckpt_last.pth.tar'.format(opt.ckpt)
		print(">>> loading ckpt len from '{}'".format(model_path_len))
		ckpt = torch.load(model_path_len)
		start_epoch = ckpt['epoch'] + 1
		err_best = ckpt['err']
		lr_now = ckpt['lr']
		net_pred.load_state_dict(ckpt['state_dict'])

		print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))

	print('>>> loading datasets')

	if not opt.is_eval:
		dataset = datasets.Datasets(opt, split=0)
		print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
		data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True)
		valid_dataset = datasets.Datasets(opt, split=1)  # changed
		print('>>> Validation dataset length: {:d}'.format(valid_dataset.__len__()))
		valid_loader = DataLoader(valid_dataset, batch_size=opt.test_batch_size, shuffle=True, num_workers=0,
		                          pin_memory=True)

	test_dataset = datasets.Datasets(opt, split=1)  # changed
	print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
	test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
	                         pin_memory=True)

	# evaluation
	if opt.is_eval:
		ret_test, _, _ = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt)
		ret_log = np.array([])
		head = np.array([])
		for k in ret_test.keys():
			ret_log = np.append(ret_log, [ret_test[k]])
			head = np.append(head, [k])
		log.save_csv_log(opt, head, ret_log, is_create=True, file_name='test_walking')
	# print('testing error: {:.3f}'.format(ret_test['m_p3d_h36']))
	# training
	if not opt.is_eval:

		if opt.is_load:
			model_path_len1 = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
			ckpt1 = torch.load(model_path_len1)
			err_best = ckpt1['err']
		else:
			err_best = 10000

		for epo in range(start_epoch, opt.epoch + 1):
			is_best = False
			if epo % opt.lr_decay == 0:
			# lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (0.1 / opt.epoch))
				lr_now = util.lr_decay_mine(optimizer, lr_now, opt.lr_gamma)
			print('>>> training epoch: {:d}'.format(epo))

			ret_train, _, _ = run_model(net_pred, optimizer, is_train=0, data_loader=data_loader, epo=epo, opt=opt)
			print('train error: {:.3f}'.format(ret_train['m_p3d_h36']))

			ret_valid, _, _ = run_model(net_pred, is_train=1, data_loader=valid_loader, opt=opt, epo=epo)
			print('validation error: {:.3f}'.format(ret_valid['m_p3d_h36']))

			ret_test, _, _ = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt, epo=epo)
			# save logs

			ret_log = np.array([epo, lr_now])
			head = np.array(['epoch', 'lr'])
			for k in ret_train.keys():
				ret_log = np.append(ret_log, [ret_train[k]])
				head = np.append(head, [k])
			for k in ret_valid.keys():
				ret_log = np.append(ret_log, [ret_valid[k]])
				head = np.append(head, ['valid_' + k])
			for k in ret_test.keys():
				ret_log = np.append(ret_log, [ret_test[k]])
				head = np.append(head, ['test_' + k])
			log.save_csv_log(opt, head, ret_log, is_create=(epo == 1))
			if ret_valid['m_p3d_h36'] < err_best:
				err_best = ret_valid['m_p3d_h36']
				is_best = True
			log.save_ckpt({'epoch': epo,
			               'lr': lr_now,
			               'err': ret_valid['m_p3d_h36'],
			               'state_dict': net_pred.state_dict(),
			               'optimizer': optimizer.state_dict()},
			              is_best=is_best, opt=opt)


def eval(opt):
	lr_now = opt.lr_now
	start_epoch = 1
	print('>>> create models')
	net_pred = model.GCN_TCN(opt=opt)
	net_pred.to(opt.cuda_idx)
	net_pred.eval()

	# load model
	model_path_len = './{}/ckpt_best.pth.tar'.format(opt.ckpt)
	print(">>> loading ckpt len from '{}'".format(model_path_len))
	ckpt = torch.load(model_path_len)

	# save weight
	# for i in ckpt['state_dict'].keys():
	#     if 'attn.proj.weight' in i:
	#         weight = ckpt['state_dict'][i].detach().cpu().numpy()
	#         ax = sns.heatmap(weight).get_figure()
	#         path = os.path.join(i + '.png')
	#         ax.savefig(path)
	net_pred.load_state_dict(ckpt['state_dict'])

	print(">>> ckpt len loaded (epoch: {} | err: {})".format(ckpt['epoch'], ckpt['err']))

	acts = ["walking", "eating", "smoking", "discussion", "directions",
	        "greeting", "phoning", "posing", "purchases", "sitting",
	        "sittingdown", "takingphoto", "waiting", "walkingdog",
	        "walkingtogether"]
	# acts = ["walking", "eating", "smoking", "discussion", "directions",
	#         "phoning", "posing", "sitting",
	#         "sittingdown", "waiting",
	#         "walkingtogether"]
	data_loader = {}
	is_create = True
	for act in acts:
		dataset = datasets.Datasets(opt=opt, split=1, actions=act)  # changed
		data_loader[act] = DataLoader(dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0,
		                              pin_memory=True)
	avg_ret_log = []
	is_create = True
	for act in acts:
		ret_test, out_seq, out_tru = run_model(net_pred, is_train=3, data_loader=data_loader[act], opt=opt)
		ret_log = np.array([act])
		head = np.array(['action'])

		for k in ret_test.keys():
			ret_log = np.append(ret_log, [ret_test[k]])
			head = np.append(head, ['test_' + k])

		avg_ret_log.append(ret_log[1:])
		log.save_csv_eval_log(opt, head, ret_log, is_create=is_create)
		log.save_npy_log(opt, opt.predict_npy, value=out_seq, is_create=is_create, file_name=act)  
		log.save_npy_log(opt, opt.truth_npy, value=out_tru, is_create=is_create, file_name=act) 
		is_create = False

	avg_ret_log = np.array(avg_ret_log, dtype=np.float64)
	avg_ret_log = np.mean(avg_ret_log, axis=0)

	write_ret_log = ret_log.copy()
	write_ret_log[0] = 'avg'
	write_ret_log[1:] = avg_ret_log
	log.save_csv_eval_log(opt, head, write_ret_log, is_create=False)


def gen_velocity(m):
    input = [m[:, 0, :] - m[:, 0, :]]  

    for k in range(m.shape[1] - 1):
        input.append(m[:, k + 1, :] - m[:, k, :])
    input = torch.stack((input)).permute(1, 0, 2)  

    return input




def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None):
	if is_train == 0:
		net_pred.train()
	else:
		net_pred.eval()

	l_p3d = 0
	if is_train <= 1:
		m_p3d_h36 = 0
	else:
		titles = (np.array(range(opt.output_n)) + 1) * 40
		m_p3d_h36 = np.zeros([opt.output_n])
	n = 0
	in_n = opt.input_n
	out_n = opt.output_n
	dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
	                     26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
	                     46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
	                     75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
	seq_in = opt.input_n
	# joints at same loc
	joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
	index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
	joint_equal = np.array([13, 19, 22, 13, 27, 30])
	index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

	itera = 1
	# idx = np.expand_dims(np.arange(seq_in + out_n), axis=1) + (
	#         out_n - seq_in + np.expand_dims(np.arange(itera), axis=0))
	st = time.time()
	for i, (p3d_h36) in tqdm(enumerate(data_loader), total=len(data_loader)):
		# print(i)
		batch_size, seq_n, _ = p3d_h36.shape
		# when only one sample in this batch
		if batch_size == 1 and is_train == 0:
			continue
		n += batch_size
		p3d_h36 = p3d_h36.float().to(opt.cuda_idx)

		input = p3d_h36[:, :in_n, dim_used].clone()

		p3d_sup_4 = p3d_h36.clone()[:, :, dim_used][:, -out_n:].reshape([-1, out_n, len(dim_used) // 3, 3])
		p3d_sup_5 = p3d_h36.clone()[:, :, dim_used][:, -out_n:].reshape([-1, out_n, len(dim_used) // 3, 3])
		p3d_sup_6 = p3d_h36.clone()[:, :, dim_used][:, -out_n:].reshape([-1, out_n, len(dim_used) // 3, 3])
		pred_final, pred_tcn, pred_gcn = net_pred(input)

		p3d_out_4 = p3d_h36.clone()[:, in_n:in_n + out_n]
		p3d_out_5 = p3d_h36.clone()[:, in_n:in_n + out_n]
		p3d_out_6 = p3d_h36.clone()[:, in_n:in_n + out_n]
		pred_final[:, :, dim_used] = pred_final
		pred_tcn[:, :, dim_used] = pred_tcn
		pred_gcn[:, :, dim_used] = pred_gcn
		pred_final[:, :, index_to_ignore] = pred_final[:, :, index_to_equal]
		pred_tcn[:, :, index_to_ignore] = pred_tcn[:, :, index_to_equal]
		pred_gcn[:, :, index_to_ignore] = pred_gcn[:, :, index_to_equal]
		pred_final = pred_final.reshape([-1, out_n, 32, 3])
		pred_tcn = pred_tcn.reshape([-1, out_n, 32, 3])
		pred_gcn = pred_gcn.reshape([-1, out_n, 32, 3])
		out_seq_1 = pred_final.cpu().data.numpy()
		out_seq_2 = pred_tcn.cpu().data.numpy()
		out_seq_3 = pred_gcn.cpu().data.numpy()

		p3d_h36 = p3d_h36.reshape([-1, in_n + out_n, 32, 3])
		out_tru = p3d_h36[:, :]
		out_tru = out_tru.cpu().data.numpy()
		pred_final = pred_all.reshape([batch_size, out_n, len(dim_used) // 3, 3])
		pred_v1 = gen_velocity(pred_final[:,2:out_n,:,:])
		pred_v2 = gen_velocity(pred_final[:,3:out_n,:,:])
		if is_train == 0:  
			alpha = torch.tensor(0.5, requires_grad=True)
			beta = torch.tensor(0.5, requires_grad=True)
			loss_jps = torch.mean(torch.norm(pred_v1 - pred_v2, dim=3))
			loss_K = torch.mean(torch.norm(pred_tcn - p3d_sup_5, dim=3))
			loss_S= torch.mean(torch.norm(pred_gcn - p3d_sup_6, dim=3))
			loss_T = loss_K**2 + loss_jps**2
			loss_all = 2*alpha**2/loss_S + 2*beta**2/loss_T + math.log(alpha) +math.log(beta)

			optimizer.zero_grad()
			loss_all.backward()
			nn.utils.clip_grad_norm_(list(net_pred.parameters()), max_norm=opt.max_norm)  
			optimizer.step()



			# update log values
			l_p3d += loss_all.cpu().data.numpy() * batch_size
			alpha.data -= opt.lr_now * alpha.grad.data
			beta.data -= opt.lr_now * beta.grad.data
		if is_train <= 1:  
			mpjpe_p3d_h36 = torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out_4, dim=3))
			m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy() * batch_size

		else:  # 测试
			mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out_4, dim=3), dim=2), dim=0)
			m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()

	ret = {}
	if is_train == 0:
		ret["l_p3d"] = l_p3d / n

	if is_train <= 1:
		ret["m_p3d_h36"] = m_p3d_h36 / n
	else:
		m_p3d_h36 = m_p3d_h36 / n
		for j in range(out_n):
			ret["#{:d}ms".format(titles[j])] = m_p3d_h36[j]
	return ret, out_seq_1, out_seq_2, out_seq_3, out_tru


if __name__ == '__main__':

	option = Options().parse()

	# option.is_load = True
	option.is_eval = True

	if not option.is_eval:
		main(opt=option)
	else:
		eval(option)
