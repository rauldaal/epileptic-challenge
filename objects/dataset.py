import cv2
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch


class EpilepticDataset(Dataset):
	def __init__(self, parquet_folder, numpy_folder, transform):
		self.folder_parquet = parquet_folder
		self.folder_numpy = numpy_folder
		self.data = None
		self.numpy_data = {}
		self.transforms = None
		self.init()
	
	def __format_filename(self, string):
		sufix = string[3:]
		if sufix.isdigit():
			return string
		else:
			indice = 0
			while indice < len(sufix) and not sufix[indice].isdigit():
				indice += 1
			return string[:3] + sufix[:indice]
 
	
	def init(self):
		parquet_files = os.listdir(self.folder_parquet)
		patients_files = [pf.split("_")[0]+"_seizure_EEGwindow_1.npz" for pf in parquet_files]
		i=0
		for parquet in parquet_files:
			print(i)
			i+=1
			df = pd.read_parquet(os.path.join(self.folder_parquet, parquet))
			df['window_id'] = df.index
			df['filename'] = df['filename'].apply(lambda x: self.__format_filename(x.split("_")[0]))
			if self.data is None:
				self.data = df
			else:
				self.data = pd.concat([self.data, df])
		print("PARQUET DONE... READING NUMPY")
		i=0
		for npz in patients_files:
			print(i)
			i+=1
			with np.load(os.path.join(self.folder_numpy, npz), mmap_mode='r',allow_pickle=True) as data:
				name = data.files[0]
				npy_data = data[name]
				self.numpy_data[npz.split("_")[0]] = npy_data

	def __getitem__(self, idx):
		row = self.data.iloc[idx]
		id, window_id, cls = row['filename'], row['window_id'], row['class']
		window = self.numpy_data[id][window_id]
		window = window.astype(np.float32)
		window = torch.from_numpy(window)  # Convertir a tensor
		cls = torch.tensor(cls, dtype=torch.float)  # Convertir a tensor
		return window, cls

	def __len__(self):
		return len(self.data)
