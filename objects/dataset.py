import cv2
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class EpilepticDataset(Dataset):
	def __init__(self, parquet_folder, numpy_folder, transform):
		self.folder_parquet = parquet_folder
		self.folder_numpy = numpy_folder
		self.data = None
		self.numpy_data = {}
		self.transforms = None
		self.init()
	
	def init(self):
		parquet_files = os.listdir(self.folder_parquet)
		patients_files = [pf.split("_")[0]+"_seizure_EEGwindow_1.npz" for pf in parquet_files]
		for parquet in parquet_files:
			df = pd.read_parquet(os.path.join(self.folder_parquet, parquet))
			df['window_id'] = df.index
			df['filename'] = df['filename'].apply(lambda x: x.split("_")[0])
			if self.data is None:
				self.data = df
			else:
				self.data = pd.concat([self.data, df])
		for npz in patients_files:
			with np.load(os.path.join(self.folder_numpy, npz), allow_pickle=True) as data:
				name = data.files[0]
				npy_data = data[name]
				self.numpy_data[npz.split("_")[0]] = npy_data

	def __getitem__(self, idx):
		row = self.data.iloc[idx]
		id, window_id, class_ = row['filename'], row['window_id'], row['class']
		window = self.numpy_data[id][window_id]
		return window, class_
	
	def __len__(self):
		return len(self.data)
