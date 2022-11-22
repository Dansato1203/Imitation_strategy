import pandas as pd
import numpy as np
import csv
import os
import argparse
import glob
from sklearn.preprocessing import MinMaxScaler

class Preprocess(object):
	def __init__(self, csv_dir):
		self.csv_dir = csv_dir

	def read_csv(self, target_csv_dir):
		df_dict = {}
		csvfiles = glob.glob(target_csv_dir + '/' +  '*.csv')
		print(csvfiles)
		for idx, f in enumerate(csvfiles):
			exec(f'df_{idx} = pd.read_csv(f)')
			print(f"df_{idx} : {f}")
			exec(f'df_dict[idx] = df_{idx}')

		return df_dict

	def save_csv(self, data_frame, csv_name):
		data_frame.to_csv(csv_name, index=False)

	def merge_csv(self, csvfiles_dict):
		for i in range(len(csvfiles_dict)):
			if i == 0:
				df1 = csvfiles_dict[0]
				df2 = csvfiles_dict[1]
			elif i == 1:
				continue
			else:
				df1 = df_merged.copy()
				df2 = csvfiles_dict[i]

			df_merged = pd.concat([df1, df2], axis=0)

		return df_merged

	def OneHotEncoding(self, data_frame, target): # TODO 列名が違う順番で生成される可能性があるので対処する
		target_data = data_frame[target]
		one_hot_data = pd.get_dummies(target_data, drop_first=True)
		target_column_idx = data_frame.columns.get_loc(target)
		one_hot_columns = one_hot_data.columns.to_numpy()
		exception_target = ['other_robot_1_role', 'other_robot_2_role', 'other_robot_3_role']

		for idx, columns_name in enumerate(one_hot_columns):
			if target in exception_target:
				new_columns = "other_robot_" + str(exception_target.index(target)+1) +  "_" + columns_name
				data_frame.insert(target_column_idx + idx, new_columns, one_hot_data.values[:, idx])
			else:
				print(columns_name)
				data_frame.insert(target_column_idx + idx, columns_name, one_hot_data.values[:, idx])

		data_frame.drop(columns=target, axis=1, inplace=True)

	def TaskName2Num(self, data_frame):
		data_frame.replace("Idling",                        0, inplace=True)
		data_frame.replace("IdleKickoff",                    1, inplace=True)
		data_frame.replace("IdleEnemyFreeKick",              2, inplace=True)
		data_frame.replace("FindLandmarksToLocalize",        3, inplace=True)
		data_frame.replace("TurnAndSearchCloseBall",         4, inplace=True)
		data_frame.replace("SearchCloseBall",                5, inplace=True)
		data_frame.replace("SearchFarBall",                  6, inplace=True)
		data_frame.replace("ApproachBallForGankenKun",       7, inplace=True)
		data_frame.replace("ApproachFreeKickTargetPos",      8, inplace=True)
		data_frame.replace("TurnAroundBallToTarget",         9, inplace=True)
		data_frame.replace("DribbleToTarget",               10, inplace=True)
		data_frame.replace("AdjustToKickPosForGankenKun",   11, inplace=True)
		data_frame.replace("KickBallToTarget",              12, inplace=True)
		data_frame.replace("MoveToDefensePos",              13, inplace=True)
		data_frame.replace("TrackBallForDefender",          14, inplace=True)
		data_frame.replace("TrackBall",                     15, inplace=True)
		data_frame.replace("WalkStraight",                  16, inplace=True)
		data_frame.replace("ApproachTargetPosForGankenKun", 17, inplace=True)
		data_frame.replace("ApproachTargetPos",             18, inplace=True)

		data_frame.replace("None",                          19, inplace=True)

	def Str2Int(self, data_frame):
		for i in range(len(data_frame)):
			print(len(data_frame.columns))
			for j in range(len(data_frame.columns)):
				tmp = data_frame.iloc[i, j]
				print(f"[{i}, {j}] tmp: {tmp}")
				data_frame.iloc[i, j] = int(tmp)
				print(type(data_frame.iloc[i, j]))

	def insert_dummyData(self, data_frame):
		df_copy = data_frame.copy()

		#dummy_data = np.array([[100000,100000,0,0,10000,10000,0,0,10,0,100000,100000,0,0,0,1,0,100000,100000,0,0,0,0,0,100000,100000,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,19]])
		dummy_data = np.array([[100000, 100000, 0, 0, 10000, 10000, "No-Role", 10, 0, 100000, 100000, 0, "None", 0, 100000, 100000, 0, "None", 0, 100000, 100000, 0, "None", "None", "None"]])
		df_dummy = pd.DataFrame(dummy_data, columns=df_copy.columns)

		insert_num = 4
		for i in range(insert_num):
			df_copy = pd.concat([df_dummy, df_copy])

		return df_copy

	def normalize(self, data_frame):
		target_df = data_frame.copy()
		target_df.drop(columns='TASK', axis=1, inplace=True)
		mm = MinMaxScaler()
		scaled_df = pd.DataFrame(mm.fit_transform(target_df), index=target_df.index, columns=target_df.columns)

		output_df = pd.concat([scaled_df, data_frame['TASK']], axis=1)
		return output_df

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--csvfiles_dir', type=str, default=None)
	parser.add_argument('-m', '--merge', action='store_true')
	parser.add_argument('-r', '--revise', action='store_true')
	parser.add_argument('-e', '--encoding', action='store_true')
	parser.add_argument('-i', '--insert', action='store_true')
	parser.add_argument('-n', '--normalize', action='store_true')
	parser.add_argument('--all', action='store_true')
	args = parser.parse_args()

	preprocess = Preprocess(None)
	csvfiles_dict = preprocess.read_csv(args.csvfiles_dir)

	if args.merge:
		import datetime
		merge_dir = args.csvfiles_dir + '/merge'
		os.makedirs(merge_dir, exist_ok=True)
		date = datetime.datetime.now()
		merge_df = preprocess.merge_csv(csvfiles_dict)
		preprocess.save_csv(merge_df, merge_dir + '/' + date.strftime("%y%m%d%H") + '-merge.csv')

	elif args.revise:
		for i in csvfiles_dict:
			target_dataframe = csvfiles_dict[i].copy()
			preprocess.TaskName2Num(target_dataframe)
			preprocess.Str2Int(target_dataframe)

	elif args.encoding:
		for i in csvfiles_dict:
			encode_target = ['role', 'PREV_TASK', 'other_robot_1_role', 'other_robot_2_role', 'other_robot_3_role']
			target_dataframe = csvfiles_dict[i].copy()

			for i, target in enumerate(encode_target):
				preprocess.OneHotEncoding(target_dataframe, target)
			preprocess.TaskName2Num(target_dataframe)
			preprocess.save_csv(target_dataframe, 'test.csv')

	elif args.insert:
		for i in csvfiles_dict:
			target_dataframe = csvfiles_dict[i].copy()
			insert_df = preprocess.insert_dummyData(target_dataframe)
			preprocess.save_csv(insert_df, 'insert_test.csv')

	elif args.normalize:
		for i in csvfiles_dict:
			target_dataframe = csvfiles_dict[i].copy()
			normalize_df = preprocess.normalize(target_dataframe)
			preprocess.save_csv(normalize_df, 'normalize_test.csv')

	elif args.all:
		processed_dir = args.csvfiles_dir + '/processed'
		os.makedirs(processed_dir, exist_ok=True)

		insert_dir = processed_dir + '/insert'
		os.makedirs(insert_dir, exist_ok=True)

		# Insert Dummy data process
		for i in csvfiles_dict:
			target_dataframe = csvfiles_dict[i].copy()
			insert_df = preprocess.insert_dummyData(target_dataframe)
			preprocess.save_csv(insert_df, insert_dir + '/' + 'insert_' + str(i) + '.csv')

		# Merge csvfile process
		merge_dir = processed_dir + '/merge'
		os.makedirs(merge_dir, exist_ok=True)

		merge_csvfiles_dict = preprocess.read_csv(insert_dir)
		merge_df = preprocess.merge_csv(merge_csvfiles_dict)
		preprocess.save_csv(merge_df, merge_dir + '/' + 'merge.csv')

		# Encode csvfiles process
		encode_dir = processed_dir + '/encode'
		os.makedirs(encode_dir, exist_ok=True)
		
		encode_csvfiles_dict = preprocess.read_csv(merge_dir)
		for i in encode_csvfiles_dict:
			encode_target = ['role', 'PREV_TASK', 'other_robot_1_role', 'other_robot_2_role', 'other_robot_3_role']
			target_dataframe = encode_csvfiles_dict[i].copy()

			for idx, target in enumerate(encode_target):
				preprocess.OneHotEncoding(target_dataframe, target)
			preprocess.TaskName2Num(target_dataframe)
			preprocess.save_csv(target_dataframe, encode_dir + '/' + 'encode.csv')

		# Normalize data process
		normalize_dir = processed_dir + '/normalize'
		os.makedirs(normalize_dir, exist_ok=True)

		normalize_csvfiles_dict = preprocess.read_csv(encode_dir)
		for i in normalize_csvfiles_dict:
			target_dataframe = normalize_csvfiles_dict[i].copy()
			normalize_df = preprocess.normalize(target_dataframe)
			preprocess.save_csv(normalize_df, normalize_dir + '/' + 'normalize.csv')

		# End process
		preprocess.save_csv(normalize_df, processed_dir + '/' +  'processed.csv')

if __name__ == '__main__':
	main()
