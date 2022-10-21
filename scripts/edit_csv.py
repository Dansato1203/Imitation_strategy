import pandas as pd
import numpy as np
import csv
import argparse

class EditCSV(object):
	def __init__(self, csv1, csv2):
		self.csv1 = csv1
		self.csv2 = csv2
		self.df1 = None
		self.df2 = None

	def read_CSV(self):
		self.df1 = pd.read_csv(self.csv1)
		if self.csv2:
			self.df2 = pd.read_csv(self.csv2)
			return self.df1, self.df2
		return self.df1

	def save_CSV(self, data_frame, csv_name):
		data_frame.to_csv(csv_name, index=False)

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

	def TaskName2Num(self, df):
		df.replace("Idling",                         0, inplace=True)
		df.replace("IdleKickoff",                    1, inplace=True)
		df.replace("IdleEnemyFreeKick",              2, inplace=True)
		df.replace("FindLandmarksToLocalize",        3, inplace=True)
		df.replace("TurnAndSearchCloseBall",         4, inplace=True)
		df.replace("SearchCloseBall",                5, inplace=True)
		df.replace("SearchFarBall",                  6, inplace=True)
		df.replace("ApproachBallForGankenKun",       7, inplace=True)
		df.replace("ApproachFreeKickTargetPos",      8, inplace=True)
		df.replace("TurnAroundBallToTarget",         9, inplace=True)
		df.replace("DribbleToTarget",               10, inplace=True)
		df.replace("AdjustToKickPosForGankenKun",   11, inplace=True)
		df.replace("KickBallToTarget",              12, inplace=True)
		df.replace("MoveToDefensePos",              13, inplace=True)
		df.replace("TrackBallForDefender",          14, inplace=True)
		df.replace("TrackBall",                     15, inplace=True)
		df.replace("WalkStraight",                  16, inplace=True)
		df.replace("ApproachTargetPosForGankenKun", 17, inplace=True)
		df.replace("ApproachTargetPos",             18, inplace=True)

		df.replace("None",                          19, inplace=True)

	def Str2Int(self):
		df = pd.read_csv(self.csv1)
		for i in range(len(df)):
			print(len(df.columns))
			for j in range(len(df.columns)):
				tmp = df.iloc[i, j]
				print(f"[{i}, {j}] tmp: {tmp}")
				df.iloc[i, j] = int(tmp)
				print(type(df.iloc[i, j]))

	def mergeCSV(self):
		import datetime

		df1 = pd.read_csv(self.csv1, header=0)
		df2 = pd.read_csv(self.csv2, header=0)

		df_merged = pd.concat([df1, df2], axis=0)

		date = datetime.datetime.now()
		df_merged.to_csv('../csvfiles/merged/' + date.strftime("%y%m%d%H")  +'-merged.csv', index = False)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-c1', '--csvfile1', type=str, default=None)
	parser.add_argument('-c2', '--csvfile2', type=str, default=None)
	parser.add_argument('-m', '--merge', action='store_true')
	parser.add_argument('-r', '--revise', action='store_true')
	parser.add_argument('-e', '--encoding', action='store_true')
	args = parser.parse_args()

	editcsv = EditCSV(args.csvfile1, args.csvfile2)
	if args.merge:
		editcsv.mergeCSV()
	elif args.revise:
		editcsv.string2num()
		editcsv.str2int()
	elif args.encoding:
		df = editcsv.read_CSV()
		encode_target = ['role', 'PREV_TASK', 'other_robot_1_role', 'other_robot_2_role', 'other_robot_3_role']

		for i, target in enumerate(encode_target):
			editcsv.OneHotEncoding(df, target)

		editcsv.TaskName2Num(df)

		editcsv.save_CSV(df, 'test.csv')

if __name__ == '__main__':
	main()
