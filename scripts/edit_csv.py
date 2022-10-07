import pandas as pd
import csv
import argparse

class EditCSV(object):
	def __init__(self, csv1, csv2):
		self.csv1 = csv1
		self.csv2 = csv2

	def readCSV(self):
		pass

	def string2num(self):
		with open(self.csv1, "r") as f:
			s = f.read()

		s = s.replace("Idling", "0")
		s = s.replace("IdleKickoff", "1")
		s = s.replace("IdleEnemyFreeKick", "2")
		s = s.replace("FindLandmarksToLocalize", "3")
		s = s.replace("TurnAndSearchCloseBall", "4")
		s = s.replace("SearchCloseBall", "5")
		s = s.replace("SearchFarBall", "6")
		s = s.replace("ApproachBallForGankenKun", "7")
		s = s.replace("ApproachFreeKickTargetPos", "8")
		s = s.replace("TurnAroundBallToTarget", "9")
		s = s.replace("DribbleToTarget", "10")
		s = s.replace("AdjustToKickPosForGankenKun", "11")
		s = s.replace("KickBallToTarget", "12")
		s = s.replace("MoveToDefensePos", "13")
		s = s.replace("TrackBallForDefender", "14")
		s = s.replace("TrackBall", "15")
		s = s.replace("WalkStraight", "16")
		s = s.replace("ApproachTargetPosForGankenKun", "17")
		s = s.replace("ApproachTargetPos", "18")

		s = s.replace("None", "19")

		s = s.replace("True", "1")
		s = s.replace("False", "0")

		with open(self.csv1, "w") as f:
			f.write(s)

	def str2int(self):
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

		data_list = []
		df1 = pd.read_csv(self.csv1, header=0)
		data_list.append(df1)
		df2 = pd.read_csv(self.csv2, header=0)
		data_list.append(df2)

		df_merged = pd.concat(data_list)

		date = datetime.datetime.now()
		df_merged.to_csv('../csvfiles/merged/' + date.strftime("%y%m%d%H")  +'-merged.csv', index = False)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-c1', '--csvfile1', type=str, default=None)
	parser.add_argument('-c2', '--csvfile2', type=str, default=None)
	parser.add_argument('-m', '--merge', action='store_true')
	parser.add_argument('-r', '--revise', action='store_true')
	args = parser.parse_args()

	editcsv = EditCSV(args.csvfile1, args.csvfile2)
	if args.merge:
		editcsv.mergeCSV()
	elif args.revise:
		editcsv.string2num()
		editcsv.str2int()

if __name__ == '__main__':
	main()
