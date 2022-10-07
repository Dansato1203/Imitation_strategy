import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import pickle
import argparse

def return_task(keys):
	# 実行時に引数のいらないもの
	if keys in [0, 1, 2, 3, 5, 6, 14, 15, 19]:
		task_dict = {
			0 : "actionbase.bodymotion.Idling",
			1 : "actionbase.bodymotion.IdleKickoff",
			2 : "IdleEnemyFreeKick", # いらないかも
			3 : "actionbase.search.FindLandmarksToLocalize",
			5 : "actionbase.search.SearchCloseBall",
			6 : "actionbase.search.SearchFarBall",
			14 : "kid.action.defenderaction.TrackBallForDefender",            
			15 : "actionbase.search.TrackBall",
			19 : None
			}
		return task_dict[keys]
	# 実行時に引数のいるもの
	elif keys in [4, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18]:
		task_dict = {
			4 : "actionbase.search.TurnAndSearchCloseBall",
			7 : "kid.action.approach.ApproachBallForGankenKun",
			8 : "kid.action.approach.ApproachFreeKickTargetPos",
			9 : "kid.action.approach.TurnAroundBallToTarget",
			10 : "kid.action.approach.DribbleToTarget",
			11 : "kid.action.approach.AdjustToKickPosForGankenKun",
			12 : "actionbase.bodymotion.KickBallToTarget",
			13 : "kid.action.defenderaction.MoveToDefensePos",
			16 : "kid.action.neutralaction.WalkStraight",
			17 : "kid.action.approach.ApproachTargetPosForGankenKun",
			18 : "kid.action.defenderaction.ApproachTargetPos"
			}
		return task_dict[keys]

def switch_csv(role):
	csv_dir = "../csvfiles/"
	model_dir = "../models/"
	if role == 'Attacker':
		role_csv = csv_dir + "edit_Attacker.csv"
		model = model_dir + "221007_imitation_Attacker.pkl"
		features = ["K_BEHIND_BALL_DEFENSIVE_LINE", "K_KNOW_BALL_POS", "K_KNOW_SELF_POS", "K_COME_BALL", "K_TARGET_IN_SHOT_RANGE", "K_ON_TARGET_POS", "K_BALL_AND_TARGET_ON_STRAIGHT_LINE", "K_HAVE_BALL", "K_BALL_IN_TARGET", "K_BALL_IN_KICK_AREA", "K_ENEMY_WITHIN_RANGE", "K_IDLE", "K_TRACKING_BALL", "K_PERMITTED_INTRUSION_CIRCLE", "K_SECONDARY_STATE_FREEZE", "K_SECONDARY_STATE_DIRECT_FREEKICK", "K_SECONDARY_STATE_KICK_TEAM_OWN", "K_SIDE_ENTRY_STATE", "K_ON_OUR_FIELD", "PREV_TASK"]
	elif role == 'Neutral':
		role_csv = csv_dir + "Neutral.csv"
		model = model_dir + "Neutral_model.pkl"
		features = ["K_BEHIND_BALL_DEFENSIVE_LINE", "K_KNOW_BALL_POS", "K_KNOW_SELF_POS" ,"K_COME_BALL", "K_TARGET_IN_SHOT_RANGE", "K_ON_TARGET_POS", "K_BALL_AND_TARGET_ON_STRAIGHT_LINE", "K_HAVE_BALL", "K_BALL_IN_TARGET", "K_BALL_IN_KICK_AREA", "K_ENEMY_WITHIN_RANGE", "K_IDLE", "K_TRACKING_BALL", "K_PERMITTED_INTRUSION_CIRCLE", "K_SECONDARY_STATE_FREEZE", "K_SECONDARY_STATE_DIRECT_FREEKICK", "K_SECONDARY_STATE_KICK_TEAM_OWN", "K_SIDE_ENTRY_STATE", "K_ON_OUR_FIELD", "K_GAME_STATE_READY", "K_GAME_STATE_SET", "K_GAME_STATE_PAUSED", "K_GAME_STATE_READY_TO_ENTER", "K_EXPLORE_AREA", "PREV_TASK"]
	elif role == 'Defender':
		role_csv = csv_dir + "Defender_csv"
		model = model_dir + "Defender_model.pkl"
		features = ["K_BEHIND_BALL_DEFENSIVE_LINE", "K_KNOW_BALL_POS", "K_KNOW_SELF_POS", "K_COME_BALL", "K_TARGET_IN_SHOT_RANGE", "K_ON_TARGET_POS", "K_BALL_AND_TARGET_ON_STRAIGHT_LINE", "K_HAVE_BALL", "K_BALL_IN_TARGET", "K_BALL_IN_KICK_AREA", "K_ENEMY_WITHIN_RANGE", "K_IDLE", "K_TRACKING_BALL", "K_PERMITTED_INTRUSION_CIRCLE", "K_SECONDARY_STATE_FREEZE", "K_SECONDARY_STATE_DIRECT_FREEKICK", "K_SECONDARY_STATE_KICK_TEAM_OWN", "K_SIDE_ENTRY_STATE", "K_ON_OUR_FIELD", "K_DEFENSE", "K_OBSERVE_BALL", "K_ON_DEFENSE_POS", "PREV_TASK"]

	return role_csv, model, features

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-r', '--role', type=str, default=None)
	args = parser.parse_args()

	csv , model, features = switch_csv(args.role)
	data_frame = pd.read_csv(csv)

	loaded_rf_model = pickle.load(open(model, 'rb'))

	X = data_frame[features]
	y = data_frame["TASK"]
	print(f"X : {X}")
	print(f"y : {y}")

	y_pred = loaded_rf_model.predict(X)
	print(f"y_pred : {y_pred}")
	#print(f"task : {return_task(int(y_pred))}")

	#print(sum(y_pred == y) / len(y))
	print(f"acc : {accuracy_score(y, y_pred)}")

	for i in range(len(y)):
		if y[i] != y_pred[i]:
			print(f"y, y_pred : {return_task(y[i])}, {return_task(y_pred[i])}")

if __name__ == '__main__':
	main()
