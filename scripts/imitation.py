import os
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import datetime
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score

import pickle

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-r', '--role', type=str, default=None)
	args = parser.parse_args()

	data_frame = pd.read_csv('../csvfiles/edit_' + args.role + '.csv')

	if args.role == 'Attacker':
		features = ["K_BEHIND_BALL_DEFENSIVE_LINE", "K_KNOW_BALL_POS", "K_KNOW_SELF_POS", "K_COME_BALL", "K_TARGET_IN_SHOT_RANGE", "K_ON_TARGET_POS", "K_BALL_AND_TARGET_ON_STRAIGHT_LINE", "K_HAVE_BALL", "K_BALL_IN_TARGET", "K_BALL_IN_KICK_AREA", "K_ENEMY_WITHIN_RANGE", "K_IDLE", "K_TRACKING_BALL", "K_PERMITTED_INTRUSION_CIRCLE", "K_SECONDARY_STATE_FREEZE", "K_SECONDARY_STATE_DIRECT_FREEKICK", "K_SECONDARY_STATE_KICK_TEAM_OWN", "K_SIDE_ENTRY_STATE", "K_ON_OUR_FIELD", "PREV_TASK"]
	elif args.role == 'Neutral':
		features = ["K_BEHIND_BALL_DEFENSIVE_LINE", "K_KNOW_BALL_POS", "K_KNOW_SELF_POS" ,"K_COME_BALL", "K_TARGET_IN_SHOT_RANGE", "K_ON_TARGET_POS", "K_BALL_AND_TARGET_ON_STRAIGHT_LINE", "K_HAVE_BALL", "K_BALL_IN_TARGET", "K_BALL_IN_KICK_AREA", "K_ENEMY_WITHIN_RANGE", "K_IDLE", "K_TRACKING_BALL", "K_PERMITTED_INTRUSION_CIRCLE", "K_SECONDARY_STATE_FREEZE", "K_SECONDARY_STATE_DIRECT_FREEKICK", "K_SECONDARY_STATE_KICK_TEAM_OWN", "K_SIDE_ENTRY_STATE", "K_ON_OUR_FIELD", "K_GAME_STATE_READY", "K_GAME_STATE_SET", "K_GAME_STATE_PAUSED", "K_GAME_STATE_READY_TO_ENTER", "K_EXPLORE_AREA", "PREV_TASK"]
	elif args.role == 'Defender':
		features = ["K_BEHIND_BALL_DEFENSIVE_LINE", "K_KNOW_BALL_POS", "K_KNOW_SELF_POS", "K_COME_BALL", "K_TARGET_IN_SHOT_RANGE", "K_ON_TARGET_POS", "K_BALL_AND_TARGET_ON_STRAIGHT_LINE", "K_HAVE_BALL", "K_BALL_IN_TARGET", "K_BALL_IN_KICK_AREA", "K_ENEMY_WITHIN_RANGE", "K_IDLE", "K_TRACKING_BALL", "K_PERMITTED_INTRUSION_CIRCLE", "K_SECONDARY_STATE_FREEZE", "K_SECONDARY_STATE_DIRECT_FREEKICK", "K_SECONDARY_STATE_KICK_TEAM_OWN", "K_SIDE_ENTRY_STATE", "K_ON_OUR_FIELD", "K_DEFENSE", "K_OBSERVE_BALL", "K_ON_DEFENSE_POS", "PREV_TASK"]

	X = data_frame[features]
	y = data_frame["TASK"]

	train_X, val_X, train_y, val_y = train_test_split(X, y)

	model = DecisionTreeClassifier(criterion='gini')

	model.fit(train_X, train_y)

	val_predictions = model.predict(val_X)

	val_mae = mean_absolute_error(val_predictions, val_y)
	print(f"正答率: {sum(val_predictions == val_y) / len(val_y)}")

	plt.figure(figsize=(16, 9))
	#plot_tree(model, feature_names = ["K_BEHIND_BALL_DEFENSIVE_LINE", "K_KNOW_BALL_POS", "K_KNOW_SELF_POS", "K_COME_BALL", "K_TARGET_IN_SHOT_RANGE", "K_ON_TARGET_POS", "K_BALL_AND_TARGET_ON_STRAIGHT_LINE", "K_HAVE_BALL", "K_BALL_IN_TARGET", "K_BALL_IN_KICK_AREA", "K_ENEMY_WITHIN_RANGE", "K_IDLE", "K_TRACKING_BALL", "K_PERMITTED_INTRUSION_CIRCLE", "K_SECONDARY_STATE_FREEZE", "K_SECONDARY_STATE_DIRECT_FREEKICK", "K_SECONDARY_STATE_KICK_TEAM_OWN", "K_SIDE_ENTRY_STATE", "K_ON_OUR_FIELD", "PREV_TASK"], class_names = ["Idling", "IdleKickoff", "IdleEnemyFreeKick", "FindLandmarksToLocalize", "TurnAndSearchCloseBall", "SearchCloseBall", "SearchFarBall", "ApproachBallForGankenKun", "TurnAroundBallToTarget", "DribbleToTarget", "AdjustToKickPosForGankenKun", "KickBallToTarget", "MovetoDefencePos", "TrackBallForDefender", "TrackBall", "WalkStraight", "ApproachTargetPosForGankenKun", "ApproachTargetPos", "None"], filled=True)
	plot_tree(model, feature_names = features, class_names = ["Idling", "IdleKickoff", "IdleEnemyFreeKick", "FindLandmarksToLocalize", "TurnAndSearchCloseBall", "SearchCloseBall", "SearchFarBall", "ApproachBallForGankenKun", "ApproachFreeKickTargetPos", "TurnAroundBallToTarget", "DribbleToTarget", "AdjustToKickPosForGankenKun", "KickBallToTarget", "MovetoDefencePos", "TrackBallForDefender", "TrackBall", "WalkStraight", "ApproachTargetPosForGankenKun", "ApproachTargetPos", "None"], filled=True)
	
	os.makedirs('../graph', exist_ok=True)
	date = datetime.datetime.now().strftime('%y%m%d')
	plt.savefig('../graph/' + date + '_' + args.role + '-tree.png', format="png", dpi=1000)

	os.makedirs('../models', exist_ok=True)
	filename = date + '_imitation_' + args.role  + '.pkl'
	pickle.dump(model,open('../models/' + filename, 'wb'))

if __name__ == '__main__':
	main()
