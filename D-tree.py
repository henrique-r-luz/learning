import pandas as pd
#dataset = pd.read_csv("data/leagues_NBA_2014_games_games.csv")
dataset = pd.read_csv("data/leagues_NBA_2014_games_games.csv", parse_dates=[0],
skiprows=[0,])
dataset.columns = ["Date", "Score Type", "Visitor Team","VisitorPts", "Home Team", "HomePts", "OT?", "Notes"]
dataset["HomeWin"] = dataset["VisitorPts"] < dataset["HomePts"]
print(dataset.ix[:10])
