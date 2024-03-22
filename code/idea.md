
# What info do I need from each team

- Recent info
- This year results

# First approach

- Get basic wins of the season, and predict who will win based on that, predict for 2023 and see results

Get winrate of all teams during regular season
Xtrain
['Team1WinRate', 'Team2WinRate'] regular season

ytrain
Winner of the match during season

Xtest
['Team1WinRate', 'Team2WinRate'] postseason games, winrate is of regular season games, match TeamId of postseason games with winrate of regular season

ytest
Winner of match in postseason

Accuracy: 0.71

- Use that year stadistics
