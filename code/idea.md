Ojo  con Howard, que no tendria que estar en el bracket final
Boise St deberia ser Colorado
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



Coje:
- Resultados de la temporada
- Resultados de la post-temporada
- Seeds (Cuadro de playoffs por temporada)


Reordena las columnnas

Cambiar W and L por T1 y T2

Duplica el numero de filas, es decir, para el primer partido hay dos filas, fila del que pierde como t1 y fila del que gana como t1

Asi obtenemos resultados de la temporada y posttemporada


Feature Engineering

Apply mean for [Season, team1]

Hacemos dos dataset de season statistics iguales, uno con t1 y otro con t2

Cojemos los datos de postemporada (los cuales ya estaban duplicados en filas)
['Season', 'DayNum', 'T1_TeamID', 'T1_Score', 'T2_TeamID' ,'T2_Score']


Juntamos con los dos datasets por los dos lados
tourney_data = pd.merge(tourney_data, season_statistics_T1, on = ['Season', 'T1_TeamID'], how = 'left')
tourney_data = pd.merge(tourney_data, season_statistics_T2, on = ['Season', 'T2_TeamID'], how = 'left')







Cojemos el win ratio de los ultimos 14 dias para cada equipo cada temporada

Sacamos team quality con una formula, GLM

Añadimos a los datos de torneo la diferencia de seed entre los equipos


## Modelos


Accuracy: 0.21
Points: 26
Number of games: 63
   TeamID      TeamName  PredictedWins  Wins
0    1163   Connecticut              6     6
1    1235       Iowa St              2     2
2    1228      Illinois              3     3
3    1120        Auburn              2     0
4    1361  San Diego St              1     2
Accuracy for predictions/stephen_a.csv: 0.33
Points for predictions/stephen_a.csv: 42

























