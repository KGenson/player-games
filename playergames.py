import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math

years = range(2000, 2020)

dict = {}

# Conver the CSVs into a dictionary of player names, positions, and rankings.
for n in years:
    df = pd.read_csv(str(n) + '.csv')

    # Removing unnecessary stuff.
    df.drop(['Age', 'Cmp', 'Att', 'Yds.1', 'Att', 'Yds.2', 'Tgt', 'Yds', 'Y/R', 'GS'], axis=1, inplace=True)

    # Converting to fantasy scores.
    df['FantasyPoints'] = df['Int']*-2 + df['Rec']*0.5 + df['Fumbles']*-2 + df['PassingYds']*0.04 + df['PassingTD']*4 + df['RushingYds']*0.1 + df['RushingTD']*6 + df['ReceivingYds']*0.1 + df['ReceivingTD']*6

    # Ranking within position.
    df['PosRank'] = df.groupby('Pos')['FantasyPoints'].rank(ascending=0,method='first')

    # Cleans out unwanted columns.
    df = df[['Player','G', 'Pos', 'PosRank']]

    # Adds a column of zeros for Games Next Year.
    df['GNY'] = 0

    # Converts games played to integer.
    df = df.astype({'G': 'int'})

    # Updates the dictionary with the year's information.
    dict.update({n:df})

posdict = {
    'WR': pd.DataFrame(columns=['PosRank','GNY']),
    'RB': pd.DataFrame(columns=['PosRank','GNY']),
    'TE': pd.DataFrame(columns=['PosRank','GNY']),
    'QB': pd.DataFrame(columns=['PosRank','GNY'])}
# rb = pd.DataFrame(columns=['PosRank','GNY'])
data = pd.DataFrame()

positions = ['RB', 'WR', 'TE', 'QB']

# Adding the games played the following year to each dictionary.
for n in range (2009, 2019):

    # We're referencing both this year and the following year.

    dfa = dict[n]
    dfb = dict[n+1]

    # Running through the cells and finding next year's games played for each name.
    # Subtract 1 for a 16 week season.
    for z in range(0, len(dfa)):
        try:
            dfa.loc[z, 'GNY'] = (dfb.loc[dfb['Player'] == dfa.loc[z, 'Player'], 'G'].values) - 1
            # Error cleaning.
            if dfa.loc[z, 'GNY'] > 15:
                dfa.loc[z, 'GNY'] = 15
        except ValueError:
            pass

    # Next up we sort the values, filter out to the desired position, set that to the index,
    # and create a new DF with that information in for all five years. 

    dfa = dfa.sort_values(by=['PosRank'])

    for m in positions:

        data = dfa[dfa['Pos']==m]

        data.reset_index(inplace=True)

        data = data.drop(data.index[100:])

        data = data[['PosRank', 'GNY']]

        posdict[m] = posdict[m].append(data, ignore_index=True)


for m in positions:
    # Define the function that the curve will be based on.
    def funca(x, a, b, c):
        return a*x**2 + b*x + c

    # Setting up the curve fit.
    xdata = posdict[m]['PosRank']
    ydata = posdict[m]['GNY']

    #Fit the data to the curve. Value c has to be bound by an upper limit of 15 since that's the max number of games.
    popta, pcova = curve_fit(funca, xdata, ydata, bounds = ((-np.inf, -np.inf, 0), (np.inf, np.inf, 15)))

    x = np.linspace(1, 100,100)

    plt.figure(figsize=(6,4))
    plt.plot(xdata, ydata, 'ro')
    plt.plot(x, funca(x, *popta), 'b')

    # R**2 Data

    # Calculate R_Squared
    residuals = ydata - funca(xdata, *popta) 
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ydata-np.mean(ydata))**2)

    r_squared = 1 - (ss_res / ss_tot)

    print(f'Formula for {m} is: {popta[0]}x^2 + {popta[1]}x + {popta[2]}\nR^2 is {r_squared}')
#    plt.show()
    posdict[m].to_csv(f'{m}.csv', index = False)