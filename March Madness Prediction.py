
import urllib2
import codecs
import numpy as np
import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
from urllib import urlopen
import urllib2
import json
from math import exp, expm1, log
from math import log
import scipy as sp
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
import itertools


# Gether point spread data from cover.com for teams participated in the NCAA March Madness 2017

data = pd.read_csv("NCAA_2003-2016_used.csv")

def get_teams(data):
    teams = []
    for i in range(0,len(data)):
        if(data['team1_teamname'][i] in teams):
            continue
        else:
            teams.append(data['team1_teamname'][i])
    return teams


team_id_cover = pd.read_csv('Teams_covers.csv') # name, id

def get_spread(team1, team2, year):
    season = str(year - 1) + '-' +  str(year)
    spread = 9999
    team1_id = 0
    spread_str = '9999 9999'

    for i in range(0,len(team_id_cover)):
        if(str(team_id_cover['name'][i]) == team1):
            team1_id = team_id_cover['id'][i]


    if(team1_id != 0):
        response = urllib2.urlopen('http://www.covers.com/pageLoader/pageLoader.aspx?page=/data/ncb/teams/pastresults/'
                               + season + '/team' + str(team1_id) + '.html')
        html = response.read()
        soup = BeautifulSoup(html,"html.parser")

        table = soup.find_all('table')[0]

        for tr in table.find_all(attrs= {'class':'datarow'}):
            for td in tr.find_all('td')[1]:
                if(team2 in td):
                    spread_str = tr.find_all('td')[4].text

        result = spread_str.split()
        spread = float(result[1])

    return spread


for i in range(0,len(data)):
    try:
        print(i)

        data.set_value(i,'point_spread',get_spread(data['team1_teamname'][i],data['team2_teamname'][i],data['Season'][i]))
    except:
        continue

data.to_csv('NCAA_with_spread.csv')

# Collect historical point spread from covers.com

teams = pd.read_csv("Teams_covers.csv")

team_name = teams['name'][0]
team_id = teams['id'][0]
year = 2003

def get_spread(team1_id, year):

    season = str(year - 1) + '-' +  str(year)
    spread = 9999
    match_count = 0

    url = 'http://www.covers.com/pageLoader/pageLoader.aspx?page=/data/ncb/teams/pastresults/'+ season + '/team' + str(team1_id) + '.html'
    response = urllib2.urlopen(url)
    #response = urllib2.urlopen('http://www.covers.com/pageLoader/pageLoader.aspx?page=/data/ncb/teams/pastresults/2002-2003/team2518.html')
    print(url)
    html = response.read()
    soup = BeautifulSoup(html,"html.parser")

    tables = soup.find_all('table')
    for table in tables:
        for tr in table.find_all(attrs= {'class':'datarow'}):
            td = tr.find_all('td')
            result = td[2].text.split()
            result = result[0]

            if(result == "L"):
                result = 0
            else:
                result = 1

            spread = td[4].text.split()
            spread = spread[1]

            if(spread == 'PK'):
                spread = 0

            if(spread != '-'):
                with open('result_spread_all_games', 'a') as csvfile:
                    csvfile.write(str(result) + '\t' + spread + '\n')
                    match_count = match_count + 1
                    print(match_count)

            else:
                continue

    return spread


for i in range(0,len(teams)):
    for j in range(2003,2017):
        try:
            get_spread(teams['id'][i], j)
        except:
            continue

 
#Logistic regression model

logreg = LogisticRegression()


df = pd.read_csv("NCAA_with_spread.csv")

variables = ['result','diff_dist','team1_log5','adjoe','adjde','opp_efg','tpp','orp','ftr']

response = ['diff_dist','team1_log5','adjoe','adjde','opp_efg','tpp','orp','ftr']

for h in range(8,len(response)+1):
    n = list(itertools.combinations(response,h))
    for a in range(0,len(n)):
        try:
            n1 = list(n[a])
            data = df[variables]

            data['intercept'] = 1.0

            train_cols = n1
            train_cols.append('intercept')
            logit = sm.Logit(data['result'],data[train_cols])
            result = logit.fit()
            # K-fold accuracy test
            X = data[train_cols]
            accuracy_mean = cross_val_score(logreg, X, data['result'], cv=10, scoring='accuracy').mean()
            a = cross_val_score(logreg, X, data['result'], cv=10, scoring='neg_log_loss')
            a = a*-1
            ll_min = a.min()
            ll_mean = a.mean()
            ll_std= a.std()

            params = result.params
            #print(result.summary())

            for i in range(0,len(df)):
                z = 0
                for j in range(0,len(train_cols)-1):
                    z = z + df[train_cols[j]][i] * params[j]
                z = z + params[-1]
                y = 1 / (1 + exp(-1 * z))
                df.set_value(i, 'm1_predict', y)

            logloss_list = []
            logloss = 0
            ll = 0

            for j in range(0,len(df)):

                win_lose = df['result'][j]
                #need log loss equation here
                ll = -1*(float(win_lose)*log(float(df['m1_predict'][j])) + ((1-float(win_lose))*log(float(1-float(df['m1_predict'][j])))))
                logloss_list.append(ll)

            for k in range(0,len(logloss_list)):
                logloss = logloss + logloss_list[k]

            logloss = logloss/len(logloss_list)
            print str(a)

        except:
            continue

# Check Logloss of the Combined model M1 + M2

data = pd.read_csv("NCAA_with_spread.csv")
predicts = pd.read_csv("Prediction_filted.csv")
model_weight = pd.read_csv("Model_weight.csv")

def predict(diff_dist,adjoe, adjde, efg, tpp, orp, ftr, opp_ftr):
    #z = 2.5847 - 0.0002 * diff_dist + 0.2404 * adjoe  -0.3137 * adjde + 6.0958 * efg + 6.1949 * orp + 2.8597 * ftr - 5.9432 * team1_log5 - 6.7155 * tpp
    z = -0.1169 - 0.0002 * diff_dist + 0.0946 * adjoe - 0.1547 * adjde + 6.0113 * efg - 6.7691 * tpp + 7.2473 * orp + 3.488 * ftr + 0.9193 * opp_ftr
    y = 1/(1+exp(-1*z))
    return y

def predict_m1(point_spread):
    z = -0.001999 + 0.16938181*point_spread
    y = 1/ (1 + exp(-1*z))
    return 1-y

def weight(m1,m2,m1_weight,m2_weight):
    result = m1*m1_weight + m2*m2_weight
    return result

p1 = 0
p2 = 0
# M1

for i in range(0,len(data)):
        #print(predicts['id'][0])
        #print(data['game_id'][354])
        #print(str(predicts['id'][0]) == str(data['game_id'][354]))
    point_spread = data['point_spread'][i]
            #predicts['prediction'][i] = predict(diff_dist, team1_log5, tpp, orp)
    p1 = predict_m1(point_spread)
    data.set_value(i, 'm1_predict', p1)

#M2
for i in range(0,len(data)):
        #print(predicts['id'][0])
        #print(data['game_id'][354])
        #print(str(predicts['id'][0]) == str(data['game_id'][354]))
        diff_dist = data['diff_dist'][i]
        team1_log5 = data['team1_log5'][i]
        orp = data['orp'][i]
        efg = data['efg'][i]
        adjoe = data['adjoe'][i]
        adjde = data['adjde'][i]
        ftr = data['ftr'][i]
        tpp = data['tpp'][i]
        opp_ftr = data['opp_ftr'][i]
        #predicts['prediction'][i] = predict(diff_dist, team1_log5, tpp, orp)
        p2 = predict(diff_dist,adjoe, adjde, efg, tpp, orp, ftr, opp_ftr)
        data.set_value(i, 'm2_predict', p2)

data.to_csv('Prediction_data.csv')

predict_result = pd.read_csv('Prediction_data.csv')


for i in range(0,len(model_weight)):
    logloss_list = []
    logloss = 0
    ll = 0
    m1_weight = model_weight['m1'][i]
    m2_weight = model_weight['m2'][i]
    for j in range(0,len(predict_result)):
        p_m1_m2 = weight(predict_result['m1_predict'][j],predict_result['m2_predict'][j],m1_weight,m2_weight)
        win_lose = predict_result['result'][j]
        #need log loss equation here
        ll = -1*(float(win_lose)*log(float(p_m1_m2)) + ((1-float(win_lose))*log(float(1-p_m1_m2))))
        logloss_list.append(ll)

    for k in range(0,len(logloss_list)):
        logloss = logloss + logloss_list[k]

    logloss = logloss/len(logloss_list)
    model_weight.set_value(i, 'weighted', logloss)
print model_weight
model_weight.to_csv('weights.csv')

# Bonus- Championship odds

data = pd.read_csv('2017_prediction.csv')
teams = pd.read_csv('2017_tourney_teams.csv')


for i in range(0,len(teams)):
    prob = []
    for j in range(0,len(data)):
        if teams['team'][i] == data['team1_id'][j]:
            prob.append(data['prediction'][j])
            teams.set_value(i, 'team_name',data['team1_name'][j])
        if teams['team'][i] == data['team2_id'][j]:
            prob.append(1-data['prediction'][j])
            teams.set_value(i, 'team_name', data['team2_name'][j])
    ave = sum(prob)/len(prob)
    teams.set_value(i,'prob_mean',ave)

teams.to_csv('teams_average_win.csv')



