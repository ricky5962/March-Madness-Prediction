


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







