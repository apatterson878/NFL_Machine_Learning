import os

import pandas as pd
import numpy as np
import json

import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

from flask import Flask, jsonify, render_template, request, flash, redirect
from flask_sqlalchemy import SQLAlchemy

from pandas.plotting import scatter_matrix

from numpy import loadtxt
import scipy
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
import seaborn as sb

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
import xgboost as xgb

from xgboost import XGBClassifier
import joblib
from joblib import dump, load
import pickle
from sklearn.preprocessing import LabelEncoder

#################################################

app = Flask(__name__)


#################################################
pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

####### OLD Series too many Parmas STARTS #######
def get_matRx_4_gvn_Year_model_series_old(year, df_4m_csv):
    groupby_Season = df_4m_csv.groupby('season')
        
    df_for_Year_Passed_1 = groupby_Season.get_group(year-1)
    groupby_Team_Avg_Values_1 = df_for_Year_Passed_1.groupby('team', as_index=False).mean()
    groupby_Team_Avg_Values_1 = groupby_Team_Avg_Values_1[['season', 'week','team', 'home', 'pass_attempts', 'pass_completions', 'pass_yards', 'net_pass_yards', 'pass_tds', 'rush_attempts', 'rush_yards', 'rush_tds', 'total_yards', 'first_downs', 'sacks', 'sacks_yards', 'pass_interceptions', 'fumbles', 'fumbles_lost', 'turnovers', 'time_of_possession', 'pentalties', 'penalty_yards', 'third_down_attempts', 'third_down_conversions', 'fourth_down_attempts', 'fourth_down_conversions','Score']]
        
    df_for_Year_Passed_2 = groupby_Season.get_group(year-2)
    groupby_Team_Avg_Values_2 = df_for_Year_Passed_2.groupby('team', as_index=False).mean()
    groupby_Team_Avg_Values_2 = groupby_Team_Avg_Values_2[['season', 'week','team', 'home', 'pass_attempts', 'pass_completions', 'pass_yards', 'net_pass_yards', 'pass_tds', 'rush_attempts', 'rush_yards', 'rush_tds', 'total_yards', 'first_downs', 'sacks', 'sacks_yards', 'pass_interceptions', 'fumbles', 'fumbles_lost', 'turnovers', 'time_of_possession', 'pentalties', 'penalty_yards', 'third_down_attempts', 'third_down_conversions', 'fourth_down_attempts', 'fourth_down_conversions','Score']]
    
    result = pd.concat([groupby_Team_Avg_Values_1, groupby_Team_Avg_Values_2], axis=0).dropna(axis=1).groupby('team', as_index=False).mean()
    result = result[['season', 'week','team', 'home', 'pass_attempts', 'pass_completions', 'pass_yards', 'net_pass_yards', 'pass_tds', 'rush_attempts', 'rush_yards', 'rush_tds', 'total_yards', 'first_downs', 'sacks', 'sacks_yards', 'pass_interceptions', 'fumbles', 'fumbles_lost', 'turnovers', 'time_of_possession', 'pentalties', 'penalty_yards', 'third_down_attempts', 'third_down_conversions', 'fourth_down_attempts', 'fourth_down_conversions','Score']]
    scaled = preprocessing.MinMaxScaler()
    scaled_result = scaled.fit_transform(result)
    return scaled_result

def prediction_4_season_model_series_old(season,model):
    if int(season) in range(2011, 2020) and int(model) in range(1, 5):
        
        win_loss_prdctd_data = pd.DataFrame(columns=['Team','0', '1', '2', '3','4', '5', '6','7', '8', '9','10', '11', '12', '13','14', '15', '16','17', '18', '19','20', '21', '22', '23','24', '25', '26','27', '28', '29', '30', '31'])
        win_loss_prdctd_data["Team"] = ['0', '1', '2', '3','4', '5', '6','7', '8', '9','10', '11', '12', '13','14', '15', '16','17', '18', '19','20', '21', '22', '23','24', '25', '26','27', '28', '29', '30', '31']
        switcher = { 
             1: 'models/LR_model.sav', 2: 'models/xgb_model.sav', 3: 'models/svm_model.sav', 4: 'models/GB_best_model.sav'
        }
        model = switcher.get(model, "nothing")
    
        #   Load the data from the csv
        df=pd.read_csv("ml_data/NFL_datasetForML.csv")
        #   Create a list named team_Names and sort them alphabetically.....It will be used for relabelling the column and row label
        team_Names = df['team'].unique()
        team_Names = team_Names.tolist()
        team_Names = sorted(team_Names)
        #   Drop unnecessary columns from the daataframe
        df_effective=df.drop(['Unnamed: 0','boxscore_id', 'roof', 'surface', 'temp', 'humidity', 'wind_chill', 'wind_mph', 'Won'], axis=1)
        #   Label encoding of two columns having categorical data
        le = LabelEncoder()
        df_effective['team'] = le.fit_transform(df_effective['team'])
        df_effective['time_of_possession'] = le.fit_transform(df_effective['time_of_possession'])

        print("\033[1m" + "Prediction table for the Season/Year %i using the `%s` !" % (season, model.split('/')[1].split('.')[0].replace('_',' ').upper()))
        #   Loading the desired model
        logit_model=joblib.load(model)
        #   getting the result of prediction by passing the season specific data(by calling the method get_matRx_4_gvn_Year_model_series_old) to the 
        #   model prepared
        LR_result=logit_model.predict_proba(get_matRx_4_gvn_Year_model_series_old(season,df_effective))
    
        for i, row in win_loss_prdctd_data.iterrows():
            for j in range(1,33):
                if(row['Team']!=str(j-1)):
                    x_Prob = LR_result[i]
                    y_Prob = LR_result[j-1]
                    # prob_diff = abs(x_Prob[1] - y_Prob[1])
                    row[j] = 1 if (x_Prob[1] > y_Prob[1]) else 0
                else:
                    row[j] = "Same Team"
                
        #   Relabel the rows
        win_loss_prdctd_data["Team"] = team_Names
        #   Add a new element Team in the list 
        team_Names.insert(0,'Team')
        #   Relabel the columns headers with the new list 
        win_loss_prdctd_data.columns = team_Names
        win_loss_prdctd_data.set_index('Team')
        return win_loss_prdctd_data
    else:
        message = print("\033[1m" + "Sorry ! The Matrix can be generated only for > 2011 and <= 2019 and 1 model 1 to 4.\n Please Enter relevant values..")
        return message
####### OLD Series too many Parmas ENDS #######


######## OLD Intermediate Series STARTS ##########

def get_matRx_4_gvn_Year_Intermediate(year, df_4m_csv):
    groupby_Season = df_4m_csv.groupby('season')
        
    df_for_Year_Passed_1 = groupby_Season.get_group(year-1)
    groupby_Team_Avg_Values_1 = df_for_Year_Passed_1.groupby('team', as_index=False).mean()
    groupby_Team_Avg_Values_1 = groupby_Team_Avg_Values_1[['season', 'week','team', 'home','total_yards', 'first_downs', 'sacks', 'turnovers', 'pentalties', 'third_down_conversions','fourth_down_conversions','Score']]
        
    df_for_Year_Passed_2 = groupby_Season.get_group(year-2)
    groupby_Team_Avg_Values_2 = df_for_Year_Passed_2.groupby('team', as_index=False).mean()
    groupby_Team_Avg_Values_2 = groupby_Team_Avg_Values_2[['season', 'week','team', 'home','total_yards', 'first_downs', 'sacks', 'turnovers', 'pentalties', 'third_down_conversions','fourth_down_conversions','Score']]
    
    result = pd.concat([groupby_Team_Avg_Values_1, groupby_Team_Avg_Values_2], axis=0).dropna(axis=1).groupby('team', as_index=False).mean()
    result = result[['season', 'week','team', 'home','total_yards', 'first_downs', 'sacks', 'turnovers', 'pentalties', 'third_down_conversions','fourth_down_conversions','Score']]
    scaled = preprocessing.MinMaxScaler()
    scaled_result = scaled.fit_transform(result)
    return scaled_result


def prediction_4_season_Intermediate(season,model):
    if int(season) in range(2011, 2020) and int(model) in range(1, 5):
        
        win_loss_prdctd_data = pd.DataFrame(columns=['Team','0', '1', '2', '3','4', '5', '6','7', '8', '9','10', '11', '12', '13','14', '15', '16','17', '18', '19','20', '21', '22', '23','24', '25', '26','27', '28', '29', '30', '31'])
        win_loss_prdctd_data["Team"] = ['0', '1', '2', '3','4', '5', '6','7', '8', '9','10', '11', '12', '13','14', '15', '16','17', '18', '19','20', '21', '22', '23','24', '25', '26','27', '28', '29', '30', '31']
        switcher = { 
             1: 'models/LR_model_dropped.sav', 2: 'models/xgb_model_dropped.sav', 3: 'models/svm_model_dropped.sav', 4: 'models/GB_best_model_dropped.sav'
        }
        model = switcher.get(model, "nothing")
    
        #Load the data from the csv
        df=pd.read_csv("ml_data/NFL_datasetForML.csv")
        #Create a list named team_Names and sort them alphabetically.....It will be used for relabelling the column and row label
        team_Names = df['team'].unique()
        team_Names = team_Names.tolist()
        team_Names = sorted(team_Names)
        #Drop unnecessary columns from the daataframe
        df_effective=df.drop(['Unnamed: 0','boxscore_id','roof', 'surface', 'temp','humidity','wind_chill','wind_mph','pass_attempts', 'pass_completions', 'net_pass_yards', 'pass_tds', 'rush_attempts', 'rush_yards', 'rush_tds', 'pass_yards', 'sacks_yards', 'pass_interceptions','fumbles', 'fumbles_lost', 'time_of_possession','penalty_yards', 'third_down_attempts', 'fourth_down_attempts','Won'], axis=1)
        #Label encoding of two columns having categorical data
        le = LabelEncoder()
        df_effective['team'] = le.fit_transform(df_effective['team'])
        

        print("\033[1m" + "Prediction table for the Season/Year %i using the `%s` !" % (season, model.split('/')[1].split('.')[0].replace('_',' ').upper()))
        #Loading the desired model
        logit_model=joblib.load(model)
        #getting the result of prediction by passing the season specific data(by calling the method get_matRx_4_gvn_Year) to the 
        #model prepared
        LR_result=logit_model.predict_proba(get_matRx_4_gvn_Year(season,df_effective))
    
        for i, row in win_loss_prdctd_data.iterrows():
            for j in range(1,33):
                if(row['Team']!=str(j-1)):
                    x_Prob = LR_result[i]
                    y_Prob = LR_result[j-1]
                    #prob_diff = abs(x_Prob[1] - y_Prob[1])
                    row[j] = 1 if (x_Prob[1] > y_Prob[1]) else 0
                else:
                    row[j] = "Same Team"
                
        #Relabel the rows
        win_loss_prdctd_data["Team"] = team_Names
        #Add a new element Team in the list 
        team_Names.insert(0,'Team')
        #Relabel the columns headers with the new list 
        win_loss_prdctd_data.columns = team_Names
        win_loss_prdctd_data.set_index('Team')
        return win_loss_prdctd_data
    else:
        message = print("\033[1m" + "Sorry ! The Matrix can be generated only for > 2011 and <= 2019 and 1 model 1 to 4.\n Please Enter relevant values..")
        return message

######## OLD Intermediate ENDS ##################


def get_matRx_4_gvn_Year(year, df_4m_csv):
    groupby_Season = df_4m_csv.groupby('season')
        
    df_for_Year_Passed_1 = groupby_Season.get_group(year-1)
    groupby_Team_Avg_Values_1 = df_for_Year_Passed_1.groupby('team', as_index=False).mean()
    groupby_Team_Avg_Values_1 = groupby_Team_Avg_Values_1[['season','team','home', 'pass_attempts', 'rush_attempts', 'rush_yards', 'total_yards', 'pass_interceptions', 'turnovers','time_of_possession', 'penalty_yards', 'fourth_down_conversions','Score']]
        
    df_for_Year_Passed_2 = groupby_Season.get_group(year-2)
    groupby_Team_Avg_Values_2 = df_for_Year_Passed_2.groupby('team', as_index=False).mean()
    groupby_Team_Avg_Values_2 = groupby_Team_Avg_Values_2[['season','team','home', 'pass_attempts', 'rush_attempts', 'rush_yards', 'total_yards', 'pass_interceptions', 'turnovers','time_of_possession', 'penalty_yards', 'fourth_down_conversions','Score']]
    
    result = pd.concat([groupby_Team_Avg_Values_1, groupby_Team_Avg_Values_2], axis=0).dropna(axis=1).groupby('team', as_index=False).mean()
    result = result[['season','team','home', 'pass_attempts', 'rush_attempts', 'rush_yards', 'total_yards', 'pass_interceptions', 'turnovers','time_of_possession', 'penalty_yards', 'fourth_down_conversions','Score']]
    scaled = preprocessing.MinMaxScaler()
    scaled_result = scaled.fit_transform(result)
    return scaled_result

def prediction_4_season(season,model):
    if int(season) in range(2011, 2020) and int(model) in range(1, 5):
        
        win_loss_prdctd_data = pd.DataFrame(columns=['Team','0', '1', '2', '3','4', '5', '6','7', '8', '9','10', '11', '12', '13','14', '15', '16','17', '18', '19','20', '21', '22', '23','24', '25', '26','27', '28', '29', '30', '31'])
        win_loss_prdctd_data["Team"] = ['0', '1', '2', '3','4', '5', '6','7', '8', '9','10', '11', '12', '13','14', '15', '16','17', '18', '19','20', '21', '22', '23','24', '25', '26','27', '28', '29', '30', '31']
        switcher = { 
             1: 'models/LR_model_dropped.sav', 2: 'models/xgb_model_dropped.sav', 3: 'models/svm_model_dropped.sav', 4: 'models/GB_best_model_dropped.sav'
        }
        model = switcher.get(model, "nothing")
    
        #   Load the data from the csv
        df=pd.read_csv("ml_data/NFL_datasetForML.csv")
        #   Create a list named team_Names and sort them alphabetically.....It will be used for relabelling the column and row label
        team_Names = df['team'].unique()
        team_Names = team_Names.tolist()
        team_Names = sorted(team_Names)
        #   Drop unnecessary columns from the daataframe
        df_effective=df.drop(['sacks_yards','pentalties','Unnamed: 0','boxscore_id','roof', 'surface', 'temp','humidity','wind_chill','wind_mph','sacks', 'first_downs', 'rush_tds', 'net_pass_yards','pass_completions', 'pass_tds', 'pass_yards','fumbles_lost', 'third_down_attempts','fumbles','week', 'third_down_conversions','fourth_down_attempts', 'Won' ], axis=1)
        #   Label encoding of two columns having categorical data
        le = LabelEncoder()
        df_effective['team'] = le.fit_transform(df_effective['team'])
        df_effective['time_of_possession'] = le.fit_transform(df_effective['time_of_possession'])
       

        print("\033[1m" + "Prediction table for the Season/Year %i using the `%s` !" % (season, model.split('/')[1].split('.')[0].replace('_',' ').upper()))
        #   Loading the desired model
        logit_model=joblib.load(model)
        #   getting the result of prediction by passing the season specific data(by calling the method get_matRx_4_gvn_Year) to the 
        #   model prepared
        LR_result=logit_model.predict_proba(get_matRx_4_gvn_Year(season,df_effective))
    
        for i, row in win_loss_prdctd_data.iterrows():
            for j in range(1,33):
                if(row['Team']!=str(j-1)):
                    x_Prob = LR_result[i]
                    y_Prob = LR_result[j-1]
                    #  prob_diff = abs(x_Prob[1] - y_Prob[1])
                    row[j] = 1 if (x_Prob[1] > y_Prob[1]) else 0
                else:
                    row[j] = "Same Team"
                
        #   Relabel the rows
        win_loss_prdctd_data["Team"] = team_Names
        #   Add a new element Team in the list 
        team_Names.insert(0,'Team')
        #   Relabel the columns headers with the new list 
        win_loss_prdctd_data.columns = team_Names
        win_loss_prdctd_data.set_index('Team')
        return win_loss_prdctd_data
    else:
        message = print("\033[1m" + "Sorry ! The Matrix can be generated only for > 2011 and <= 2019 and 1 model 1 to 4.\n Please Enter relevant values..")
        return message
#################################################
# Database Setup
#################################################

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///datasource/rawdata/nfl.db"
db = SQLAlchemy(app)

#-------ALL Html Template Roots-STARTS-----------------
@app.route("/")
def index():
    """Return the HEAT-MAP homepage."""
    return render_template("index.html")
#-------ALL Html Template Roots-ENDS-----------------

#-------ALL Html Predication Data-STARTS-----------------
@app.route("/prediction/<season>/<ml_type>")
def get_prediction_bySeason(season, ml_type):
    if int(season) < 2011:
        return jsonify([])
    if int(season) > 2018:
        return jsonify([])
    # Season = 2018 & below ml_type = 1
    prediction_Result = prediction_4_season(int(season), int(ml_type))
    prediction_Result = prediction_Result.set_index('Team')
    teams = ['atl','buf', 'car','chi', 'cin', 'cle', 'clt', 'crd', 'dal', 'den', 'det', 'gnb',
        'htx','jax','kan','mia','min','nor','nwe','nyg','nyj','oti','phi','pit','rai','ram','rav','sdg','sea','sfo','tam','was'
    ]
    data = []
    for team in teams:
        #select count(*), sum(case winner when 'atl' then 1 else 0 end) as win_count from game_info where season=2018 and (home='atl' or away='atl')
        #select home, away from game_info where season=2018 and (home='atl' or away='atl')
        #When in Loop wheather away of home choose the opponent team to get the Win/Loss
        #Aggregate the Wins / Number of games = 16 > then calculate percentage of Prediction for Each Team.
        results = db.engine.execute("select sum(case winner when :team then 1 else 0 end)\
            as win_count from game_info where season=:season\
            and (home=:team or away=:team)", {'season': season, 'team':team})
        team_win_count = 0
        for row in results:
            team_win_count = row[0]

        team_array = []
        results2 = db.engine.execute("select home, away from game_info where season=:season\
            and (home=:team or away=:team)", {'season': season, 'team':team})
        for row in results2:
            team_dict = {}
            team_dict["left_team"] = row[0]
            team_dict["right_team"] = row[1]
            team_array.append(team_dict)
        
        #Now Loop Through team_array and get value of Win/Loss From prediction_Result
        predicted_team_aggregate_wins = 0
        countOfGames = 0
        for row in team_array:
            right_team = ""
            if(team != row["left_team"]):
                right_team = row["left_team"]
            if(team != row["right_team"]):
                right_team = row["right_team"]
            countOfGames = countOfGames + 1
            value = prediction_Result.loc[[team], [right_team]]
            if right_team == 'bye':
                continue
            #Conditional Check
            if team != right_team:
                predicted_team_aggregate_wins = int(predicted_team_aggregate_wins) + int(value.iloc[0,0])
            else:
                #This Scope should never be reached.
                print("### Team cannot Play against the Same Team")
        #Eval
        # +ve or -ve Prediction and percentage.
        positive_or_negative_count = predicted_team_aggregate_wins - team_win_count
        positive_or_negative = "+"
        if positive_or_negative_count < 0:
            positive_or_negative = "-"


        #Compile a New Data Array or JSON, return the Results for Display..
        teamMap = {
            "team": team,
            "team_win_count":team_win_count,
            "positive_or_negative_count":positive_or_negative_count,
            "positive_or_negative":positive_or_negative,
            "predicted_win_count":predicted_team_aggregate_wins,
            "countOfGames": countOfGames
        }
        data.append(teamMap)
    return jsonify(data)


# Assemble Object in the Main JSON
def getSeasonJson(seasonNumber, col1, col2, prediction_Result):
    seasonWinLoss = -1
    if col2 != "bye":
        seasonWinLoss = prediction_Result.loc[col1,col2]
    season = {
        "Season":seasonNumber,
        "MainTeam":col1,
        "PlayedAgainst":col2,
        "WinOrLoss":seasonWinLoss
    } 
    return season

@app.route("/prediction/current/<ml_type>")
def get_prediction_currentSeason(ml_type):
    # Season = 2019 ml_type = 1
    prediction_Result = prediction_4_season(2019, int(ml_type))
    prediction_Result = prediction_Result.set_index('Team')
    # Get the Preloaded Current Season Playoffs.
    df_schedule = pd.read_csv("ml_data/schedule2019.csv")
    # New Dict
    data_array = []
    for i in df_schedule.itertuples(): 
        #print(i[0]) Ignore default Index.
        #17 Seasons total
        season1 = getSeasonJson(1,i[1],i[2],prediction_Result)
        season2 = getSeasonJson(2,i[1],i[3],prediction_Result)
        season3 = getSeasonJson(3,i[1],i[4],prediction_Result)
        season4 = getSeasonJson(4,i[1],i[5],prediction_Result)
        season5 = getSeasonJson(5,i[1],i[6],prediction_Result)
        season6 = getSeasonJson(6,i[1],i[7],prediction_Result)
        season7 = getSeasonJson(7,i[1],i[8],prediction_Result) 
        season8 = getSeasonJson(8,i[1],i[9],prediction_Result) 
        season9 = getSeasonJson(9,i[1],i[10],prediction_Result) 
        season10 = getSeasonJson(10,i[1],i[11],prediction_Result) 
        season11 = getSeasonJson(11,i[1],i[12],prediction_Result)
        season12 = getSeasonJson(12,i[1],i[13],prediction_Result) 
        season13 = getSeasonJson(13,i[1],i[14],prediction_Result) 
        season14 = getSeasonJson(14,i[1],i[15],prediction_Result) 
        season15 = getSeasonJson(15,i[1],i[16],prediction_Result) 
        season16 = getSeasonJson(16,i[1],i[17],prediction_Result) 
        season17 = getSeasonJson(17,i[1],i[18],prediction_Result) 
        data = {
            "MainTeam": i[1], 
            "Seasons": [season1, season2, season3, 
                season4, season5, season6, season7,
                season8, season9, season10, season11,
                season12, season13, season14, season15,
                season16, season17]
        }
        data_array.append(data) 
    return jsonify(data_array)

#-------ALL Html Predication Data-ENDS-----------------


#---ROUTES bar and sparkline and Partial Main HeatMap- STARTS----
@app.route("/teamstat/metadata")
def get_static_metadata():
    """Return All Season MetadataValues."""

    results = db.engine.execute("select st.season\
        from stats_team st\
        group by st.season order by st.season desc limit 8 ")
    seasons = []
    for row in results:
        seasons.append(row[0])

    model_dict = {
        "1": "Logistic Regression",
        "2": "XGBoost Model",
        "3": "Support Vector Model",
        "4": "Gradient Tree Boosting"
    }

    data = {
        "seasons": seasons,
        "models": model_dict
    }

    return jsonify(data)
#---ROUTES bar and sparkline and Partial Main HeatMap- ENDS----

#------MANDATORY---Run Main APP-----
if __name__ == "__main__":
    app.run(debug=True)
