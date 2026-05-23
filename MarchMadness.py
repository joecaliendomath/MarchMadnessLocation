"""
NCAA Tournament Travel Distance Study
Author: Joseph Caliendo
Created: 5/16/2026 - Finished 5/20/2026

Study Overview:
    This script investigates whether travel distance disparity between teams
    in NCAA Tournament games is a statistically significant predictor of 
    game outcomes against the spread. 
    
    Three distance metrics are examined:
    1. Distance Ratio (DistRatio)  - Multiplicative difference in travel distance
    2. Standard Deviation (STD)    - Games where distance gap exceeds 1 STD 
                                     relative to all of the distance travelled by all teams in all years.
    3. Combined (BOTH)             - Games qualifying under both the DistRatio and STD criteria
     
    For each metric, results are broken down by:
        - Overall success rate
        - High seed advantage vs Low seed advantage (Advantage = Less Travel)
        - Seed Groupings by 4. (1-4,5-8,9-12,13-16)
     
    Statistical significance is assessed using a one-sided binomial test against
    a null hypothesis of p=0.5    

Terminology:
    LS = Lower Seed
    HS = Higher Seed
    Outcome = How much the lower seed covered the spread by(+ means they covered, - means they didn't')
    Success = 1 if the team with the travel advantage covered the spread, 0 otherwise
"""

#=============================================================================
#IMPORTS
#=============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats



#=============================================================================
#SECTION 1: DATA LOADING AND CLEANING
#=============================================================================

#Load raw game data from CSV
df = pd.read_csv("Data - Sheet1.csv")

#Drop any extraneous columns beyond column index 12 (keep only relevant fields)
df = df.drop(columns=df.columns[13:])

#Distance columns are stored as strings with commas (1,200 ---> 1200.)
df['Distance from LS'] = df['Distance from LS'].str.replace(",","").astype(float)
df['Distance from HS'] = df['Distance from HS'].str.replace(",","").astype(float)

#Drop the 13th column (index 12) which is not needed for analysis
df = df.drop(columns=df.columns[12])

#Calculate the Distance Ratio: how many times farther the LS travels vs the HS
#DistRatio > 1 means LS travels farther (HS has location advantage)
#DistRatio < 1 means HS travels farther (LS has location advantage)
df['DistRatio'] = df["Distance from LS"] / df["Distance from HS"]


#=============================================================================
#SECTION 2: DISTANCE RATIO ANALYSIS (DistRatio > 2 or < 0.5)
#=============================================================================
#Filter to games where one team travels at least twice as far as the other.
#DistRatio > 2: LS travels more than 2x the distance of HS (HS has location advantage)
#DistRatio < 0.5: HS travels more than 2x the distance of LS (LS has location advantage)

two_df = df[(df['DistRatio'] > 2) | (df['DistRatio'] < 0.5)]

#Rename score columns for clarity
two_df = two_df.rename(columns={'FavScore': 'LSScore'})
two_df = two_df.rename(columns={'DawgScore': 'HSScore'})

#Calculate Outcome
two_df['Outcome'] = two_df['LSScore'] - two_df["HSScore"] + two_df["Spread"]

#Define Success: did the team with the travel advantage cover the spread?
#If DistRatio > 2 (HS closer), success = HS covers = Outcome < 0
#If DistRatio < 0.5 (LS closer), success = LS covers = Outcome > 0
two_df['Success'] = np.where(((two_df['Outcome'] < 0) & (two_df['DistRatio'] > 2)) |((two_df['Outcome'] > 0) & (two_df['DistRatio'] < .5)),1, 0)

#Removed Index 409(push)
two_df = two_df.drop(index=409)

#OVERALL DISTRATIO RESULTS
suc = two_df['Success'].sum()
trials = two_df['Success'].count()
print("===========================================================")
print("DISTANCE RATIO ANALYSIS (Ratio > 2x or < 0.5x)")
print("===========================================================")
print(f"Number of Successes: {suc}")
print(f"Number of Trials:    {trials}")
print(f"Success Rate:        {suc/trials*100}%")
#One-sided binomial test
result = stats.binomtest(suc, trials, p=0.5)
print(f"P-Value (one-sided): {result.pvalue/2}")

#Split by which team has the travel advantage
#high_df: LS travels farther (DistRatio > 2), so HS has location advantage
#low_df:  HS travels farther (DistRatio < .5), so LS has location advantage
high_df = two_df[(two_df["DistRatio"] > 1)]
low_df = two_df[(two_df["DistRatio"] < 1)]

high_suc = high_df['Success'].sum()
high_trials = high_df['Success'].count()
low_suc = low_df['Success'].sum()
low_trials = low_df['Success'].count()

print("-----------------------------------------------------------")
print("High Seed Advantage in DistRatio")
print("-----------------------------------------------------------")
print(f"DistRatio Number of High Seed Successes:    {high_suc}")
print(f"DistRatio Number of High Seed Advantages:   {high_trials}")
print(f"DistRatio High Seeded Success Rate:         {100*high_suc/high_trials}%")
result.high = stats.binomtest(high_suc, high_trials, p=0.5)
print(f"DistRatio High Seeded P-Value:              {result.high.pvalue/2}")

print("-----------------------------------------------------------")
print("Low Seed Advantage in DistRatio")
print("-----------------------------------------------------------")
print(f"DistRatio Number of Low Seed Successes:     {low_suc}")
print(f"DistRatio Number of Low Seed Advantages:    {low_trials}")
print(f"DistRatio Low Seeded Success Rate:          {100*low_suc/low_trials}%")
result.low = stats.binomtest(low_suc, low_trials, p=0.5)
result.high.contemp = stats.binomtest(high_suc, high_trials, p=suc/trials)
print(f"DistRatio Low Seeded P-Value:               {result.low.pvalue/2}")

#4-SEED GROUP BREAKDOWN (1-4, 5-8, 9-12, 13-16)
onefour_df = low_df[(low_df['LS'] <= 4)]             #Seeds 1-4 have location advantage
fiveeight_df = low_df[(low_df['LS'] > 4)]            #Seeds 5-8 have location advantage
ninetwelve_df = high_df[(high_df['HS'] <= 12)]       #Seeds 9-12 have location advantage
thirteensixteen_df = high_df[(high_df['HS'] > 12)]   #Seeds 13-16 have location advantage

of_suc = onefour_df['Success'].sum()
of_trials = onefour_df['Success'].count()
fe_suc = fiveeight_df['Success'].sum()
fe_trials = fiveeight_df['Success'].count()
nt_suc = ninetwelve_df['Success'].sum()
nt_trials = ninetwelve_df['Success'].count()
ts_suc = thirteensixteen_df['Success'].sum()
ts_trials = thirteensixteen_df['Success'].count()

print("-----------------------------------------------------------")
print(f"DistRatio Number of 1-4 Seed Successes:     {of_suc}")
print(f"DistRatio Number of 1-4 Seed Advantages:    {of_trials}")
print(f"DistRatio 1-4 Seeded Success Rate:          {100*of_suc/of_trials}%")
result.of = stats.binomtest(of_suc, of_trials, p=0.5)
print(f"DistRatio 1-4 Seeded P-Value:               {result.of.pvalue/2}")

print("-----------------------------------------------------------")
print(f"DistRatio Number of 5-8 Seed Successes:     {fe_suc}")
print(f"DistRatio Number of 5-8 Seed Advantages:    {fe_trials}")
print(f"DistRatio 5-8 Seeded Success Rate:          {100*fe_suc/fe_trials}%")
result.fe = stats.binomtest(fe_suc, fe_trials, p=0.5)
print(f"DistRatio 5-8 Seeded P-Value:               {result.fe.pvalue/2}")

print("-----------------------------------------------------------")
print(f"DistRatio Number of 9-12 Seed Successes:    {nt_suc}")
print(f"DistRatio Number of 9-12 Seed Advantages:   {nt_trials}")
print(f"DistRatio 9-12 Seeded Success Rate:         {100*nt_suc/nt_trials}%")
result.nt = stats.binomtest(nt_suc, nt_trials, p=0.5)
print(f"DistRatio 9-12 Seeded P-Value:              {result.nt.pvalue}")

print("-----------------------------------------------------------")
print(f"DistRatio Number of 13-16 Seed Successes:   {ts_suc}")
print(f"DistRatio Number of 13-16 Seed Advantages:  {ts_trials}")
print(f"DistRatio 13-16 Seeded Success Rate:        {100*ts_suc/ts_trials}%")
result.ts = stats.binomtest(ts_suc, ts_trials, p=0.5)
print(f"DistRatio 13-16 Seeded P-Value:             {result.ts.pvalue/2}")


#=============================================================================
#SECTION 3: STANDARD DEVIATION ANALYSIS
#=============================================================================
#Calculate the mean and standard deviation of ALL travel distances (both teams combined).
avg_dist = pd.concat([df['Distance from LS'], df['Distance from HS']]).mean()
std = pd.concat([df['Distance from LS'], df['Distance from HS']]).std()

print("===========================================================")
print("DISTANCE DISTRIBUTION STATISTICS (All Games, Both Teams)")
print("===========================================================")
print(f"Mean Distance: {avg_dist:.2f} miles")
print(f"Standard Deviation: {std:.2f} miles")


#Filter to games where the absolute distance gap between teams exceeds 1 standard deviation
std_df = df[(abs(df['Distance from LS'] - df['Distance from HS']) > std)]

#Rename columns again
std_df = std_df.rename(columns={'FavScore': 'LSScore'})
std_df = std_df.rename(columns={'DawgScore': 'HSScore'})

std_df['Outcome'] = std_df['LSScore'] - std_df["HSScore"] + std_df["Spread"]

#Determines Success variable for each game
std_df['Success'] = np.where(((std_df['Outcome'] < 0) & (std_df['Distance from LS'] > std_df['Distance from HS'])) |((std_df['Outcome'] > 0) & (std_df['Distance from LS'] < std_df['Distance from HS'])),1, 0)

stdsuc = std_df['Success'].sum()
stdtrials = std_df['Success'].count()
print("===========================================================")
print("1 STD DISTANCE GAP ANALYSIS")
print("===========================================================")
print(f"STD Number of Successes:           {stdsuc}")
print(f"STD Number of Trials:              {stdtrials}")
print(f"STD Success Rate:                  {stdsuc/stdtrials*100}%")
result.std = stats.binomtest(stdsuc, stdtrials, p=0.5)
print(f"STD P-Value (one-sided):           {result.std.pvalue/2}")


#STD Seed Breakdown
#Split by which team has the location advantage, then by seed bracket
std_high_df = std_df[(std_df["Distance from LS"] > std_df["Distance from HS"])]  
std_low_df  = std_df[(std_df["Distance from LS"] < std_df["Distance from HS"])]  

std_onefour_df       = std_low_df[(std_low_df['LS'] <= 4)]
std_fiveeight_df     = std_low_df[(std_low_df['LS'] > 4)]
std_ninetwelve_df    = std_high_df[(std_high_df['HS'] <= 12)]
std_thirteensixteen_df = std_high_df[(std_high_df['HS'] > 12)]

std_of_suc = std_onefour_df['Success'].sum()
std_of_trials = std_onefour_df['Success'].count()
std_fe_suc = std_fiveeight_df['Success'].sum()
std_fe_trials = std_fiveeight_df['Success'].count()
std_nt_suc = std_ninetwelve_df['Success'].sum()
std_nt_trials = std_ninetwelve_df['Success'].count()
std_ts_suc = std_thirteensixteen_df['Success'].sum()
std_ts_trials = std_thirteensixteen_df['Success'].count()

std_high_suc = std_high_df['Success'].sum()
std_high_trials = std_high_df['Success'].count()
std_low_suc = std_low_df['Success'].sum()
std_low_trials = std_low_df['Success'].count() 

print("-----------------------------------------------------------")
print("STD — High Seed Advantage (LS travels farther)")
print("-----------------------------------------------------------")
print(f"STD Number of High Seed Successes:    {std_high_suc}")
print(f"STD Number of High Seed Advantages:   {std_high_trials}")
print(f"STD High Seeded Success Rate:         {100*std_high_suc/std_high_trials}%")
result.std_high = stats.binomtest(std_high_suc, std_high_trials, p=0.5)
print(f"STD High Seeded P-Value:              {result.std_high.pvalue/2}")

print("-----------------------------------------------------------")
print("STD — Low Seed Advantage (HS travels farther)")
print("-----------------------------------------------------------")
print(f"STD Number of Low Seed Successes:     {std_low_suc}")
print(f"STD Number of Low Seed Advantages:    {std_low_trials}")
print(f"STD Low Seeded Success Rate:          {100*std_low_suc/std_low_trials}%")
result.std_low = stats.binomtest(std_low_suc, std_low_trials, p=0.5)
print(f"STD Low Seeded P-Value:               {result.std_low.pvalue/2}")

print("-----------------------------------------------------------")
print(f"STD 1-4 Seed Successes:               {std_of_suc}")
print(f"STD 1-4 Seed Advantages:              {std_of_trials}")
print(f"STD 1-4 Seeded Success Rate:          {100*std_of_suc/std_of_trials}%")
result.std_of = stats.binomtest(std_of_suc, std_of_trials, p=0.5)
print(f"STD 1-4 Seeded P-Value:               {result.std_of.pvalue/2}")

print("-----------------------------------------------------------")
print(f"STD 5-8 Seed Successes:               {std_fe_suc}")
print(f"STD 5-8 Seed Advantages:              {std_fe_trials}")
print(f"STD 5-8 Seeded Success Rate:          {100*std_fe_suc/std_fe_trials}%")
result.std_fe = stats.binomtest(std_fe_suc, std_fe_trials, p=0.5)
print(f"STD 5-8 Seeded P-Value:               {result.std_fe.pvalue/2}")

print("-----------------------------------------------------------")
print(f"STD 9-12 Seed Successes:              {std_nt_suc}")
print(f"STD 9-12 Seed Advantages:             {std_nt_trials}")
print(f"STD 9-12 Seeded Success Rate:         {100*std_nt_suc/std_nt_trials}%")
result.std_nt = stats.binomtest(std_nt_suc, std_nt_trials, p=0.5)
print(f"STD 9-12 Seeded P-Value:              {result.std_nt.pvalue/2}")

print("-----------------------------------------------------------")
print(f"STD 13-16 Seed Successes:             {std_ts_suc}")
print(f"STD 13-16 Seed Advantages:            {std_ts_trials}")
print(f"STD 13-16 Seeded Success Rate:        {100*std_ts_suc/std_ts_trials}%")
result.std_ts = stats.binomtest(std_ts_suc, std_ts_trials, p=0.5)
print(f"STD 13-16 Seeded P-Value:             {result.std_ts.pvalue/2}")

#=============================================================================
#SECTION 4: COMBINED ANALYSIS (DistRatio AND STD criteria both met)
#=============================================================================
#Games that qualify under BOTH the Distance Ratio filter and the STD filter.
both_df = two_df[two_df.index.isin(std_df.index)]

both_high_df = both_df[(both_df["Distance from LS"] > both_df["Distance from HS"])]
both_low_df  = both_df[(both_df["Distance from LS"] < both_df["Distance from HS"])]

both_suc = both_df['Success'].sum()
both_trials = both_df['Success'].count()

print("===========================================================")
print("COMBINED ANALYSIS (DistRatio AND 1 STD criteria both met)")
print("===========================================================")
print(f"BOTH Number of Successes:             {both_suc}")
print(f"BOTH Number of Trials:                {both_trials}")
print(f"BOTH Success Rate:                    {both_suc/both_trials*100}%")
result.both = stats.binomtest(both_suc, both_trials, p=0.5)
print(f"BOTH P-Value (one-sided):             {result.both.pvalue/2}")

print("-----------------------------------------------------------")
both_high_suc = both_high_df['Success'].sum()
both_high_trials = both_high_df['Success'].count()
print(f"BOTH High Seed Successes:             {both_high_suc}")
print(f"BOTH High Seed Advantages:            {both_high_trials}")
print(f"BOTH High Seeded Success Rate:        {100*both_high_suc/both_high_trials}%")
result.both_high = stats.binomtest(both_high_suc, both_high_trials, p=0.5)
print(f"BOTH High Seeded P-Value:             {result.both_high.pvalue/2}")

print("-----------------------------------------------------------")
both_low_suc = both_low_df['Success'].sum()
both_low_trials = both_low_df['Success'].count()
print(f"BOTH Low Seed Successes:              {both_low_suc}")
print(f"BOTH Low Seed Advantages:             {both_low_trials}")
print(f"BOTH Low Seeded Success Rate:         {100*both_low_suc/both_low_trials}%")
result.both_low = stats.binomtest(both_low_suc, both_low_trials, p=0.5)
print(f"BOTH Low Seeded P-Value:              {result.both_low.pvalue/2}")

print("-----------------------------------------------------------")
both_onefour_df       = both_low_df[(both_low_df['LS'] <= 4)]
both_fiveeight_df     = both_low_df[(both_low_df['LS'] > 4)]
both_ninetwelve_df    = both_high_df[(both_high_df['HS'] <= 12)]
both_thirteensixteen_df = both_high_df[(both_high_df['HS'] > 12)]

both_of_suc = both_onefour_df['Success'].sum()
both_of_trials = both_onefour_df['Success'].count()
both_fe_suc = both_fiveeight_df['Success'].sum()
both_fe_trials = both_fiveeight_df['Success'].count()
both_nt_suc = both_ninetwelve_df['Success'].sum()
both_nt_trials = both_ninetwelve_df['Success'].count()
both_ts_suc = both_thirteensixteen_df['Success'].sum()
both_ts_trials = both_thirteensixteen_df['Success'].count()

print(f"BOTH 1-4 Seed Successes:              {both_of_suc}")
print(f"BOTH 1-4 Seed Advantages:             {both_of_trials}")
print(f"BOTH 1-4 Seeded Success Rate:         {100*both_of_suc/both_of_trials}%")
result.both_of = stats.binomtest(both_of_suc, both_of_trials, p=0.5)
print(f"BOTH 1-4 Seeded P-Value:              {result.both_of.pvalue/2}")

print("-----------------------------------------------------------")
print(f"BOTH 5-8 Seed Successes:              {both_fe_suc}")
print(f"BOTH 5-8 Seed Advantages:             {both_fe_trials}")
print(f"BOTH 5-8 Seeded Success Rate:         {100*both_fe_suc/both_fe_trials}%")
result.both_fe = stats.binomtest(both_fe_suc, both_fe_trials, p=0.5)
print(f"BOTH 5-8 Seeded P-Value:              {result.both_fe.pvalue/2}")

print("-----------------------------------------------------------")
print(f"BOTH 9-12 Seed Successes:             {both_nt_suc}")
print(f"BOTH 9-12 Seed Advantages:            {both_nt_trials}")
print(f"BOTH 9-12 Seeded Success Rate:        {100*both_nt_suc/both_nt_trials}%")
result.both_nt = stats.binomtest(both_nt_suc, both_nt_trials, p=0.5)
print(f"BOTH 9-12 Seeded P-Value:             {result.both_nt.pvalue/2}")

print("-----------------------------------------------------------")
print(f"BOTH 13-16 Seed Successes:            {both_ts_suc}")
print(f"BOTH 13-16 Seed Advantages:           {both_ts_trials}")
print(f"BOTH 13-16 Seeded Success Rate:       {100*both_ts_suc/both_ts_trials}%")
result.both_ts = stats.binomtest(both_ts_suc, both_ts_trials, p=0.5)
print(f"BOTH 13-16 Seeded P-Value:            {result.both_ts.pvalue/2}")

