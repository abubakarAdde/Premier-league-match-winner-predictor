#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


matches = pd.read_csv("matches.csv", index_col=0)


# In[4]:


matches.head()


# 

# In[5]:


matches.shape


# In[6]:


matches["team"].value_counts()


# In[7]:


matches[matches["team"] == "Liverpool"]


# In[8]:


matches["round"].value_counts()


# In[9]:


matches.dtypes


# In[10]:


matches["date"] = pd.to_datetime(matches["date"])


# In[11]:


matches


# In[12]:


matches.dtypes


# In[13]:


matches["venue_code"] = matches["venue"].astype("category").cat.codes


# In[14]:


matches


# In[15]:


matches["opp_code"] = matches["opponent"].astype("category").cat.codes


# In[16]:


matches


# In[17]:


matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")


# In[18]:


matches


# In[19]:


matches["day_code"] = matches["date"].dt.dayofweek


# In[20]:


matches


# In[21]:


matches["target"] = (matches["result"] =="W").astype("int")


# In[22]:


matches


# In[23]:


from sklearn.ensemble import RandomForestClassifier


# In[24]:


rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)


# In[25]:


train = matches[matches["date"] < '2022-01-01']


# In[26]:


test = matches[matches["date"] > '2-22-01-01']


# In[27]:


predictors = ["venue_code", "opp_code", "hour", "day_code"]


# In[28]:


rf.fit(train[predictors], train["target"])


# In[29]:


preds = rf.predict(test[predictors])


# In[30]:


from sklearn.metrics import accuracy_score


# In[31]:


acc = accuracy_score(test["target"], preds)


# In[32]:


acc


# In[33]:


combined = pd.DataFrame(dict(actual=test["target"], prediction=preds))


# In[34]:


pd.crosstab(index=combined["actual"], columns=combined["prediction"])


# In[35]:


from sklearn.metrics import precision_score


# In[36]:


precision_score(test["target"], preds)


# In[37]:


grouped_matches = matches.groupby("team")


# In[38]:


group = grouped_matches.get_group("Manchester City")


# In[39]:


group


# In[40]:


def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(2, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group


# In[41]:


cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]


# In[42]:


new_cols


# In[43]:


rolling_averages(group, cols, new_cols)


# In[44]:


matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))


# In[45]:


matches_rolling


# In[46]:


matches_rolling = matches_rolling.droplevel('team')


# In[47]:


matches_rolling


# 

# In[48]:


matches_rolling.index = range(matches_rolling.shape[0])


# In[49]:


matches_rolling


# In[51]:


def make_predictions(data, predictors):
    train = data[data["date"] < '2022-01-01']
    test = data[data["date"] > '2022-01-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
    precision = precision_score(test["target"], preds)
    return combined, precision


# 

# In[52]:


combined, precision = make_predictions(matches_rolling, predictors + new_cols)


# In[53]:


precision


# In[54]:


combined


# In[55]:


combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)


# In[56]:


combined


# In[59]:


class MissingDict(dict):
    __missing__ = lambda self, key: key
map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Wolverhampton Wanderers": "Wolves",
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
}
mapping = MissingDict(**map_values)


# In[61]:


mapping["West Ham United"]


# In[62]:


combined["new_team"] = combined["team"].map(mapping)


# In[63]:


combined


# In[65]:


merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"])


# In[66]:


merged


# In[67]:


merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 0)]["actual_x"].value_counts()


# In[ ]:




