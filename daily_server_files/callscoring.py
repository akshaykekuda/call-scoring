# load packages
from warnings import filterwarnings
import os
import numpy as np
import pandas as pd
import sys
import re

import pyodbc
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

def main(day_to_run):
  filterwarnings('ignore')

  d1 = (day_to_run + timedelta(-6)).strftime('%Y-%m-%d')
  d2 = (day_to_run + timedelta(6)).strftime('%Y-%m-%d')

  # import calls info from HQDBSQLPRD02
  cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=PGESQLFIDC01\MSSQLSERVEREQ;uid=sasvc-DDSG;pwd=:iLIioXrsd')
  
  phonedb = cnxn.cursor()
  
  sql = """
  SELECT distinct [CallId], [InitiatedDateTimeGMT], [AssignedWorkGroup], [LocalUserId], [LocalName], r.RecordingId
  FROM [I3_IC].[dbo].[calldetail_viw] c JOIN I3_IC.dbo.RecordingDetail_viw r ON c.CallId = r.InteractionId
  WHERE [InitiatedDateTimeGMT] BETWEEN '"""+str(d1)+"""' AND '"""+str(d2)+"""' AND r.ICUserID != ' -'
  """
  
  phonedb.execute(sql)
  
  info = pd.DataFrame.from_records(phonedb.fetchall(),columns=['CallId','InitiatedDateTimeGMT','AssignedWorkGroup','Local User','Local Name','RecordingId'])
  cnxn.close()
  
  # import customer service calls
  list_df = []
  for i in np.arange(1, 6):
      fpath = '/mnt/callratings/daily_scores/'+str(day_to_run)+'/server0'+str(i)+'.csv'
      if(os.path.exists(fpath)): 
          df = pd.read_csv(fpath)
          list_df.append(df)
  
  if len(list_df)==0:
    raise Exception("No daily score files found")
    return

  data = pd.concat(list_df, ignore_index=False)
  
  data['id'] = data['id'].astype('str')
  info['CallId'] = info['CallId'].astype('str')
  info['RecordingId'] = info['RecordingId'].astype('str')

  # Genesys recording id always takes this format.
  recordingid_regex = r'[1-5]-([a-zA-Z0-9]{8,8}-[a-zA-Z0-9]{4,4}-[a-zA-Z0-9]{4,4}-[a-zA-Z0-9]{4,4}-[a-zA-Z0-9]{12,12})'   
  check_for_rid = re.compile(recordingid_regex)

  # This conditional is required because file naming conventions don't seem to be 
  # consistent for certain machines. This code used to assume that file names
  # took the form [InteractionId]_[Department].opus This was not always the case.
  # Some machines export the call files to [RecordingId].opus
  if check_for_rid.match(data['id'].iloc[0]) is not None:
    data['id'] = data['id'].apply(lambda x: check_for_rid.match(x).group(1).upper())   
    merge_var = 'RecordingId'
  else:
    merge_var = 'CallId'
    # data['id'] = data['id'].apply(lambda x: x.split('-')[1].split('_')[0])

  res = data.merge(info, how='left', left_on='id', right_on=merge_var)
  
  # This line is a hacky fix to deal with inconsistent file names. The old joins broke if the 
  # opus files has a recordingid in the filename and not the interaction id.
  res['id']=res['CallId']

  res.drop('CallId', axis=1, inplace=True)
  res.drop('RecordingId',axis=1,inplace=True)
  res = res.dropna()
  res['DateTime'] = pd.to_datetime(res['InitiatedDateTimeGMT'])
  res  = res.drop(['InitiatedDateTimeGMT'], axis=1)

  # res.columns = ['Interaction ID', 'CS_Score', 'Sale_Score', 'DateTime', 'Type', 'Local User', 'Local Name', 'Score']
  # res2 = res[['Interaction ID', 'DateTime', 'Type', 'Score', 'Local User', 'Local Name']]
  
  
  # res2['Interaction ID'] = res2['Type'] + '_' + res2['Interaction ID']
  # res2['DateTime'] = pd.to_datetime(res2['DateTime'])
  
  d = day_to_run
  filename = "/mnt/callratings/upload_to_onedrive/Call_" + str(d.month).zfill(2) + str(d.day).zfill(2) + ".csv"
  res.to_csv(filename, index=False,encoding="utf-8")
  
  upload_name = "/mnt/callratings/testmnt/Call_" + str(d.month).zfill(2) + str(d.day).zfill(2) + ".csv"
  res.to_csv(upload_name, index=False, encoding="utf-8")


if __name__ == "__main__":
  # This script defaults to being used for a daily run. 
  # By passing in a date of the for YYYY-MM-DD it will backdate.
  if len(sys.argv)==1:
    # Time delta of -2 is used here because it is scoring the transcribed
    # calls from the previous days run. 
    # If the transcription routine runs on 2020-08-03 it will transcribe calls
    # from 2020-08-01.
    main(datetime.today().date()+timedelta(-2))
  else:
    # This conditional allows you to specify the date of the transcription run you want to score.
    # For example, if you pass 2020-08-20 it will score the calls that were transcribed on 
    # 2020-08-20, NOT the calls that were recorded on 2020-08-20.
    day2run=datetime.strptime(sys.argv[1],"%Y-%m-%d").date()
    main(day2run)
