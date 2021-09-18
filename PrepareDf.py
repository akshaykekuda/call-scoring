import pandas as pd
import pickle
import os

transcripts_dir = '/Users/akshaykekuda/Desktop/CSR-SA/manual_score_transcriptions/output_dual/output'

def prepare_df():

    df = pd.read_pickle("/Users/akshaykekuda/Desktop/CSR-SA/ScoringDetail_viw_subscore.p")
    df = df.sort_values(by= ['RecordingDate', 'QGroupSequence', 'QuestionSequence']).copy()
    cols = ['QGroupSequence', 'QGroupName','InteractionIdKey', 'QuestionSequence', 'QuestionText', 'QuestionType', 
            'QuestionPromptType', 'QuestionWeight', 'QuestionMin', 'QuestionMax', 'AnswerScore', 'RawAnswer', 'DisplayAnswer', 
            'UserComments']
    calls_df = df[(df.QuestionnaireName == 'Call Interaction')].copy()
    q_df = calls_df[cols]
    temp = q_df[0:10]
    temp = temp.reset_index(drop=True)
    q_text =[]
    for index, row in temp.iterrows():
        q_text.append(row['QuestionText'])

    score_df = pd.DataFrame()
    score_df['WorkgroupQueue'] = calls_df.WorkgroupQueue[::10]
    score_df['RecordingDate'] = calls_df.RecordingDate[::10]
    score_df['CombinedPercentileScore'] = calls_df.CombinedPercentileScore[::10].astype(float).round(2)
    score_df['Category'] = (score_df['CombinedPercentileScore'] > 75).apply(lambda x:int(x))
    score_df.index = calls_df.InteractionIdKey[::10]

    for i in range(10):
        score_df[q_text[i]] = (calls_df.AnswerScore[i::10]).values
    score_df = score_df.loc[~score_df.index.duplicated(keep='last')]

    print("Dataframe creation done")
    """
    score_comment_df = pd.DataFrame()
    for row in calls_df.itertuples():
        score_comment_df.loc[row.InteractionIdKey,row.QuestionText] = row.UserComments
        score_comment_df.loc[row.InteractionIdKey,'CombinedPercentileScore'] = row.CombinedPercentileScore
        score_comment_df.loc[row.InteractionIdKey,'Category'] = score_df.loc[row.InteractionIdKey,'Category']
    """

    df = pd.DataFrame(columns=['text', 'file_name'])
    for file in os.listdir(transcripts_dir):
        if file.endswith('.txt'):
            file_loc = transcripts_dir + '//' + file
            f = open(file_loc, 'r')
            tscpt = f.read()
            f.close()
            if len(tscpt) == 0:
                print("empty file")
                continue
            arr = file.split("_")
            id = arr[1].split('.')[0]
            if id in score_df.index:
                    try:
                        df.loc[id, score_df.columns] = score_df.loc[id]
                        df.loc[id, ['text', 'file_name']] = [tscpt, file_loc]
                    except:
                        print(score_df.loc[id])
    df = df.dropna()
    h = len(df[df.Category == 1])
    l = len(df) - h
    temp = df[df.Category == 1].sample(h -l).index
    df_sampled = df.drop(temp)
    print('training and dev sample size = {} good ={} bad ={}'.format(len(df_sampled),l,l))
    print('test sample size = {}'.format(len(df.loc[temp])))
    remaining = df.index.difference(temp, sort=False)
    return df_sampled
