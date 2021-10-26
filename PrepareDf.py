import pandas as pd
import os

transcripts_dir = '/Users/akshaykekuda/Desktop/CSR-SA/manual_score_transcriptions/output_dual/output'

def prepare_score_df(path_to_p):
    df = pd.read_pickle(path_to_p)
    df = df.sort_values(by=['RecordingDate', 'QGroupSequence', 'QuestionSequence']).copy()
    cols = ['QGroupSequence', 'QGroupName', 'InteractionIdKey', 'QuestionSequence', 'QuestionText', 'QuestionType',
            'QuestionPromptType', 'QuestionWeight', 'QuestionMin', 'QuestionMax', 'AnswerScore', 'RawAnswer',
            'DisplayAnswer',
            'UserComments']
    calls_df = df[(df.QuestionnaireName == 'Call Interaction')].copy()
    q_df = calls_df[cols]
    temp = q_df[0:10]
    temp = temp.reset_index(drop=True)
    q_text = []
    for index, row in temp.iterrows():
        q_text.append(row['QuestionText'])

    score_df = pd.DataFrame()
    score_df['WorkgroupQueue'] = calls_df.WorkgroupQueue[::10]
    score_df['RecordingDate'] = calls_df.RecordingDate[::10]
    # change baseline score range accordingly
    # overall score between 0 and 1
    # score_df['CombinedPercentileScore'] = (calls_df.CombinedPercentileScore[::10]/100).astype(float).round(4)
    # score_df['Category'] = (score_df['CombinedPercentileScore'] > 0.75).apply(lambda x: int(x))
    # overall score between 0 and 100
    score_df['CombinedPercentileScore'] = (calls_df.CombinedPercentileScore[::10]).astype(float).round(2)
    score_df['Category'] = (score_df['CombinedPercentileScore'] > 75).apply(lambda x: int(x))
    score_df.index = calls_df.InteractionIdKey[::10]
    calls_df.AnswerScore = calls_df.AnswerScore.astype('int')
    for i in range(10):
        criteria = q_text[i]
        q_max = int(calls_df.QuestionMax.iloc[i])
        score_df[criteria] = (calls_df.AnswerScore[i::10]).values
        score_df[criteria] = score_df[criteria].apply(lambda x: 1 if x >= q_max else 0) #used as binary class

    score_df = score_df.loc[~score_df.index.duplicated(keep='last')]

    score_comment_df = pd.DataFrame()
    score_comment_df['WorkgroupQueue'] = calls_df.WorkgroupQueue[::10]
    score_comment_df['RecordingDate'] = calls_df.RecordingDate[::10]
    score_comment_df.index = calls_df.InteractionIdKey[::10]
    for i in range(10):
        criteria = q_text[i]
        score_comment_df[criteria] = (calls_df.UserComments[i::10]).values
    score_comment_df = score_comment_df.loc[~score_comment_df.index.duplicated(keep='last')]

    print("Dataframe creation done")

    return score_df, score_comment_df, q_text

def prepare_trancript_score_df(score_df, q_text):
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
                    df.loc[id, score_df.columns] = score_df.loc[id]
                    df.loc[id, ['text', 'file_name']] = [tscpt, file_loc]
    df.loc[:, q_text] = df.loc[:, q_text].astype(int)
    df = df.dropna()
    return df

def balance_df(df):
    h = len(df[df.Category == 1])
    l = len(df) - h
    temp = df[df.Category == 1].sample(h -l).index
    df_sampled = df.drop(temp)
    print('training and dev sample size = {} good ={} bad ={}'.format(len(df_sampled),l,l))
    print('test sample size = {}'.format(len(df.loc[temp])))
    remaining = df.index.difference(temp, sort=False)
    return df_sampled, df.loc[temp]


def prepare_baseline_df():
    cs_df = pd.read_csv('/Users/akshaykekuda/Desktop/CSR-SA/Baseline_Scores/predicted_score.csv', names=['InteractionId', 'score', 'len', 'quote_score'])
    sales_df = pd.read_csv('/Users/akshaykekuda/Desktop/CSR-SA/Baseline_Scores/predicted_sales_score.csv', names=['InteractionId', 'score', 'len', 'quote_score'])
    cs_df.InteractionId = cs_df.InteractionId.apply(lambda x: x.split('_')[1])
    sales_df.InteractionId = sales_df.InteractionId.apply(lambda x: x.split('_')[1])
    cs_df = cs_df.set_index('InteractionId')
    sales_df = sales_df.set_index('InteractionId')
    return cs_df, sales_df