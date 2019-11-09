"""
This file parse raw csv file and output processed train/test data set
"""
import pandas as pd
import numpy as np


def paser(raw_path, fluency_path):
    csv_name = raw_path
    df = pd.read_csv(csv_name)
    df_fluency = pd.read_csv(fluency_path)


    # code
    codes_inline = []
    codes_pre_line = []
    codes_pre_count = []

    # hyperlink
    hyperlinks = []

    #Edit and Label
    edits = []
    labels = []

    # others
    parsed_CommentCount = []
    parsed_BodyLength = []
    parsed_UserReputation = []
    parsed_UserViews = []
    parsed_UserUpVotes = []
    parsed_UserDownVotes = []



    for i in range(df.shape[0]): # go through each answer post

        # label:
        answer_id = df['Id'][i]  # int
        accepted_id = df['ParentAcceptedAnswerId'][i]  # float (np.nan)
        if np.isnan(accepted_id):
            #labels.append(np.nan)
            continue # if the question owner did not select a preferred answer post, do not use this instance
        else:
            if answer_id == int(accepted_id):
                labels.append('1')
            else:
                labels.append('0')


        # code:
        body = df['Body'][i]
        code_start = 0
        code_end = 0
        code_pre_line = 0
        code_pre_count = 0
        code_inline = 0

        while True:
            code_start = body.find('<code>', code_end)
            if code_start == -1:
                break
            code_end = body.find('</code>', code_start)
            code = body[code_start: code_end+7]
            if body[code_start-5: code_start] == '<pre>':
                code_line = code.split('\n')
                # if '' in code_line: # Remove blank lines
                    # while '' in code_line:
                    #    code_line.remove('')
                code_line_count = len(code_line)-1 # Last line as </code>
                code_pre_line += code_line_count
                code_pre_count += 1
            else:
                code_inline += 1
        codes_inline.append(code_inline)
        codes_pre_line.append(code_pre_line)
        codes_pre_count.append(code_pre_count)


        # hyperlink
        hyperlink_pos = -1
        hyperlink = 0
        while True:
            hyperlink_pos = body.find('<a href=', hyperlink_pos + 1)
            if hyperlink_pos == -1:
                break
            hyperlink += 1
            # print(i, body[hyperlink_pos:hyperlink_pos + 50])
        hyperlinks.append(hyperlink)


        # edit:
        last_edit_date = df['LastEditDate'][i]  # str or float (np.nan)
        if isinstance(last_edit_date, float):
            if np.isnan(last_edit_date):
                edits.append('0')
            else:
                edits.append('1')
        else:
            edits.append('1')


        # others:
        #CommentCount','BodyLength', 'UserReputation','UserViews','UserUpVotes','UserDownVotes'
        parsed_CommentCount.append(df['CommentCount'][i])
        parsed_BodyLength.append(df['BodyLength'][i])
        parsed_UserReputation.append(df['UserReputation'][i])
        parsed_UserViews.append(df['UserViews'][i])
        parsed_UserUpVotes.append(df['UserUpVotes'][i])
        parsed_UserDownVotes.append(df['UserDownVotes'][i])



    # create a output panda dataframe
    output = pd.DataFrame(list(zip(parsed_CommentCount,parsed_BodyLength, parsed_UserReputation,parsed_UserViews,parsed_UserUpVotes,parsed_UserDownVotes,codes_inline,codes_pre_count,codes_pre_line,hyperlinks,edits,labels)), columns =['parsed_CommentCount','parsed_BodyLength', 'parsed_UserReputation','parsed_UserViews','parsed_UserUpVotes','parsed_UserDownVotes','InlineCode','BlockCode','BlockCodeLine','Hyperlink','Edit','Label'])
    output = pd.concat([df_fluency, output], axis=1, sort=False) # include the fluency cols


    output_name = csv_name.replace('.csv','_merged.csv')
    output.to_csv(output_name)





if __name__ == '__main__':
    paser("../Example Data/one_day_2018-03-01_2018-03-02.csv", "../Language Model/one_day_2018-03-01_2018-03-02_fluency.csv")
    paser("../Example Data/one_day_2018-06-01_2018-06-02.csv", "../Language Model/one_day_2018-06-01_2018-06-02_fluency.csv")
    paser("../Example Data/one_month_2018-04-01_2018-05-01.csv", "../Language Model/one_month_2018-04-01_2018-05-01_fluency.csv")





