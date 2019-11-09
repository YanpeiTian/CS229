"""
This file parse raw csv file and output processed train/test data set
"""
import pandas as pd
import numpy as np
import wordsegUtil


def parser(file_path,file):
    csv_name = file_path
    df = pd.read_csv(csv_name)


    for i in range(df.shape[0]): # go through each answer post
        if df['Id'][i]==df['ParentAcceptedAnswerId'][i]:
            body = df['Body'][i]
            line=wordsegUtil.cleanLine(body)
            file.write(line)


def main():
    file=open("training.txt",'w')
    # parser('one_month_2018-04-01_2018-05-01.csv',file)
    parser('one_month_2018-07-01_2018-08-01.csv',file)
    file.close()

if __name__ == '__main__':
    main()
