import pandas as pd


def combine_csvs(all_file_paths, output_path):

    df = None # the final combined dataframe


    for path in all_file_paths:
        if df is None:
            df = pd.read_csv(path)
        else:
            df2 = pd.read_csv(path)
            df = pd.concat([df, df2], axis=0, sort=False)

        # df = df.drop(df.columns[0], axis=1) # drop the panda df index
    df.to_csv(output_path, index=False)
    return








if __name__ == '__main__':
    combine_csvs(["/Users/yanhaojiang/Desktop/CS229/Example Data/two_month_2017-07-01_2017-09-01.csv","/Users/yanhaojiang/Desktop/CS229/Example Data/two_month_2017-09-01_2017-11-01.csv","/Users/yanhaojiang/Desktop/CS229/Example Data/two_month_2017-11-01_2018-01-01.csv"], "/Users/yanhaojiang/Desktop/CS229/Example Data/half_year_2017-07-01_2018-01-01.csv")
