import sys
import wordsegUtil
import pandas as pd
import numpy as np

FILE_PATH='/Users/yanhaojiang/Desktop/CS229/Example Data/half_year_2017-07-01_2018-01-01.csv'

# REPL and main entry point
def repl(unigramCost, bigramCost):
    '''REPL: read, evaluate, print, loop'''

    while True:
        sys.stdout.write('>> ')
        line = sys.stdin.readline().strip()
        if not line:
            break

        cmdAndLine = line.split(None, 1)
        cmd, line = cmdAndLine[0], ' '.join(cmdAndLine[1:])

        print('')

        if cmd == 'help':
            print('Usage: <command> [arg1, arg2, ...]')
            print('')
            print('Commands:')
            print(('\n'.join(a + '\t\t' + b for a, b in [
                ('help', 'This'),
                ('ug', 'Query unigram cost function, treating input as a single word'),
                ('bg', 'Call bigram cost function on the last two words of the input'),
            ])))
            print('')
            print('Enter empty line to quit')

        elif cmd == 'ug':
            grams=tuple(wordsegUtil.cleanLine(line))
            cost=0
            for i in range(len(grams)):
                cost+=unigramCost(grams[i])
            cost=cost/len(grams)
            print(cost)

        elif cmd == 'bg':
            grams = tuple(wordsegUtil.words(line))
            if len(grams)<2:
                print("Text too short. Enter again.")
                print('')
                continue
            cost=bigramCost(wordsegUtil.SENTENCE_BEGIN, grams[0])
            for i in range(1,len(grams)):
                cost+=bigramCost(grams[i-1], grams[i])
            cost=cost/len(grams)
            print(cost)

        else:
            print(('Unrecognized command:', cmd))

        print('')

def calculate(unigramCost, bigramCost):
    csv_name = FILE_PATH
    df = pd.read_csv(csv_name)

    ug = []
    bg = []

    for i in range(df.shape[0]):
        accepted_id = df['ParentAcceptedAnswerId'][i]
        if np.isnan(accepted_id):
            continue

        line = df['Body'][i]

        # Unigram cost.
        grams=tuple(wordsegUtil.cleanLine(line))
        cost=0
        for j in range(len(grams)):
            cost+=unigramCost(grams[j])
        cost=cost/len(grams)
        ug.append(cost)

        cost=bigramCost(wordsegUtil.SENTENCE_BEGIN, grams[0])
        for j in range(1,len(grams)):
            cost+=bigramCost(grams[j-1], grams[j])
        cost=cost/len(grams)
        bg.append(cost)


    # create a output panda dataframe
    output = pd.DataFrame(list(zip(ug,bg)), columns =['unigramCost','bigramCost'])


    output_name = csv_name.replace('.csv','_fluency.csv')
    output.to_csv(output_name)

def main():

    corpus = 'training.txt'

    sys.stdout.write('Training language cost functions [corpus: %s]... ' % corpus)
    sys.stdout.flush()

    unigramCost, bigramCost = wordsegUtil.makeLanguageModels(corpus)

    print('Done!')
    print('')

    # repl(unigramCost, bigramCost)
    calculate(unigramCost, bigramCost)

if __name__ == '__main__':
    main()
