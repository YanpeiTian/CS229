import sys
import wordsegUtil

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

def main():

    corpus = input('Please enter training text: ')

    sys.stdout.write('Training language cost functions [corpus: %s]... ' % corpus)
    sys.stdout.flush()

    unigramCost, bigramCost = wordsegUtil.makeLanguageModels(corpus)

    print('Done!')
    print('')

    repl(unigramCost, bigramCost)

if __name__ == '__main__':
    main()
