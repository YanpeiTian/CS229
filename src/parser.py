"""
This file parse raw csv file and output processed train/test data set
"""
import pandas as pd
import numpy as np


# following are for bert:
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig, BertModel
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import io
import matplotlib.pyplot as plt
#% matplotlib inline

from bs4 import BeautifulSoup


# following are the pypi bert embeddings:
from bert_embedding import BertEmbedding




def clear_text(body):
    """
    clear the hyper links in a paragraph
    :param body:
    :return:
    """
    soup = BeautifulSoup(body, features="html.parser")
    for a in soup.findAll('a'):
        # print(a)
        # del a['href']
        a.replaceWithChildren()

    # for code in soup.findAll('code'):
    #     # print(a)
    #     # del a['href']
    #     print("888888888888888888")
    #     print(code)
    #     print("888888888888888888")
    #     #code.replaceWithChildren()
    #
    #     del code

    return str(soup)


def bert_feature_extraction_pypi(bodys):
    # used the new word embedding package:
    print("extracting feature for bert features..")
    #print("The shape of input: ", bodys.shape)


    # We need to add special tokens at the beginning and end of each sentence for BERT to work properly
    bodys = [clear_text(body) for body in bodys] # clear the hyperlinks

    bert_embedding = BertEmbedding(max_seq_length=512)

    result = bert_embedding(bodys)
    num_bodys = len(result)

    whole_body_vectors = []
    for i in range(len(result)): # for every body
        token_vectors = result[i][1] # should be vectors of length (768,)
        token_vectors = np.stack(token_vectors, axis=0) # stack all token's vector together

        whole_body_vector = np.mean(token_vectors, axis=0) # take average of all tokens
        whole_body_vectors.append(whole_body_vector)
        print("processed body: ",i*100/num_bodys,"%")
    # print(len(whole_body_vectors), whole_body_vectors[0].shape)



    # stack all body vectors together and become a numpy matrix
    whole_body_vectors = np.stack(whole_body_vectors, axis=0)
    # convert the np matrix to be panda dataframe
    print(whole_body_vectors.shape)
    df_text_vectors = pd.DataFrame(whole_body_vectors)

    return df_text_vectors



def bert_feature_extraction(bodys):
    """

    :param bodys: body coloumn in data frames read from csv
    :return: panda dataframes
    """

    print("extracting feature for bert..")
    print("size of input: ", len(bodys))

    # Create sentence and label lists
    #bodys = df.Body.values

    # We need to add special tokens at the beginning and end of each sentence for BERT to work properly
    bodys = ["[CLS] " + clear_text(body) + " [SEP]" for body in bodys]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    tokenized_texts = [tokenizer.tokenize(sent) for sent in bodys]

    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

    # Pad our input tokens
    input_ids = pad_sequences(input_ids, maxlen=512, dtype="long", truncating="post", padding="post")

    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)


    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([input_ids])
    masks_tensors = torch.tensor([attention_masks])

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased')

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()



    # find the vector representation for each text body
    text_vectors = []
    for i in range(tokens_tensor.shape[1]):  # for every text bodys:
        encoded_layers = None
        with torch.no_grad():
            # encoded_layers, _ = model(tokens_tensor, token_type_ids=None, attention_mask=masks_tensors)
            encoded_layers, _ = model(tokens_tensor[:, i, :], token_type_ids=None,
                                      attention_mask=masks_tensors[:, i, :])

        # Concatenate the tensors for all layers. We use `stack` here to
        # create a new dimension in the tensor.
        token_embeddings = torch.stack(encoded_layers, dim=0)

        # token_embeddings.size()

        # Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(token_embeddings, dim=1)

        # token_embeddings.size()

        # Finally, we can switch around the "layers" and "tokens" dimensions with permute.
        # Swap dimensions 0 and 1.
        token_embeddings = token_embeddings.permute(1, 0, 2)
        # token_embeddings.size()

        # Sentence Vectors To get a single vector for our entire sentence we have multiple application-dependent strategies, but a simple approach is to average the second to last hiden layer of each token producing a single 768 length vector.
        # `encoded_layers` has shape [12 x 1 x 22 x 768]

        # `token_vecs` is a tensor with shape [22 x 768]
        token_vecs = encoded_layers[-2][0]  # second to last layer
        # print(token_vecs.shape)

        # Calculate the average of all 512 token vectors.
        sentence_embedding = torch.mean(token_vecs, dim=0)
        # print (sentence_embedding)

        print("processed body: ", i)


        text_vectors.append(sentence_embedding)

    # stack all vectors together
    text_vectors = torch.stack(text_vectors, dim=0)

    # convert the pytorch tensor to data frames
    df_text_vectors = text_vectors.numpy()
    df_text_vectors = pd.DataFrame(df_text_vectors)

    return df_text_vectors






def paser(raw_path, fluency_path):
    csv_name = raw_path
    df = pd.read_csv(csv_name)
    df_fluency = pd.read_csv(fluency_path)

    #################
    # find the bert vectors:
    # df = df.head(20)
    # df_fluency = df_fluency.head(20)

    bodys_a = []
    bodys_q = []
    for i in range(df.shape[0]): # clean the instances without selected best answer

        accepted_id = df['ParentAcceptedAnswerId'][i]  # float (np.nan)
        if np.isnan(accepted_id):
            #labels.append(np.nan)
            continue # if the question owner did not select a preferred answer post, do not use this instance
        else:
            bodys_a.append(df['Body'][i])
            bodys_q.append(df['ParentBody'][i])


    # extract bert features for both question and ans
    df_berts_a = bert_feature_extraction(bodys_a) # extract bert feature from ans
    df_berts_q = bert_feature_extraction(bodys_q) # from question body



    #################



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
    output = pd.concat([df_berts_a, df_berts_q, df_fluency['unigramCost'],df_fluency['bigramCost'], output], axis=1, sort=False) # include the fluency cols


    output_name = csv_name.replace('.csv','_merged.csv')
    output.to_csv(output_name)





if __name__ == '__main__':
    #paser("../Example Data/one_day_2018-03-01_2018-03-02.csv", "../Language Model/one_day_2018-03-01_2018-03-02_fluency.csv")
    #paser("../Example Data/one_day_2018-06-01_2018-06-02.csv", "../Language Model/one_day_2018-06-01_2018-06-02_fluency.csv")
    paser("../Example Data/one_month_2018-04-01_2018-05-01.csv", "../Language Model/one_month_2018-04-01_2018-05-01_fluency.csv")











