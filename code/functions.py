import pandas as pd
import numpy as np
import json
import re
import itertools
from collections import Counter
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from spacy.matcher import PhraseMatcher
import json
import seaborn as sns
import bisect
from scipy import stats
import matplotlib.pyplot as plt


def apply_tucker_jones_tokenizer(df):    
    tokens = df.bio.apply(lambda x: re.split("[^a-zA-Z0-9/'`’-]", x)) + df.display_name.apply(lambda x: re.split("[^a-zA-Z0-9/'`’-]", x))
    tokens = tokens.apply(lambda l: [x for x in l if not x == ''])    
    return(tokens)    

def apply_whitespace_tokenizer_find_gendered_words(df, gen):
    nlp = English()
    tokenizer = nlp.tokenizer   
    tokens  = df.bio.apply(tokenizer)
    
    matcher = PhraseMatcher(nlp.vocab)    
    matcher.add('Gendered words', [nlp(w) for w in gen])
    out = tokens.apply(lambda x: matcher(nlp(x)))
    df['gendered_words'] = ['/'.join([tokens[i][y[1]:y[2]].text for y in x]) if len(x)>0 else None for i,x in enumerate(out)]
    return(df)


def load_terms(term_file):
    # load as scalar
    terms = pd.read_csv(term_file, header=None).squeeze()
    
    # make all terms lower case
    terms = terms.str.lower()
    
    return terms

def load_all_terms(pronoun_path):

    gen = set(load_terms(f'{pronoun_path}/gendered_words.csv'))
    pro = set(load_terms(f'{pronoun_path}/standard_pronouns.csv'))

    with open(f'{pronoun_path}/gender_dict.json', 'r') as fp:
        gender_dict = json.loads(fp.read())

    term_to_gender = dict()

    for gender, term_list in gender_dict.items():
        for term in term_list:
            gender = gender.capitalize()

            if gender == 'Enby':
                gender = 'Non-binary'

            term_to_gender[term] = gender
            
    return(gen, pro, term_to_gender)

def find_pronoun_combinations(df):    

    nb_pro = {"they/them", "they/them/theirs", "they/them/their", 'them/they', 'they/'}
    mixed_pro = {"she/they", "he/they", "they/she", "they/he", "she/her/they/them", "she/her/they", 
            "he/him/they/them", "he/him/they", "she/them", "he/she/they", 
            "she/they/he", 'he/them', 'they/she/he', 'they/them/she/her'}
    other_pro = nb_pro.union(mixed_pro)
    fem_pro = {"she/her", "she/her/hers", "she/hers", "she/her/ella", "she/", "she/ella", "her/she", "she/her/"}
    male_pro = {"he/him", 'he/him/his', 'he/his', 'he/him/', 'he/', 'him/he', 'he/him/el'}

    all_pro = male_pro.union(fem_pro).union(other_pro)
    
    df["theythem_pronouns"] = df.tokens.apply(lambda l: len(set(l).intersection(nb_pro)) > 0)
    df["mixed_pronouns"] = df.tokens.apply(lambda l: len(set(l).intersection(mixed_pro)) > 0)

    df["hehim_pronouns"] = df.tokens.apply(lambda l: len(set(l).intersection(male_pro)) > 0)
    df["sheher_pronouns"] = df.tokens.apply(lambda l: len(set(l).intersection(fem_pro)) > 0)
    df['anypronoun'] = df.tokens.apply(lambda l: len(set(l).intersection(all_pro)) > 0)
    
    df['pronouns'] = df.tokens.apply(lambda l: list(set(l).intersection(all_pro)))
    
    return(df)

def estimate_genders_ling(df, term_to_gender):
    pronoun_gender = list()
    word_gender = list()
    pronoun_word_gender = list()

    for i, row in df.iterrows():
        
        # gender determined by pronoun
        pronoun_val = np.nan
        if row['anypronoun']:
            if row['sheher_pronouns']:
                pronoun_val = 'Female'
            elif row['hehim_pronouns']:
                pronoun_val = 'Male'
            elif row['theythem_pronouns']:
                pronoun_val = 'Non-binary'
            else:
                pronoun_val = 'Mixed_pro'

        pronoun_gender.append(pronoun_val)

        # gender determinied by keyword:
        words = row['gendered_words']
        if words:
            genders = set([term_to_gender[word] for word in words.split('/')])
            if len(genders) == 1:
                word_val = list(genders)[0]
            else:
                word_val = 'Mixed_gen_word'

        else:
            word_val = np.nan

        word_gender.append(word_val)

        # gender from words and pronouns

        genders = {x for x in [pronoun_val, word_val] if pd.notna(x)}

        if len(genders) == 1:
            pronoun_word_gender.append(list(genders)[0])
        elif len(genders) > 1:
            pronoun_word_gender.append('Different_pro_gen_words')
        else:
            pronoun_word_gender.append(np.nan)
            
    return(pronoun_gender, word_gender, pronoun_word_gender)


def clean_handcoded_df(df, coders, verbose=False):
    # drop extra header rows
    codes = df.loc[2:].copy()
    
    # get column names for coding questions
    all_questions = codes.columns[17:]
    
    # drop 2nd instance of user who was included twice
    questions = [item for item in all_questions if '.' not in item]
    
    # get column names for gender questions
    gender_questions = [item for item in questions if item.isnumeric()]
    
    # get column names for questions about being trans
    trans_questions = [item for item in questions if not item.isnumeric()]
        
    # add column for unique coder
    codes['Coder'] = [coders[ip] for ip in codes['IPAddress']]
    
    ################ GENDER ###################
    # New dataframe with just gender questions
    # columns: 
    #       ResponseId (coder)
    #       UserID (person being coded)
    #       Code: Code assigned 
    gender_df = pd.melt(codes, id_vars='Coder', value_vars=gender_questions,
                        var_name='TweetId', value_name='Code')
    
    # New dataframe with just questions about being trans
    # columns: 
    #       ResponseId (coder)
    #       UserID (person being coded)
    #       Code: Code assigned 
    trans_df = pd.melt(codes, id_vars='Coder', value_vars=trans_questions,
                       var_name='TweetId', value_name='Code')
    
    if verbose:
        # print counts of questions/columns of each type
        print(f'{len(all_questions)} total questions')
        print(f'{len(gender_questions)} about gender')
        print(f'{len(trans_questions)} about being trans\n')

        # print size of dataframes
        print(f'{len(gender_df)} codings of gender (users X coders)')
        print(f'{len(trans_df)} codings of being transgender (users X coders)')
    
    return gender_df, trans_df

def load_handcoded_files(file_list, coders, verbose=False):
    gender_dfs = list()
    trans_dfs = list()

    for file in file_list:
        df = pd.read_csv(file)

        # clean and split
        if verbose:
            print(f'\nProcessing {file}...')
        gender_df, trans_df = clean_handcoded_df(df, coders, verbose)

        gender_dfs.append(gender_df)
        trans_dfs.append(trans_df)
        
    gender_df = pd.concat(gender_dfs, ignore_index=True)
    trans_df = pd.concat(trans_dfs, ignore_index=True)
    
    # 3 users were included in both the all coded group and Batch A, so drop those
    gender_df.drop_duplicates(subset=['Coder', 'TweetId'], keep='last', inplace=True)
    trans_df.drop_duplicates(subset=['Coder', 'TweetId'], keep='last', inplace=True)

    # trans df needs some extra cleaning because tweet id is followed by _conf
    trans_df['TweetClean'] = [item.split('_')[0] for item in trans_df['TweetId']]
    trans_df = trans_df[['Coder', 'TweetClean', 'Code']]
    trans_df.rename(columns={'TweetClean':'TweetId'}, inplace=True)
        
    return gender_df, trans_df

def code_check(df, topic, col_names):
    code = list()
    coders = list()
    code_count = list()

    for i, row in df.iterrows():
        codes = row[col_names]
        codes.dropna(inplace=True)

        # number of people who gave a code
        n_coders = len(codes)
        coders.append(n_coders)

        # unique codes
        unique_codes = set(codes)
        code_count.append(len(unique_codes))

        # if everyone agrees
        if len(unique_codes) == 1:
            code.append(codes[0])
        else:
            code.append('Mixed')
            
    # append as columns to dataframe
    df[f'code_{topic}'] = code
    df[f'code_count_{topic}'] = code_count
    
    if topic == 'trans':
        df['n_coders'] = coders
    
    return df    

def clean_handcoding(raw_df, topic):    
    
    # pivot so one user per row
    df = pd.pivot(raw_df, 
                  index='TweetId',
                  columns='Coder',
                  values='Code')

    # rename columns with what is being coded
    new_names = {col:f'{col}_{topic}' for col in df.columns}
    df.rename(columns=new_names, inplace=True)

    df.reset_index(inplace=True)
    
    # check codes
    df = code_check(df, topic, new_names.values())

    print(f'Total of {len(df)} users coded for {topic}')
    
    return df

def recode_genders(df): 
    df.loc[df.voter_file_sex == "Unknown", "voter_file_sex"] = "Not sure"

    df["pronoun_word_gender_group"] = df["pronoun_word_gender"]    
    df.loc[df.pronoun_word_gender_group.isin({"Mixed_gen_word", "Mixed_pro", "Different_pro_gen_words"} ),
                       "pronoun_word_gender_group"] = "Mixed"
    
    if "pronoun_word_gender_2023" in df.columns:
        df["pronoun_word_gender_2023_group"] = df["pronoun_word_gender_2023"]    
        df.loc[df.pronoun_word_gender_2023_group.isin({"Mixed_gen_word", "Mixed_pro", "Different_pro_gen_words"} ),
                           "pronoun_word_gender_2023_group"] = "Mixed"
    
    return(df)


def calc_stats(var, df, dec = 2):
    return(
        df.groupby(var, dropna = False).agg(
            mean_retweet_avg= ('retweet_avg', np.mean),
            median_retweet_avg= ('retweet_avg', np.median),
            mean_likes_avg= ('likes_avg', np.mean),
            median_likes_avg= ('likes_avg', np.median),
            mean_followers= ('followers', np.mean),
            median_followers= ('followers', np.median),
            mean_ntweets= ('n_tweets', np.mean),
            median_ntweets= ('n_tweets', np.median),
            n_users= ('n_tweets', 'count')
        ).round(2)            
    )

def dist_log_bin(x,n_bins, x_0 = False, x_n = False):
    if not x_0 and not x_n:
        x_0 = min(x)
        x_n = max(x)

    q_0 = np.log(x_0)
    q_n = np.log(x_n)
    
    ## total range and length in log bins
    D = q_n - q_0
    L = D/n_bins
    
    ### bin limits and central points in log space
    q = [q_0 + L*i for i in range(0,n_bins)]
    q.append(q_n)
    Q = [(q[i]+q[i+1])/2 for i in range(0, n_bins)]
    
    ### bin limits and central points in linear space
    b = [round(np.exp(qi),10) for qi in q]
    b[-1] = b[-1]+0.00001
    X = [np.exp(Qi) for Qi in Q]
    
    ### I now count how many x there is in each bin, calculate the length of each bin, and use that to calculate the Ys
    S = Counter([bisect.bisect(b, xi)-1 for xi in x])
    l = [b[i+1]-b[i] for i in range(0,n_bins)]
    Y = [S.get(i,0)/(l[i]*len(x)) for i in range(0, n_bins)]
    
    X = np.asarray(X)
    Y = np.asarray(Y)
    l = np.asarray(l)
    
    return(X,Y, l)

def log_bin(data, nbins=20, x_0 = False, x_n = False):
    data = data.loc[data > 0]
    x,y,l = dist_log_bin(data, nbins, x_0, x_n)
    x = x[y > 0]
    y = y[y > 0]
    return(x,y)

palette = sns.color_palette("colorblind", 10)

def plot_hist(var, df, group, n_bins=20, log = "loglog", x_0 = False, x_n = False,
              title = False, path = False, groups = False, size = (8,6)):
    plt.figure(figsize=size)
    
    df_cop = df.loc[(~pd.isna(df[group]))]
    if not groups:
        groups = df_cop[group].unique()
    for attr in groups :
        if "Female" in attr:
            color = palette[0]
        if "Male" in attr:
            color = palette[1]
        if "Non-binary" in attr:
            color = palette[2]
        if "Mixed" == attr or "Not sure" == attr:
            color = palette[3]
        else:
            color = colors[4]
        if not attr == None:           
            data = df_cop.loc[(df[group] == attr), var]
            if log == "loglog":
                x,y = log_bin(data, n_bins, x_0, x_n)
                plt.loglog(x,y, linestyle='-', marker='o',label = attr, alpha = 0.8)
                
            if log == "linear" or log =="semilogy":
                if x_0 and x_n:
                    bins = np.linspace(x_0, x_n, n_bins+1)
                else:
                    bins = n_bins
                y,x = np.histogram(data, bins = bins)
                x = (x[1:] + x[:-1])/2
                y = y/len(data)
                if log == "semilogy":
                    x = x[y > 0]
                    y = y[y > 0]
                plt.plot(x,y, linestyle='-', marker='o',label = attr, alpha = 0.8)
                if log == "semilogy":
                    plt.yscale("log")  
    plt.legend()
    if title:
        plt.title(title, fontdict = {'fontsize':12,
                                     'fontweight':"bold"}, pad = 20)
    if var == "retweet_avg":
        plt.xlabel("Average number of retweets per tweet")
    if var == "followers":
        plt.xlabel("Number of followers")
    if var == "likes_avg":
        plt.xlabel("Average number of likes per tweet")
    
    plt.ylabel("Probability density")

    if path:
        plt.savefig(path, dpi = 200, bbox_inches="tight")
        
    return(plt)

def plot_mul_hist(df, group, n_bins=20, log = "loglog",
              title = False, path = False, groups = False, size = (8,6)):
    
    fig, ax = plt.subplots(1, 3, figsize = size)    
    df_cop = df.loc[(~pd.isna(df[group]))]
    if not groups:
        groups = df_cop[group].unique()
        
    for i,var in enumerate(["retweet_avg", "likes_avg", "followers"]):
        
        for attr in groups:
            if "Female" in attr:
                color = palette[0]
            elif "Male" in attr:
                color = palette[1]
            elif "Non-binary" in attr:
                color = palette[2]
            elif "Mixed" in attr:
                color = palette[3]
            else:
                color = palette[4]
                
            if not attr == None:           
                data = df_cop.loc[(df[group] == attr), var]
                x_0 = min([x for x in df_cop[var] if x >0])
                x_n = max(df_cop[var])+1
                
                if log == "loglog":
                    x,y = log_bin(data, n_bins, x_0, x_n)
                    ax[i].loglog(x,y, linestyle='-', marker='o',label = attr, alpha = 0.8, c = color)

                if log == "linear" or log =="semilogy":
                    if x_0 and x_n:
                        bins = np.linspace(x_0, x_n, n_bins+1)
                    else:
                        bins = n_bins
                    y,x = np.histogram(data, bins = bins)
                    x = (x[1:] + x[:-1])/2
                    y = y/len(data)
                    if log == "semilogy":
                        x = x[y > 0]
                        y = y[y > 0]
                    ax[i].plot(x,y, linestyle='-', marker='o',label = attr, alpha = 0.8, color = color)
                    if log == "semilogy":
                        plt.yscale("log")  
                        
                if var == "retweet_avg":
                    ax[i].set_xlabel("Average number of retweets per tweet")
                if var == "followers":
                    ax[i].set_xlabel("Number of followers")
                if var == "likes_avg":
                    ax[i].set_xlabel("Average number of likes per tweet")
                    
    handles, labels = ax[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.4, 0.2))
    fig.tight_layout()
        
    if title:
        fig.title(title, fontdict = {'fontsize':12,
                                     'fontweight':"bold"}, pad = 20)
    ax[0].set_ylabel("Probability density")

    if path:
        plt.savefig(path, dpi = 200, bbox_inches="tight")
        
    return(plt)



def run_mann(df, attr, group1, group2, var, alt = "two-sided"):
    out = stats.mannwhitneyu(
        df.loc[df[attr] == group1, var], 
        df.loc[df[attr] == group2, var],
        alternative = alt,
    )
    return((out[0], out[1]))

def run_all_mann(df, attr, group1, group2, alt = "two-sided"):
    likes = run_mann(df, attr, group1, group2, "likes_avg", alt)
    retweets = run_mann(df, attr, group1, group2, "retweet_avg", alt)
    followers = run_mann(df, attr, group1, group2, "followers", alt)
    return({"likes":likes, 
            "retweets": retweets, 
            "followers" :followers})