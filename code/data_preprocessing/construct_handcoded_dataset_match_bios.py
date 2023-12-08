import numpy as np
import pandas as pd
import glob
import json
import gzip
import re
import functions as f

mainpath = ''

# where raw handcoding data is stored
inpath = f'{mainpath}/data/raw/hand_coding'
all_coding_file = f'{inpath}/CodingTest_AllCoders.csv'

batchA_files = [f'{inpath}/A{i}_coding.csv' for i in range(1,5)]
batchB_files = [f'{inpath}/B{i}_coding.csv' for i in range(1,5)]
all_files = [all_coding_file] + batchA_files + batchB_files

# use IP address to figure out unique coders
# This is better than response ID which changes with each survey
coders = {
    'IP1a' : 'Coder1', # BatchA
    'IP1b'  : 'Coder1', # BatchA
    'IP2' : 'Coder2',  # BatchA
    'IP3' : 'Coder3',  # BatchB
    'IP4a' : 'Coder4',     # BatchB
    'IP4b': 'Coder4',    # BatchB
    'IP4c': 'Coder4'     # Initial coding
}

gender_df, trans_df = f.load_handcoded_files(all_files, coders)

gender_df = f.clean_handcoding(gender_df, 'gender')
trans_df = f.clean_handcoding(trans_df, 'trans')

# merge handcoding
# all users are in both dataframes so do outer
coding = gender_df.merge(trans_df, on='TweetId', how='outer')
print(f'Final count of {len(coding)} hand coded users')


# Get User Ids and other data from newest tweets

# raw tweets (collected 2023)
rawpath= f'{mainpath}/data/raw/raw_tweets'

all_tweets = set(coding['TweetId'])
user_details = dict()

for filepath in glob.glob(f'{rawpath}/*.json.gzip'):
    filename = filepath.split('/')[-1]
    user_id = filename.split('_')[0]
    
    with gzip.open(filepath, 'r') as fp:
        data = json.loads(fp.read().decode())
        
        if 'data' in data:
            for item in data['data']:
                tweet_id = item['id']
                
                if tweet_id in all_tweets:
                    
                    # save this user's data
                    if data['includes']['users'][0]['id'] == user_id:
                        user_details[tweet_id] = {
                            'userid': data['includes']['users'][0]['id'],
                            'display_name': data['includes']['users'][0]['name'],
                            'handle': data['includes']['users'][0]['username'],
                            'bio': data['includes']['users'][0]['description'],
                            'verified': data['includes']['users'][0]['verified'],
                            'followers': data['includes']['users'][0]['public_metrics']['followers_count'],
                            'following': data['includes']['users'][0]['public_metrics']['following_count'],
                            'tweet_count': data['includes']['users'][0]['public_metrics']['tweet_count'],
                            'listed_count': data['includes']['users'][0]['public_metrics']['listed_count']
                        }
                        
                        try:                            
                            user_details[tweet_id]['location']=data['includes']['users'][0]['location']
                        except:
                            user_details[tweet_id]['location']='None'


print(f'{len(user_details)} users matched to tweets.')

# turn into dataframe
user_df = pd.DataFrame.from_dict(user_details, orient='index')
user_df.reset_index(inplace=True)
user_df.rename(columns={'index':'TweetId'}, inplace=True)

# Merge coding with most recent tweet data and clean bios
final = user_df.merge(coding, on='TweetId', how='inner')

final["bio"] = final["bio"].str.lower().apply(lambda x: re.sub("\n"," ",x))
final["bio"] = final["bio"].str.lower().apply(lambda x: re.sub("\r"," ",x))
final["bio"] = final["bio"].str.lower().apply(lambda x: re.sub("\t"," ",x))

# write handcoded dataset with 2023 tweet info
final.to_csv(f'{mainpath}/data/handcoded_data_2023_info.tsv', sep = "\t")

### finish cleaning bios and apply bios matching process

final = final.loc[~pd.isna(final.bio)] # remove empty bios
final.bio = final.bio.str.lower() # lower all text in bios
colnames = list(final.columns) # record list of column names
final = final.reset_index()[colnames] # reset index and drop duplicate index column

final['bio'] = final['bio'].fillna('')
final['handle'] = final['handle'].fillna('')

final["tokens"] = f.apply_tucker_jones_tokenizer(final)
final = f.find_pronoun_combinations(final)

pronoun_path = f'{mainpath}/data/pronouns'
gen,pro,term_to_gender = f.load_all_terms(pronoun_path)
final = f.apply_whitespace_tokenizer_find_gendered_words(final, gen)

pronoun_gender, word_gender, pronoun_word_gender = f.estimate_genders_ling(final, term_to_gender)
final['pronoun_gender_2023'] = pronoun_gender
final['word_gender_2023'] = word_gender
final['pronoun_word_gender_2023'] = pronoun_word_gender

# Write to file
final.to_csv(f'{mainpath}/data/handcoded_data_2023_info_matched.tsv', sep = "\t")

