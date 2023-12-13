import pandas as pd
import re
path = '..'
import sys
sys.path.append(path)
import functions as f


bios = pd.read_csv(f"{path}/data/raw/bios_2021.tsv", sep = "\t", lineterminator = '\n')
bios = bios.loc[~pd.isna(bios.bio)]
bios = bios.loc[~pd.isna(bios.display_name)]

bios["bio"] = bios["bio"].str.lower().apply(lambda x: re.sub("\n"," ",x))
bios["bio"] = bios["bio"].str.lower().apply(lambda x: re.sub("\r"," ",x))
bios["bio"] = bios["bio"].str.lower().apply(lambda x: re.sub("\t"," ",x))

bios["display_name"] = bios["display_name"].str.lower()

# drop weird rows with misplaced columns
bios = bios.loc[~(bios["age"] == "age")]
bios = bios.loc[(bios.voter_file_sex.isin({"Male","Female","Unknown"}))]

bios = bios.reset_index().drop(["index"], axis=1)
bios["tokens"] = f.apply_tucker_jones_tokenizer(bios)
bios = f.find_pronoun_combinations(bios)

pronoun_path = f'{path}/data/pronouns'
gen,pro,term_to_gender = f.load_all_terms(pronoun_path)
bios = f.apply_whitespace_tokenizer_find_gendered_words(bios, gen)

pronoun_gender, word_gender, pronoun_word_gender = f.estimate_genders_ling(bios, term_to_gender)
bios["pronoun_gender"] = pronoun_gender
bios["word_gender"] = word_gender
bios["pronoun_word_gender"] = pronoun_word_gender

bios.to_csv(f'{path}/data/panel_bios_2021_matched.tsv', sep = "\t")