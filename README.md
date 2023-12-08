This repository contains the code used for the paper *Categorizing the Non-Categorical:  The Challenges of Studying Gendered Phenomena Online* published for the Special issue *Gender Gaps in Digital Spaces* in the *Journal of Computer Mediated Communication*. In addition to the code, the repository includes two anonymized datasets allowing for a partial reproduction of the results of the paper. 

# Data
Anomyzied versions of our datasets are shared in the `data` folder. Our anonymized data drops user names, handles, ids, and bios, but retains indicators of linguistic signals. Additionally, the numerical age variable is transformed to a categorical range for all users and noise taken from a normal distribution is added to the average number of likes and retweets as well as to the total number followers and tweets. Note that the addition of this noise willl result in slightly different attention and amplification measures than reported in our paper. The data is available in two separate files:
* `panel_bios_anonymized.tsv` contains anonymized data from panel members, joined to the handcoding results and the attention measures
* `decahose_bios_anonymized.tsv` contains anonymized data from the decahose sample

# Code

## data pre-processing
The folder `code/data_preprocessing` includes code used in our data collection and cleaning steps. While we are unable to provide raw datasets due to privacy concerns, we include these files so that others may implement a similar pipeline with their own data. Our larger panel and decahose datasets are housed on a Hadoop cluster and our primary datasets are extracted using pyspark.
*  `spark_gender_bios.py` : retrieves user bios from our panel and joins this with voter data
* `spark_gender_bios_decahose.py` retrieves user bios from the decahose
*  `spark_attention_measures` calculates the attention measures over all 2021 tweets for all panel users.

Next, we identify linguistic features present in bios and merge our handcoded data into a single dataset. Specifically:
*  `generate_match_panel_bios.py` detects linguistic features in the output of `spark_gender_bios.py` 
* `generate_match_decahose_bios.py` detects linguistic features in the output of `spark_gender_bios_decahose.py`
* `construct_handcoded_dataset_match_bios.py` gathers all handcoding data into a single dataset with the aggregated handcoding results and also applies the detection of linguistic signals to the 2023 bios used for handcoding. 

Note that these three scripts use functions from the `code/functions.py` script.

For the purposes of data sharing, we then apply the code in `Anonymize_datasets.ipynb` to transform the datasets used for the paper into anonymized datasets. These anonymized datasets are included in `data` folder.

# Analysis
Our analysis is conducted in three jupyter notebooks available within the `code` folder. All three files take the anonymized data available in the `data` folder and rely on `code/functions.py` for aspects of the analysis. 
* `analysis_ling_signals_measures_match.ipynb` :  Analyzes the linguistic signals in the data
* `analysis_handcoding_matching.ipynb` : Calculates intercoder reliablity 
* `analysis_attention_measures.ipynb` : Calculates attention measures

