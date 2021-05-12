#!usr/bin/python

'''
Invesitgating and devloping bias assessment in Sysrev texts
'''

# Use python 3 env with lime - lime_env

# Load modules
import os
import os.path
import numpy as np
import pandas as pd
import random
import datetime
import re

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.porter import PorterStemmer as ps

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib 

import tensorflow as tf
import tensorflow_hub as hub

from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, SpatialDropout1D
from keras.models import Model
from keras.models import Sequential
# from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model

from keras.layers import Activation, Dropout
from keras import regularizers


from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from keras import utils


####
## Functions
####

embed = hub.load("../../../Carnivore_trends/Data/universal-sentence-encoder-large_5") 


bold_re = re.compile(r'</*b>')
italic_re = re.compile(r'</*i>')
typewrite_re = re.compile(r'</*tt>')
under_re = re.compile(r'</*u>')
emph_re = re.compile(r'</*em>')
big_re = re.compile(r'</*big>')
small_re = re.compile(r'</*small>')
strong_re = re.compile(r'</*strong>')
sub_re = re.compile(r'</*sub>')
sup_re = re.compile(r'</*sup>')
inf_re = re.compile(r'</*inf>')
para_re = re.compile(r'</*p>')

regex_list = [bold_re, italic_re, typewrite_re, under_re, emph_re, big_re, small_re, strong_re, sub_re, sup_re, inf_re] 

def rm_html_tags(txt):

	for i in regex_list:
		txt = i.sub('', txt)
	
	return txt


def sp_rm_stop(text):
	if type(text) == str:
		text = remove_stopwords(" ".join(simple_preprocess(text, deacc = True, min_len = 3)))
	
	return(text)	

def sp_(text):
	if type(text) == str:
		text = " ".join(simple_preprocess(text, deacc = True, min_len = 0, max_len = 25))
	return(text)	


def select_unique_articles(df):
	df["N_rep"] = df.article_id.groupby(df.article_id).transform('count')

	# Multiple articles, from different people...
	df["All_agree"] = df.groupby(df.article_id).answer.transform('nunique')#nunique()#.eq(1)

	# df.loc[df.All_agree == 1, ["article_id", "article_title", "answer"]]

	# Reduce to articless where people agree
	df_sub = df.loc[df.All_agree == 1]
	df_sub = df_sub.drop_duplicates(subset = ["article_id"]).reset_index(drop = True)
	
	df_sub["y"] = np.nan
	df_sub.y[df_sub.answer == "true"] = 1
	df_sub.y[df_sub.answer == "false"] = 0

	return(df_sub)

def proc_titles(df):
	df["sp_title"] = df.article_title.apply(sp_) # for google enc
	df["proc_title"] = df.article_title.apply(sp_rm_stop) # for lr
	df["proc_title_stem"] = df.proc_title.apply(ps().stem_sentence) # for lr
	return(df)


def lr_mod(df, vect_type):
	# Get n - balanced true and false...
	# n_samp = min(df.y.value_counts())
	# # Sub sample
	# pos = df.loc[df.y==1].sample(n=n_samp, random_state=seed)
	# neg = df.loc[df.y==0].sample(n=n_samp, random_state=seed)
	# df = pd.concat([pos, neg]).reset_index(drop = True)

	vect = None
	if vect_type == "count":
		vect = CountVectorizer(analyzer = 'word', ngram_range = (1, 2), max_df = 0.85)
	if vect_type == "tfidf":
		vect = TfidfVectorizer(analyzer = 'word', ngram_range = (1, 2), max_df = 0.85)
	
	x = vect.fit_transform(df.proc_title_stem)
	lr = LogisticRegression(penalty='l2', class_weight='balanced').fit(x, df.y)

	feat_df = pd.DataFrame({"Feature" : np.array(vect.get_feature_names()),
							"Coefficient" : lr.coef_[0],
							"Index" : np.arange(0, len(lr.coef_[0]))})
	feat_df = feat_df.sort_values(by = "Coefficient", ascending = False)
	feat_df = feat_df.reset_index(drop = True)
	return(feat_df)


def encode(le, labels):
	enc = le.transform(labels)
	return utils.to_categorical(enc)

def decode(le, one_hot):
	dec = np.argmax(one_hot, axis=1)
	return le.inverse_transform(dec)

def nn_build(inp_dim, nlayers, nnodes, dropout, kern_reg, n_out, out_type):
	inp = Input(shape=(inp_dim,), dtype = "float32")
	# input_text = Input(shape=(1,), dtype=tf.string)
	# x = Lambda(UniversalEmbedding, output_shape=(512, ))(input_text)
	x = Dense(nnodes, activation='relu',
								kernel_regularizer = kern_reg)(inp)
	x = Dropout(dropout)(x)
	
	for i in range(0,(nlayers-1)):
		x = Dense(nnodes, activation='relu',
								kernel_regularizer = kern_reg)(x)
		x = Dropout(dropout)(x)
	

	if out_type == "class":
		pred = Dense(n_out, activation='softmax')(x)
		model = Model(inputs=[inp], outputs=pred)
		if n_out>1:
			model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		elif n_out == 1:
			model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	elif out_type == "cont":
		pred = Dense(n_out)(x) 	
		model = Model(inputs=[inp], outputs=pred)
		model.compile(loss='mean_squared_error', optimizer='adam')
	
	return(model)



def gnn_fit(df, k):

	# Get n - balanced true and false...
	n_samp = min(df.y.value_counts())
	# # Sub sample
	pos = df.loc[df.y==1].sample(n=n_samp, random_state=1)
	neg = df.loc[df.y==0].sample(n=n_samp, random_state=1)
	df = pd.concat([pos, neg]).reset_index(drop = True)


	# fit full model
	print("Embedding texts")
	# np.arange(0,len(x),500)

	x = df.sp_title
	y = df.y

	tmp_indx = np.arange(0, int(len(x)/500+1))
	tmp_indx = np.multiply(tmp_indx, 500)
	tmp_indx = tmp_indx.tolist()
	tmp_indx.append(len(x)+1)

		# print(tmp_indx)
		# For each chunk, calc cosine simlarity and add max scores to list 
	emb_ls = []
	for i in range(len(tmp_indx)-1):
		tmp_x = x[tmp_indx[i]: tmp_indx[i+1]]
		emb_ls.append(embed(tmp_x))

	x_emb = tf.concat(emb_ls, axis = 0)
	# embed(x)

	# maybe try encoding labels...
	le = LabelEncoder()
	le.fit(y)

	gnn_mod = None
	gnn_mod = nn_build(512, 2, 1024, 0.3, "l2", 2, "class")
	gnn_mod.fit(np.array(x_emb),
				np.asarray(encode(le, y)), 
				epochs = 20, validation_split = 0)

	# k fold cv,
	cv = StratifiedKFold(n_splits=k, shuffle = True, random_state = 1)

	out_ls = []
	fold_id = 1
	print("Cross-validation")
	for (tr, te) in cv.split(x, y):
		print ("\tFold: ", str(fold_id))
	
		# get tr data
		tr_emb = np.array(tf.gather(x_emb, tr))
		tr_y = np.asarray(encode(le, y[tr]))
		n_tr = tr_y.shape[0]

		# and test data
		te_emb = np.array(tf.gather(x_emb, te))
		te_y = y[te].tolist()
		n_te = len(te_y)

		tmp_gnn = None
		tmp_gnn = nn_build(512, 2, 1024, 0.3, "l2", 2, "class")
		tmp_gnn.fit(#np.array(tr_emb), 
					# np.asarray(encode(le, y[tr])),
					tr_emb, 
					tr_y, 
					epochs = 20, validation_split = 0)

		gnn_te_pred = tmp_gnn.predict(np.array(te_emb))[:,1].tolist()
		print(gnn_te_pred)

		# Make it a long df, row per te_txt
		# store output to df
		out_df = pd.DataFrame({ "article_id"    :   df.article_id[te],
								"proc_title"	: 	x[te],
								"proc_title_stem":  df.proc_title_stem[te],
								"N_folds"		:	[k]*n_te,
								"Fold_id"		:	[fold_id]*n_te,
								"N_train"		:	[n_tr]*n_te,
								"N_test"		:	[n_te]*n_te,
								"Classifier"	:	["gnn"]*n_te,
								"Test_class"	:	te_y,
								"Test_pred"		:	gnn_te_pred
								})
		
		out_ls.append(out_df) 

		fold_id += 1

	# bind dfs and return
	out_df = pd.concat(out_ls)

	# return...
	return(out_df, gnn_mod)

####
## Main Code
####

## Load sysrev data
dat = pd.read_csv("../Data/public_sysrev_data.csv", sep = ";")
dat.columns

## Subset to consider incl v exclude
len(dat.label_question.unique()) # 2122
len(dat.label_shortname.unique()) # 2015
len(dat.article_title.unique()) # 330323
# len(dat.title_proc.unique()) # 318496
# len(dat.title_proc_stem.unique()) # 318327

len(dat.article_id.unique()) # 369915

len(dat.project_id.unique()) # 1018


dat["includ"] = dat.label_question.str.contains("includ", case = False)

dat.label_shortname.str.contains("include")


dat.loc[dat.label_question == 'Include this article?'].shape
dat.loc[dat.label_shortname == 'Include'].shape
# Same number of rows...

len(dat.loc[dat.label_shortname == 'Include', "project_name"].unique())
# 947
dat.loc[dat.label_shortname == 'Include', "answer"].unique()
# array(['false', 'true', nan], dtype=object)

###
## Whilst there are various other label questions, good to start with simple yes/no
### 

## Subset to this "include" group
incl_dat = dat.loc[dat.label_shortname == 'Include']
# Only if answer is not na do we include
incl_dat = incl_dat.loc[incl_dat.answer.notna()]

# Reduce to projects with >100 articles
incl_dat["N_docs"] = incl_dat.project_id.groupby(incl_dat.project_id).transform('count')
sum(incl_dat.project_id.value_counts()>100)
incl_dat = incl_dat.loc[incl_dat.N_docs>100]


# Pick a couple of example projects
# 1. 'EntoGEM: a systematic map of global insect population and biodiversity trends'
# 2. 'Climate change impacts on human health'
# 3. 'Fragmentation effects on North American mammal species systematic review and meta-analysis'
# 4. 'Nature-based solutions to climate change adaptation in cities'

ento_df = incl_dat.loc[incl_dat.project_name == "EntoGEM: a systematic map of global insect population and biodiversity trends"]
clim_df = incl_dat.loc[incl_dat.project_name == "Climate change impacts on human health"]
frag_df = incl_dat.loc[incl_dat.project_name == "Fragmentation effects on North American mammal species systematic review and meta-analysis"]
nbs_df = incl_dat.loc[incl_dat.project_name == "Nature-based solutions to climate change adaptation in cities"]

# Only consider articles where all assesors agree on classification - simple initil pass
ento_df = select_unique_articles(ento_df)
clim_df = select_unique_articles(clim_df)
frag_df = select_unique_articles(frag_df)
nbs_df = select_unique_articles(nbs_df) # too few

sum(ento_df.y)/ento_df.shape[0] # 2459, 0.43
sum(clim_df.y)/clim_df.shape[0] # 1219, 0.69
sum(frag_df.y)/frag_df.shape[0] # 352, 0.21

ento_df = proc_titles(ento_df)
clim_df = proc_titles(clim_df)
frag_df = proc_titles(frag_df)

ento_lr_feat_count = lr_mod(ento_df, "count")
clim_lr_feat_count = lr_mod(clim_df, "count")
frag_lr_feat_count = lr_mod(frag_df, "count")

ento_lr_feat_tfidf = lr_mod(ento_df, "tfidf")
clim_lr_feat_tfidf = lr_mod(clim_df, "tfidf")
frag_lr_feat_tfidf = lr_mod(frag_df, "tfidf")

## NN model - based on google encoder
ento_gnn_cv, ento_gnn_mod = gnn_fit(ento_df, 5)
clim_gnn_cv, clim_gnn_mod = gnn_fit(clim_df, 5)
frag_gnn_cv, frag_gnn_mod = gnn_fit(frag_df, 5)


from lime import lime_text
from lime.lime_text import LimeTextExplainer

class_names = ["Irrelevant", "Relevant"]

# Need a function which takes raw texts and nn and outputs prediction
def ento_gnn_predict(inp_text):
	inp_emb = embed(inp_text)
	pred = ento_gnn_mod.predict(inp_emb)
	return(pred)

def clim_gnn_predict(inp_text):
	inp_emb = embed(inp_text)
	pred = clim_gnn_mod.predict(inp_emb)
	return(pred)

def frag_gnn_predict(inp_text):
	inp_emb = embed(inp_text)
	pred = frag_gnn_mod.predict(inp_emb)
	return(pred)

# ento_gnn_mod.predict(embed())

def gnn_predict(inp_text, mod):
	inp_emb = embed(inp_text)
	pred = mod.predict(inp_emb)
	return(pred)

# explainer = LimeTextExplainer(class_names=class_names)
# exp = explainer.explain_instance(ento_df.sp_title[1], gnn_predict(mod = ento_gnn_mod), 
# 								num_features=10)
# exp.as_list()

# specify text col and true col

def lime_sample(text_df, text_col, true_col, n_samp, class_names, pred_func):
	
	idxs = random.sample(range(text_df.shape[0]), n_samp)
	
	out_ls = []
	for idx in idxs:
		explainer = LimeTextExplainer(class_names=class_names)
		tmp_exp = explainer.explain_instance(text_df[text_col][idx], pred_func, 
								num_features=10)

		tmp_exp_ls = tmp_exp.as_list()
		tmp_df = pd.DataFrame({
			"term" :[x[0] for x in tmp_exp_ls],
			"value"  :[x[1] for x in tmp_exp_ls],
			"idx"	 :idx,
			"text"	 :text_df[text_col][idx],
			"y"		 :text_df[true_col][idx],
			"pred"	 :tmp_exp.predict_proba[1]
			})

		out_ls.append(tmp_df)

	return(pd.concat(out_ls).reset_index(drop = True))


# Sample texts from all data - may use proc_title - no stop words and no short words??
# interrested in how changing words changes preds, not what actual preds are...
ento_p_lime = lime_sample(ento_df.loc[ento_df.y == 1].reset_index(drop = True), 
						"sp_title", "y", 100, class_names, ento_gnn_predict)
ento_n_lime = lime_sample(ento_df.loc[ento_df.y == 0].reset_index(drop = True), 
						"sp_title", "y", 100, class_names, ento_gnn_predict)

ento_p_lime1 = lime_sample(ento_df.loc[ento_df.y == 1].reset_index(drop = True), 
						"proc_title", "y", 100, class_names, ento_gnn_predict)
ento_n_lime1 = lime_sample(ento_df.loc[ento_df.y == 0].reset_index(drop = True), 
						"proc_title", "y", 100, class_names, ento_gnn_predict)


ento_lime_df = pd.concat([ento_p_lime, ento_n_lime]).reset_index(drop = True).groupby('term', as_index=False)['value'].mean().reset_index(drop = True)
ento_lime_df1 = pd.concat([ento_p_lime1, ento_n_lime1]).reset_index(drop = True).groupby('term', as_index=False)['value'].mean().reset_index(drop = True)

ento_lime_df.sort_values("value", ascending = False)[:50]
ento_lime_df1.sort_values("value")

ento_lr_feat_tfidf
ento_lr_feat_count


clim_p_lime = lime_sample(clim_df.loc[clim_df.y == 1].reset_index(drop = True), 
						"sp_title", "y", 100, class_names, clim_gnn_predict)
clim_n_lime = lime_sample(clim_df.loc[clim_df.y == 0].reset_index(drop = True), 
						"sp_title", "y", 100, class_names, clim_gnn_predict)
clim_lime = pd.concat([clim_p_lime, clim_n_lime]).reset_index(drop = True).groupby('term', as_index=False)['value'].mean().reset_index(drop = True)

frag_p_lime = lime_sample(frag_df.loc[frag_df.y == 1].reset_index(drop = True), 
						"sp_title", "y", 100, class_names, frag_gnn_predict)
frag_n_lime = lime_sample(frag_df.loc[frag_df.y == 0].reset_index(drop = True), 
						"sp_title", "y", 100, class_names, frag_gnn_predict)
frag_lime = pd.concat([frag_p_lime, frag_n_lime]).reset_index(drop = True).groupby('term', as_index=False)['value'].mean().reset_index(drop = True)



ento_lime_df.sort_values("value", ascending = False)[:50]
ento_lr_feat_count.loc[:50,]

clim_lime.sort_values("value", ascending = False)[:50]
clim_lr_feat_count.loc[:50,]

frag_lime.sort_values("value", ascending = False)[:50]
frag_lr_feat_count.loc[:50,]


# Save output
ento_lime_df.to_csv("../Results/ento_lime.csv", index = False)
ento_lr_feat_count.to_csv("../Results/ento_lr_count.csv", index = False)
ento_lr_feat_tfidf.to_csv("../Results/ento_lr_tfidf.csv", index = False)

clim_lime.to_csv("../Results/clim_lime.csv", index = False)
clim_lr_feat_count.to_csv("../Results/clim_lr_count.csv", index = False)
clim_lr_feat_tfidf.to_csv("../Results/clim_lr_tfidf.csv", index = False)

frag_lime.to_csv("../Results/frag_lime.csv", index = False)
frag_lr_feat_count.to_csv("../Results/frag_lr_count.csv", index = False)
frag_lr_feat_tfidf.to_csv("../Results/frag_lr_tfidf.csv", index = False)



# Sample from trining texts only?? - ie. <dat>_gnn_cv
# ...
# ento_gnn_cv, text_col = "proc_title", true_col = "Test_class"


