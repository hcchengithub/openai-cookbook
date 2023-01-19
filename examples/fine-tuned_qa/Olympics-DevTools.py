#
# Conditional invoke forth debugger
# 
    __file__ = "OpenAI cookbook examples olympics"
    try:
        get_ipython() # developing in jupyterlab
        PROD = False
        get_ipython().run_line_magic('run', 'forth.py')

        # These are my path envs. Magic 裡面可以引用 global variables by $varname or {varname} 
        jupyter   = get_ipython().run_line_magic('env', 'jupyter')
        onedrive  = get_ipython().run_line_magic('env', 'onedrive')
        github    = get_ipython().run_line_magic('env', 'github')
        downloads = get_ipython().run_line_magic('env', 'downloads')

        get_ipython().run_line_magic('f', 's" {jupyter}\\I study Pandas.ipynb" path\\to/ . cr')
        get_ipython().run_line_magic('f', 's" {github}\\hubble2-nearest-neighbor\\DataGrab_for_with_synonyms_and_chipset.ipynb" path\\to/ . cr')
        get_ipython().run_line_magic('f', 's" {github}\\hubble2-nearest-neighbor\\Hubble2NN_with_synonyms_and_chipset.ipynb" path\\to/ . cr')
        get_ipython().run_line_magic('f', 's" {github}\\hubble2-nearest-neighbor\\Hubble2NN_DevTools.py" path\\to/ . cr')
        get_ipython().run_line_magic('f', 's" {jupyter}\\DevTools.py" path\\to/ . cr')

    except:
        PROD = True


    # In[ ]:


    # import nltk
    # from nltk.stem import PorterStemmer
    from functools import reduce
    import pandas as pd
    import numpy as np
    import sys, json, datetime, re, os, time, logging, bisect, pickle
    from elasticsearch import Elasticsearch
    from elasticsearch_dsl import Search

    # Orange3 for vectorization
    from collections import OrderedDict
    import Orange
    from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable, StringVariable
    from orangecontrib.text import Corpus
    from orangecontrib.text.vectorization.bagofwords import BowVectorizer

    # Wall time 2.6s


    # # Setup 
    # setup.py 與 DataGrab 共用

    # In[ ]:


    import setup
    print(".feather cache file for today: %s" % setup.pathname)


    # In[ ]:


    # ## logging 

    import logging    
    FORMAT = '%(levelname)s %(asctime)s %(pathname)s Line:%(lineno)d %(message)s'
    def reset_logging(level=logging.INFO, pathname=setup.path+"Hubble2NN.log"):
        # Clear the log file 
        for i in logging.root.handlers: logging.root.removeHandler(i) # 一定要先把舊的清空，basicConfig 才有效。
        logging.basicConfig(level=level, filename=pathname if PROD else None, filemode='w', format=FORMAT)
    reset_logging()


    # # 用 elasticsearch_dsl 抓 hubble2 轉成 DataFrame .feather 檔
    # 檔名像這樣：
    # 
    #     2020-3-1.feather
    #     2020-2-1.feather
    #     2020-1-1.feather
    #     2019-12-1.feather
    #     2019-11-1.feather
    #     2019-10-1.feather
    #     2019-9-1.feather

    # ## Function - 從 Elasticsearch 上把指定日期區段的 TRs 抓下來以 Pandas DataFrame 存 .feather 檔

\ Save restore data to save more than an hour of time !!

    pathname = r"c:\Users\8304018\Downloads\pages.pkl"
    # Save pages
    with open(pathname,"wb") as f:
        pickle.dump(pages, f)

    # Restore pages
    with open(pathname, 'rb') as fi:
        pages = pickle.load(fi)

