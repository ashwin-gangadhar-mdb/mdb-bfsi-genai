import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
import numpy as np

from typing import List
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.chains import RetrievalQA,LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain import OpenAI

from pymongo import MongoClient
from dotenv import load_dotenv
import os
import joblib
from glob import glob

load_dotenv()

def get_embedding_model(vcol):
    repo_id = "hkunlp/instructor-base"
    hf = HuggingFaceInstructEmbeddings(model_name=repo_id, cache_folder="tmp/")
    hf.embed_instruction = "Represent the document for retrieval of personalized credit cards:"
    vectorstore = MongoDBAtlasVectorSearch(vcol, hf)
    return vectorstore

MONGO_CONN=os.environ.get("MONGO_CONNECTION_STRING")
client = MongoClient(MONGO_CONN)
col = client["bfsi-genai"]["credit_history"]
vcol = client["bfsi-genai"]["cc_products"]

llm = ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo")
llm4= OpenAI(temperature=0, model_name="gpt-4")
pipeline= [
    {"$addFields":{
            "score":{"$meta":"searchScore"}
        }
    },
    {
        '$group': {
            '_id': '$title', 
            'source': {
                '$first': '$source'
            }, 
            'title': {
                '$first': '$title'
            }, 
            'text': {
                '$push': '$text'
            },
            'score':{"$sum": "$score"}
        }
    }, {
        '$addFields': {
            'text': {
                '$reduce': {
                    'input': '$text', 
                    'initialValue': '', 
                    'in': {
                        '$concat': [
                            '$$value', '$$this'
                        ]
                    }
                }
            }
        }
    },
    {
        "$sort":{
            "score": -1
        }
    },
    {
        '$limit': 1
    }
]
retriever = get_embedding_model(vcol).as_retriever(search_type='similarity',search_kwargs={'k': 3})
recommender_retriever = MultiQueryRetriever.from_llm(retriever=retriever,llm=llm4)
model = joblib.load("model/classifier.jlb")
imp_idx = np.argsort(-1 * model.feature_importances_)

def get_user_n_model_info(age, dlq_status=False):
    age = int(age)
    if not dlq_status:
        df = pd.DataFrame.from_records(col.find({"$and":[{"age": {"$gt":age}}, {"age": {"$lt":age+10}},{"MonthlyIncome": {"$gt":0}},{"SeriousDlqin2yrs":0}]}, {"_id":0,"Unnamed: 0":0, "SeriousDlqin2yrs":0}).limit(1))
    else:
        df = pd.DataFrame.from_records(col.find({"$and":[{"age": {"$gt":age}}, {"age": {"$lt":age+10}}, {"MonthlyIncome": {"$gt":0}},{"SeriousDlqin2yrs":1}]}, {"_id":0,"Unnamed: 0":0, "SeriousDlqin2yrs":0}).limit(1))
    feature_importance = "\n".join(i for i in list(map(lambda x:f"Columns:{x[0]}  Prob score for decision making:{x[1]}" ,zip(df.columns[imp_idx], model.feature_importances_[imp_idx]))))
    features = df.to_dict(orient="records")[0]
    dlq = model.predict_proba(df)[:,1][0]
    return feature_importance,features,dlq

def get_user_n_model_info_by_id(id):
    df = pd.DataFrame.from_records(col.find({"Unnamed: 0":id}, {"_id":0,"Unnamed: 0":0, "SeriousDlqin2yrs":0}).limit(1))
    feature_importance = "\n".join(i for i in list(map(lambda x:f"Columns:{x[0]}  Prob score for decision making:{x[1]}" ,zip(df.columns[imp_idx], model.feature_importances_[imp_idx]))))
    features = df.to_dict(orient="records")[0]
    dlq = model.predict_proba(df)[:,1][0]
    return feature_importance,features,dlq

def generate_flow(feature_importance, features, dlq):
    # explain
    prompt_template_for_explain = f"""
##Instruction: Taking into account the Definitions of various fields and their respective values a model is trained to predict weather a person will expericen delinquency or not in the next 2 years. Below both the values that was input to the model and the result produced by the model are provided. As a bank employee response to the candidate, It is expected to provide a detailed reason in layman language as to why a Credit request was rejected or processed given the profile of the candidate. Also while providing reason do not mention the use of AI model being used for decision making.

##Definitions
RevolvingUtilizationOfUnsecuredLines=Total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits DataType=ratio between 0-1
age=Age of borrower in years DataType=integer
NumberOfTime30-59DaysPastDueNotWorse=Number of times borrower has been 30-59 days past due but no worse in the last 2 years. DataType=integer
DebtRatio=Monthly debt payments, alimony,living costs divided by monthy gross income DataType=ratio between 0-1
MonthlyIncome=Monthly income in INR DataType=real
NumberOfOpenCreditLinesAndLoans=Number of Open loans (installment like car loan or mortgage) and Lines of credit (e.g. credit cards) DataType=integer
NumberOfTimes90DaysLate=Number of times borrower has been 90 days or more past due. DataType=integer
NumberRealEstateLoansOrLines=Number of mortgage and real estate loans including home equity lines of credit DataType=integer
NumberOfTime60-89DaysPastDueNotWorse=Number of times borrower has been 60-89 days past due but no worse in the last 2 years. DataType=integer
NumberOfDependents=Number of dependents in family excluding themselves (spouse, children etc.) DataType=integer
SeriousDlqin2yrs=Person experienced 90 days past due delinquency or worse  DataType=ratio between 0-1

##Feature importace of the model used:
{feature_importance}

##Values for given profile to be use to predict the Result(SeriousDlqin2yrs) with a reason
{features}

## Model Result
SeriousDlqin2yrs={dlq}

##Reason in step by step points:
"""
    response = llm.predict(prompt_template_for_explain)

    # recommendations
    user_profile_based_card_template=f"""
    ##Instruction: Given the user profile recommended credit cards that will best fit the user profile.

    ## User profile:
    {response}

    ## Credit card features and eligibility criteria recommendations:
    """
    rec = recommender_retriever.get_relevant_documents(user_profile_based_card_template)
    card_suggestions= ""
    for r in rec:
        card_suggestions += f'- Card name:{" ".join(r.metadata["title"].split("-"))} card \n  Card Features:{r.page_content} +\n'

    recomendations_template=f"""
    ##Instruction: Given the user profile recommended credit cards that will best fit the user profile. 
    Provide card by card reasons(concise) as to why the credit card is suggested to the user.

    ## User profile:
    {response}

    ## Recommended Credit cards
    {card_suggestions}

    ## Recommendations=Output as Json with card name as Key and concise reasons point by point as Value:
    """
    out = llm.predict(recomendations_template)

    out = list(filter(lambda x:x!="",out.split("\n")[1:-1]))

    return {"out": out, "response":response}

if __name__=="__main__":
    print("test")
    print(get_user_n_model_info_by_id(132))
    # df = pd.DataFrame.from_records((col.find({"Unnamed: 0":1}, {"_id":0,"Unnamed: 0":0, "SeriousDlqin2yrs":0})))
    # feature_importance = "\n".join(i for i in list(map(lambda x:f"Columns:{x[0]}  Prob score for decision making:{x[1]}" ,zip(df.columns[imp_idx], model.feature_importances_[imp_idx]))))
    # features = df.to_dict(orient="records")[0]
    # dlq = model.predict_proba(df)[:,1][0]*100
    # print(generate_flow(feature_importance, features ,dlq))

