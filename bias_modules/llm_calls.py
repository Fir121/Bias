from openai import OpenAI
import os
import dotenv

dotenv.load_dotenv(dotenv.find_dotenv())

client = OpenAI(
    api_key=os.environ.get("OPENAIKEY"),
)

class ModelHandler:
    def __init__(self):
        pass

    def generate_response(self, prompt):
        messages = [
            {"role": "user", "content": prompt},
        ]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo" if os.environ.get("DEBUG")=="true" else "gpt-4o",
            messages=messages,
            temperature=0.3,
        )

        return response.choices[0].message.content
    
    def message_ai(self, messages):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo" if os.environ.get("DEBUG")=="true" else "gpt-4o",
            messages=messages,
        )

        return response.choices[0].message.content
    
class Constants:
    @staticmethod
    def get_classifier_determination_system_prompt(classes_col_name, text_col_name, classes_array):
        return f"I received a dataset containing a text column named {text_col_name} and a label column named \
{classes_col_name}. The classes in the label column are {classes_array}.\
\n\
Output a single sentence stating the use case for what this dataset is used to predict."
    
    def get_bad_bias(dataset_desc, word, class_dist):
        d_sentence = " and ".join([f"{x['count']} times for the category \"{x['class']}\"" for x in class_dist])

        return f"I am building a model using a dataset described as - \"{dataset_desc}\", and want you to perform the work of a bias detection tool. \
Upon examination of the dataset, I have found a word with an irregular distribution. The word is \"{word}\". \"{word}\ appears {d_sentence}\" \
\n\
\n\
Output VALID if this is normal for a model of this type, if this word is INTENTIONALLY biased for this use-case. \
Output INVALID if this word might be causing an unintentional bias in the dataset.\
\n\
\n\
For example in a dataset about food reviews, words like \"delicious\" or \"tasty\" or \"bad\" or \"stale\" are expected to affect the bias \
and would be VALID but words like \"female\" or \"waiter\" may be causing a harmful, unintentional bias and would be considered INVALID\
\n\
Only output the word INVALID or VALID in your response. Do not provide an explanation."
    
    @staticmethod
    def get_col_name(column_names, type_to_find):
        col_sentence = ", ".join(column_names)

        return f"I have a dataset with the following columns: {col_sentence}.\
I want to know which column is most likely to be the {type_to_find}.\
\n\nOutput the name of the {type_to_find} only as a single word.\
\n\nFor example, if the column is named serial_no then output only serial_no"
    
    ## Make a conclusion for the pre training model, a conclusion for the post training model
    @staticmethod
    def post_analysis_prompt(model_desc, model_type, accuracy, demographic_parity_std, text_col, label_col, pred_col):
        return f"I have created a text based classification model. Where the text column is named {text_col}, the label column is named {label_col} and the predicted outputs are stored in a column named {pred_col}.\
\n\nMy model is a {model_type} model. When I generate a summary of my model I receive the following description: {model_desc}.\n\n\
The model has an accuracy of {accuracy}% and a demographic parity standard deviation of {demographic_parity_std}.\
\n\nTell me in length about the potential bias that could be there in my model. Give me steps to mitigate them."
    
    @staticmethod
    def pre_analysis_prompt(model_desc, text_col, label_col, balanced, balanced_dist, chi_df, lbalanced, length_df):
        return f"I am going to create a text based classification model. The model is described as {model_desc}. The text column is named {text_col} and the label column is named {label_col}.\
\n\nThe dataset is {'' if balanced else 'not '}balanced. The distribution of the classes is given by this pandas df: {balanced_dist.to_dict()}.\
\n\nThe chi square test results are as follows: {chi_df.to_dict()}. \nThis is a custom test created to find potential biased words in the dataset using advanced and reliable NLP techniques.\
\n\nThe text length classifier deviation is {'' if lbalanced else 'not '}balanced. The results are as follows: {length_df.to_dict()}.\
\n\nTell me in length about the potential bias that could be there in my model's dataset. Give me steps to mitigate them before I start building the actual model."
    
    @staticmethod
    def detbias_sysprompt():
        return "You are DetBias, an AI assistant designed to help users with bias detection and mitigation. You are located on a website where users get insights on their text classification datasets and models via statistical and NLP methods."