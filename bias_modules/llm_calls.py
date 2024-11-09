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