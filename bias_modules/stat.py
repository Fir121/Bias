import pandas as pd
from bias_modules.llm_calls import ModelHandler, Constants

def class_balance_checker(df, class_col):
    class_dist = df[class_col].value_counts()

    balanced = False
    if class_dist.std() / class_dist.mean() < 0.1:
        balanced = True
    return class_dist, balanced

def chi_square_test(ddesc, df, text_col, class_col): # modified version with word frequency and analysis
    classes = df[class_col].unique().tolist()

    words = df[text_col].str.split(expand=True).stack().value_counts()
    words = words[words > 10]
    
    new_df = {}

    for word in words.index:
        try:
            sanitized_word = word.strip().replace('"', "").replace("'", "").replace(".", "").replace(",", "").replace("!", "").replace("?", "").replace(":", "").replace(";", "").replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace("{", "").replace("}", "").replace("/", "").replace("\\", "").replace("|", "").replace("_", "").replace("-", "").replace("=", "").replace("+", "").replace("*", "").replace("^", "").replace("&", "").replace("%", "").replace("$", "").replace("#", "").replace("@", "").replace("~", "").replace("`", "").replace("<", "").replace(">", "").replace(" ", "")
            count = df[text_col].str.contains(sanitized_word).groupby(df[class_col]).sum()
        except:  # noqa: E722
            continue
        if count.sum() <= int(0.01*len(df)):
            continue
        deviation = count.std() / count.mean()
        if deviation > 1:
            new_df[sanitized_word] = {"std": deviation, **count.to_dict()}
    
    new_df = [{**{"word": key}, **value} for key, value in new_df.items()]
    new_df = pd.DataFrame(new_df)
    new_df = new_df.sort_values("std", ascending=False)[:10]

    model = ModelHandler()

    for index, row in new_df.iterrows():
        word = row["word"]
        fdist = []
        for class_ in classes:
            fdist.append({"count": row[class_], "class": class_})
        
        prompt = Constants.get_bad_bias(ddesc, word, fdist)
        response = model.generate_response(prompt)

        new_df.loc[index, "response"] = response
        
    new_df = new_df.drop(columns=["std"])
    new_df = new_df.reset_index(drop=True)
    return new_df



