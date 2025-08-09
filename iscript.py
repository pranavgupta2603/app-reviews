import pandas as pd
from openai import OpenAI
import warnings
import json
warnings.filterwarnings("ignore")
client = OpenAI(api_key="")

df = pd.read_csv("data_with_categories_with_gpt_4_o_mini_multi.csv")
df["Category"] = df["Category"].apply(lambda x: eval(x) if isinstance(x, str) else x)

# do a set of all categories

def get_set_cats(df):
    
    df["Category"] = df["Category"].apply(lambda x: eval(x) if isinstance(x, str) else [x])
    categories = set()
    for i in df["Category"]:
        if isinstance(i, list):
            for j in i:
                categories.add(j)
        else:
            categories.add(i)
            
    # print the categories
    string = ""
    for idx, i in enumerate(categories):
        string += f"{idx+1}. {i}\n"
    return string

def generate_review_classification_prompt(app: str, review: str, cats: str) -> str:

    prompt = f"""You are a smart **multi-label review classification assistant**.  
You will be given ① a fixed set of category labels (created earlier) and ② a user-written review from an App Store application.  
Your task is to assign **all relevant categories** from the provided list to each **meaningful** review. A review may belong to **multiple categories** if applicable.

---

### 1  Category Labels (use *exactly* as written)

{cats}

*(If a review is meaningless, spammy, or a duplicate, label it **Irrelevant/Spam** instead.)*
- Random or repeated words (e.g., "hi hi hi", "❤️❤️❤️")
- One-word or vague comments (e.g., "nice", "ok", "good app") that lack context.
- Duplicate or near-duplicate content.

---

### 2  Instructions

1. **Read each review carefully.**  
2. **Select all applicable category labels** from the list above (1 or more).  
3. If the review is meaningless, duplicated, or pure emoji/random characters, assign **only** `Irrelevant/Spam`.  
4. If none of the existing categories are a good fit for the review, you are allowed to **create a new category** that accurately describes it.  
5. Output only the mapping in the exact format shown below.

---

### 3  Output Format (strict)

{{"Category": ["<Category Label 1>", "<Category Label 2>", ...]}}

If the review is meaningless or spam:

{{"Category": ["Irrelevant/Spam"]}}

Here is the review:  
{app}: {review}"""


    return prompt.strip()

#print(generate_review_classification_prompt("Kiwi", "lol", get_set_cats())) 
file_name = "new_data_full_copy_multi_new_cats.csv"
#new_df = pd.read_csv("data_full_copy_multi_new_cats.csv")

#new_df.to_csv(file_name, index=False)

new_df = pd.read_csv(file_name)
print(get_set_cats(new_df))

# Print the initial DataFrame

#print(new_df)
for i in range(0, len(new_df)):
    new_df = pd.read_csv(file_name)
    #print(new_df["Category"][i])
    if new_df["Category"][i] != "[]":  # Check if the category is already assigned
        continue  # Skip if the category is already assigned
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": generate_review_classification_prompt(new_df["App"][i], new_df["Review"][i], get_set_cats(new_df))}
        ]
    )
    
    # Parse the response to extract categories
    print(response.choices[0].message.content)
    
    json_response = json.loads(response.choices[0].message.content)
    category = json_response.get("Category", [])
    print(f"Category for row {i}: {category}")
    #add into a Category column
    new_df.at[i, "Category"] = category
    new_df.to_csv(file_name, index=False)