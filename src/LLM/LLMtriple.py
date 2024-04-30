from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_cpp import Llama
import ast
import spacy
import string
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Load the English language model
nlp = spacy.load("en_core_web_sm")

llm = Llama(model_path="/Users/ananyahooda/.cache/lm-studio/models/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/mistral-7b-instruct-v0.2.Q4_K_S.gguf",  
            n_ctx=2048,
            n_gpu_layers=-1,
            n_batch=512,
            callback_manager=callback_manager,
            verbose=True)

# prompt_template = '''<s>[INST] <<SYS>>
# Assistant is an expert JSON builder designed to assist with a tasks of Triple extraction from text.

# Assistant is able to extract all the triples from the text.

# Assistant is able to handle inverse relations and multi-hop relations also.

# Assistant returns the response as the list of dictionaries only always and dictionaries consisting of keys called "head", "type" and "tail" in ever dictionary in the list

# Here are some previous conversations between the Assistant and User:

# User: Hey how are you today?
# Assistant: I'm good thanks, how are you?
# User: Can you extract all the triplets from this text: John Wilkes Booth , who assassinated President Lincoln , was an actor .
# Assistant:  [{{"head": "President Lincoln", "type": "killed by", "tail": "John Wilkes Booth"}}, {{"head": "John Wilkes Booth", "type": "assassinated", "tail": "President Lincoln"}},{{"head": "John Wilkes Booth", "type": "occupation", "tail": "actor"}}, {{"head": "actor", "type": "occupation of", "tail": "John Wilkes Booth"}}]
# User: Also give triples for text: "Marie Magdefrau Ferraro , 50 , of Bethany , Conn. , was shot to death Thursday when two bandits armed with assault rifles emerged from nearby bushes and began firing at a van carrying a Connecticut Audubon Society wildlife wild tour group .
# Assistant: [{{"head": "Marie Magdefrau Ferraro", "type": "residence", "tail": "Bethany"}}, {{"head": "Marie Magdefrau Ferraro", "type": "residence", "tail": "Conn." }},{{"head": "Bethany", "type": "location", "tail": "Conn."}},] 
# User: Extract triples for text: Barrack Obama is husband of Michelle Obama. Barrack Obama lives in USA. Barrack Obama lives with his wife.
# Assistant: [
#     {{"head": "Barrack Obama", "type": "husband of", "tail": "Michelle Obama"}},
#     {{"head": "Michelle Obama", "type": "wife of", "tail": "Barrack Obama"}}
#     {{"head": "Barrack Obama", "type": "residence", "tail": "USA"}},
#     {{"head": "Barrack Obama", "type": "lives with", "tail": "Michelle Obama"}}
#     {{"head": "Michelle Obama", "type": "residence", "tail": "USA"}}
# ]

# User: Thanks, Bye!
# Assistant: See you later, the Chat is closed.
# <</SYS>>

# {0}[/INST]'''

prompt_template = '''<s>[INST] <<SYS>>
Assistant is an expert JSON builder designed to assist with a tasks of Triple extractions from text.

Assistant is able to perform coreference resolution on the text so that no pronouns are in the text and those pronouns are replaced with the corresponding nouns. Pronouns like he, she, they, her, him, them, we should not be present. They should be replaced with their corresponding nouns.

Assistant is able to extract all the triples from the text

Assistant is able to handle inverse relations and transitive relations also.

Assistant returns the response as the list of dictionaries only always and dictionaries consisting of keys called "head", "type" and "tail" in ever dictionary in the list. "head" and "tail" should only contain nouns.

Assistant only returns JSON objects and not any extra explanation or text.

Here are some previous conversations between the Assistant and User:

User: Hey how are you today?
Assistant: I'm good thanks, how are you?
User: Can you extract all the triplets from this text: John Wilkes Booth , who assassinated President Lincoln , was an actor .
Assistant:  [{{"head": "President Lincoln", "type": "killed by", "tail": "John Wilkes Booth"}}, {{"head": "John Wilkes Booth", "type": "assassinated", "tail": "President Lincoln"}},{{"head": "John Wilkes Booth", "type": "occupation", "tail": "actor"}}, {{"head": "actor", "type": "occupation of", "tail": "John Wilkes Booth"}}]
User: Also give triples for text: "Marie Magdefrau Ferraro , 50 , of Bethany , Conn. , was shot to death Thursday when two bandits armed with assault rifles emerged from nearby bushes and began firing at a van carrying a Connecticut Audubon Society wildlife wild tour group .
Assistant: [{{"head": "Marie Magdefrau Ferraro", "type": "residence", "tail": "Bethany"}}, {{"head": "Marie Magdefrau Ferraro", "type": "residence", "tail": "Conn." }},{{"head": "Bethany", "type": "location", "tail": "Conn."}},] 
User: Extract triples for text: Barrack Obama is husband of Michelle Obama. Barrack Obama lives in USA. Barrack Obama lives with his wife.
Assistant: [
    {{"head": "Barrack Obama", "type": "husband of", "tail": "Michelle Obama"}},
    {{"head": "Michelle Obama", "type": "wife of", "tail": "Barrack Obama"}}
    {{"head": "Barrack Obama", "type": "residence", "tail": "USA"}},
    {{"head": "Barrack Obama", "type": "lives with", "tail": "Michelle Obama"}}
    {{"head": "Michelle Obama", "type": "residence", "tail": "USA"}}
]
User: Please give triples for: Ananya does not study at IIT Madras. She studies at IIT Bhilai. Arunya is batchmate of Ananya. They are good friends.
Assistant: [{{'head': 'Ananya', 'type': 'does not study at', 'tail': 'IIT Bhilai'}}, {{'head': 'Ananya', 'type': 'studies at', 'tail': 'IIT Bhilai'}}, {{'head': 'Arunya', 'type': 'studies at', 'tail': 'IIT Bhilai'}},
            {{'head': 'Ananya', 'type': 'batchmate of', 'tail': 'Arunya'}}, {{'head': 'Arunya', 'type': 'batchmate of', 'tail': 'Ananya'}}, {{'head': 'Arunya', 'type': 'friend of', 'tail': 'Ananya'}}, {{'head': 'Ananya', 'type': 'friend of', 'tail': 'Arunya'}}]]
User:  Please give triples for: John, who works at Google, collaborated with Maria, who works at Microsoft, to solve a difficult math problem with the help of LLMs.
Assistant: [{{'head': 'John', 'type': 'collaborated with', 'tail': 'Maria'}},{{'head': 'Maria', 'type': 'collaborated with', 'tail': 'John'}}, {{'head': 'John', 'type': 'works at', 'tail': 'Google'}},
            {{'head': 'Maria', 'type': 'works at', 'tail': 'Microsoft'}}, {{'head': 'Maria', 'type': 'used tool', 'tail': 'LLM'}}, {{'head': 'John', 'type': 'used tool', 'tail': 'LLM'}}, {{'head': 'LLM', 'type': 'used by', 'tail': 'Maria'}}, {{'head': 'LLM', 'type': 'used by', 'tail': 'John'}}, {{'head': 'Maria', 'type': 'solved', 'tail': 'difficult math problem'}},
            {{'head': 'John', 'type': 'solved', 'tail': 'difficult math problem'}}, {{'head': 'LLM', 'type': 'helped with', 'tail': 'solve a difficult math problem'}}]
User: Thanks, Bye!
Assistant: See you later, the Chat is closed.
<</SYS>>

{0}[/INST]'''

#John, who works at Google, collaborated with Maria, who works at Microsoft, to solve a difficult math problem with the help of LLMs.

def process_command1(command):
    # Put user command into prompt
    prompt = prompt_template.format("User: " + command)
    # Send command to the model
    output = llm(prompt, max_tokens=2000, stop=["User:"])
    response = (output['choices'][0]['text']).strip()
    try:
        response = ast.literal_eval(response)
    except Exception as e:
        print(e)
    # No json match, just return response
    return response 

# def extract_triples(text):
#     # Process the text
#     doc = nlp(text)

#     # Create an empty list to store the sentences
#     sentences = []

#     # Iterate over the sentences in the document and append them to the list
#     for sent in doc.sents:
#         # Remove punctuation marks and escape characters from the sentence
#         sentence = sent.text.translate(str.maketrans("", "", string.punctuation))
#         sentence = sentence.replace("\n", "").replace("\r", "").replace("\t", "")
#         sentences.append(sentence)

#     # Extract triples for each sentence
#     triples = []
#     for sentence in sentences:
#         triple = process_command1(f"Extract triples for text: {sentence}")
#         triples.extend(triple)

#     return triples

@app.route('/extract_triples', methods=['GET', 'POST'])
def extract_triples_api():
    if request.method == 'POST':
        text = request.json['text']
    else:
        text = request.args.get('text', '')
    triples = process_command1(text)
    return jsonify(triples)

if __name__ == '__main__':
    app.run(debug=True, port=8001)
