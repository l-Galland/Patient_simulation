import pandas as pd
from openai import OpenAI

def read_prompt_csv(role):
  if role == 'client':
      filename = 'prompts/client_prompts.csv'
  df = pd.read_csv(filename)
  intent_detail_list = []
  for index, row in df.iterrows():
      print(row)
      positive_examples = [row['positive example 1'],row['positive example 2'], row['positive example 3'], row['positive example 4'], row['positive example 5']]

      intent_detail_list.append({'intent': row['intent'].strip(),'definition': row['definition'],'positive_examples': positive_examples})
  return intent_detail_list


def get_completion_from_messages_local(messages, temperature=0.7):
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")


    response = client.chat.completions.create(
                                            model="local-model",
                                            messages=messages,
                                            temperature=temperature,
                                        )

    return response.choices[0].message.content

def create_message_client_generation_conditionned_da(intent_detail_list, intent, context):
    intent_definition_list = []
    intent_example_list = []
    for intent_detail in intent_detail_list:
        intent_text = intent_detail['intent']
        definition_text = intent_detail['definition'].replace("\\", "")
        positive_example_list = intent_detail['positive_examples']

        for ex in positive_example_list:
            if len(ex) > 3:
                intent_example_list.append(f"{ex}\n Intent: {intent_text}")
        intent_definition_list.append(f" {intent_text}: {definition_text} ")
    intent_definition = ";\n".join(intent_definition_list)
    exemples = ";\n".join(intent_example_list)
    system_prompt_template = f""" You are a patient talking with a therapist. Your task is to generate the patient's next utterance and respect the intent given \
                                    from one of the following predefined categories:\n {intent_definition}\
                                     
                                            ####
                                              Here are some examples:\n {exemples}
                                            ####
                                     The dialogue is happening orally, use a oral style language with hesitation, repetition, and deviation.\ 
                                     You will only respond with one patient's utterance. Do not provide explanations or notes. Make only one proposition.\
                                     The response must be short, no more than 2 or 3 sentences.\
                                     Use [laughter] to signal a patient laughter, [laughs] to signal a patient laugh, [sighs] to signal a patient sigh, [gasps] to signal a patient gasps, [clears throat] to signal a patient clears throat, — or ... for hesitation, and CAPITALIZATION for emphasis of a word.\
                                     
                                            <<<
                                              Context : {context}
                                              Generate the patient's next utterance with the intent: {intent}
                                            >>>
                                            Patient's utterance: """

    messages = [{'role': 'system', 'content': system_prompt_template}]
    return messages


def create_message_client_generation_unconditionned(intent_detail_list, n_turn, context):

    intent_example_list = []
    for intent_detail in intent_detail_list:

        positive_example_list = intent_detail['positive_examples']

        for ex in positive_example_list:
            if len(ex) > 3:
                intent_example_list.append(f"{ex}")

    exemples = ";\n".join(intent_example_list)
    system_prompt_template = f""" You are a patient talking with a therapist. Your task is to generate the patient's next utterance \
                                        
                                            ####
                                              Here are some examples:\n {exemples}
                                            ####
                                     The dialogue is happening orally, use a oral style language with hesitation, repetition, and deviation.\ 
                                     You will only respond with one patient's utterance. Do not provide explanations or notes. Make only one proposition.\
                                     The response must be short, no more than 2 or 3 sentences.\
                                     There is on average 62 turns in a dialog. You are at turn {n_turn}.\
                                     Use [laughter] to signal a patient laughter, [laughs] to signal a patient laugh, [sighs] to signal a patient sigh, [gasps] to signal a patient gasps, [clears throat] to signal a patient clears throat, — or ... for hesitation, and CAPITALIZATION for emphasis of a word.\

                                            <<<
                                            
                                              Context : {context}
                                              Generate the patient's next utterance:
                                            >>>
                                            Patient's utterance: """

    messages = [{'role': 'system', 'content': system_prompt_template}]
    return messages


def create_message_client_generation_conditionned_type(intent_detail_list, type, n_turn, context):

    if 25-int(n_turn) < 12:
        time = "end"
    else:
        time = "begninng"

    if type == "Resistant to change":
        type_def = "You are a patient resistant to change. You are not willing to change your behavior or your way of thinking. You are not open to new ideas or suggestions."
    elif type == "Open to change":
        type_def = "You are a patient open to change. You are willing to change your behavior or your way of thinking. You are open to new ideas or suggestions. "
    elif type == "Receptive":
        type_def = "You are a patient receptive to therapy, you change you mind during the conversation. "
        if time == "end":
            type_def = type_def + "This is the end of the conversation, You are now more open to change. You are willing to change your behavior or your way of thinking. You are open to new ideas or suggestions. "
        else:
            type_def = type_def + "This is the beginning of the conversation, You are resistant to change. You are not willing to change your behavior or your way of thinking. You are not open to new ideas or suggestions. "
    intent_example_list = []
    for intent_detail in intent_detail_list:

        positive_example_list = intent_detail['positive_examples']

        for ex in positive_example_list:
            if len(ex) > 3:
                intent_example_list.append(f"{ex}")

    exemples = ";\n".join(intent_example_list)
    system_prompt_template = f""" You are a patient talking with a therapist. Your task is to generate the patient's next utterance \
                                            ####
                                              Here are some examples:\n {exemples}
                                            ####
                                     The dialogue is happening orally, use a oral style language with hesitation, repetition, and deviation.\ 
                                     You will only respond with one patient's utterance. Do not provide explanations or notes. Make only one proposition.\
                                     The response must be short, no more than 2 or 3 sentences.\
                                     {type_def}\
                                     There is on average 25 turns in a dialog. You are at turn {n_turn}.\
                                     Use [laughter] to signal a patient laughter, [laughs] to signal a patient laugh, [sighs] to signal a patient sigh, [gasps] to signal a patient gasps, [clears throat] to signal a patient clears throat, — or ... for hesitation, and CAPITALIZATION for emphasis of a word.\

                                            <<<
                                              Context : {context}
                                              Generate the patient's next utterance:
                                            >>>
                                            Patient's utterance: """

    messages = [{'role': 'system', 'content': system_prompt_template}]
    return messages