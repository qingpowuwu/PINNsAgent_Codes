from pathlib import Path
import sys
current_dir = Path(__file__).parent.absolute()
pinnacle_root = current_dir.parent
sys.path.append(str(pinnacle_root))

from utils.util_colors import RED, BLUE, YELLOW, GREEN, DARKGREEN, RESET, GRAY, NUMBLUE
from argparse import Namespace
import textwrap

def print_dict(dictionary, indent=4):
    """
    Prints a dictionary in a human-readable format with indentation for nested dictionaries.
    Prints quotes around keys and string values, with colors for keys and string values.

        key: dark green
        string value: GRAY
        other value: blue
        list/iter brackets: YELLOW
        set/frozenset brackets: YELLOW

    Args:
        dictionary (dict): The dictionary to be printed.
        indent (int, optional): The initial indentation level for the dictionary. Defaults to 4.
    """
    for key, value in dictionary.items():
        print(" " * indent + DARKGREEN + '"' + str(key) + '":' + RESET, end="")

        if isinstance(value, dict):
            print()
            print_dict(value, indent + 4)
        elif isinstance(value, (list, tuple)):
            print(" " + YELLOW + "[" + RESET, end="")
            for elem in value:
                if isinstance(elem, str):
                    print(GRAY + '"' + elem + '"' + RESET + ", ", end="")
                else:
                    print(str(elem) + ", ", end="")
            print(YELLOW + "]" + RESET)
        elif isinstance(value, (set, frozenset)):
            print(" " + YELLOW + "{" + RESET, end="")
            for elem in value:
                if isinstance(elem, str):
                    print(GRAY + '"' + elem + '"' + RESET + ", ", end="")
                else:
                    print(str(elem) + ", ", end="")
            print(YELLOW + "}" + RESET)
        else:
            if isinstance(value, str):
                print(" " + GRAY + '"' + value + '"' + RESET)
            else:
                print(" " + NUMBLUE + str(value) + RESET)

# # Example usage:
# my_dict = {
#     "name": "John Doe",
#     "age": 30,
#     "address": {
#         "street": "123 Main St",
#         "city": "Anytown",
#         "state": "CA",
#         "zip": "12345",
#     },
#     "skills": ["python", "javascript", "html", "css"],
#     "employment": {
#         "company": "Acme Inc.",
#         "position": "Software Engineer",
#         "salary": 80000,
#     },
# }

# print_dict(my_dict)


def print_namespace(namespace, indent=4):
    """
    Prints an argparse.Namespace object in a human-readable format with indentation for nested objects.
    Prints quotes around keys and string values, with colors for keys, string values, and other values.

        key: dark green
        string value (including list/iter string elements): GRAY
        other value: blue
        list/iter brackets: YELLOW
        set/frozenset brackets: YELLOW

    Args:
        namespace (Namespace): The argparse.Namespace object to be printed.
        indent (int, optional): The initial indentation level for the namespace. Defaults to 4.
    """
    for key, value in vars(namespace).items():
        print(" " * indent + DARKGREEN + '"' + str(key) + '":' + RESET, end="")

        if isinstance(value, Namespace):
            # If the value is a Namespace, print a newline and recursively call print_namespace
            print()
            print_namespace(value, indent + 4)
        elif isinstance(value, dict):
            # If the value is a dictionary, print a newline and recursively call print_dict
            print()
            print_dict(value, indent + 4)
        elif isinstance(value, (list, tuple)):
            # If the value is a list or tuple, print its elements with GRAY for strings
            print(" " + YELLOW + "[" + RESET, end="")
            for elem in value:
                if isinstance(elem, str):
                    print(GRAY + '"' + elem + '"' + RESET + ", ", end="")
                else:
                    print(str(elem) + ", ", end="")
            print(YELLOW + "]" + RESET)
        elif isinstance(value, (set, frozenset)):
            # If the value is a set or frozenset, print its elements with GRAY for strings
            print(" " + YELLOW + "{" + RESET, end="")
            for elem in value:
                if isinstance(elem, str):
                    print(GRAY + '"' + elem + '"' + RESET + ", ", end="")
                else:
                    print(str(elem) + ", ", end="")
            print(YELLOW + "}" + RESET)
        else:
            # If the value is a string, print it with quotes and GRAY color
            if isinstance(value, str):
                print(" " + GRAY + '"' + value + '"' + RESET)
            else:
                # Otherwise, print the value without quotes and blue color
                print(" " + NUMBLUE + str(value) + RESET)


# # Example usage:
# args = Namespace(task='handwriting', log_dir='./logs/20240503-195711', work_dir='./workspace/20240503-195711', max_steps=20, max_time=1000, device=2, python='python', interactive=False, resume=None, resume_step=0, agent_type='DSAgent', llm_name='gpt-3.5-turbo-16k', fast_llm_name='gpt-3.5-turbo-16k', edit_script_llm_name='gpt-3.5-turbo-16k', edit_script_llm_max_tokens=4000, agent_max_steps=50, actions_remove_from_prompt={'Execute Script', 'Edit Script (AI)', 'Reflection', 'List Files', 'Copy File', 'Undo Edit Script', 'Inspect Script Lines', 'Retrieval from Research Log', 'Append Summary to Research Log', 'Reflection'}, actions_add_to_prompt=set(), no_retrieval=True, valid_format_entires=None, max_steps_in_context=3, max_observation_steps_in_context=3, max_retries=5, langchain_agent='zero-shot-react-description', config_path='/data/Project_3_Science_Agent/5_DS-Agent/development/MLAgentBench/config/config.yaml', valid_format_entries=None, **{'fast-llm-name': 'gpt-3.5-turbo-16k'})

# print_namespace(args)

#%% 只打印最多10行的文本



def print_limited_lines(text, max_lines=10):
    lines = text.split('\n')
    if len(lines) <= max_lines:
        print(f'{GRAY}\n*completion* = \n{text}{RESET}\n')
    else:
        limited_lines = '\n'.join(lines[:max_lines])
        print(f'{GRAY}\n*completion* = \n{limited_lines}\n...{RESET}\n')

# long_text = "This is a long text\n" * 20
# print_limited_lines(long_text)

#%%
