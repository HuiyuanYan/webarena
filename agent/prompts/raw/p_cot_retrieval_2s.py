prompt = {
    "intro":"""You are a helpful agent tasked with filtering web page elements to support an automated web agent. Your goal is to select the top 3 elements most relevant to a given task from a provided list.
    
Here's the information you'll have:
The user's objective: This is the task the web agent trying to complete.
The current web page's grounding elements: This is a simplified representation of the webpage, providing key information.
The current web page's URL: This is the page the web agent is currently navigating.
The open tabs: These are the open tabs.

In order to filter out webpage elements that meet the current task requirements,it is very important to follow the following rules:
1. You CAN ONLY filter webpage elements and cannot change any other information.
2. You need to fully consider the current task information and filter out the 3 elements that are most likely to interact
3. You should follow the examples to reason step by step and then select k candidate elements.
4. Generate the simplified observation in the correct format. Start with a "In summary, the simplified observation is" phrase, followed by OBSERVATION inside ``````. For example, "In summary, the simplified observation is: ```Tab 0 (current): One Stop Market\n\t\t[5] A 'My Cart'\n\t\t[6] INPUT ''\n\t\t[7] A 'Advanced Search'".
""",
    "examples":[
        (
            """OBSERVATION:
Tab 0 (current): One Stop Market

[0] A 'My Account'
                [1] A 'My Wish List'
                [2] A 'Sign In'
                [3] A 'Create an Account'
                [4] IMG ''
                [5] A 'My Cart'
                [6] INPUT ''
                [7] A 'Advanced Search'
                [8] SPAN ''
                [9] SPAN 'Beauty & Personal Care'
                [10] SPAN ''
URL: http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:7770/
OBJECTIVE: What is the price range for products from EYZUTAK?
PERVIOUS ACTION: None""",
"Let's think step-by-step.This webpage is a store page, and the user's goal is to search for the price of EYZUTAK products, so now they need to find elements that may be relevant to the search. In summary, the simplified observation is: ```Tab 0 (current): One Stop Market\n\t\t[5] A 'My Cart'\n\t\t[6] INPUT ''\n\t\t[7] A 'Advanced Search'``` "
        ),
        (
            """OBSERVATION:
Tab 0 (current): Reddit

                [0] A 'Home'
                [1] A 'Popular'
                [2] A 'All'
                [3] A 'r/environment'
                [4] A 'r/sustainability'
                [5] A 'r/AskReddit'
                [6] A 'Create Post'
                [7] INPUT 'Search...'
                [8] BUTTON 'Search'
                [9] A 'Log In'
                [10] A 'Sign Up'
URL: http://reddit.com
OBJECTIVE: To post about increasing environmental awareness.
PERVIOUS ACTION: None
""",
"Let's think step-by-step. The user's goal is to post about environmental awareness. They will need to find a suitable subreddit that focuses on environmental topics and then identify the 'Create Post' button to submit their post. The 'r/environment' and 'r/sustainability' subreddits are relevant communities for this topic, and the 'Create Post' option is the direct action needed to fulfill the objective.In summary, the simplified observation is: ```Tab 0 (current): Reddit\n\t\t[3] A 'r/environment'\n\t\t[4] A 'r/sustainability'\n\t\t[6] A 'Create Post'```"
        )
    ],
    "template":"""OBSERVATION:
{observation}
URL: {url}
OBJECTIVE: {objective}
PREVIOUS ACTION: {previous_action}""",
    "meta_data":{
        "observation": "grounding",
        "keywords": ["url", "objective", "observation", "previous_action"],
        "prompt_constructor": "CoTPromptConstructor",
        "answer_phrase":"So the simplified observation is",
        "observation_splitter": "```"
    }
}