"""Script to run end-to-end evaluation on the benchmark"""
import argparse
import glob
import json
import logging
import os
import random
import subprocess
import tempfile
import time
import traceback
from PIL import Image
import numpy as np
from pathlib import Path

import openai

from agent import (
    Agent,
    PromptAgent,
    TeacherForcingAgent,
    construct_agent
)
from agent.prompts import *
from retriever.retriever import construct_retriever
from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.actions import is_equivalent
from browser_env.auto_login import get_site_comb_from_filepath
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)
from evaluation_harness import evaluator_router
from retriever.retriever import PromptBasedRetriever

LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = f"{LOG_FOLDER}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random.randint(0, 10000)}.log"

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Set the log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the browser"
    )
    parser.add_argument(
        "--slow_mo",
        type=int,
        default=0,
        help="Slow down the browser by the specified amount",
    )
    parser.add_argument(
        "--action_set_tag", default="id_accessibility_tree", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=["accessibility_tree", "html","grounding", "image"],
        default="accessibility_tree",
        help="Observation type",
    )
    parser.add_argument(
        "--current_viewport_only",
        action="store_true",
        help="Only use the current viewport for the observation",
    )
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=720)
    parser.add_argument("--save_trace_enabled", action="store_true")
    parser.add_argument("--save_format_trace_enabled", action="store_true") # modified
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)

    parser.add_argument("--max_steps", type=int, default=30)

    # agent config
    parser.add_argument("--agent_type", type=str, default="prompt")
    parser.add_argument(
        "--instruction_path",
        type=str,
        default="agents/prompts/state_action_agent.json",
    )
    parser.add_argument(
        "--parsing_failure_th",
        help="When concesecutive parsing failure exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeating_action_failure_th",
        help="When concesecutive repeating action exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )

    # retrieval config
    parser.add_argument("--r_mode", type=int,default=0,help="0: Do not retrieve environmental information; 1: Using language models for retrieval; 2: Using a text embedding model for retrieval.")
    
    # retrieval config for prompt based model
    parser.add_argument("--r_instruction_path",type=str,default="agent/prompts/raw/p_cot_retrieval_2s.py")
    parser.add_argument("--r_provider", type=str, default="openai")
    parser.add_argument("--r_model", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("--r_mode", type=str, default="chat")
    parser.add_argument("--r_temperature", type=float, default=1.0)
    parser.add_argument("--r_top_p", type=float, default=0.9)
    parser.add_argument("--r_context_length", type=int, default=0)
    parser.add_argument("--r_max_tokens", type=int, default=384)
    parser.add_argument("--r_stop_token", type=str, default=None)
    parser.add_argument(
        "--r_max_retry",
        type=int,
        help="max retry times to perform generations when parsing fails",
        default=1,
    )
    parser.add_argument(
        "--r_max_obs_length",
        type=int,
        help="when not zero, will truncate the observation to this length before feeding to the model",
        default=1920,
    )

    # retrieval config for text embedding model
    parser.add_argument("--r_model_name_or_path", type=str, help="Model name or path for EmbeddingBased mode")
    parser.add_argument("--r_tokenizer_name_or_path", type=str, help="Tokenizer name or path for EmbeddingBased mode")
    parser.add_argument("--r_max_length", type=int, help="Maximum length for EmbeddingBased mode")
    parser.add_argument("--r_padding", action="store_true", help="Whether to pad sequences for EmbeddingBased mode")
    parser.add_argument("--r_truncation", action="store_true", help="Whether to truncate sequences for EmbeddingBased mode")
    parser.add_argument("--r_k_threshold",type=int,default=3)

    # lm config
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=384)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument(
        "--max_retry",
        type=int,
        help="max retry times to perform generations when parsing fails",
        default=1,
    )
    parser.add_argument(
        "--max_obs_length",
        type=int,
        help="when not zero, will truncate the observation to this length before feeding to the model",
        default=1920,
    )
    parser.add_argument(
        "--model_endpoint",
        help="huggingface model endpoint",
        type=str,
        default="",
    )

    # example config
    parser.add_argument("--test_start_idx", type=int, default=0)
    parser.add_argument("--test_end_idx", type=int, default=1000)

    # logging related
    parser.add_argument("--result_dir", type=str, default="")
    args = parser.parse_args()

    # check the whether the action space is compatible with the observation space
    if ( args.observation_type == "accessibility_tree" or args.observation_type == "grounding") \
        and args.action_set_tag != "id_accessibility_tree":
        raise ValueError(
            f"Action type {args.action_set_tag} is incompatible with the observation type {args.observation_type}"
        )

    return args


def early_stop(
    trajectory: Trajectory, max_steps: int, thresholds: dict[str, int]
) -> tuple[bool, str]:
    """Check whether need to early stop"""

    # reach the max step
    num_steps = (len(trajectory) - 1) / 2
    if num_steps >= max_steps:
        return True, f"Reach max steps {max_steps}"

    last_k_actions: list[Action]
    action_seq: list[Action]

    # Case: parsing failure for k times
    k = thresholds["parsing_failure"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    if len(last_k_actions) >= k:
        if all(
            [
                action["action_type"] == ActionTypes.NONE
                for action in last_k_actions
            ]
        ):
            return True, f"Failed to parse actions for {k} times"

    # Case: same action for k times
    k = thresholds["repeating_action"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    action_seq = trajectory[1::2]  # type: ignore[assignment]

    if len(action_seq) == 0:
        return False, ""

    last_action: Action = action_seq[-1]

    if last_action["action_type"] != ActionTypes.TYPE:
        if len(last_k_actions) >= k:
            if all(
                [
                    is_equivalent(action, last_action)
                    for action in last_k_actions
                ]
            ):
                return True, f"Same action for {k} times"

    else:
        # check the action sequence
        if (
            sum([is_equivalent(action, last_action) for action in action_seq])
            >= k
        ):
            return True, f"Same typing action for {k} times"

    return False, ""


def extract_format_trace(
    current_prompt:APIInput,
    action_str:str,
    response:str
)->dict:
    trace_dict = dict()
    try:
        for message in current_prompt:
            if message["role"] == "system" and "name" not in message:
                # instruction field
                trace_dict["instruction"] = message["content"]
            if message["role"] == "user":
                # input field
                trace_dict["input"] = message["content"]
            
        trace_dict["output"] = action_str
        trace_dict["response"] = response
        return trace_dict
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        raise ValueError("save trace failed")

def test(
    args: argparse.Namespace,
    agent: Agent | PromptAgent | TeacherForcingAgent,
    config_file_list: list[str],
    retriever: PromptBasedRetriever = None
) -> None:
    scores = []
    max_steps = args.max_steps

    early_stop_thresholds = {
        "parsing_failure": args.parsing_failure_th,
        "repeating_action": args.repeating_action_failure_th,
    }

    env = ScriptBrowserEnv(
        headless=not args.render,
        slow_mo=args.slow_mo,
        observation_type=args.observation_type,
        current_viewport_only=args.current_viewport_only,
        viewport_size={
            "width": args.viewport_width,
            "height": args.viewport_height,
        },
        save_trace_enabled=args.save_trace_enabled,
        sleep_after_execution=args.sleep_after_execution,
    )
    for config_file in config_file_list:
        try:
            render_helper = RenderHelper(
                config_file, args.result_dir, args.action_set_tag
            )

            # get intent
            with open(config_file) as f:
                _c = json.load(f)
                intent = _c["intent"]
                task_id = _c["task_id"]
                # automatically login
                if _c["storage_state"]:
                    cookie_file_name = os.path.basename(_c["storage_state"])
                    comb = get_site_comb_from_filepath(cookie_file_name)
                    temp_dir = tempfile.mkdtemp()
                    # subprocess to renew the cookie
                    subprocess.run(
                        [
                            "python",
                            "browser_env/auto_login.py",
                            "--auth_folder",
                            temp_dir,
                            "--site_list",
                            *comb,
                        ]
                    )
                    _c["storage_state"] = f"{temp_dir}/{cookie_file_name}"
                    assert os.path.exists(_c["storage_state"])
                    # update the config file
                    config_file = f"{temp_dir}/{os.path.basename(config_file)}"
                    with open(config_file, "w") as f:
                        json.dump(_c, f)
            
            logger.info(f"[Config file]: {config_file}")
            logger.info(f"[Intent]: {intent}")

            agent.reset(config_file)
            trajectory: Trajectory = []
            obs, info = env.reset(options={"config_file": config_file})

            state_info: StateInfo = {"observation": obs, "info": info}
            trajectory.append(state_info)

            meta_data = {
                "action_history": ["None"],
                "reflexion":["None"]
            }

            format_traces_list = []
            retrieval_obs_list = []
            while True:      
                early_stop_flag, stop_info = early_stop(
                    trajectory, max_steps, early_stop_thresholds
                )
                if early_stop_flag:
                    action = create_stop_action(f"Early stop: {stop_info}")
                else:
                    obs_after = retriever.simplify_observation(
                        trajectory,intent,meta_data=meta_data
                    )
                    trajectory[-1]["observation"]["text"] = obs_after
                        
                    try:
                        action = agent.next_action(
                            trajectory, intent, meta_data=meta_data
                        )
                        current_prompt = agent.__current_prompt__
                    except ValueError as e:
                        # get the error message
                        action = create_stop_action(f"ERROR: {str(e)}")
                trajectory.append(action)
                action_str = get_action_description(
                    action,
                    state_info["info"]["observation_metadata"],
                    action_set_tag=args.action_set_tag,
                    prompt_constructor=agent.prompt_constructor
                    if isinstance(agent, PromptAgent)
                    else None,
                )

                if args.save_format_trace_enabled:
                    new_trace = extract_format_trace(current_prompt,action_str,agent.__current_reponse__)
                    format_traces_list.append(new_trace)
                    
                    # retrieval
                    retrieval_meta_data = retriever.get_retriever_meta_data().copy()
                    retrieval_meta_data["step"] = len(retrieval_obs_list)
                    retrieval_obs_list.append(
                            retrieval_meta_data.copy()
                    )
                    

                render_helper.render(
                    action, state_info, meta_data, args.render_screenshot
                )

                reflexion = agent.get_reflexion()
                
                meta_data["reflexion"].append(reflexion)

                meta_data["action_history"].append(action_str)

                if action["action_type"] == ActionTypes.STOP:
                    break
         
                obs, _, terminated, _, info = env.step(action)
                state_info = {"observation": obs, "info": info}
                trajectory.append(state_info)

                if terminated:
                    # add a action place holder
                    trajectory.append(create_stop_action(""))
                    break

            evaluator = evaluator_router(config_file)
            score = evaluator(
                trajectory=trajectory,
                config_file=config_file,
                page=env.page,
                client=env.get_page_client(env.page),
            )

            scores.append(score)

            if score == 1:
                logger.info(f"[Result] (PASS) {config_file}")
            else:
                logger.info(f"[Result] (FAIL) {config_file}")

            if args.save_trace_enabled:
                env.save_trace(
                    Path(args.result_dir) / "traces" / f"{task_id}.zip"
                )

            if args.save_format_trace_enabled:
                
                if score == 1:
                    result_str = "PASS"
                    format_traces_list.append({"result":"PASS"})
                else:
                    result_str = "FAIL"
                format_traces_list.append(
                    {
                        "meta_data":{
                            "intent":intent,
                            "result":result_str
                        }
                    }
                )
                format_traces_dir = Path(args.result_dir) / "format_traces" / f"task_{task_id}"
                if not (format_traces_dir).exists():
                    (format_traces_dir).mkdir(parents=True)

                logger.info(f"[Saving format traces...] Dir: {format_traces_dir}")

                # save text trace
                format_traces_path = Path(format_traces_dir) / f"{task_id}.json"
                with open(format_traces_path,"w") as f:
                    json.dump(format_traces_list,f,indent=4)
                

                # save screenshot
                screenshot_dir = Path(format_traces_dir) / "screenshot"
                if not (screenshot_dir).exists():
                    (screenshot_dir).mkdir(parents=True)

                for i,item in enumerate(trajectory):
                    if i % 2 ==0:
                        image_arr = item["observation"]["image"]
                        image = Image.fromarray(image_arr.astype(np.uint8))
                        filename = f'screenshot_{i//2}.png'
                        image.save(os.path.join(screenshot_dir, filename))
                
                # save retrieval info
                retrieval_path = Path(format_traces_dir) / f"retrieval_{task_id}.json"
                with open(retrieval_path,"w") as f:
                    json.dump(retrieval_obs_list,f,indent=4)
                

                
                logger.info(f"[Format trace saved] Dir: {format_traces_dir}")



        except openai.OpenAIError as e:
            logger.info(f"[OpenAI Error] {repr(e)}")
        except Exception as e:
            logger.info(f"[Unhandled Error] {repr(e)}]")
            import traceback

            # write to error file
            with open(Path(args.result_dir) / "error.txt", "a") as f:
                f.write(f"[Config file]: {config_file}\n")
                f.write(f"[Unhandled Error] {repr(e)}\n")
                f.write(traceback.format_exc())  # write stack trace to file

        render_helper.close()

    env.close()
    logger.info(f"Average score: {sum(scores) / len(scores)}")


def prepare(args: argparse.Namespace) -> None:
    # convert prompt python files to json
    from agent.prompts import to_json

    to_json.run()

    # prepare result dir
    result_dir = args.result_dir
    if not result_dir:
        result_dir = (
            f"cache/results_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
        )
    if not Path(result_dir).exists():
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        args.result_dir = result_dir
        logger.info(f"Create result dir: {result_dir}")

    if not (Path(result_dir) / "traces").exists():
        (Path(result_dir) / "traces").mkdir(parents=True)
    
    if not (Path(result_dir) / "format_traces").exists():
        (Path(result_dir) / "format_traces").mkdir(parents=True)

    # log the log file
    with open(os.path.join(result_dir, "log_files.txt"), "a+") as f:
        f.write(f"{LOG_FILE_NAME}\n")


def get_unfinished(config_files: list[str], result_dir: str) -> list[str]:
    result_files = glob.glob(f"{result_dir}/*.html")
    task_ids = [
        os.path.basename(f).split(".")[0].split("_")[1] for f in result_files
    ]
    unfinished_configs = []
    for config_file in config_files:
        task_id = os.path.basename(config_file).split(".")[0]
        if task_id not in task_ids:
            unfinished_configs.append(config_file)
    return unfinished_configs


def dump_config(args: argparse.Namespace) -> None:
    config_file = Path(args.result_dir) / "config.json"
    if not config_file.exists():
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=4)
            logger.info(f"Dump config to {config_file}")


if __name__ == "__main__":
    args = config()
    args.sleep_after_execution = 2.0
    prepare(args)

    test_file_list = []
    st_idx = args.test_start_idx
    ed_idx = args.test_end_idx
    for i in range(st_idx, ed_idx):
        test_file_list.append(f"config_files/{i}.json")
    if "debug" not in args.result_dir:
        test_file_list = get_unfinished(test_file_list, args.result_dir)

    if len(test_file_list) == 0:
        logger.info("No task left to run")
    else:
        print(f"Total {len(test_file_list)} tasks left")
        args.render = False
        args.render_screenshot = True
        #args.save_trace_enabled = True

        args.current_viewport_only = True
        dump_config(args)

        agent = construct_agent(args)
        retriever = construct_retriever(args)
        test(args, agent, test_file_list,retriever)
