# 环境配置

## 1.虚拟环境配置
首先按照原项目的文档设置虚拟环境，然后需要更新/下载如下包：
```shell
pip install openai --upgrade

pip install transfomers --upgrade

pip install dotenv
```

## 2.本地模型下载
由于ollama的模型格式(`.ai`)与常规模型参数格式不一致，而本项目需要用到本地模型`llama3:8b`的参数作为tokenizer，所以需要下载两次模型。

首先常规下载模型到本地：
```python
from modelscope import snapshot_download
# 下载模型参数
model_dir=snapshot_download('LLM-Research/Meta-Llama-3-8B-Instruct') 
print(model_dir) #该目录即为设置的tokenizer目录
```

在`llms/tokenizers.py`下，即可通过本地加载分词器：
```python
# line 12
if provider == "openai":
            if model_name == "llama3:8b": #本地模型的名字
                self.tokenizer = AutoTokenizer.from_pretrained(
                    os.environ.get("MODEL_DIR")
                ) # 通过AutoTokenizer进行加载
            else:
                self.tokenizer = tiktoken.encoding_for_model(model_name)
```

然后下载ollama：
```shell
export OLLAMA_HOST="0.0.0.0:XXXX" #设置ollama端口，非必须，默认为11434
export OLLAMA_MODELS= "YOU_DIR" #设置ollama下载模型的路径，建议设置

curl -fsSL https://ollama.com/install.sh | sh #下载ollama
```

下载完成后，执行命令：
```shell
ollama start
```
启动ollama。

然后执行命令：
```shell
ollama run llama3:8b
```
下载模型，之后亦可以在ollama启动后，运行此命令与模型在命令行交互。

## 3.设置环境变量。
不同于原项目，修改为通过`dotenv`加载项目下的`.env`文件作为环境变量。
在项目目录下新建`.env`文件并填入以下字段（示例，根据需求修改）：
```python
OPENAI_API_KEY = "ollama"  # 使用本地模型的话，这个随便填
OPENAI_BASE_URL = "http://localhost:11434/v1/" # 端口设置为ollama的端口
MODEL_DIR = "/root/autodl-tmp/Meta-Llama-3-8B-Instruct" # 目录设置为分词模型目录
REDDIT = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:9999"
GITLAB = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:8023"
MAP = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:3000"
SHOPPING = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:7770"
SHOPPING_ADMIN = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:7780/admin"
WIKIPEDIA = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
HOMEPAGE = "PASS"
```

## 4. 运行
设置了参数`--save_format_trace_enabled`，添加则会保存format trace记录。同时关闭了原项目默认开启的`--save_trace_enabled`（记录浏览器trace，没用），以节省空间。

一个运行的示例如下：
```shell
python run.py  --instruction_path agent/prompts/jsons/p_cot_id_actree_2s.json  --test_start_idx 9  --test_end_idx 10  --model llama3:8b  --result_dir results --save_format_trace_enabled
```
然后会在`results/format_traces/task_9`下看到记录的内容。

值得注意的是，`evaluator`使用的`must_include`会用到`nltk`的某些词库，若出现相关报错，可自行下载。
