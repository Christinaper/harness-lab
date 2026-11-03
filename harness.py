"""harness.py — 第 5 步:上下文管理(先量,再管)(仍然只有一个 bash 工具)

累计功能:
    第 2 步  权限层:每条命令执行前由人确认;子进程不继承 DEEPSEEK_API_KEY
    第 3 步  记录层:实际发生的事写进 logs/*.jsonl(完整输出),结束时 harness 自己打印命令清单
    第 4 步  健壮性:工具出错一律变成"观察"回传;先记后做;超时杀整个进程组;用真正的 bash
    第 5 步  上下文管理:
             a. 量:每轮打印 token 用量(来自 API 返回的 usage),并累计
             b. 管:工具输出回传给模型前,过长就只保留头和尾,完整输出仍在日志里
             c. 告知:把命令超时、输出上限写进工具描述,模型不必靠撞墙才知道

运行:
    pip install openai
    export DEEPSEEK_API_KEY=sk-...
    mkdir sandbox && cd sandbox          # 务必在一个一次性目录里跑
    python ../harness.py "创建 hello.py,打印 hello,然后运行它"

做对照实验(同一个任务,取消截断,看 token 差多少):
    HARNESS_MAX_OUTPUT=100000000 python ../harness.py "..."

整个 harness 就是 main() 里的那个 for 循环:
    模型决定 -> (人确认) -> (先记) -> 执行工具 -> (截断) -> (再记) -> 把结果放回消息 -> 再让模型决定
"""
import datetime
import json
import os
import signal
import subprocess
import sys

from openai import OpenAI

MODEL = "deepseek-v4-flash"   # 若报模型不存在,去 DeepSeek 文档确认当前的模型名
# 子进程不继承 API key:否则模型执行一句 `env` 就能把 key 读进对话,再发给 API 提供方
SAFE_ENV = {k: v for k, v in os.environ.items() if k != "DEEPSEEK_API_KEY"}
MAX_STEPS = 20                # 最简单的"刹车":防止模型无限循环烧钱
TIMEOUT = 30                  # 单条命令的最长运行秒数
MAX_OUTPUT = int(os.environ.get("HARNESS_MAX_OUTPUT", "8000"))   # 回传给模型的单条工具输出上限(字符)
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

# 告诉模型有哪些工具,以及它们的限制。模型只能"请求"调用,真正执行的是下面的 run_bash。
TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "在当前目录执行一条 bash 命令,返回退出码、stdout 和 stderr。"
            f"单条命令最多运行 {TIMEOUT} 秒,超时会被终止。"
            f"输出超过 {MAX_OUTPUT} 字符时只保留开头和结尾,"
            "所以输出可能很多时,请先用 head/tail/grep/wc 缩小范围。"
        ),
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "要执行的命令"}},
            "required": ["command"],
        },
    },
}]

SYSTEM = (
    "You are a coding agent. Use the bash tool to complete the task. "
    "When finished, reply with a short summary and no tool call. "
    "Reply in the same language as the user's task."
)

_auto_approve = False
_log_file = None
_executed = []                                # [(command, 是否被允许执行)],供结束时的 harness 记录使用
_usage = {"prompt": 0, "completion": 0}       # 累计 token 用量


def log(event: dict) -> None:
    """把实际发生的事原样追加到 JSONL。这是记录,不是模型自己的叙述。"""
    if _log_file:
        with open(_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def usage_of(resp) -> dict:
    """取出 API 返回的用量;取不到就返回空,不影响主流程。"""
    u = getattr(resp, "usage", None)
    if u is None:
        return {}
    if hasattr(u, "model_dump"):
        return u.model_dump()
    return dict(vars(u))


def clip(output: str) -> str:
    """输出过长时只保留头和尾,中间用一句说明替代(完整输出仍在日志里)。
    为什么留头和尾:开头通常交代了上下文,报错和最终结果通常在结尾。"""
    if len(output) <= MAX_OUTPUT:
        return output
    half = MAX_OUTPUT // 2
    cut = len(output) - 2 * half
    return (output[:half]
            + f"\n...[已截断中间 {cut} 个字符;请用 grep/head/tail/wc 缩小范围再看]...\n"
            + output[-half:])


def print_report() -> None:
    """harness 自己的记录,不经过模型。可以和模型的最终总结对照着看。"""
    ran = [c for c, ok in _executed if ok]
    denied = [c for c, ok in _executed if not ok]
    print(f"\n[harness 记录] 实际执行 {len(ran)} 条命令,拒绝 {len(denied)} 条")
    for c in ran:
        print("  ✔ " + c[:120].replace("\n", " ⏎ "))
    for c in denied:
        print("  ✘ " + c[:120].replace("\n", " ⏎ "))
    if _usage["prompt"]:
        # 每一轮都会把整个历史重新发一遍,所以"累计输入"远大于"上下文大小"
        print(f"  累计输入 {_usage['prompt']} / 输出 {_usage['completion']} tokens")


def ask_permission(command: str) -> bool:
    """权限闸门:这是确定性的代码,不经过模型,模型无法说服它放行。"""
    global _auto_approve
    if _auto_approve:
        return True
    ans = input("  允许执行? [y=允许 / n=拒绝 / a=本次运行全部允许] ").strip().lower()
    if ans == "a":
        _auto_approve = True
        return True
    return ans == "y"


def _kill_group(p: subprocess.Popen) -> None:
    try:
        os.killpg(p.pid, signal.SIGKILL)   # 杀整个进程组,不只是 shell
    except ProcessLookupError:
        pass


def run_bash(command: str) -> str:
    """真正执行命令的地方。任何失败都以文本形式返回,绝不向上抛异常:
    对 harness 来说,工具出错是模型需要看到的"观察",不是 harness 该崩溃的理由。"""
    try:
        p = subprocess.Popen(
            command, shell=True, executable="/bin/bash",     # 不写 executable 会用 /bin/sh(Ubuntu 上是 dash),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,  # 与工具描述里的"bash"不符,[[ ]]、数组等写法会失败
            text=True, encoding="utf-8", errors="replace",   # 输出被截到半个汉字也不会解码失败
            env=SAFE_ENV,
            start_new_session=True,                           # 独立进程组,超时才能整组杀掉
        )
    except Exception as e:
        return f"error: failed to start command: {type(e).__name__}: {e}"
    try:
        out, err = p.communicate(timeout=TIMEOUT)
        return f"exit={p.returncode}\n{out}{err}"
    except subprocess.TimeoutExpired:
        _kill_group(p)
        try:
            out, err = p.communicate(timeout=5)   # 取回超时前已产生的输出
        except Exception:
            out, err = "", ""
        return (f"error: command timed out after {TIMEOUT}s (whole process group killed)\n"
                f"[partial output]\n{out}{err}")
    except Exception as e:
        _kill_group(p)
        return f"error: {type(e).__name__}: {e}"


def main(task: str) -> None:
    global _log_file
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    os.makedirs(LOG_DIR, exist_ok=True)
    _log_file = os.path.join(LOG_DIR, datetime.datetime.now().strftime("run-%Y%m%d-%H%M%S-%f.jsonl"))
    log({"event": "task", "task": task, "model": MODEL, "max_output": MAX_OUTPUT})
    print(f"[日志] {_log_file}")

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": task},
    ]

    try:
        for step in range(MAX_STEPS):
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                # V4 默认开启思考模式;思考模式下的工具循环要求回传 reasoning_content,
                # 先关掉,避开这个坑(之后单独作为一步打开)。
                extra_body={"thinking": {"type": "disabled"}},
            )
            msg = resp.choices[0].message

            usage = usage_of(resp)
            p_tok, c_tok = usage.get("prompt_tokens"), usage.get("completion_tokens")
            if p_tok is not None:
                _usage["prompt"] += p_tok
                _usage["completion"] += c_tok or 0
                print(f"[第 {step + 1} 轮] 输入 {p_tok} / 输出 {c_tok} tokens")

            # 把模型这一轮的输出原样记入历史,下一轮它才"记得"自己做过什么
            entry = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {"id": c.id, "type": "function",
                     "function": {"name": c.function.name, "arguments": c.function.arguments}}
                    for c in msg.tool_calls
                ]
            messages.append(entry)
            log({"event": "assistant", "step": step + 1, "content": entry["content"],
                 "tool_calls": entry.get("tool_calls", []), "usage": usage})

            # 没有工具调用 = 模型认为做完了
            if not msg.tool_calls:
                print(f"\n[完成,用了 {step + 1} 轮]\n{msg.content}")
                return

            # 执行每个工具调用,结果以 role=tool 放回消息(每个 tool_call 都必须有回应)
            for c in msg.tool_calls:
                try:
                    command = json.loads(c.function.arguments)["command"]
                    if not isinstance(command, str):
                        raise TypeError("command must be a string")
                except (ValueError, KeyError, TypeError) as e:
                    # 模型给出的参数格式不对:也是一条观察,告诉它,让它重来
                    output = (f"error: invalid tool arguments ({type(e).__name__}); "
                              'expected JSON like {"command": "ls"}')
                    log({"event": "tool_error", "step": step + 1,
                         "arguments": c.function.arguments, "output": output})
                    print(f"! 工具参数格式错误,已回传给模型: {str(c.function.arguments)[:100]}")
                    messages.append({"role": "tool", "tool_call_id": c.id, "content": output})
                    continue

                print(f"$ {command}")
                approved = ask_permission(command)
                _executed.append((command, approved))                      # 先记后做 ①
                log({"event": "tool_call", "step": step + 1,               # 先记后做 ②
                     "command": command, "approved": approved})
                if approved:
                    output = run_bash(command)
                else:
                    # 被拒绝也要作为"观察"回传,模型才知道要换条路,而不是卡住
                    output = ("denied: the user refused to run this command. "
                              "Do not retry it; try a different approach or explain.")
                model_output = clip(output)   # 模型看到的(可能被截断)
                log({"event": "tool_result", "step": step + 1,
                     "output": output,                           # 日志里是完整输出
                     "sent_to_model_chars": len(model_output)})

                shown = output[:500]
                if len(output) > 500:
                    note = f"完整输出共 {len(output)} 字符,见日志"
                    if len(model_output) < len(output):
                        note += f";回传给模型时已截断为 {len(model_output)} 字符"
                    shown += f"\n…(屏幕只显示前 500 字符;{note})"
                print(shown)
                messages.append({"role": "tool", "tool_call_id": c.id, "content": model_output})

        print(f"\n[达到 {MAX_STEPS} 轮上限,强制停止]")
    finally:
        print_report()   # 正常结束、达到上限、Ctrl+C、意外崩溃都会打印


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('用法: python harness.py "你的任务"')
    main(" ".join(sys.argv[1:]))