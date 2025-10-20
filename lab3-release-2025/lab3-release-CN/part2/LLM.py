import re
from Maze import Maze
from openai import OpenAI


# TODO: Replace this with your own prompt.
your_prompt = """
基本规则：
你控制吃豆人（P）在迷宫中移动
迷宫由数字组成：0=空地，1=墙，2=豆子
鬼魂（G）固定不动，碰到就输
撞墙也输
移动方式：
每次只能选一个方向移动一步
方向对应数字：
0=上（y-1）
1=下（y+1）
2=左（x-1）
3=右（x+1）
胜负条件：
赢：吃完所有豆子
输：撞墙/撞鬼/超90步
计分规则：
50步内吃完：100分
51-90步吃完：60-99分
超90步：0分
决策建议：
优先吃最近的豆子
避开鬼魂（虽然它不动）
少走回头路
目标：运用启发算法或者其他有效算法经过科学合理计算后再选择，一定要在50步内吃完所有豆子并且不撞墙不撞鬼
"""

# Don't change this part.
output_format = """
输出必须是0-3的整数，上=0，下=1，左=2，右=3。
*重点*：(5,5)的上方是(4,5)，下方是(6,5)，左方是(5,4)，右方是(5,6)。
输出格式为：
“分析：XXXX。
动作：0（一个数字，不能出现其他数字）。”
"""

prompt = your_prompt + output_format


def get_game_state(maze: Maze, places: list, available: list) -> str:
    """
    Convert game state to natural language description.
    """
    description = ""
    for i in range(maze.height):
        for j in range(maze.width):
            description += f"({i},{j})="
            if maze.grid[i, j] == 0:
                description += f"空地"
            elif maze.grid[i, j] == 1:
                description += "墙壁"
            else:
                description += "豆子"
            description += ","
        description += "\n"
    places_str = ','.join(map(str, places))
    available_str = ','.join(map(str, available))
    state = f"""当前游戏状态（坐标均以0开始）：\n1、迷宫布局（0=空地,1=墙,2=豆子）：\n{description}\n2、吃豆人位置：{maze.pacman_pos[4]}\n3、鬼魂位置：{maze.pacman_pos[3]}\n4、曾经走过的位置：{places_str}\n5、可用方向：{available_str}\n"""
    return state


def get_ai_move(client: OpenAI, model_name: str, maze: Maze, file, places: list, available: list) -> int:
    """
    Get the move from the AI model.
    :param client: OpenAI client instance.
    :param model_name: Name of the AI model.
    :param maze: The maze object.
    :param file: The log file to write the output.
    :param places: The list of previous positions.
    :param available: The list of available directions.
    :return: The direction chosen by the AI.
    """
    state = get_game_state(maze, places, available)

    file.write("________________________________________________________\n")
    file.write(f"message:\n{state}\n")
    print("________________________________________________________")
    print(f"message:\n{state}")

    print("Waiting for AI response...")
    all_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": state
            }
        ],
        stream=False,
        temperature=.0
    )
    info = all_response.choices[0].message.content

    file.write(f"AI response:\n{info}\n")
    print(f"AI response:\n{info}")
    numbers = re.findall(r'\d+', info)
    choice = numbers[-1]
    return int(choice), info
