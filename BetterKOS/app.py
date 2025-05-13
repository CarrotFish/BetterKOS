import asyncio, json
from typing import Type
from BetterKOS.core import BetterKOS

async def main(App: Type[BetterKOS]):
    # 读取config.json文件
    config = {}
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    async with App(config['robot_ip'], config['robot_port']) as kos:
        try:
            await kos.load_session(config['model_file'])
            input('回车以初始化imu')
            await kos.init_imu()
            input('回车以开始循环')
            await kos.loop()
        except:
            pass
        await kos.reset()

def run(App: Type[BetterKOS]):
    """运行程序，传入BetterKOS的子类本身（不需要实例化）"""
    asyncio.run(main(App))