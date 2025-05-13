import asyncio, json

async def main(App):
    # 读取config.json文件
    config = {}
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    async with App(config['robot_ip'], config['robot_port']) as kos:
        try:
            await kos.load_session(config['model_file'])
            input('press to continue')
            await kos.loop()
        except:
            pass
        await kos.reset()

def run(App):
    asyncio.run(main(App))