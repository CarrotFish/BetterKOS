import sys
sys.path.append('.')

from BetterKOS.core import BetterKOS
from BetterKOS.app import run

class App(BetterKOS):
    async def update(self):
        if self.frame%100==0: print(f'[FRAME {self.frame}] update before')
        # 设置移动命令
        self.move_commands[0] = 0
        self.move_commands[1] = 0.1
        self.move_commands[2] = 0
    async def update_after(self):
        if self.frame%100==0: print(f'[FRAME {self.frame}] update after')

if __name__ == '__main__':
    run(App)