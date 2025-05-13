
from BetterKOS.core import BetterKOS
from BetterKOS.app import run

class App(BetterKOS):
    async def update(self):
        if self.frame%100==0: print(f'[FRAME {self.frame}] update before')
    async def update_after(self):
        if self.frame%100==0: print(f'[FRAME {self.frame}] update after')

if __name__ == '__main__':
    run(App)