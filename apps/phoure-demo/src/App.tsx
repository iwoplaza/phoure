import { Provider } from 'jotai';

import { ControlsSidebar } from '@/components/ControlsSidebar';
import { Statistics } from '@/components/Statistics';
import { NavAside } from '@/components/NavAside';
import GameViewport from './GameEngine/GameViewport';
import { store } from './store';

function App() {
  return (
    <Provider store={store}>
      <div className="flex bg-slate-200 min-h-screen">
        <ControlsSidebar />
        <NavAside />
        <div className="flex-1 flex justify-center items-center">
          <div>
            <Statistics />
            <GameViewport />
          </div>
        </div>
      </div>
    </Provider>
  );
}

export default App;
