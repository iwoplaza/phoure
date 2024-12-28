import { Provider } from 'jotai';
import { HeartIcon } from 'lucide-react';

import { SidebarProvider, SidebarTrigger } from '@/components/ui/sidebar';
import { ControlsSidebar } from '@/components/ControlsSidebar';
import { Statistics } from '@/components/Statistics';
import { NavAside } from '@/components/NavAside';
import GameViewport from './GameEngine/GameViewport';
import { store } from './store';

function App() {
  return (
    <SidebarProvider defaultOpen>
      <Provider store={store}>
        <ControlsSidebar />
        <main className="relative flex-1 flex bg-slate-200 min-h-screen justify-center items-center">
          <header className="absolute top-2 inset-x-0 h-10 flex justify-center items-center">
            <img
              className="h-8"
              src="/phoure-logo-light.svg"
              alt="phoure logo"
            />
            <h1 className="hidden">phoure</h1>
          </header>
          <div className="absolute top-2 left-2">
            <SidebarTrigger />
          </div>
          <div>
            <Statistics />
            <GameViewport />
          </div>
        </main>
        <NavAside />
        <footer className="absolute bottom-2 right-2 text-sm font-poppins">
          <p className="whitespace-nowrap text-nowrap">
            Made with <HeartIcon className="w-4 h-4 inline-block" /> by Iwo
            Plaza
          </p>
        </footer>
      </Provider>
    </SidebarProvider>
  );
}

export default App;
