import { Provider } from 'jotai';
import { HeartIcon } from 'lucide-react';

import { store } from 'src/lib/store.ts';
import { SidebarProvider, SidebarTrigger } from 'src/components/ui/sidebar.tsx';
import GameViewport from 'src/lib/GameEngine/GameViewport.tsx';
import { ControlsSidebar } from './ControlsSidebar.tsx';
import { Statistics } from './Statistics.tsx';
import { NavAside } from './NavAside.tsx';

export function DemoApp() {
  return (
    <SidebarProvider defaultOpen>
      <Provider store={store}>
        <ControlsSidebar />
        <main className="relative flex-1 flex bg-slate-200 min-h-screen justify-center items-center">
          <header className="absolute top-2 inset-x-0 h-10 flex justify-center items-center">
            <img
              className="h-8"
              src="/phoure/phoure-logo-light.svg"
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
