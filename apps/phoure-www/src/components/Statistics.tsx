import FPSCounter from './FpsCounter.tsx';

export function Statistics() {
  return (
    <div className='bg-slate-700 rounded-t-lg px-2 py-1'>
      <FPSCounter />
    </div>
  );
}
