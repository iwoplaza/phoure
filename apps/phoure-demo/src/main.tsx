import './style.css';

import React from 'react';
import ReactDOM from 'react-dom/client';

import App from './App.tsx';

// biome-ignore lint/style/noNonNullAssertion: <it's okay, we're sure that the element exists>
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
