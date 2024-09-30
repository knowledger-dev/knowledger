import React from 'react';
import CaptureBar from './components/CaptureBar';
import { ToastContainer } from 'react-toastify';  // Import Toastify container

function App() {
  return (
    <div className="h-screen flex justify-center items-center bg-gray-100">
      <CaptureBar />
      <ToastContainer />  {/* Render the toast container for notifications */}
    </div>
  );
}

export default App;
