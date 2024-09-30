// import CaptureBar from "./components/CaptureBar";
import { ToastContainer } from "react-toastify"; // Import Toastify container
// import SearchBar from "./components/SearchBar";
import CaptureBar from "./components/CaptureBar";

import { BrowserRouter, Route, Routes } from "react-router-dom";
import Search from "./pages/Search";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route
          path="/"
          element={
            <div>
              <Search />
              <ToastContainer />
              {/* Render the toast container for notifications */}
            </div>
          }
        />
        <Route
          path="/search-og"
          element={
            <div className="h-screen flex justify-center items-center bg-gray-100">
              <CaptureBar />
              <ToastContainer />
              {/* Render the toast container for notifications */}
            </div>
          }
        />
        <Route
          path="*"
          element={
            <div className="h-screen flex justify-center items-center bg-gray-100">
              <h1 className="text-3xl font-bold">404 - Not Found</h1>
            </div>
          }
        />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
