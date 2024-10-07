// import CaptureBar from "./components/CaptureBar";
import { ToastContainer } from "react-toastify"; // Import Toastify container
// import SearchBar from "./components/SearchBar";
import CaptureBar from "./components/CaptureBar";

import { BrowserRouter, Route, Routes } from "react-router-dom";
import Search from "./pages/Search";
import NavBar from "./molecules/NavBar";

function App() {
  return (
    <main className="flex justify-center gap-0 bg-white dark:bg-black">
      <NavBar />
      <BrowserRouter>
        <Routes>
          <Route
            path="/"
            element={
              <section className="w-full h-screen flex justify-center items-center">
                <Search />
              </section>
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
              <div className="h-screen flex justify-center items-center bg-gray-100 w-full">
                <h1 className="text-3xl font-bold">404 - Not Found</h1>
              </div>
            }
          />
        </Routes>
      </BrowserRouter>
    </main>
  );
}

export default App;
