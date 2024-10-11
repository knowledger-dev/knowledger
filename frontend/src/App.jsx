// import CaptureBar from "./components/CaptureBar";
import { ToastContainer } from "react-toastify"; // Import Toastify container
// import SearchBar from "./components/SearchBar";
import CaptureBar from "./components/CaptureBar";

import { HashRouter, Routes, Route, Navigate } from "react-router-dom";

import Search from "./pages/Search";
import NavBar from "./molecules/NavBar";
import { useAuthContext } from "./hooks/useAuthContext";
import Login from "./pages/Login";
import Signup from "./pages/Signup";

function App() {
  const { user } = useAuthContext();

  return (
    <HashRouter>
      <main className="flex justify-center gap-0 bg-white dark:bg-black">
        <NavBar />
        <Routes>
          <Route
            path="/"
            element={
              user ? (
                <section className="w-full h-screen flex justify-center items-center">
                  <Search />
                </section>
              ) : (
                <Navigate to="/login" />
              )
            } // renavigates to a route if there is no user
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
            path="/login"
            element={
              !user ? (
                <section className="w-full h-screen flex justify-center items-center">
                  <Login />
                </section>
              ) : (
                <Navigate to="/" />
              )
            }
          />
          <Route
            path="/signup"
            element={
              !user ? (
                <section className="w-full h-screen flex justify-center items-center">
                  <Signup />
                </section>
              ) : (
                <Navigate to="/" />
              )
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
      </main>
    </HashRouter>
  );
}

export default App;
