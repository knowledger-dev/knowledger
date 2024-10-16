// import CaptureBar from "./components/CaptureBar";
import { ToastContainer } from "react-toastify"; // Import Toastify container
// import SearchBar from "./components/SearchBar";
import CaptureBar from "./components/CaptureBar";

import { HashRouter, Routes, Route, Navigate } from "react-router-dom";

import NavBar from "./molecules/NavBar";
import { useAuthContext } from "./hooks/useAuthContext";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import Capture from "./pages/Capture";
import { useEffect, useRef, useState } from "react";
import IconButton from "./atoms/IconButton";
import { AiOutlineMoon, AiOutlineSun } from "react-icons/ai";
import Bar from "./molecules/Bar";
import GraphPage from "./pages/GraphPage";

function App() {
  // User
  const { user } = useAuthContext();

  // Dark Mode References
  const isDarkModeRef = useRef(
    localStorage.theme === "dark" || !("theme" in localStorage)
  );

  const [isDarkMode, setIsDarkMode] = useState(isDarkModeRef.current);

  function handleClick() {
    if (isDarkModeRef.current) {
      document.documentElement.classList.remove("dark");
      localStorage.theme = "light";
      setIsDarkMode(false);
    } else {
      document.documentElement.classList.add("dark");
      localStorage.theme = "dark";
      setIsDarkMode(true);
    }

    isDarkModeRef.current = !isDarkModeRef.current;
  }

  useEffect(() => {
    if (isDarkModeRef.current) {
      document.documentElement.classList.add("dark");
      setIsDarkMode(true);
    } else {
      document.documentElement.classList.remove("dark");
      setIsDarkMode(false);
    }
  }, [isDarkModeRef]);

  // Data Handling
  const [data, setData] = useState({
    nodes: [],
    links: [],
  });

  const handleChangeData = (newData) => {
    setData({
      nodes: newData.nodes,
      links: newData.links,
    });
  };

  // Search Bar

  const [isInputFocused, setIsInputFocused] = useState(false);
  const [isPaletteOpen, setIsPaletteOpen] = useState(false);
  const [isBarOpen, setIsBarOpen] = useState(false);
  const [focusedNode, setFocusedNode] = useState(null);

  const setClickOutsideListener = (handler) => {
    document.addEventListener("mousedown", handler);
  };

  const removeClickOutsideListener = (handler) => {
    document.removeEventListener("mousedown", handler);
  };

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (isBarOpen && !event.target.closest("#bar")) {
        setIsBarOpen(false);
      }
    };

    if (isBarOpen) {
      setClickOutsideListener(handleClickOutside);
    } else {
      removeClickOutsideListener(handleClickOutside);
    }

    return () => {
      removeClickOutsideListener(handleClickOutside);
    };
  }, [isBarOpen]);

  const [currentMode, setCurrentMode] = useState(() => {
    const savedMode = localStorage.getItem("currentMode");
    return ["capture", "search", "searchai"].includes(savedMode)
      ? savedMode
      : "search"; // Default to "search" mode if not valid
  });

  // Routes
  return (
    <HashRouter>
      <main className="flex justify-between gap-0 bg-white dark:bg-black">
        <NavBar />
        <IconButton
          onClick={handleClick}
          className="absolute right-0 p-5 top-0 z-50"
        >
          {isDarkMode ? (
            <AiOutlineSun size={25} className="text-white" />
          ) : (
            <AiOutlineMoon size={25} className="text-white" />
          )}
        </IconButton>

        <Routes>
          <Route
            path="/capture"
            element={
              user ? (
                <section className="w-full h-screen flex flex-col justify-center items-center relative">
                  <Bar
                    isPaletteOpen={isPaletteOpen}
                    setIsPaletteOpen={setIsPaletteOpen}
                    isBarOpen={isBarOpen}
                    setIsBarOpen={setIsBarOpen}
                    removeClickOutsideListener={removeClickOutsideListener}
                    currentMode={currentMode}
                    setCurrentMode={setCurrentMode}
                    isDarkMode={isDarkMode}
                    handleChangeData={handleChangeData}
                    setFocusedNode={setFocusedNode}
                    setIsInputFocused={setIsInputFocused}
                    isInputFocused={isInputFocused}
                  />
                  <Capture />
                </section>
              ) : (
                <Navigate to="/login" />
              )
            } // renavigates to a route if there is no user
          />

          <Route
            path="/"
            element={
              user ? (
                <section className="w-full h-screen flex justify-center items-center relative">
                  <Bar
                    isPaletteOpen={isPaletteOpen}
                    setIsPaletteOpen={setIsPaletteOpen}
                    isBarOpen={isBarOpen}
                    setIsBarOpen={setIsBarOpen}
                    removeClickOutsideListener={removeClickOutsideListener}
                    currentMode={currentMode}
                    setCurrentMode={setCurrentMode}
                    isDarkMode={isDarkMode}
                    handleChangeData={handleChangeData}
                    setFocusedNode={setFocusedNode}
                    setIsInputFocused={setIsInputFocused}
                    isInputFocused={isInputFocused}
                  />

                  <GraphPage
                    isDarkMode={isDarkMode}
                    data={data}
                    handleChangeData={handleChangeData}
                    currentMode={currentMode}
                    setIsPaletteOpen={setIsPaletteOpen}
                    setIsBarOpen={setIsBarOpen}
                    setIsInputFocused={setIsInputFocused}
                    focusedNode={focusedNode}
                    setFocusedNode={setFocusedNode}
                  />
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
                <section className="w-full h-screen flex justify-center items-center relative">
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
                <section className="w-full h-screen flex justify-center items-center relative">
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
              <div className="h-screen flex justify-center items-center bg-gray-100 w-full relative">
                <h1 className="text-3xl font-bold">404 - Not Found</h1>
              </div>
            }
          />
        </Routes>
        <ToastContainer
          position="top-right"
          className="z-50"
          autoClose={5000}
          hideProgressBar={false}
          newestOnTop={false}
          closeOnClick
          rtl={false}
          pauseOnFocusLoss
          draggable
          pauseOnHover
        />
      </main>
    </HashRouter>
  );
}

export default App;
