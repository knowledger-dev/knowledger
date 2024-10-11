import { useRef, useEffect, useState } from "react";
import Bar from "../molecules/Bar";
import { AiOutlineSun, AiOutlineMoon } from "react-icons/ai";
import IconButton from "../atoms/IconButton";
import Graph from "../molecules/Graph";

export default function Search() {
  const [focusedNode, setFocusedNode] = useState(null);
  const [data, setData] = useState({
    nodes: [],
    links: [],
  });

  const isDarkModeRef = useRef(
    localStorage.theme === "dark" || !("theme" in localStorage)
  );

  const [isDarkMode, setIsDarkMode] = useState(isDarkModeRef.current);

  useEffect(() => {
    if (isDarkModeRef.current) {
      document.documentElement.classList.add("dark");
      setIsDarkMode(true);
    } else {
      document.documentElement.classList.remove("dark");
      setIsDarkMode(false);
    }
  }, [isDarkModeRef]);

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

  const handleChangeData = (newData) => {
    setData({
      nodes: newData.nodes,
      links: newData.links,
    });
  };

  const [isInputFocused, setIsInputFocused] = useState(false);

  const [currentMode, setCurrentMode] = useState(() => {
    const savedMode = localStorage.getItem("currentMode");
    return ["capture", "search", "searchai"].includes(savedMode)
      ? savedMode
      : "search"; // Default to "search" mode if not valid
  });

  useEffect(() => {
    localStorage.setItem("currentMode", currentMode);
  }, [currentMode]);

  return (
    <>
      <div className="fixed w-full z-0 inset-2">
        {data.nodes.length !== 0 && (
          <>
            <Graph
              data={data}
              isDarkMode={isDarkMode}
              focusedNode={focusedNode}
              setFocusedNode={setFocusedNode}
            />
          </>
        )}
      </div>
      <IconButton onClick={handleClick} className="absolute right-0 p-5 top-0">
        {isDarkMode ? (
          <AiOutlineSun size={25} className="text-white" />
        ) : (
          <AiOutlineMoon size={25} className="text-white" />
        )}
      </IconButton>
      <section
        className={`font-inter font-semibold w-full fixed z-10 transition-transform duration-500 ${
          data.nodes.length !== 0
            ? "bottom-[5%]"
            : "top-1/2 transform -translate-y-1/2"
        }`}
      >
        <div className="flex flex-col justify-center items-center">
          <h1
            className={`text-5xl p-10 text-black dark:text-white text-center max-md:text-xl transition-opacity duration-300 ${
              isInputFocused ? "opacity-100" : "opacity-30"
            }`}
          >
            {currentMode === "capture"
              ? "Capture your thoughts"
              : currentMode === "search"
              ? "Search your notes"
              : currentMode === "searchai"
              ? "Ask your agent"
              : "Mode is currently undetermined"}
          </h1>

          <Bar
            currentMode={currentMode}
            setCurrentMode={setCurrentMode}
            isDarkMode={isDarkMode}
            handleChangeData={handleChangeData}
            setFocusedNode={setFocusedNode}
            setIsInputFocused={setIsInputFocused}
            isInputFocused={isInputFocused}
          />
        </div>
      </section>
    </>
  );
}
