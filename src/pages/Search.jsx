import { useRef, useEffect, useState } from "react";
import Bar from "../molecules/Bar";
import { AiOutlineSun, AiOutlineMoon } from "react-icons/ai";
import IconButton from "../atoms/IconButton";
import Graph from "../molecules/Graph";

export default function Search() {
  const [focusedNode, setFocusedNode] = useState(null);
  const [data, setData] = useState({
    nodes: [{ id: 1, name: "OG", val: 1 }],
    links: [],
  });

  const isDarkModeRef = useRef(
    localStorage.theme === "dark" || !("theme" in localStorage)
  );

  const [isDarkMode, setIsDarkMode] = useState(isDarkModeRef.current);

  useEffect(() => {
    if (isDarkModeRef.current) {
      document.documentElement.classList.add("dark");
      setIsDarkMode(false);
    } else {
      document.documentElement.classList.remove("dark");
      setIsDarkMode(true);
    }
  }, [isDarkModeRef]);

  function handleClick() {
    if (isDarkModeRef.current) {
      document.documentElement.classList.remove("dark");
      localStorage.theme = "light";
      setIsDarkMode(true);
    } else {
      document.documentElement.classList.add("dark");
      localStorage.theme = "dark";
      setIsDarkMode(false);
    }

    isDarkModeRef.current = !isDarkModeRef.current;
  }

  const handleChangeData = (newData) => {
    setData((prevData) => ({
      nodes: [...prevData.nodes, ...newData.nodes],
      links: [...prevData.links, ...newData.links],
    }));
  };

  const [isInputFocused, setIsInputFocused] = useState(false);

  const [isCaptureMode, setIsCaptureMode] = useState(() => {
    const savedMode = localStorage.getItem("isCaptureMode");
    return savedMode !== null ? JSON.parse(savedMode) : false;
  });

  return (
    <>
      <div className="fixed w-full z-0 inset-2">
        <Graph
          data={data}
          isDarkMode={isDarkMode}
          focusedNode={focusedNode}
          setFocusedNode={setFocusedNode}
        />
      </div>
      <IconButton onClick={handleClick} className="absolute right-0 p-5 top-0">
        {isDarkMode ? (
          <AiOutlineSun size={25} className="text-white" />
        ) : (
          <AiOutlineMoon size={25} className="text-white" />
        )}
      </IconButton>
      <section className="font-inter font-semibold w-full relative z-10">
        <div className="flex flex-col justify-center items-center">
          <h1
            className={`text-5xl p-10 text-black dark:text-white text-center max-md:text-xl transition-opacity duration-300 ${
              isInputFocused ? "opacity-100" : "opacity-30"
            }`}
          >
            {isCaptureMode ? "Capture your thoughts" : "Ask your agent"}
          </h1>

          <Bar
            isCaptureMode={isCaptureMode}
            setIsCaptureMode={setIsCaptureMode}
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
