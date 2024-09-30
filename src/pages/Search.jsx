import { useRef, useEffect, useState } from "react";
import Bar from "../molecules/Bar";
import { AiOutlineSun, AiOutlineMoon } from "react-icons/ai";
import IconButton from "../atoms/IconButton";

export default function Search() {
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

  return (
    <section className="font-inter font-semibold">
      <IconButton onClick={handleClick} className="absolute right-0 p-5">
        {isDarkMode ? (
          <AiOutlineSun size={25} className="text-white" />
        ) : (
          <AiOutlineMoon size={25} className="text-white" />
        )}
      </IconButton>
      <div className="flex flex-col justify-center items-center w-[100%] h-screen bg-white dark:bg-black">
        <h1 className="text-5xl p-10 text-black dark:text-white">
          Hi, what do you want to know?
        </h1>
        <Bar />
      </div>
    </section>
  );
}
