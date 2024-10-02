import { useCallback, useEffect } from "react";
import Mousetrap from "mousetrap";
import ButtonHalo from "../atoms/ButtonHalo";
import {
  AiOutlineArrowRight,
  AiOutlineSearch,
  AiOutlineUpload,
} from "react-icons/ai";
import IconButton from "../atoms/IconButton";
import { useState } from "react";
import { GiBrain } from "react-icons/gi";

// Main functional component
import PropTypes from "prop-types";

export default function Bar({ isDarkMode }) {
  // State to keep track of dark mode, initialized from localStorage

  // State to keep track of Capture Mode, initialized from localStorage
  const [isCaptureMode, setIsCaptureMode] = useState(() => {
    const savedMode = localStorage.getItem("isCaptureMode");
    return savedMode !== null ? JSON.parse(savedMode) : false;
  });

  // State to keep track of input focus
  const [isFocused, setIsFocused] = useState(false);

  // Function to toggle Capture Mode and save it to localStorage
  const onChangeMode = useCallback(() => {
    const newMode = !isCaptureMode;
    setIsCaptureMode(newMode);
    localStorage.setItem("isCaptureMode", JSON.stringify(newMode));
  }, [isCaptureMode]);

  // Function to handle file upload
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      console.log(file);
    }
  };

  // Function to focus the input element
  const focusInput = () => {
    document.querySelector("#search").focus();
  };

  // useEffect to bind and unbind keyboard shortcuts
  useEffect(() => {
    Mousetrap.bind("ctrl+m", onChangeMode);
    Mousetrap.bind("ctrl+space", focusInput);

    return () => {
      Mousetrap.unbind("ctrl+m");
      Mousetrap.unbind("ctrl+space");
    };
  }, [isCaptureMode, onChangeMode]);

  return (
    <>
      <section
        className={`w-[40%] flex justify-center py-2 px-4 bg-slate-300 rounded-3xl dark:bg-gray-800 gap-2 transition-[shadow,transform] duration-300 ${
          isFocused
            ? "shadow-lg shadow-purple-500/50 transform -translate-y-2 transition-[shadow,transform] duration-300"
            : "transition-[shadow,transform] duration-300"
        }`}
      >
        <ButtonHalo onChangeMode={onChangeMode}>
          {/* Change icon of button depending on Capture Mode */}
          {isCaptureMode ? (
            <GiBrain
              size={25}
              className="text-white transition-transform duration-300"
            />
          ) : (
            <AiOutlineSearch
              size={25}
              className="text-white transition-transform duration-300"
            />
          )}
        </ButtonHalo>

        {/* Change text input placeholder depending on Capture Mode */}
        <input
          id="search"
          className="w-[80%] py-4 px-2 bg-slate-300 text-black dark:text-white dark:bg-gray-800 rounded-2xl text-lg focus:outline-none focus:ring-0 transition-none duration-300 autofill:bg-slate-300 dark:autofill:bg-gray-800"
          placeholder={isCaptureMode ? "Capture your thoughts..." : "Search..."}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          style={{
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: `${isDarkMode ? "black" : "white"}`,
            transition: "background-color 5000s ease-in-out 0s",
          }}
        />

        {/* Render different icons depending on Capture Mode */}
        {isCaptureMode ? (
          <>
            <IconButton
              onClick={() => document.getElementById("fileInput").click()}
            >
              <AiOutlineUpload
                size={25}
                className="text-white transition-transform duration-300"
              />
            </IconButton>
            <input
              type="file"
              id="fileInput"
              style={{ display: "none" }}
              onChange={handleFileUpload}
            />
            <IconButton onClick={() => {}}>
              <AiOutlineArrowRight
                size={25}
                className="text-white transition-transform duration-300"
              />
            </IconButton>
          </>
        ) : (
          <IconButton onClick={() => {}}>
            <AiOutlineArrowRight
              size={25}
              className="text-white transition-transform duration-300"
            />
          </IconButton>
        )}
      </section>
    </>
  );
}

Bar.propTypes = {
  isDarkMode: PropTypes.bool.isRequired,
};
