import { useRef, useEffect } from "react";
const { ipcRenderer } = window.require("electron");
import PropTypes from "prop-types";

const FocusableInput = ({
  isCaptureMode,
  setIsInputFocused,
  isDarkMode,
  setSearch,
  onKeyDown,
  isInputFocused,
  search,
}) => {
  const inputRef = useRef(null);

  useEffect(() => {
    // Listen for the "focus-input" event from the Electron main process
    ipcRenderer.on("focus-input", () => {
      if (inputRef.current) {
        if (document.activeElement === inputRef.current) {
          inputRef.current.blur(); // Unfocus the input if it is already focused
        } else {
          inputRef.current.focus(); // Focus the input when the event is received
        }
      }
    });

    // Cleanup the event listener when the component unmounts
    return () => {
      ipcRenderer.removeAllListeners("focus-input");
    };
  }, []);

  useEffect(() => {
    if (isInputFocused) {
      inputRef.current.focus();
    } else {
      inputRef.current.blur(); // programmatically blur the input
    }
  }, [isInputFocused]);

  return (
    <>
      <input
        ref={inputRef}
        id="search"
        className={`w-[80%] py-4 px-2 bg-slate-300 text-black dark:text-white dark:bg-gray-800 rounded-2xl text-lg focus:outline-none focus:ring-0 transition-none duration-300 autofill:bg-slate-300 dark:autofill:bg-gray-800 max-md:text-sm`}
        placeholder={isCaptureMode ? "Capture your thoughts..." : "Search..."}
        onFocus={() => setIsInputFocused(true)}
        onBlur={() => setIsInputFocused(false)}
        value={search}
        style={{
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: `${
            isDarkMode
              ? search.length > 0
                ? "black"
                : "gray"
              : search.length > 0
              ? "white"
              : "gray"
          }`,
          transition: "background-color 5000s ease-in-out 0s",
        }}
        onKeyDown={onKeyDown}
        onChange={(e) => setSearch(e.target.value)}
      />
    </>
  );
};

FocusableInput.propTypes = {
  isCaptureMode: PropTypes.bool.isRequired,
  isDarkMode: PropTypes.bool.isRequired,
  setSearch: PropTypes.func.isRequired,
  onKeyDown: PropTypes.func.isRequired,
  setIsInputFocused: PropTypes.func.isRequired,
  isInputFocused: PropTypes.bool.isRequired,
  search: PropTypes.string.isRequired,
};

export default FocusableInput;
