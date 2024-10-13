import { useRef, useEffect } from "react";
import PropTypes from "prop-types";

const FocusableInput = ({
  currentMode,
  setIsInputFocused,
  isDarkMode,
  setSearch,
  onKeyDown,
  isInputFocused,
  search,
}) => {
  const inputRef = useRef(null);
  const isElectron = typeof window !== "undefined" && window.require;

  useEffect(() => {
    if (isElectron) {
      const { ipcRenderer } = window.require("electron");
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
    }
  }, [isElectron]);

  useEffect(() => {
    if (isInputFocused) {
      inputRef.current.focus();
    } else {
      inputRef.current.blur(); // programmatically blur the input
    }
  }, [isInputFocused]);

  // on first render, focus the input
  useEffect(() => {
    setIsInputFocused(true);
  }, []);

  const getPlaceholderText = () => {
    if (!isInputFocused) {
      switch (currentMode) {
        case "capture":
          return "Capture your thoughts...";
        case "search":
          return "Search...";
        case "searchai":
          return "Search with AI...";
        default:
          return "";
      }
    }
    return "";
  };

  return (
    <>
      {/* TODO: Format of rows doesn't revert after submitting a multi-lined query or input */}
      <textarea
        ref={inputRef}
        id="search"
        rows="1"
        className={`w-[100%] py-4 px-2 bg-slate-300 text-black dark:text-white dark:bg-gray-800 rounded-2xl text-lg focus:outline-none focus:ring-0 transition-none duration-300 autofill:bg-slate-300 dark:autofill:bg-gray-800 max-md:text-sm break-words resize-none`}
        placeholder={getPlaceholderText()}
        onFocus={() => setIsInputFocused(true)}
        onBlur={() => setIsInputFocused(false)}
        value={search}
        style={{
          color: `${
            isDarkMode
              ? search.length > 0
                ? "white"
                : "gray"
              : search.length > 0
              ? "black"
              : "gray"
          }`,
          wordWrap: "break-word",
          overflow: "hidden",
          maxHeight: "5em", // Set the maximum height to 5 lines
          overflowY: "auto", // Enable vertical scrolling
        }}
        onKeyDown={onKeyDown}
        onChange={(e) => {
          setSearch(e.target.value);
          e.target.style.height = "auto";
          e.target.style.height = `${e.target.scrollHeight}px`;
        }}
      />
    </>
  );
};

FocusableInput.propTypes = {
  currentMode: PropTypes.oneOf(["capture", "search", "searchai"]).isRequired,
  isDarkMode: PropTypes.bool.isRequired,
  setSearch: PropTypes.func.isRequired,
  onKeyDown: PropTypes.func.isRequired,
  setIsInputFocused: PropTypes.func.isRequired,
  isInputFocused: PropTypes.bool.isRequired,
  search: PropTypes.string.isRequired,
};

export default FocusableInput;
