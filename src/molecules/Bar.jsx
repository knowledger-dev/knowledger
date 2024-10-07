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
import FocusableInput from "../atoms/FocusableInput";

export default function Bar({
  isDarkMode,
  handleChangeData,
  setFocusedNode,
  isInputFocused,
  setIsInputFocused,
  isCaptureMode,
  setIsCaptureMode,
}) {
  const [search, setSearch] = useState("");
  const [num, setNum] = useState(2);
  // State to keep track of dark mode, initialized from localStorage

  // State to keep track of Capture Mode, initialized from localStorage

  // Function to toggle Capture Mode and save it to localStorage
  const onChangeMode = useCallback(() => {
    const newMode = !isCaptureMode;
    setIsCaptureMode(newMode);
    localStorage.setItem("isCaptureMode", JSON.stringify(newMode));
  }, [isCaptureMode, setIsCaptureMode]);

  // Function to handle file upload
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      console.log(file);
    }
  };

  // useEffect to bind and unbind keyboard shortcuts
  useEffect(() => {
    Mousetrap.bindGlobal("ctrl+m", onChangeMode);

    return () => {
      Mousetrap.unbind("ctrl+m");
    };
  }, [isCaptureMode, onChangeMode]);

  const onSubmit = (event) => {
    if (isCaptureMode) {
      event.preventDefault();
      console.log(search);
      const randomValue = Math.floor(Math.random() * 12) + 2;
      handleChangeData({
        nodes: [{ id: num, name: search, val: randomValue }],
        links: [
          { source: num, target: num - 1 },
          { source: num, target: 1 },
        ],
      });
      setNum(num + 1);
      setSearch("");
    } else {
      setFocusedNode(search);
      setIsInputFocused(false);
      setSearch("");
    }
  };

  return (
    <>
      <section
        className={`max-xl:w-[80%] w-[40%] flex justify-center py-2 px-4 bg-slate-300 rounded-3xl dark:bg-gray-800 gap-2 transition-[shadow,transform] duration-300 ${
          isInputFocused
            ? "shadow-lg shadow-purple-500/50 transform -translate-y-2 transition-[shadow,transform] duration-300"
            : "transition-[shadow,transform] duration-300"
        }`}
      >
        <ButtonHalo onChangeMode={onChangeMode}>
          {isCaptureMode ? (
            <GiBrain
              size={window.innerWidth < 768 ? 20 : 25}
              className="text-white transition-transform duration-300"
            />
          ) : (
            <AiOutlineSearch
              size={window.innerWidth < 768 ? 20 : 25}
              className="text-white transition-transform duration-300"
            />
          )}
        </ButtonHalo>
        <FocusableInput
          isDarkMode={isDarkMode}
          setIsInputFocused={setIsInputFocused}
          isCaptureMode={isCaptureMode}
          setSearch={setSearch}
          onKeyDown={(e) => {
            if (e.key === "Enter" && isInputFocused) {
              onSubmit(e);
            }
          }}
          isInputFocused={isInputFocused}
          search={search}
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
            <IconButton
              onClick={(e) => {
                onSubmit(e);
              }}
            >
              <AiOutlineArrowRight
                size={25}
                className="text-white transition-transform duration-300"
              />
            </IconButton>
          </>
        ) : (
          <IconButton>
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
  handleChangeData: PropTypes.func.isRequired,
  setFocusedNode: PropTypes.func.isRequired,
  isInputFocused: PropTypes.bool.isRequired,
  setIsInputFocused: PropTypes.func.isRequired,
  setIsCaptureMode: PropTypes.func.isRequired,
  isCaptureMode: PropTypes.bool.isRequired,
};
